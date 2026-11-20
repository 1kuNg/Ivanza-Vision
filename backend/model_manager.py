"""Gerencia os modelos por dispositivo com troca CPU/CUDA em tempo de execução.

Uma única lock serializa análises, cadastros Few-Shot e a troca de dispositivo.
Trocar hardware invalida pesos e vetores de referência em cache, nunca as imagens.
"""
import gc
import subprocess
import threading

from vision_core import DinoExtractor, DinoV3Extractor, MODELS, SAM_NAME, Sam2Segmenter
from extra_models_core import EXTRA_ENGINES
from .session_store import LRU

EXTRA_NAMES = dict(zip(EXTRA_ENGINES, ('Grounding DINO Tiny', 'Depth Anything V2 Small', 'SigLIP 2 Base', 'RapidOCR')))
ALL_MODELS = {**MODELS, **{v: EXTRA_ENGINES[k][0] for k, v in EXTRA_NAMES.items()}}
EXTRA_KEYS = {v: k for k, v in EXTRA_NAMES.items()}


def nvidia_driver_info():
    """Best-effort: NVIDIA no Windows não prova que o PyTorch tem CUDA."""
    try:
        command = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if command.returncode == 0 and command.stdout.strip():
            first = command.stdout.strip().splitlines()[0].split(',', 1)
            return dict(name=first[0].strip(), driver=first[1].strip() if len(first) > 1 else '')
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


class ModelManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.models = {}
        self.features = {}
        self.reference_detections = LRU(128)
        self.device_preference = 'auto'
        self._nvidia_info = None
        self._nvidia_checked = False

    @staticmethod
    def _cuda_available():
        import torch
        return bool(torch.cuda.is_available())

    def effective_device(self):
        """Resolve o destino antes de carregar qualquer modelo novo."""
        if self.device_preference == 'cpu':
            return 'cpu'
        if self._cuda_available():
            return 'cuda'
        return 'cpu'

    def get(self, name, token=None):
        # Chamador mantém self.lock durante todas as inferências.
        if name not in self.models:
            device = self.effective_device()
            if name == 'DINOv2 Small':
                model = DinoExtractor(device=device)
            elif name == SAM_NAME:
                model = Sam2Segmenter(device=device)
            elif name in MODELS:
                model = DinoV3Extractor(device=device, token=token, model_id=MODELS[name])
            elif name == 'RapidOCR':
                # rapidocr-onnxruntime permanece em CPU neste projeto.
                model = EXTRA_ENGINES[EXTRA_KEYS[name]][1]()
            else:
                model = EXTRA_ENGINES[EXTRA_KEYS[name]][1](device=device)
            self.models[name] = model
        return self.models[name]

    def _unload(self):
        """Executar somente sob self.lock."""
        import torch
        self.models.clear()
        self.features.clear()
        self.reference_detections.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self):
        with self.lock:
            self._unload()

    def set_device(self, preference):
        """Troca entre CPU, CUDA e Auto sem reiniciar o servidor.

        A mudança espera a inferência em andamento encerrar e descarrega modelos.
        Não usa fallback silencioso quando CUDA foi selecionada explicitamente.
        """
        if preference not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Dispositivo inválido: escolha Auto, CPU ou GPU NVIDIA.')
        with self.lock:
            if preference == 'cuda' and not self._cuda_available():
                raise ValueError(
                    'CUDA não está disponível no PyTorch deste ambiente. '
                    'Abra Diagnóstico GPU; se necessário instale o PyTorch CUDA '
                    'com habilitar_cuda_windows.cmd e reinicie o Lab uma vez.'
                )
            before = self.effective_device()
            target = 'cpu' if preference == 'cpu' else 'cuda' if self._cuda_available() else 'cpu'
            if preference != self.device_preference or before != target:
                self._unload()
            self.device_preference = preference
        return self.status()

    def status(self):
        import torch

        available = self._cuda_available()
        if not self._nvidia_checked:
            self._nvidia_info = nvidia_driver_info()
            self._nvidia_checked = True
        built = bool(getattr(torch.backends.cuda, 'is_built', lambda: False)())
        cuda_version = getattr(torch.version, 'cuda', None)
        device = self.effective_device()
        name = None
        vram = None
        if available:
            try:
                name = torch.cuda.get_device_name(0)
                vram = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                            total=torch.cuda.get_device_properties(0).total_memory)
            except Exception:
                # Mesmo com um driver defeituoso, o diagnóstico deve abrir.
                available = False
                if device == 'cuda':
                    device = 'cpu'
        driver_info = self._nvidia_info
        if available:
            issue = ''
        elif not built or not cuda_version:
            issue = ('PyTorch instalado sem suporte CUDA (CPU-only). '
                     'Use o instalador CUDA incluído e reinicie o Lab após a instalação.')
        elif driver_info:
            issue = ('O PyTorch possui CUDA, mas não consegue acessar a NVIDIA. '
                     'Verifique driver, compatibilidade da GPU e versão CUDA instalada.')
        else:
            issue = ('CUDA não está disponível. Verifique se a placa NVIDIA aparece no '
                     'nvidia-smi, os drivers e se o PyTorch tem suporte CUDA.')
        return dict(
            device=device.upper(),
            device_preference=self.device_preference,
            cuda_available=available,
            torch_version=torch.__version__,
            torch_cuda_version=cuda_version,
            cuda_built=built,
            gpu_name=name or (driver_info['name'] if driver_info else None),
            nvidia_driver=driver_info['driver'] if driver_info else None,
            nvidia_detected=bool(driver_info) or available,
            cuda_issue=issue,
            loaded_models=list(self.models.copy()),
            model_devices={key: obj.device.upper() for key, obj in self.models.copy().items()},
            vram=vram,
        )
