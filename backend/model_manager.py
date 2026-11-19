"""Modelos lazy e fila única para inferência, cache e limpeza de VRAM."""
import gc
import threading

from vision_core import DinoExtractor, DinoV3Extractor, MODELS, SAM_NAME, Sam2Segmenter
from extra_models_core import EXTRA_ENGINES

EXTRA_NAMES = dict(zip(EXTRA_ENGINES, ('Grounding DINO Tiny', 'Depth Anything V2 Small', 'SigLIP 2 Base', 'RapidOCR')))
ALL_MODELS = {**MODELS, **{v: EXTRA_ENGINES[k][0] for k, v in EXTRA_NAMES.items()}}
EXTRA_KEYS = {v:k for k,v in EXTRA_NAMES.items()}


class ModelManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.models = {}
        self.features = {}

    def get(self, name, token=None):
        # O chamador mantém lock durante a carga e toda inferência.
        if name not in self.models:
            if name == 'DINOv2 Small':
                model = DinoExtractor()
            elif name == SAM_NAME:
                model = Sam2Segmenter()
            elif name in MODELS:
                model = DinoV3Extractor(token=token, model_id=MODELS[name])
            else:
                model = EXTRA_ENGINES[EXTRA_KEYS[name]][1]()
            self.models[name] = model
        return self.models[name]

    def clear(self):
        import torch
        with self.lock:
            self.models.clear()
            self.features.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def status(self):
        import torch
        cuda = torch.cuda.is_available()
        return dict(device='CUDA' if cuda else 'CPU', loaded_models=list(self.models.copy()),
                    vram=dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                              total=torch.cuda.get_device_properties(0).total_memory) if cuda else None)
