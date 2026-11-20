"""Smoke tests sem download de pesos ou dependências pesadas.

Executar: python ci/device_manager_smoke.py
Injeta apenas doubles mínimos para verificar o gerenciador de dispositivos.
"""
import importlib
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

state = {'cuda': False}
fake_torch = types.ModuleType('torch')
fake_torch.__version__ = '2.9.0+cpu'
fake_torch.version = types.SimpleNamespace(cuda=None)
fake_torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(is_built=lambda: state['cuda']))
fake_torch.cuda = types.SimpleNamespace(
    is_available=lambda: state['cuda'],
    get_device_name=lambda index: 'NVIDIA Fake Test GPU',
    memory_allocated=lambda: 0,
    memory_reserved=lambda: 0,
    get_device_properties=lambda index: types.SimpleNamespace(total_memory=8 * 1024**3),
    empty_cache=lambda: None,
)
sys.modules['torch'] = fake_torch

class Model:
    def __init__(self, device='cpu', **kwargs):
        self.device = device
        self.kwargs = kwargs

class OCR:
    device = 'cpu'
    def __init__(self):
        pass

vision = types.ModuleType('vision_core')
vision.DinoExtractor = Model
vision.DinoV3Extractor = Model
vision.Sam2Segmenter = Model
vision.MODELS = {'DINOv2 Small': 'test-v2', 'DINOv3 Small': 'test-v3'}
vision.SAM_NAME = 'SAM 2.1 Tiny'
sys.modules['vision_core'] = vision

extra = types.ModuleType('extra_models_core')
extra.EXTRA_ENGINES = {
    'grounding_dino': ('grounding-model', Model),
    'depth_anything': ('depth-model', Model),
    'siglip2': ('siglip-model', Model),
    'rapidocr': ('ocr-model', OCR),
}
sys.modules['extra_models_core'] = extra

session_store = types.ModuleType('backend.session_store')
class LRU:
    def __init__(self, *args):
        self.cleared = False
    def clear(self):
        self.cleared = True
session_store.LRU = LRU
sys.modules['backend.session_store'] = session_store
mm = importlib.import_module('backend.model_manager')
mm.nvidia_driver_info = lambda: {'name': 'NVIDIA Fake Test GPU', 'driver': 'fake'}


class DeviceTest(unittest.TestCase):
    def setUp(self):
        state['cuda'] = False
        self.manager = mm.ModelManager()

    def test_cpu_fallback_and_gpu_unavailable_message(self):
        self.assertEqual(self.manager.status()['device'], 'CPU')
        self.assertIn('CPU-only', self.manager.status()['cuda_issue'])
        with self.assertRaisesRegex(ValueError, 'CUDA não está disponível'):
            self.manager.set_device('cuda')
        self.assertEqual(self.manager.device_preference, 'auto')
        self.assertEqual(self.manager.get('DINOv2 Small').device, 'cpu')

    def test_gpu_and_cpu_switch_releases_models(self):
        state['cuda'] = True
        model = self.manager.get('DINOv2 Small')
        self.assertEqual(model.device, 'cuda')
        self.manager.features['ref'] = 'vector'
        self.manager.set_device('cpu')
        self.assertFalse(self.manager.models)
        self.assertFalse(self.manager.features)
        self.assertTrue(self.manager.reference_detections.cleared)
        self.assertEqual(self.manager.get('DINOv3 Small').device, 'cpu')
        self.assertEqual(self.manager.get('DINOv2 Small').device, 'cpu')
        self.manager.set_device('cuda')
        self.assertEqual(self.manager.status()['device'], 'CUDA')
        self.assertEqual(self.manager.get('DINOv2 Small').device, 'cuda')

    def test_ocr_is_cpu_only_and_gpu_models_follow_selection(self):
        state['cuda'] = True
        self.manager.set_device('cuda')
        self.assertEqual(self.manager.get('RapidOCR').device, 'cpu')
        self.assertEqual(self.manager.get('Grounding DINO Tiny').device, 'cuda')
        self.assertEqual(self.manager.get('Depth Anything V2 Small').device, 'cuda')
        self.assertEqual(self.manager.get('SigLIP 2 Base').device, 'cuda')
        devices = self.manager.status()['model_devices']
        self.assertEqual(devices['RapidOCR'], 'CPU')
        self.assertEqual(devices['Grounding DINO Tiny'], 'CUDA')

    def test_auto_returns_to_cuda_and_invalid_choice_is_rejected(self):
        state['cuda'] = True
        self.manager.set_device('cpu')
        self.manager.set_device('auto')
        self.assertEqual(self.manager.effective_device(), 'cuda')
        with self.assertRaisesRegex(ValueError, 'Dispositivo inválido'):
            self.manager.set_device('unknown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
