"""Regressões do laboratório com dados pequenos e sem baixar checkpoints na suíte."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from dinov2_core import DinoExtractor, Features, heatmap_images, inference_autocast, inspect, prepare_image
from hf_auth import redact_secret
from lab_workflows import SkuBank, inspection_recipe
from vision_core import (
    DINOV3_CONVNEXT_ID, DINOV3_ID, DinoV3Extractor, benchmark_dino, dino_image_size,
    sam_mask_diagnostics, sam_position_comparison,
)


@pytest.mark.parametrize('extractor_class,patch_size', [(DinoExtractor, 14), (DinoV3Extractor, 16)])
@pytest.mark.parametrize('registers', [2, 4])
def test_vits_skip_the_configured_number_of_registers(monkeypatch, extractor_class, patch_size, registers):
    import sys
    import torch

    size = 2 * patch_size
    tokens = torch.tensor([[3., 0.]] + [[100., 100.]] * registers + [[0., 2.]] * 4)

    class Loader:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls()

        config = SimpleNamespace(patch_size=patch_size, num_register_tokens=registers)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            if 'images' in kwargs:
                assert kwargs['do_resize'] is kwargs['do_center_crop'] is False
                assert kwargs['images'].size == (size, size)
                return {'pixel_values': torch.zeros(1, 3, size, size)}
            assert tuple(kwargs['pixel_values'].shape) == (1, 3, size, size)
            return SimpleNamespace(last_hidden_state=tokens[None])

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        Dinov2WithRegistersModel=Loader, AutoModel=Loader, AutoImageProcessor=Loader))
    extractor = extractor_class(device='cpu')
    feature = extractor.extract(Image.new('RGB', (40, 25)), size=size, precision='FP16')
    np.testing.assert_array_equal(feature.global_vector, [1, 0])
    np.testing.assert_array_equal(feature.patches, [[0, 1]] * 4)
    assert feature.grid == (2, 2)


@pytest.mark.parametrize('packed', [False, True])
def test_convnext_spatial_layouts_and_global_pooler(packed):
    import torch
    ch_first = torch.tensor([[[3., 0.], [4., 0.]], [[0., 5.], [0., 6.]]])
    global_vector = torch.tensor([[3., 4.]])
    spatial = torch.cat((global_vector, ch_first.flatten(1).T)) if packed else ch_first
    extractor = object.__new__(DinoV3Extractor)
    extractor.torch = torch
    extractor.device = 'cpu'
    extractor.is_convnext, extractor.patch_size = True, 32

    def process(**kwargs):
        assert kwargs['do_resize'] is kwargs['do_center_crop'] is False
        return {'pixel_values': torch.zeros(1, 3, 64, 64)}

    extractor.processor = process
    extractor.model = lambda **kwargs: SimpleNamespace(last_hidden_state=spatial[None], pooler_output=global_vector)
    feature = extractor.extract(Image.new('RGB', (80, 40)), size=64)
    assert feature.grid == (2, 2)
    np.testing.assert_allclose(feature.global_vector, [0.6, 0.8])
    np.testing.assert_array_equal(feature.patches, [[1, 0], [0, 1], [1, 0], [0, 1]])


def test_letterbox_excludes_padding_from_metrics_and_reference_matching():
    image = Image.new('RGB', (80, 20), 'white')
    padded, geometry = prepare_image(image, 56, 14, with_geometry=True)
    assert padded.getpixel((0, 0)) == (114, 114, 114)
    assert geometry.box == (0, 21, 56, 35)
    valid = geometry.valid_patch_mask((4, 4))
    assert valid.sum() == 8
    normal = Features(np.array([1, 0]), np.tile([1, 0], (16, 1)), (4, 4), geometry)
    changed = normal.patches.copy()
    changed[~valid.ravel()] = [-1, 0]
    result = inspect([normal], Features(normal.global_vector, changed, normal.grid, geometry))
    assert np.isnan(result.patch_distances[~valid]).all()
    assert result.p95_distance == result.mean_distance == result.max_distance == 0
    # An artificially perfect match exists only in reference padding, which must be excluded.
    normal.patches[~valid.ravel()] = [0, 1]
    changed[valid.ravel()] = [0, 1]
    result = inspect([normal], Features(normal.global_vector, changed, normal.grid, geometry), 'regioes')
    assert result.p95_distance == result.mean_distance == result.max_distance == 1


@pytest.mark.parametrize('mode', ['posicao', 'regioes'])
def test_sam_masks_limit_dino_metrics_and_heatmap_and_convert_mm(mode):
    image = Image.new('RGB', (560, 280), 'white')
    _, geometry = prepare_image(image, 56, 14, with_geometry=True)
    foreground = np.zeros((image.height, image.width), dtype=bool)
    foreground[4:12, 4:20] = True
    covered = geometry.valid_patch_mask((4, 4), foreground)
    assert covered.sum() == 1
    normal = Features(np.array([1, 0]), np.tile([1, 0], (16, 1)), (4, 4), geometry)
    query = Features(normal.global_vector, np.tile([-1, 0], (16, 1)), normal.grid, geometry)
    result = inspect([normal], query, mode, sample_foreground=foreground, reference_patch_masks=[covered])
    assert result.p95_distance == result.mean_distance == result.max_distance == 2
    assert np.isfinite(result.patch_distances).sum() == 1
    overlay, heatmap = heatmap_images(image, result.patch_distances, geometry=geometry, foreground=foreground)
    assert overlay.size == heatmap.size == image.size
    assert np.all(np.asarray(overlay)[~foreground] == 255)
    assert np.any(np.asarray(overlay)[foreground] < 255)
    measurement = sam_mask_diagnostics(foreground, [], px_per_mm=4)
    assert (measurement['area_mm2'], measurement['width_mm'], measurement['height_mm']) == (8, 4, 2)
    with pytest.raises(ValueError, match='Nenhum patch válido'):
        inspect([normal], query, mode, sample_foreground=np.zeros_like(foreground))


def test_cache_keys_include_checkpoint_resolution_precision_and_provided_hash(monkeypatch):
    class Extractor:
        device = 'cuda'
        model_id = DINOV3_ID

        def extract(self, image, *, size, precision):
            return Features(np.array([1, 0]), np.array([[1, 0]]), (1, 1))

    def fail(*args, **kwargs):
        pytest.fail('Uma referência com hash conhecido não deve ser codificada no benchmark.')

    monkeypatch.setattr(Image.Image, 'save', fail)
    monkeypatch.setattr(Image.Image, 'tobytes', fail)
    cache = {}
    image = Image.new('RGB', (2, 2))
    extractor = Extractor()
    for model_id, size, precision in [
        (DINOV3_ID, 224, 'FP16'), (DINOV3_ID, 384, 'FP16'), (DINOV3_ID, 384, 'FP32'),
        (DINOV3_CONVNEXT_ID, 384, 'FP16'),
    ]:
        extractor.model_id = model_id
        arg = dict(ref_hashes=['hash-original'], image_size=size, precision=precision)
        assert benchmark_dino(extractor, [image], image, 'posicao', cache, **arg)[1]['cache_hits'] == 0
        assert (model_id, size, precision, 'hash-original') in cache
        assert benchmark_dino(extractor, [image], image, 'posicao', cache, **arg)[1]['cache_hits'] == 1
    assert len(cache) == 4


def test_gpu_precision_uses_cuda_autocast_only_for_fp16():
    calls = []
    fake = SimpleNamespace(float16='float16', amp=SimpleNamespace(
        autocast=lambda *args, **kwargs: calls.append((args, kwargs)) or nullcontext()))
    with inference_autocast(fake, 'cpu', 'FP16'):
        pass
    with inference_autocast(fake, 'cuda', 'FP32'):
        pass
    with inference_autocast(fake, 'cuda', 'FP16'):
        pass
    assert calls == [(('cuda',), {'dtype': 'float16'})]


def test_sku_classification_rejects_unknown_and_shows_margin():
    bank = SkuBank.from_examples([('A', [[1, 0, 0], [0, 0, 3]]), ('B', [[0, 2, 0]])])
    result = bank.classify(np.array([0.1, 0.2, 2]), 0.9)
    assert result.label == 'A'
    assert result.margin == pytest.approx(result.by_class['A'] - result.by_class['B'])
    assert bank.classify(np.array([0.7, 0.7, 0]), 0.9).label == 'Desconhecido'
    assert bank.classify(np.array([0, 1, 0]), 1).label == 'B'
    with pytest.raises(ValueError, match='nomes diferentes'):
        SkuBank.from_examples([('A', [[1, 0]]), ('a', [[0, 1]])])


def test_sam_cell_dino_score_and_recipe():
    img = Image.new('RGB', (80, 40))
    regions = [(0., 0., 0.5, 1.), (0.5, 0., 1., 1.)]
    rows = sam_position_comparison(img, img, regions, dino_vectors=[([1, 0], [1, 0]), ([1, 0], [0, 1])])
    assert [row['Diferença visual (%)'] for row in rows] == [0, 50]
    assert [row['suspeita'] for row in rows] == [False, True]
    recipe = inspection_recipe('DINOv2 Small', 2, 'FP16', 'posicao', 0.42, regions, [], 4,
                               hybrid=True, cell_method='dino_por_celula', cell_model='DINOv3 ConvNeXt Tiny',
                               cell_threshold=5)
    assert recipe['resolucao'] == dino_image_size('DINOv2 Small', 2) == 518
    assert recipe['comparacao_grade']['model_id'] == DINOV3_CONVNEXT_ID
    assert recipe['limiar_calibrado']['valor'] == 0.42
    assert recipe['px_por_mm'] == 4
    assert recipe['rois'][0] == {'x0': 0., 'y0': 0., 'x1': 0.5, 'y1': 1.}
    assert recipe['segmentacao_sam']['filtrar_patches_dino']


def test_redacts_implicit_huggingface_token():
    assert redact_secret('Falha com hf_0123456789abcdef', None) == 'Falha com [token oculto]'
