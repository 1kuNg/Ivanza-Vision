from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dinov2_core import Features
from vision_core import clean_points, draw_masks, object_crops, display_points, benchmark_dino, MaskResult


def test_multiple_masks_overlay_boxes_and_crops():
    im = Image.new('RGB', (80, 50), (110, 110, 110))
    m1, m2 = np.zeros((50, 80), bool), np.zeros((50, 80), bool)
    m1[8:30, 6:27] = True
    m2[12:40, 40:68] = True
    overlay, areas, boxes = draw_masks(im, [m1, m2], [(0.20,0.32),(0.7,0.45)])
    assert overlay.size == (80, 50)
    assert len(areas) == 2 and all(0 < a < 1 for a in areas)
    assert boxes == [(6,8,27,30),(40,12,68,40)]
    seg = MaskResult([m1,m2],['Item 1','Item 2'],areas,boxes,overlay,0.1)
    crops = object_crops(im, seg)
    assert len(crops) == 2
    assert crops[0].width >= 21 and crops[0].height >= 22
    assert crops[1].width >= 28 and crops[1].height >= 28


def test_points_limit_and_coordinates():
    assert clean_points([(0, 0), (1, 1)]) == [(0.,0.),(1.,1.)]
    with pytest.raises(ValueError, match='12'):
        clean_points([(0.5,0.5)] * 13)
    with pytest.raises(ValueError):
        clean_points([(1.2,0.4)])
    assert display_points(Image.new('RGB',(80,80)),[(0.5,0.5)]).size==(80,80)


class DummyExtractor:
    model_id = 'test-model'
    device = 'cpu'

    def __init__(self):
        self.calls = 0
    def extract(self, image, *, size, precision):
        self.calls += 1
        vec = np.array([1.,0.,0.],dtype=np.float32)
        patches = np.tile(vec,(4,1))
        return Features(vec, patches, (2,2))


def test_benchmark_dino_cache_is_reused():
    im = Image.new('RGB',(40,40),'white')
    dummy = DummyExtractor()
    cache = {}
    analysis, metrics = benchmark_dino(dummy,[im],im,'posicao',cache)
    assert dummy.calls == 2
    assert metrics['cache_hits'] == 0
    assert analysis.global_similarity == pytest.approx(1.0)
    _, metrics = benchmark_dino(dummy,[im],im,'posicao',cache)
    assert dummy.calls == 3
    assert metrics['cache_hits'] == 1


def test_sam2_multiple_points_single_image():
    from vision_core import sam2_prompt_arrays
    pts, labels = sam2_prompt_arrays([(0.2, 0.5), (0.7, 0.8)], 101, 201)
    assert len(pts) == len(labels) == 1  # Uma imagem
    assert len(pts[0]) == len(labels[0]) == 2  # Dois objetos
    assert pts[0][0] == [[20.0, 100.0]]
    assert pts[0][1] == [[70.0, 160.0]]
    assert labels == [[[1], [1]]]
