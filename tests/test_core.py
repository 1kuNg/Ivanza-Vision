import io

import numpy as np
from PIL import Image
import pytest

from dinov2_core import Features, csv_patch_scores, heatmap_images, inspect, open_image, prepare_image


def feat(vectors, global_vector=None, grid=(2, 2)):
    p = np.asarray(vectors, dtype=np.float32)
    return Features(
        global_vector=np.asarray(global_vector or [1., 0.], dtype=np.float32),
        patches=p,
        grid=grid,
    )


def test_identical_image_features_have_zero_distance():
    sample = feat([[1, 0], [0, 1], [1, 0], [0, 1]])
    result = inspect([sample], sample)
    assert result.global_similarity == pytest.approx(1, abs=1e-6)
    assert np.max(result.patch_distances) == pytest.approx(0, abs=1e-6)


def test_fixed_and_free_mode_differ_with_swapped_patches():
    reference = feat([[1, 0], [1, 0], [0, 1], [0, 1]])
    swapped = feat([[0, 1], [0, 1], [1, 0], [1, 0]])
    fixed = inspect([reference], swapped, mode="posicao")
    free = inspect([reference], swapped, mode="regioes")
    assert np.mean(fixed.patch_distances) > 0.9
    assert np.max(free.patch_distances) < 1e-6


def test_selects_best_reference_same_position():
    reference_1 = feat([[1, 0]] * 4)
    reference_2 = feat([[0, 1]] * 4)
    sample = feat([[0, 1]] * 4)
    result = inspect([reference_1, reference_2], sample)
    assert result.count_references == 2
    assert result.max_distance == pytest.approx(0, abs=1e-6)


def test_validates_bad_inputs():
    sample = feat([[1, 0]] * 4)
    with pytest.raises(ValueError):
        inspect([], sample)
    with pytest.raises(ValueError):
        inspect([sample], sample, mode="wrong")


def test_image_processing_and_overlay():
    img = Image.new("RGB", (480, 240), (230, 220, 210))
    assert prepare_image(img).size == (224, 224)
    distances = np.full((16, 16), 0.2, dtype=np.float32)
    distances[7, 6] = 0.8
    marked, heat = heatmap_images(img, distances, threshold=0.25)
    assert marked.size == img.size
    assert heat.size == img.size
    assert np.mean(np.asarray(marked)[90:110, 80:100, 0]) > 0
    assert csv_patch_scores(distances).count(b"\n") == 257


def test_open_bytes_image():
    bio = io.BytesIO()
    Image.new("RGB", (9, 8)).save(bio, format="PNG")
    assert open_image(bio.getvalue()).size == (9, 8)
