from version import VERSION
"""Pós-processamento determinístico e compatibilidade da receita da v0.4."""
import json

import numpy as np
import pytest
import torch

from extra_models_core import (
    DEPTH_ID, GROUNDING_ID, OCR_ID, SIGLIP_ID, Detection, OcrReading, depth_cell_means,
    depth_mask, filter_ocr, normalize_depth, normalize_grounding_prompt, peak_depth_point,
    sam_center_prompts, scene_alerts, scene_states, sigmoid_probabilities,
)
from lab_workflows import inspection_recipe, recipe_with_extras
from vision_core import sam_grid_regions


def test_grounding_prompt_handles_separators_empty_and_duplicate_items():
    assert normalize_grounding_prompt(' Screw, LABEL. cap.,  , LABEL . SCREW ') == 'screw. label. cap.'
    assert normalize_grounding_prompt(' , .., ') == ''
    assert normalize_grounding_prompt(' Safety   CAP\t. ') == 'safety cap.'
    points = sam_center_prompts([Detection('cap', 0.7, (1, 1, 4, 4))], (5, 5))
    assert points == [[(0.5, 0.5, 1)]]


def test_depth_normalization_cut_cell_statistics_and_peak():
    depth = normalize_depth([[4, 4, 7, 10], [4, 4, 7, 16]])
    assert depth.min() == 0 and depth.max() == 1
    np.testing.assert_array_equal(depth_mask(depth, 0.5), [[False, False, False, True],
                                                           [False, False, False, True]])
    cells = sam_grid_regions(1, 2, (0., 0., 1., 1.))
    assert depth_cell_means(depth, cells) == pytest.approx([0., 0.5])
    assert peak_depth_point(depth) == (1., 1.)
    assert peak_depth_point(np.ones((1, 1))) == (0., 0.)
    np.testing.assert_array_equal(normalize_depth([[4, 4]]), [[0, 0]])


def test_siglip_independent_scores_and_multilabel_alerts():
    states = scene_states(' Wet floor\nOpen panel\nwet floor\nEmpty conveyor\n')
    assert states == ('wet floor', 'open panel', 'empty conveyor')
    probs = sigmoid_probabilities(torch.tensor([[4., 3., -4.]], dtype=torch.float16))[0]
    assert probs.sum().item() > 1
    rows = scene_alerts(states, probs.tolist(), 0.8)
    assert [row['Alerta'] for row in rows] == [True, True, False]
    assert rows[0]['Probabilidade'] == pytest.approx(0.9820, abs=1e-4)


def test_ocr_filters_confidence_searches_and_supports_anchors():
    pol = ((1, 1), (3, 1), (3, 3), (1, 3))
    readings = [OcrReading('LOTE-125', .5, pol), OcrReading('ETIQUETA LOTE-789', .9, pol),
                OcrReading('LOTE-003', .49, pol)]
    returned = filter_ocr(readings, .5, r'LOTE-\d+')
    assert [(item.text, item.regex_ok) for item in returned] == [
        ('LOTE-125', True), ('ETIQUETA LOTE-789', True)]
    assert filter_ocr(readings,.5,r'^LOTE-\d+$')[1].regex_ok is False
    assert filter_ocr(readings)[0].regex_ok is None
    assert all(item.regex_ok is None for item in readings)
    with pytest.raises(ValueError, match='Regex inválido'):
        filter_ocr(readings, .5, '[')


def test_receita_optional_extras_and_original_fields_roundtrip():
    configs = {
        'grounding_dino': {'prompt': 'screw. label.', 'threshold': .3, 'text_threshold': .25},
        'depth_anything': {'depth_threshold': .55, 'apply_to_dino': True},
        'siglip2': {'states': ['wet floor', 'empty conveyor'], 'alert_threshold': .60},
        'rapidocr': {'regex': r'LOTE-\d+', 'min_confidence': .5, 'input_mode': 'grounding_rois'},
    }
    args = ('DINOv2 Small', 0, 'FP16', 'posicao', .3, [], [], 2.0)
    old = inspection_recipe(*args)
    assert old['schema_version'] == 1
    assert 'modelos_extras' not in old
    new = json.loads(json.dumps(inspection_recipe(*args, extra_settings=configs)))
    assert new['schema_version'] == 2 and new['versao_lab'] == VERSION
    for key in old.keys() - {'schema_version'}:
        assert new[key] == old[key]
    for (key, config), model_id in zip(configs.items(), [GROUNDING_ID, DEPTH_ID, SIGLIP_ID, OCR_ID]):
        assert new['modelos_extras'][key] == {**config, 'model_id': model_id}
    assert recipe_with_extras(configs)['modelos_extras'] == new['modelos_extras']
