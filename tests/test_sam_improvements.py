import numpy as np
import pytest
from PIL import Image
from vision_core import (clean_sam_objects, sam2_prompt_arrays, draw_masks,
                         MaskResult, sam_mask_diagnostics, sam_detail_images,
                         select_mask_candidates, display_sam_prompts)


def test_multiple_positive_negative_points_heterogeneous():
    objects = [[(0.25, 0.25, 1), (0.6, 0.6, 0), (0.4, 0.4, 1)],
               [(0.75, 0.8, 1)]]
    pts, labels = sam2_prompt_arrays(objects, 101, 201)
    assert pts == [[[[25., 50.], [60., 120.], [40., 80.]], [[75., 160.]]]]
    assert labels == [[[1,0,1],[1]]]
    assert clean_sam_objects(objects) == objects


def test_legacy_one_point_per_object():
    pts, labels = sam2_prompt_arrays([(0.25, 0.5),(0.9,0.1)], 101, 201)
    assert pts == [[[[25.,100.]], [[90.,20.]]]]
    assert labels == [[[1],[1]]]


def test_invalid_negative_only_rejected():
    with pytest.raises(ValueError, match='ponto dentro'):
        clean_sam_objects([[(0.4,0.4,0)]])
    with pytest.raises(ValueError):
        clean_sam_objects([[(0.4,0.4,2)]])
    with pytest.raises(ValueError):
        clean_sam_objects([[(1.4,0.4,1)]])


def test_candidate_selection_updates_overlay_without_inference():
    im = Image.new('RGB',(100,70),'grey')
    a = np.zeros((70,100),bool)
    b = np.zeros_like(a)
    a[5:12,5:12] = True
    b[15:45,10:40] = True
    overlay, areas, boxes = draw_masks(im,[a], [[(.2,.4,1)]])
    old = MaskResult([a], ['Item 1'], areas, boxes, overlay, .2,
                     [[a,b]], [[.4,.91]], [0], [[(.2,.4,1)]])
    selected = select_mask_candidates(im,old,[1])
    assert selected.areas[0] > old.areas[0]
    assert selected.boxes[0] == (10,15,40,45)
    assert selected.chosen_indices == [1]
    assert old.chosen_indices == [0]
    with pytest.raises(ValueError):
        select_mask_candidates(im,old,[2])


def test_diagnostics_flag_bad_mask_and_no_false_approval():
    mask = np.zeros((100,100),bool)
    mask[10:40,10:40] = True
    checked = sam_mask_diagnostics(mask,[(.2,.2,1),(.9,.9,0)], .93)
    assert checked['status'] == 'Verificar visualmente'
    assert checked['positive_ok'] == '1/1'
    assert checked['negative_ok'] == '1/1'
    wrong = sam_mask_diagnostics(mask,[(.95,.95,1),(.15,.15,0)], .98)
    assert wrong['status'] == 'Revisar'
    assert 'Ponto positivo fora da máscara' in wrong['warnings']
    assert 'Ponto negativo dentro da máscara' in wrong['warnings']
    empty = sam_mask_diagnostics(np.zeros_like(mask),[(.1,.1,1)],0.8)
    assert any('vazia' in w for w in empty['warnings'])


def test_3_detail_views_and_empty_mask():
    im=Image.new('RGB',(100,100),'blue')
    mask=np.zeros((100,100),bool)
    mask[30:70,40:80] = True
    orig, binary, isolated = sam_detail_images(im,mask)
    assert orig.size == binary.size == isolated.size
    assert np.asarray(binary).max() == 255
    assert np.asarray(binary).min() == 0
    assert display_sam_prompts(im,[[(0.5,0.5,1),(0.1,0.1,0)]]).size == im.size
    a,b,c=sam_detail_images(im,np.zeros_like(mask))
    assert a.size == b.size == c.size == (160,160)


def test_sam_segmenter_fully_mocked_multiple_candidates(monkeypatch):
    torch = pytest.importorskip('torch')
    from vision_core import Sam2Segmenter

    class Inputs(dict):
        def to(self, device):
            return self

    class MockProcessor:
        def __call__(self, *, images, input_points, input_labels, return_tensors):
            assert len(input_points) == len(input_labels) == 1
            assert len(input_points[0]) == 2
            assert input_labels[0] == [[1, 0], [1]]
            return Inputs(original_sizes=torch.tensor([[60, 100]]))

        def post_process_masks(self, pred_masks, original_sizes):
            return [pred_masks[0]]

    class MockModel:
        calls = 0
        def __call__(self, **kwargs):
            self.calls += 1
            pred = torch.zeros((1, 2, 3, 60, 100),dtype=torch.float32)-1
            pred[0, 0, 0, 5:35, 5:35] = 1
            pred[0, 0, 1, 10:30, 10:30] = 1
            pred[0, 1, 2, 20:48, 55:88] = 1
            scores = torch.tensor([[[.3,.9,.7],[.2,.4,.8]]])
            return type('Result', (), {'pred_masks': pred, 'iou_scores': scores})()

    inst = object.__new__(Sam2Segmenter)
    inst.torch = torch
    inst.device = 'cpu'
    inst.processor = MockProcessor()
    inst.model = MockModel()
    image = Image.new('RGB', (100,60), 'grey')
    prompts = [[(.2,.3,1),(.9,.9,0)],[(.65,.55,1)]]
    seg = inst.segment(image, prompts)
    assert inst.model.calls == 1
    assert len(seg.masks) == 2
    assert seg.chosen_indices == [1, 2]
    assert len(seg.candidates[0]) == 3
    assert seg.boxes == [(10,10,30,30),(55,20,88,48)]
    assert seg.candidate_scores[1][2] == pytest.approx(.8)
