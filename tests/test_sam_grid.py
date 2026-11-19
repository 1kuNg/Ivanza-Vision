"""Novos testes da v0.2.3: uma peça por objeto, grade e presença por referência."""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from vision_core import (
    sam_grid_regions, sam_grid_objects, sam_grid_preview, sam_position_comparison,
    sam_clip_mask, sam2_prompt_arrays, sam_detail_images, Sam2Segmenter,
)

ROOT = Path(__file__).resolve().parent.parent


def test_eight_independent_sam_prompts_for_bandeja():
    regions = sam_grid_regions(2,4)
    objects = sam_grid_objects(regions)
    assert len(objects) == 8
    assert all(len(obj)==1 and obj[0][2]==1 for obj in objects)
    points, labels = sam2_prompt_arrays(objects,920,530)
    assert len(points)==len(labels)==1
    assert len(points[0]) == len(labels[0]) == 8
    assert all(lbl == [1] for lbl in labels[0])
    assert len({tuple(prompt[0]) for prompt in points[0]})==8


def test_grid_validation():
    with pytest.raises(ValueError): sam_grid_regions(4,4)
    with pytest.raises(ValueError): sam_grid_regions(2,4,(.8,.3,.2,.9))
    assert len(sam_grid_regions(3,4)) == 12


def test_grid_masks_do_not_overlap_and_show_expected_rois():
    regions=sam_grid_regions(2,4)
    full=np.ones((530,920),dtype=bool)
    masks=[sam_clip_mask(full,region) for region in regions]
    occupancy=np.sum(np.stack(masks),axis=0)
    assert occupancy.max()==1
    assert all(m.any() for m in masks)
    frame=Image.open(ROOT/'exemplos'/'bandeja_faltando.png')
    annotated=sam_grid_preview(frame,regions)
    assert annotated.size==frame.size
    original, binary, isolated=sam_detail_images(frame,masks[5],region=regions[5])
    assert original.size==binary.size==isolated.size
    assert original.width < frame.width//2


def test_missing_part_position_flagged_only_in_six():
    normal=Image.open(ROOT/'exemplos'/'bandeja_normal.png')
    missing=Image.open(ROOT/'exemplos'/'bandeja_faltando.png')
    regions=sam_grid_regions(2,4)
    results=sam_position_comparison(missing,normal,regions)
    assert len(results)==8
    assert [result['Posição'] for result in results if result['suspeita']] == ['#6']
    assert results[5]['Diferença visual (%)'] > 7
    same=sam_position_comparison(normal,normal,regions)
    assert all(not r['suspeita'] for r in same)


def test_reference_must_be_aligned():
    with pytest.raises(ValueError):
        sam_position_comparison(Image.new('RGB',(200,100)),Image.new('RGB',(100,200)),sam_grid_regions(2,4))


def test_sam_single_forward_for_eight_isolated_objects_mocked():
    torch = pytest.importorskip('torch')
    class Inputs(dict):
        def to(self,device): return self
    class Processor:
        def __call__(self,*,images,input_points,input_labels,return_tensors):
            assert len(input_points[0])==len(input_labels[0])==8
            assert all(len(x)==1 for x in input_points[0])
            return Inputs(original_sizes=torch.tensor([[60,100]]))
        def post_process_masks(self,pred_masks,original_sizes):
            return [pred_masks[0]]
    class Model:
        calls=0
        def __call__(self,**kwargs):
            self.calls+=1
            pred=torch.ones((1,8,1,60,100))
            scores=torch.full((1,8,1),.8)
            return type('O',(),{'pred_masks':pred,'iou_scores':scores})()
    inst=object.__new__(Sam2Segmenter)
    inst.torch=torch
    inst.device='cpu'
    inst.processor=Processor()
    inst.model=Model()
    regions=sam_grid_regions(2,4)
    image=Image.new('RGB',(100,60),'grey')
    result=inst.segment(image,sam_grid_objects(regions),regions=regions)
    assert inst.model.calls==1
    assert len(result.masks)==8
    assert np.stack(result.masks).sum(axis=0).max() == 1
    assert result.regions == regions
    assert all(area > 0 for area in result.areas)
