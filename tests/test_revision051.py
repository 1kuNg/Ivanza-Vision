import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from backend import api, llm_assistant as llm
from backend.contracts import Settings
from backend.session_store import Store
from extra_models_core import GroundingDetector, depth_cell_means, peak_depth_point
from vision_core import Sam2Segmenter, sam_mask_diagnostics


class Inputs(dict):
    def to(self, device):
        return self


def test_sam_box_prompt_has_no_synthetic_point():
    import torch

    seen = {}
    class Processor:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return Inputs(original_sizes=torch.tensor([[16, 24]]))

        def post_process_masks(self, masks, sizes):
            return [masks[0]]

    segmenter = Sam2Segmenter.__new__(Sam2Segmenter)
    segmenter.device, segmenter.torch, segmenter.processor = 'cpu', torch, Processor()
    mask = torch.zeros((1, 1, 1, 16, 24))
    mask[..., 3:12, 4:18] = 1
    mask[..., 6:10, 8:14] = 0
    segmenter.model = lambda **kwargs: SimpleNamespace(pred_masks=mask, iou_scores=torch.tensor([[[.95]]]))
    result = segmenter.segment(Image.new('RGB',(24,16)), boxes=[(.10,.20,.75,.80)])
    assert seen['input_boxes'][0][0] == pytest.approx([2.4,3.2,18,12.8])
    assert 'input_points' not in seen and 'input_labels' not in seen
    assert result.prompts == [[]]
    assert not sam_mask_diagnostics(result.masks[0],result.prompts[0])['warnings']


def test_grounding_nms_filters_overlap_before_returning_rois():
    import torch

    class Processor:
        def __call__(self, **kwargs):
            return Inputs()

        def post_process_grounded_object_detection(self, outputs, threshold, text_threshold, target_sizes):
            return [dict(text_labels=['arruela','arruela','flange'],
                         scores=torch.tensor([.9,.8,.7]),
                         boxes=torch.tensor([[2,2,20,20],[3,3,21,21],[24,24,34,34]]))]

    detector=GroundingDetector.__new__(GroundingDetector)
    detector.processor, detector.device, detector.torch = Processor(), 'cpu', torch
    detector.model=lambda **kwargs: SimpleNamespace(logits=torch.zeros(1), pred_boxes=torch.zeros(1))
    assert [d.label for d in detector.detect(Image.new('RGB',(40,40)),'arruela,flange')] == ['arruela','flange']


def test_robust_depth_peak_and_nonempty_cells():
    depth = np.zeros((12, 12), dtype=np.float32)
    depth[5:10,5:10]=.8
    depth[0,0]=1.
    depth[11,11]=float('nan')
    x,y=peak_depth_point(depth)
    assert .4 < x < .85 and .4 < y < .85
    assert depth_cell_means(depth,[(1.,1.,1.,1.),(.45,.45,.45,.45)]) == pytest.approx([0,.8])


def test_assistant_sends_natural_language_redacts_and_validates_combined_proposal(monkeypatch):
    calls=[]
    def chat(session, system, data, temperature):
        calls.append(data)
        return '<think>oculto</think>{"action":"update_params","params":{"models":["SAM 2.1 Tiny"],"mode":"regioes","opacity":0.45,"use_depth_filter":true,"ocr_confidence":0.8,"sam_cell_method":"dino_por_celula"},"summary":"Alterações de bancada"}'
    monkeypatch.setattr(llm,'chat',chat)
    session={'hf':'hf_abcdefghijklmnop'}
    proposal=llm.parse_command(session,'Coloque SAM nas regiões e senha: "secreto local"',Settings())
    assert calls[0]['instruction']=='Coloque SAM nas regiões e senha: [segredo oculto]'
    assert calls[0]['settings']['grounding_prompt']
    assert 'SAM 2.1 Tiny' in calls[0]['ALL_MODELS'] and calls[0]['PIPELINES']
    assert proposal['params']['models']==['SAM 2.1 Tiny'] and proposal['params']['opacity']==.45
    assert proposal['requires_confirmation']
    llm.generate_report(session,{},'Por que o P95 aumentou? hf_abcdefghijklmnop')
    assert calls[1]['foco']=='Por que o P95 aumentou? [token oculto]'
    for params in ({'opacity':2},{'models':['forjado']},{'ocr_rois':'sim'},{'mode':'global'},
                   {'unexpected':True},{'siglip_states':[]}):
        with pytest.raises(ValueError):
            llm.parse_proposal(json.dumps(dict(action='update_params',params=params,summary='X')))


def test_llm_transport_fallback_and_thinking(monkeypatch):
    traces=[]
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['timeout']==45
        def __enter__(self):
            return self
        def __exit__(self,*args):
            pass
        def post(self,url,**kwargs):
            traces.append(kwargs['json'])
            import httpx
            return httpx.Response(200,request=httpx.Request('POST',url),
                                  json={'choices':[{'message':{'content':'<think>oculto</think> Parecer'}}]})
    monkeypatch.setattr(llm.httpx,'Client',Client)
    session={'llm':dict(base_url='https://openrouter.ai/api/v1',model='a:free,b:free',key='segredo')}
    assert llm.chat(session,'system',{},.1)=='Parecer'
    assert traces[0]['models']==['a:free','b:free'] and traces[0]['max_tokens']==2400


def test_webcam_continuous_does_not_evict_static_references(monkeypatch):
    store = Store()
    store.images.capacity=2
    def jpeg(color):
        buffer=io.BytesIO()
        Image.new('RGB',(16,16),(color,0,0)).save(buffer,format='JPEG')
        return buffer.getvalue()

    reference=store.upload(jpeg(0))['image_id']
    captured=[]
    monkeypatch.setattr(api,'store',store)
    monkeypatch.setattr(api,'analyze',lambda m,s,d,session,redact: captured.append((d.sample_id,session)) or {})
    monkeypatch.setattr(api.camera,'continuous',lambda interval,operation: [operation(jpeg(i)) for i in range(3,90,3)])
    session={}
    request=SimpleNamespace(state=SimpleNamespace(session=session))
    api.continuous(api.CameraAnalysis(ref_ids=[reference]),request)
    assert len(captured)>5 and all(s is session for _,s in captured)
    assert store.images.get(reference) is not None and len(store.images.items)==1
    assert len(store.webcam_frames.items)==5
    assert store.image(captured[-1][0]).size==(16,16)


def test_grounding_pipeline_detects_reference_in_its_own_coordinates(monkeypatch):
    from backend import analysis
    from backend.contracts import AnalysisInput
    from backend.model_manager import ModelManager
    from extra_models_core import Detection

    store=Store()
    def photo(color):
        data=io.BytesIO()
        Image.new('RGB',(20,20),color).save(data,format='PNG')
        return store.upload(data.getvalue())['image_id']

    sample,reference=photo('black'),photo('red')
    found=[]
    prompts=[]
    def detect(image,*args,**kwargs):
        is_reference=image.getpixel((0,0))[0]>0
        found.append(is_reference)
        return [Detection('componente',.95,(12,4,19,14) if is_reference else (2,4,9,14))]

    def segment(image,points,**kwargs):
        prompts.append(kwargs['boxes'])
        mask=np.ones((20,20),dtype=bool)
        return SimpleNamespace(masks=[mask],candidates=[[mask]],elapsed=.01)

    models={analysis.GROUND:SimpleNamespace(detect=detect,device='cpu'),
            analysis.SAM_NAME:SimpleNamespace(segment=segment,device='cpu'),
            'DINOv3 Small':SimpleNamespace(device='cpu')}
    manager=ModelManager()
    monkeypatch.setattr(manager,'get',lambda name,token=None:models[name])
    monkeypatch.setattr(analysis,'sam_output',lambda *args:{})
    monkeypatch.setattr(analysis,'heatmap_output',lambda *args:{})
    result=SimpleNamespace(global_similarity=1.,mean_distance=0.,p95_distance=0.,max_distance=0.,
                           patch_distances=np.zeros((1,1)))
    monkeypatch.setattr(analysis,'benchmark_dino',lambda *args,**kwargs:(result,{'imagem_s':.01}))
    settings=AnalysisInput(sample_id=sample,ref_ids=[reference],models=['DINOv3 Small'])
    for _ in range(2):
        response=analysis.analyze(manager,store,settings,{},llm.redact,'grounding_sam_dino')
        assert all(row['status']=='ok' for row in response['benchmark'])
    assert found==[False,True,False]
    assert prompts==[[(.1,.2,.45,.7)],[(.6,.2,.95,.7)]]*2
