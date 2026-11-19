import asyncio
import json
from pathlib import Path

import httpx
import pytest

from backend.api import app
from version import VERSION
from backend.contracts import Settings
from backend.llm_assistant import build_safe_telemetry_payload, parse_proposal
from backend.session_store import Store


def test_telemetry_firewall_uses_only_numeric_metrics_and_calibrated_parameters():
    name = 'RapidOCR'
    result = dict(
        benchmark=[dict(model=name, device='CPU', inference_ms=12.5, filename=r'C:\clientes\ACME.png')],
        results={name: dict(overlay='data:image/png;base64,AAAA',
                            metrics=dict(regex_ok=[True, False, 'ACME'], probabilities=[.7,float('nan')],
                                         token='secret', read_count=3),
                            rows=[dict(text='ACME'),dict(text='LOTE-917'),dict(text='11/02/2027')])})
    recipe=Settings(grounding_prompt='ACME',ocr_regex='secret')
    payload=build_safe_telemetry_payload(result,recipe)
    serialized=json.dumps(payload,allow_nan=False)
    for forbidden in ('secret','ACME','base64','LOTE-917','11/02/2027'):
        assert forbidden not in serialized
    assert payload['results'][0]['metrics']['regex_ok']==[True,False]
    assert build_safe_telemetry_payload(result,recipe,allow_ocr=True)['results'][0]['ocr_readings']==[
        {'text':text,'confidence':None,'regex_ok':None} for text in ('ACME','LOTE-917','11/02/2027')]


def test_assistant_parses_fenced_valid_action_and_refuses_invalid_actions():
    p=parse_proposal('```json\n{"action":"update_params","params":{"resolution":512,"threshold":0.30},'
                     '"summary":"Ajustar análise"}\n```')
    assert p.requires_confirmation and p.params=={'resolution':512,'threshold':.3}
    for action, params in [('select_models',{'models':['other']}),('run_analysis',{'path':'local'}),
                           ('run_pipeline',{'pipeline_id':'unknown'}),('update_params',{'resolution':420}),
                           ('export_recipe',{'commands':[]}),('run_shell',{})]:
        with pytest.raises(ValueError):
            parse_proposal(json.dumps(dict(action=action,params=params,summary='x')))
    with pytest.raises(ValueError):
        parse_proposal('{"action":"run_analysis","params":{},"summary":"x","requires_confirmation":false}')


def test_content_addressed_upload_and_api_boundaries():
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as http:
            data=Path('exemplos/painel_normal.png').read_bytes()
            one=(await http.post('/api/images/upload',files={'file':('a.png',data,'image/png')})).json()
            two=(await http.post('/api/images/upload',files={'file':('b.png',data,'image/png')})).json()
            assert one==two
            assert (await http.get(one['url'])).headers['content-type']=='image/png'
            invalid=await http.post('/api/analyze',json={'sample_id':one['image_id'],'resolution':420})
            assert invalid.status_code==422
            secret='token-should-stay-local'
            assert (await http.post('/api/assistant/config',json={
                'base_url':'http://127.0.0.1:11434/v1','model':'local','key':secret})).status_code==200
            status=await http.get('/api/system/status')
            assert status.json()['llm_configured'] and status.json()['version']==VERSION
            assert secret not in status.text
            assert (await http.post('/api/recipe/export',headers={'origin':'https://untrusted.example'},json={})).status_code==403
            assert (await http.post('/api/recipe/export',json={})).json()['versao_lab']==VERSION
            assert len((await http.get('/api/guide')).json()['cases'])==16
    asyncio.run(check())


def test_image_lru_keeps_recent_entries():
    images=Store().images
    images.capacity=2
    images.put('first',1)
    images.put('second',2)
    assert images.get('first')==1
    images.put('third',3)
    assert images.get('second') is None and images.get('first')==1
