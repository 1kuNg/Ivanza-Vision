import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import get_token
from pydantic import Field, SecretStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from extra_models_core import normalize_grounding_prompt, compile_ocr_regex
from hf_auth import verify_dinov3_access, save_dinov3_login
from industrial_guide import MATRIX, CASES, EXTRA_CASES
from lab_workflows import SkuBank, inspection_recipe, recipe_with_extras
from vision_core import (DINO_STRIDES, DINOV3_ID, DINOV3_CONVNEXT_ID, SAM_NAME, cached_reference_feature,
                         sam_grid_regions, sam_grid_objects, select_mask_candidates)
from .analysis import analyze, depth_output, heatmap_output, sam_output, size_for, DEPTH
from .contracts import AnalysisInput, Hash, PIPELINES, PipelineInput, Settings, Strict, Unit, Resolution, Precision
from .llm_assistant import build_safe_telemetry_payload, config, generate_report, parse_command, redact, safe_url
from .model_manager import ALL_MODELS, ModelManager
from .session_store import Store
from .webcam_service import Webcam

ROOT = Path(__file__).resolve().parents[1]
VERSION = '0.5.0'
store, manager, camera = Store(), ModelManager(), Webcam()
examples = {p.name:p for p in (ROOT/'exemplos').glob('*.png')}


@asynccontextmanager
async def lifespan(app):
    yield
    try:
        await asyncio.to_thread(camera.stop)
    finally:
        # Aguarda qualquer inferência em curso antes de descarregar os pesos.
        await asyncio.to_thread(manager.clear)


app = FastAPI(title='Ivanza Vision Lab',version=VERSION,lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])


@app.middleware('http')
async def session_middleware(request, call_next):
    if not request.url.path.startswith('/api/'):
        return await call_next(request)
    origin = request.headers.get('origin')
    if origin and (urlsplit(origin).netloc != request.headers.get('host') and
                   origin not in ('http://localhost:5173','http://127.0.0.1:5173')):
        return JSONResponse({'detail':'Origem não autorizada.'},status_code=403)
    key, session = store.session(request.cookies.get('vision_session'))
    request.state.session = session
    response = await call_next(request)
    if key != request.cookies.get('vision_session'):
        response.set_cookie('vision_session',key,httponly=True,samesite='strict')
    return response


@app.exception_handler(RequestValidationError)
async def invalid(request, error):
    fields = ['.'.join(map(str,e['loc'])) for e in error.errors()]
    return JSONResponse({'detail':'Parâmetros inválidos: '+redact(', '.join(fields),request.state.session)},status_code=422)


@app.exception_handler(ValueError)
async def invalid_value(request, error):
    return JSONResponse({'detail':redact(error,request.state.session)},status_code=400)


@app.get('/api/system/status')
def status(request:Request):
    s=request.state.session
    c=config(s)
    return dict(**manager.status(), version=VERSION, models=ALL_MODELS, pipelines=PIPELINES,
                hf_configured=bool(s.get('hf') or get_token()),
                llm_configured=bool(c['key']), base_url=c['base_url'], llm_model=c['model'])


@app.post('/api/system/vram/clear')
def clear():
    manager.clear()
    return {'ok':True}


class HF(Strict):
    token: SecretStr | None = None
    checkpoint: Literal['vit','convnext']='vit'


def hf_login(data, request, persist):
    s=request.state.session
    token=data.token.get_secret_value().strip() if data.token else s.get('hf','')
    try:
        meth=save_dinov3_login if persist else verify_dinov3_access
        username=meth(token or None,model_id=DINOV3_CONVNEXT_ID if data.checkpoint=='convnext' else DINOV3_ID)
    except Exception:
        raise ValueError('Falha ao verificar acesso ao Hugging Face. Confira o token e a licença.') from None
    if token:
        s['hf']=token
    return dict(configured=True,username=username)


@app.post('/api/auth/hf/verify')
def hf_verify(data:HF,request:Request):
    return hf_login(data,request,False)


@app.post('/api/auth/hf/save')
def hf_save(data:HF,request:Request):
    return hf_login(data,request,True)


@app.get('/api/examples')
def list_examples():
    return [dict(name=name,**store.upload(file.read_bytes())) for name,file in examples.items()]


@app.get('/api/examples/{name}')
def example(name:str):
    if name not in examples:
        raise HTTPException(404)
    return Response(examples[name].read_bytes(),media_type='image/png')


@app.post('/api/images/upload')
def upload(file:UploadFile=File(...)):
    data=file.file.read(30*1024**2+1)
    if len(data)>30*1024**2:
        raise HTTPException(413,'Limite de 30 MB por imagem.')
    try:
        return store.upload(data)
    except (OSError, ValueError):
        raise ValueError('Imagem inválida ou acima de 12 megapixels.') from None


@app.get('/api/images/{key}')
def image(key:Hash):
    entry=store.images.get(key)
    if entry is None:
        raise HTTPException(404,'Imagem expirada.')
    return Response(entry[1],media_type='image/png')


@app.get('/api/artifacts/{key}')
def artifact(key:Hash):
    value=store.artifacts.get(key)
    if value is None:
        raise HTTPException(404,'Resultado expirado.')
    return Response(value,media_type='image/png')


@app.post('/api/analyze')
def run(data:AnalysisInput,request:Request):
    return analyze(manager,store,data,request.state.session,redact)


@app.post('/api/pipeline/run')
def pipeline(data:PipelineInput,request:Request):
    return analyze(manager,store,data,request.state.session,redact,data.pipeline_id)


class Candidate(Strict):
    run_id:str=Field(max_length=32)
    choices:list[int]=Field(max_length=12)
    index:int=Field(0,ge=0,le=11)


def find_view(run_id,name):
    item=store.views.get((run_id,name))
    if item is None:
        raise HTTPException(404,'Resultado expirado. Execute a análise novamente.')
    return item


@app.post('/api/sam/select-candidate')
def candidate(data:Candidate):
    ident,result,scale=find_view(data.run_id,SAM_NAME)
    if data.index>=len(result.masks):
        raise ValueError('Índice de peça inválido.')
    source=store.image(ident)
    selected=select_mask_candidates(source,result,data.choices)
    return sam_output(store,source,selected,scale,data.index)


class Preview(Strict):
    run_id:str=Field(max_length=32)
    model:str=Field(max_length=64)
    threshold:float=Field(ge=.02,le=.90)
    opacity:Unit
    depth_threshold:Unit=.50


@app.post('/api/preview')
def preview(data:Preview):
    if data.model==DEPTH:
        depth,regions=find_view(data.run_id,DEPTH)
        return depth_output(store,depth,data.depth_threshold,regions)
    if data.model not in DINO_STRIDES:
        raise ValueError('Modelo sem controles de mapa.')
    ident,result=find_view(data.run_id,data.model)
    return heatmap_output(store,store.image(ident),result,data.threshold,data.opacity)


class SkuClass(Strict):
    name:str=Field(min_length=1,max_length=70)
    image_ids:list[Hash]=Field(min_length=1,max_length=10)


class SkuSetup(Strict):
    model:str='DINOv2 Small'
    precision:Precision='fp16'
    resolution:Resolution=224
    classes:list[SkuClass]=Field(min_length=2,max_length=6)


class Prediction(Strict):
    sample_id:Hash
    threshold:float=Field(.75,ge=-1,le=1)


@app.post('/api/sku/register')
def sku_register(data:SkuSetup,request:Request):
    s=request.state.session
    if data.model not in DINO_STRIDES:
        raise ValueError('Escolha um backbone DINO.')
    size=size_for(data.model,data.resolution)
    p=data.precision.upper()
    with manager.lock:
        try:
            model=manager.get(data.model,s.get('hf'))
            bank=SkuBank.from_examples([(item.name,[
                cached_reference_feature(model,store.image(key),key,manager.features,size,p)[0].global_vector
                for key in item.image_ids]) for item in data.classes])
        except Exception as exc:
            raise ValueError(redact(exc,s)) from None
        s['sku']=(data.model,size,p,bank)
    return dict(classes=bank.labels)


@app.post('/api/sku/predict')
def sku_predict(data:Prediction,request:Request):
    s=request.state.session
    if not s.get('sku'):
        raise ValueError('Cadastre o banco primeiro.')
    name,size,p,bank=s['sku']
    with manager.lock:
        try:
            vec=manager.get(name,s.get('hf')).extract(store.image(data.sample_id),size=size,precision=p)
            return asdict(bank.classify(vec.global_vector,data.threshold))
        except Exception as exc:
            raise ValueError(redact(exc,s)) from None


@app.delete('/api/sku/clear')
def clear_sku(request:Request):
    request.state.session.pop('sku',None)
    return dict(ok=True)


class CameraStart(Strict):
    index:int=Field(0,ge=0,le=12)
    width:Literal[320,640,960,1280]=640


class CameraAnalysis(Settings):
    ref_ids:list[Hash]=Field(default_factory=list,max_length=10)
    sku:bool=False
    sku_threshold:float=Field(.75,ge=-1,le=1)
    interval:float=Field(1.,ge=.25,le=60)


@app.post('/api/webcam/start')
def camera_start(data:CameraStart):
    camera.start(data.index,data.width)
    return dict(ok=True)


@app.post('/api/webcam/stop')
def camera_stop():
    camera.stop()
    return dict(ok=True)


def latest_frame():
    if not camera.latest:
        raise ValueError(camera.error or 'Aguardando quadro da webcam.')
    return camera.latest[1]


@app.post('/api/webcam/capture-ref')
def capture():
    return store.upload(latest_frame())


@app.get('/api/webcam/stream')
async def stream():
    async def generate():
        last=None
        while camera.thread and camera.thread.is_alive() and not camera.capture_stop.is_set():
            item=camera.latest
            if item and item[0]!=last:
                last=item[0]
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+item[1]+b'\r\n'
            await asyncio.sleep(.025)
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame',
                             headers={'Cache-Control':'no-store'})


def analyze_frame(frame,data,request):
    ident=store.upload(frame)['image_id']
    if data.sku:
        return {'sku':sku_predict(Prediction(sample_id=ident,threshold=data.sku_threshold),request)}
    params=data.model_dump(exclude={'interval','sku','sku_threshold'})
    return run(AnalysisInput(**params,sample_id=ident),request)


@app.post('/api/webcam/analyze-latest')
def analyze_latest(data:CameraAnalysis,request:Request):
    return analyze_frame(latest_frame(),data,request)


@app.post('/api/webcam/continuous')
def continuous(data:CameraAnalysis,request:Request):
    def operation(frame):
        try:
            return analyze_frame(frame,data,request)
        except Exception as exc:
            return dict(error=redact(exc,request.state.session))
    camera.continuous(data.interval,operation)
    return dict(ok=True)


@app.delete('/api/webcam/continuous')
def stop_continuous():
    camera.stop_analysis()
    return dict(ok=True)


@app.get('/api/webcam/events')
async def events():
    async def generate():
        previous=None
        while camera.capture_stop and not camera.capture_stop.is_set():
            current=camera.result
            marker=(current[0] if current else None,camera.error)
            if marker!=previous:
                previous=marker
                data=current[1] if current else {}
                if camera.error:
                    data={**data,'camera_error':camera.error}
                yield f'data: {json.dumps(data,ensure_ascii=False)}\n\n'
            await asyncio.sleep(.15)
    return StreamingResponse(generate(),media_type='text/event-stream',headers={'Cache-Control':'no-store'})


@app.post('/api/recipe/export')
def recipe(data:Settings):
    extra={}
    if 'Grounding DINO Tiny' in data.models:
        extra['grounding_dino']=dict(prompt=normalize_grounding_prompt(data.grounding_prompt),
                                     threshold=data.grounding_threshold,text_threshold=data.grounding_text_threshold)
    if DEPTH in data.models or data.use_depth_filter:
        extra['depth_anything']=dict(depth_threshold=data.depth_threshold,apply_to_dino=data.use_depth_filter)
    if 'SigLIP 2 Base' in data.models:
        extra['siglip2']=dict(states=data.siglip_states,alert_threshold=data.siglip_threshold)
    if 'RapidOCR' in data.models:
        compile_ocr_regex(data.ocr_regex)
        extra['rapidocr']=dict(regex=data.ocr_regex,min_confidence=data.ocr_confidence,
                               input_mode='grounding_rois' if data.ocr_rois else 'full_image')
    core=next((n for n in data.models if n in DINO_STRIDES or n==SAM_NAME),None)
    regions=sam_grid_regions(data.grid_rows,data.grid_cols,data.grid_bounds) if data.grid_rows else None
    obj=inspection_recipe(core,(224,384,512).index(data.resolution),data.precision.upper(),data.mode,data.threshold,
        regions,sam_grid_objects(regions) if regions else data.points,data.px_per_mm,hybrid=data.use_sam_mask_filter,
        cell_method=data.sam_cell_method,cell_model=data.sam_cell_model if data.sam_cell_method=='dino_por_celula' else None,
        cell_threshold=data.sam_cell_threshold,extra_settings=extra) if core else recipe_with_extras(extra)
    obj.update(versao_lab=VERSION, box_prompts=data.boxes)
    return JSONResponse(obj,headers={'Content-Disposition':'attachment; filename="receita_ivanza.json"'})


@app.get('/api/guide')
def guide():
    return dict(matrix=MATRIX.strip(),cases=[dict(zip(('title','model','complexity','inputs','steps'),row))
        for row in [*CASES,*EXTRA_CASES]], note='OWLv2 e SAM 3 aceitam exemplos visuais para peças difíceis de nomear.')


class LLMSettings(Strict):
    base_url:str=Field(max_length=300)
    model:str=Field(min_length=1,max_length=150)
    key:SecretStr|None=None


@app.post('/api/assistant/config')
def configure_llm(data:LLMSettings,request:Request):
    s=request.state.session
    s['llm']=dict(base_url=safe_url(data.base_url),model=data.model,
                  key=data.key.get_secret_value() if data.key is not None else config(s)['key'])
    return dict(ok=True,configured=bool(s['llm']['key']))


class Report(Strict):
    run_id:str=Field(max_length=32)
    focus:str=Field('',max_length=600)
    allow_ocr:bool=False


class Command(Strict):
    instruction:str=Field(min_length=1,max_length=1400)
    settings:Settings


@app.post('/api/assistant/report')
def assistant_report(data:Report,request:Request):
    item=store.rounds.get(data.run_id)
    if item is None:
        raise HTTPException(404,'Execute uma análise antes de gerar o parecer.')
    telemetry=build_safe_telemetry_payload(*item,allow_ocr=data.allow_ocr)
    return dict(markdown=generate_report(request.state.session,telemetry,data.focus))


@app.post('/api/assistant/command')
def assistant_command(data:Command,request:Request):
    return parse_command(request.state.session,data.instruction,data.settings)


app.mount('/',StaticFiles(directory=ROOT/'frontend'/'dist',html=True,check_dir=False),name='frontend')
