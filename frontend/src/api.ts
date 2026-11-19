export const DINO = ['DINOv2 Small','DINOv3 Small','DINOv3 ConvNeXt Tiny']
export const SAM = 'SAM 2.1 Tiny'
export const GROUND = 'Grounding DINO Tiny'
export const DEPTH = 'Depth Anything V2 Small'
export const SIGLIP = 'SigLIP 2 Base'
export const OCR = 'RapidOCR'
export const GROUPS = [['Anomalia',DINO],['Segmentação',[SAM]],['Detecção / 2.5D / Texto',[GROUND,DEPTH,SIGLIP,OCR]]] as const
export type Point = [number,number,0|1]
export type Box = [number,number,number,number]
export type Photo = {image_id:string,width:number,height:number,url:string,name?:string}
export type Settings = {
  models:string[], precision:'fp16'|'fp32', resolution:224|384|512, threshold:number, opacity:number,
  mode:'posicao'|'regioes', use_sam_mask_filter:boolean, px_per_mm:number, points:Point[][], boxes:Box[],
  grid_rows:number,grid_cols:number,grid_bounds:Box, sam_cell_method:string,sam_cell_model:string,sam_cell_threshold:number,
  grounding_prompt:string,grounding_threshold:number,grounding_text_threshold:number,depth_threshold:number,use_depth_filter:boolean,
  siglip_states:string[],siglip_threshold:number,ocr_confidence:number,ocr_regex:string,ocr_rois:boolean
}
export const initial:Settings={models:[DINO[0]],precision:'fp16',resolution:224,threshold:.30,opacity:.70,mode:'posicao',
  use_sam_mask_filter:false,px_per_mm:0,points:[],boxes:[],grid_rows:0,grid_cols:0,grid_bounds:[.10,.16,.90,.86],
  sam_cell_method:'escala_de_cinza',sam_cell_model:DINO[0],sam_cell_threshold:7,grounding_prompt:'screw. label. cap.',
  grounding_threshold:.30,grounding_text_threshold:.25,depth_threshold:.50,use_depth_filter:false,
  siglip_states:['open electrical panel','closed electrical panel','wet floor','empty conveyor'],siglip_threshold:.50,
  ocr_confidence:.50,ocr_regex:'',ocr_rois:false}
export type ModelResult = {error?:string,overlay?:string,details?:Record<string,string>,metrics?:Record<string,number|number[]|boolean[]>,
  choices?:number[],candidates?:number[][],rows?:Record<string,unknown>[],positions?:Record<string,unknown>[],timing_ms?:Record<string,number>}
export type Round={run_id:string,sample_id:string,benchmark:{model:string,inference_ms:number|null,load_ms?:number,device:string|null,resolution?:number|null,status:string}[],results:Record<string,ModelResult>,pipeline?:string}
export type System={version:string,device:string,loaded_models:string[],vram?:{allocated:number,reserved:number,total:number},hf_configured:boolean,llm_configured:boolean,base_url:string,llm_model:string,pipelines:Record<string,string>}
export type Prediction={label:string,similarity:number,margin:number,by_class:Record<string,number>}
export async function api<T>(path:string,data?:unknown,method=data===undefined?'GET':'POST'):Promise<T>{
  const response=await fetch('/api'+path,{method,headers:data && !(data instanceof FormData)?{'Content-Type':'application/json'}:undefined,
    body:data===undefined?undefined:data instanceof FormData?data:JSON.stringify(data)})
  const json=await response.json()
  if(!response.ok) throw Error(typeof json.detail==='string'?json.detail:'Não foi possível concluir a solicitação.')
  return json as T
}
export function upload(file:File){const data=new FormData();data.append('file',file);return api<Photo>('/images/upload',data)}
export function save(name:string,value:string,type='application/json'){
  const url=URL.createObjectURL(new Blob([value],{type})); const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)
}
export const fmt=(x:number|null|undefined)=>x==null?'—':x.toLocaleString('pt-BR',{maximumFractionDigits:1})+' ms'
export type Patch=(value:Partial<Settings>)=>void
