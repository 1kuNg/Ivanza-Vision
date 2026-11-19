import {useEffect,useState} from 'react'
import ReactMarkdown from 'react-markdown'
import {api,DINO,fmt,save,upload,type Photo,type Prediction,type Round,type Settings} from './api'
import {Bars,Results} from './results'
import {Range} from './controls'

export function Files({label,multiple=false,change}:{label:string,multiple?:boolean,change:(files:File[])=>void}){
  return <label className="button upload">{label}<input type="file" aria-label={label} accept="image/*" multiple={multiple} onChange={e=>{const files=Array.from(e.target.files||[]);if(files.length)change(files);e.target.value=''}}/></label>
}
function ClassResult({value}:{value:Prediction}){
  return <><div className="split"><h3>{value.label}</h3><span className="mono">Margem {value.margin.toFixed(3)}</span></div>
    <Bars values={Object.entries(value.by_class).map(([k,v])=>[k,v])}/></>
}
export function Sku({settings,task}:{settings:Settings,task:(fn:()=>Promise<void>)=>void}){
  const [classes,setClasses]=useState([{name:'OK',photos:[] as Photo[]},{name:'NG',photos:[] as Photo[]}])
  const [model,setModel]=useState(DINO[0]),[active,setActive]=useState(false),[sample,setSample]=useState<Photo>(),[limit,setLimit]=useState(.75),[result,setResult]=useState<Prediction>()
  function edit(i:number,v:Partial<(typeof classes)[number]>){setClasses(items=>items.map((item,j)=>i===j?{...item,...v}:item))}
  return <div className="split-panels">
    <section className="card"><h2>Cadastro de classes</h2><label>Backbone<select value={model} onChange={e=>setModel(e.target.value)}>{DINO.map(m=><option key={m}>{m}</option>)}</select></label>
    {classes.map((c,i)=><div key={i} className="class-entry"><input aria-label={'Nome da classe '+(i+1)} value={c.name} onChange={e=>edit(i,{name:e.target.value})}/><Files label={'Fotos · '+c.photos.length} multiple change={files=>task(async()=>edit(i,{photos:await Promise.all(files.slice(0,10).map(upload))}))}/>{classes.length>2&&<button onClick={()=>setClasses(cs=>cs.filter((_,j)=>j!==i))}>Remover</button>}</div>)}
    <div className="toolbar"><button disabled={classes.length>=6} onClick={()=>setClasses([...classes,{name:'',photos:[]}])}>+ Classe</button>
      <button className="primary" disabled={classes.some(c=>!c.name.trim()||!c.photos.length)} onClick={()=>task(async()=>{await api('/sku/register',{model,classes:classes.map(c=>({name:c.name,image_ids:c.photos.map(p=>p.image_id)})),resolution:settings.resolution,precision:settings.precision});setActive(true);setResult(undefined)})}>Registrar banco</button>
      <button onClick={()=>task(async()=>{await api('/sku/clear',undefined,'DELETE');setActive(false);setResult(undefined)})}>Limpar</button></div>
    </section><section className="card"><div className="split"><h2>Teste e similaridade</h2>{active&&<span className="good">Banco disponível</span>}</div>
      <Range label="Limiar de desconhecido" value={limit} min={-1} set={setLimit}/><div className="toolbar"><Files label="Imagem de teste" change={files=>task(async()=>{setSample(await upload(files[0]));setResult(undefined)})}/>
      <button className="primary" disabled={!sample} onClick={()=>task(async()=>setResult(await api('/sku/predict',{sample_id:sample?.image_id,threshold:limit})))}>Classificar</button></div>
      {sample&&<img className="sku-image" src={sample.url} alt="Amostra SKU"/>}{result&&<ClassResult value={result}/>}</section>
  </div>
}
export function Webcam({settings,refIds,addRef,edit,task,onRound}:{settings:Settings,refIds:string[],addRef:(p:Photo)=>void,edit:(p:Photo)=>void,task:(fn:()=>Promise<void>)=>void,onRound:(r:Round)=>void}){
  const [running,setRunning]=useState(false),[continuous,setContinuous]=useState(false),[index,setIndex]=useState(0),[width,setWidth]=useState(640)
  const [interval,setIntervalValue]=useState(1),[sku,setSku]=useState(false),[limit,setLimit]=useState(.75)
  const [round,setRound]=useState<Round>(),[prediction,setPrediction]=useState<Prediction>(),[error,setError]=useState(''),[history,setHistory]=useState<number[]>([])
  const callback=useState(()=>({onRound}))[0];callback.onRound=onRound
  function apply(data:Round&{sku?:Prediction,error?:string,camera_error?:string}){
    setError(data.camera_error||data.error||'')
    if(data.sku)setPrediction(data.sku)
    if(data.run_id){setRound(data);callback.onRound(data);setHistory(old=>[...old.slice(-39),data.benchmark.reduce((sum,r)=>sum+(r.inference_ms||0),0)])}
  }
  useEffect(()=>{
    if(!running)return
    const source=new EventSource('/api/webcam/events')
    source.onmessage=e=>apply(JSON.parse(e.data))
    return ()=>{source.close();void api('/webcam/stop',{}).catch(()=>{})}
  },[running])
  const payload={...settings,ref_ids:refIds,sku,sku_threshold:limit,interval}
  return <><div className="card toolbar">
    <label>Câmera <input type="number" min="0" max="12" value={index} disabled={running} onChange={e=>setIndex(+e.target.value)}/></label>
    <label>Largura <select value={width} disabled={running} onChange={e=>setWidth(+e.target.value)}>{[320,640,960,1280].map(n=><option key={n}>{n}</option>)}</select></label>
    <button className="primary" disabled={running} onClick={()=>task(async()=>{await api('/webcam/start',{index,width});setError('');setRunning(true)})}>▶ Iniciar</button>
    <button disabled={!running} onClick={()=>task(async()=>{await api('/webcam/stop',{});setRunning(false);setContinuous(false)})}>■ Parar</button>
    <label className="check"><input type="checkbox" checked={sku} onChange={e=>setSku(e.target.checked)}/> SKU</label>
    {sku&&<label>Limiar <input type="number" min="-1" max="1" step=".01" value={limit} onChange={e=>setLimit(+e.target.value)}/></label>}
    <button disabled={!running} onClick={()=>task(async()=>addRef(await api('/webcam/capture-ref',{})))}>Capturar referência</button>
    {settings.models.includes('SAM 2.1 Tiny')&&<button disabled={!running} onClick={()=>task(async()=>edit(await api('/webcam/capture-ref',{})))}>Marcar quadro</button>}
    <button disabled={!running} onClick={()=>task(async()=>apply(await api('/webcam/analyze-latest',payload)))}>Analisar quadro</button>
    <label>Intervalo (s)<input type="number" min=".25" max="60" step=".25" value={interval} onChange={e=>setIntervalValue(+e.target.value)}/></label>
    <button disabled={!running} className={continuous?'selected':''} onClick={()=>task(async()=>{if(continuous)await api('/webcam/continuous',undefined,'DELETE');else await api('/webcam/continuous',payload);setContinuous(!continuous)})}>{continuous?'Parar análise contínua':'Análise contínua'}</button>
  </div>{error&&<p className="error">{error}</p>}<div className="split-panels camera">
    <section className="card"><h2>Imagem ao vivo</h2>{running?<img className="live" src="/api/webcam/stream" alt="Webcam ao vivo"/>:<div className="empty">Câmera parada</div>}</section>
    <section className="card"><h2>Último resultado</h2>{sku&&prediction?<ClassResult value={prediction}/>:round?<Results round={round} s={settings} compact/>:<div className="empty">Aguardando análise</div>}</section>
  </div><details className="card"><summary>Telemetria / Histórico</summary>
    {history.length>0&&<><p className="mono">Última inferência somada {fmt(history.at(-1))} · {history.length} quadros</p><svg className="spark" viewBox="0 0 400 70" role="img" aria-label="Histórico de latência"><polyline fill="none" stroke="#f07126" strokeWidth="2" points={history.map((x,i)=>`${i*400/Math.max(1,history.length-1)},${68-62*x/Math.max(1,...history)}`).join(' ')}/></svg>
    <button onClick={()=>save('telemetria_webcam.json',JSON.stringify(history))}>Baixar JSON</button></>}
  </details></>
}
type GuideData={matrix:string,cases:{title:string,model:string,complexity:string,inputs:string,steps:string[]}[],note:string}
export function Guide(){
  const [data,setData]=useState<GuideData>(),[search,setSearch]=useState(''),[error,setError]=useState('')
  useEffect(()=>{api<GuideData>('/guide').then(setData).catch(e=>setError(e.message))},[])
  if(!data)return <p>{error||'Carregando guia…'}</p>
  const rows=data.matrix.split('\n').slice(2).map(x=>x.trim().split('|').slice(1,-1).map(x=>x.trim()))
  const term=search.trim().toLowerCase()
  return <section className="guide"><div className="split"><h2>Guia Industrial</h2><input type="search" placeholder="Buscar por técnica ou modelo" value={search} onChange={e=>setSearch(e.target.value)}/></div>
    <details className="card" open={!term}><summary>Matriz de modelos e licenças</summary><div className="table-scroll"><table><thead><tr><th>Modelo</th><th>Aplicação</th><th>Licença</th></tr></thead><tbody>{rows.filter(row=>row.join(' ').toLowerCase().includes(term)).map(row=><tr key={row[0]}>{row.map((s,i)=><td key={i}><ReactMarkdown>{s}</ReactMarkdown></td>)}</tr>)}</tbody></table></div></details>
    {data.cases.filter(c=>JSON.stringify(c).toLowerCase().includes(term)).map(c=><details className="card" key={c.title}><summary>{c.title}</summary><p className="muted">{c.model} · {c.complexity}</p><p>{c.inputs}</p><ol>{c.steps.map((step,i)=><li key={i}><ReactMarkdown>{step}</ReactMarkdown></li>)}</ol></details>)}
    <p className="muted">{data.note}</p>
  </section>
}
