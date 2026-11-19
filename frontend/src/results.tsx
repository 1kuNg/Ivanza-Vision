import {useEffect,useRef,useState} from 'react'
import {api,DEPTH,DINO,fmt,SAM,SIGLIP,save,type ModelResult,type Round,type Settings} from './api'

const labels:Record<string,string>={global_similarity:'Similaridade',mean_distance:'Distância média',p95_distance:'Distância p95',max_distance:'Distância máx.',mask_count:'Máscaras',box_count:'Caixas',read_count:'Leituras',alert_count:'Alertas',foreground_ratio:'Primeiro plano',peak_depth:'Pico',area_mm2:'Área (mm²)',iou_scores:'IoU',regex_ok:'Regex',probabilities:'Probabilidades'}
const metric=(v:number|boolean)=>typeof v==='boolean'?v?'OK':'NG':Number.isInteger(v)?String(v):v.toFixed(3)
export function Metrics({value,deep=false}:{value:ModelResult,deep?:boolean}) {
  return <div className="metrics">{Object.entries(value.metrics||{}).filter(([,v])=>deep||!Array.isArray(v)).slice(0,deep?20:3).map(([key,val])=><div key={key}><span>{labels[key]||key}</span><strong>{Array.isArray(val)?val.map(metric).join(' · '):metric(val)}</strong></div>)}</div>
}
export function Table({rows}:{rows:Record<string,unknown>[]}){
  if(!rows.length)return null
  const cols=[...new Set(rows.flatMap(row=>Object.keys(row)))]
  return <div className="table-scroll"><table><thead><tr>{cols.map(k=><th key={k}>{({text:'Texto',confidence:'Confiança',regex_ok:'Regex',label:'Classe',score:'Score',box:'Caixa',area_mm2:'Área (mm²)',width_mm:'Largura (mm)',height_mm:'Altura (mm)',mean_depth:'Profundidade média'} as Record<string,string>)[k]||k.replaceAll('_',' ')}</th>)}</tr></thead><tbody>{rows.map((row,i)=><tr key={i}>{cols.map(k=><td key={k}>{typeof row[k]==='number'?metric(row[k] as number):typeof row[k]==='boolean'?metric(row[k] as boolean):row[k]==null?'—':typeof row[k]==='object'?JSON.stringify(row[k]):String(row[k])}</td>)}</tr>)}</tbody></table></div>
}
export function Bars({values}:{values:[string,number,boolean?][]}) {
  return <div className="bars">{values.map(([label,v,alert],i)=><div key={i}><span title={label}>{label}</span><div className="track"><div className={alert?'alert':''} style={{width:Math.max(0,Math.min(1,v))*100+'%'}}/></div><b>{(100*v).toFixed(1)}%</b></div>)}</div>
}
type View={zoom:number,x:number,y:number}
const defaultView={zoom:1,x:0,y:0}
function ImagePane({url,view,adjust}:{url:string,view:View,adjust:(p:View)=>void}){
  const pane=useRef<HTMLDivElement>(null)
  const drag=useRef<{x:number,y:number,v:View}|null>(null)
  const viewRef=useRef(view), adjustRef=useRef(adjust)
  viewRef.current=view
  adjustRef.current=adjust

  // Native non-passive listener: the browser must not scroll the entire page
  // when the pointer is over an inspection image.
  useEffect(()=>{
    const element=pane.current
    if(!element)return
    function wheel(e:WheelEvent){
      e.preventDefault()
      e.stopPropagation()
      const v=viewRef.current
      const nextZoom=Math.min(8,Math.max(1,v.zoom*Math.exp(-e.deltaY*.002)))
      const factor=nextZoom/v.zoom
      const b=element!.getBoundingClientRect()
      const x=e.clientX-(b.left+b.width/2)
      const y=e.clientY-(b.top+b.height/2)
      const next={zoom:nextZoom,x:x-(x-v.x)*factor,y:y-(y-v.y)*factor}
      viewRef.current=next
      adjustRef.current(next)
    }
    element.addEventListener('wheel',wheel,{passive:false})
    return ()=>element.removeEventListener('wheel',wheel)
  },[])

  function zoom(factor:number){
    const v=viewRef.current
    const nextZoom=Math.min(8,Math.max(1,v.zoom*factor))
    const ratio=nextZoom/v.zoom
    adjust({zoom:nextZoom,x:v.x*ratio,y:v.y*ratio})
  }
  return <div className="zoom-shell">
    <div className="zoom-toolbar"><small>Scroll na imagem: zoom · arraste para mover · duplo clique: ajustar</small>
      <div className="zoom-actions">
        <button type="button" aria-label="Diminuir zoom" onClick={()=>zoom(1/1.5)}>−</button>
        <span aria-label="Nível de zoom">{Math.round(view.zoom*100)}%</span>
        <button type="button" aria-label="Aumentar zoom" onClick={()=>zoom(1.5)}>+</button>
        <button type="button" onClick={()=>adjust(defaultView)}>Ajustar</button>
      </div>
    </div>
    <div ref={pane} className="image-pane" aria-label="Resultado com zoom independente da página"
      onDoubleClick={()=>adjust(defaultView)}
      onPointerDown={e=>{if(view.zoom>1&&e.button===0){e.currentTarget.setPointerCapture(e.pointerId);drag.current={x:e.clientX,y:e.clientY,v:view}}}}
      onPointerMove={e=>{const d=drag.current;if(d)adjust({zoom:d.v.zoom,x:d.v.x+e.clientX-d.x,y:d.v.y+e.clientY-d.y})}}
      onPointerUp={()=>{drag.current=null}}
      onPointerCancel={()=>{drag.current=null}}
      onLostPointerCapture={()=>{drag.current=null}}>
      <img src={url} alt="Saída do modelo" draggable={false}
        style={{transform:'translate('+view.x+'px, '+view.y+'px) scale('+view.zoom+')'}}/>
    </div>
  </div>
}
function ImageLightbox({url,label,close}:{url:string,label:string,close:()=>void}){
  const [view,setView]=useState(defaultView)
  useEffect(()=>{
    function key(e:KeyboardEvent){if(e.key==='Escape')close()}
    const before=document.body.style.overflow
    document.body.style.overflow='hidden'
    document.addEventListener('keydown',key)
    return ()=>{document.body.style.overflow=before;document.removeEventListener('keydown',key)}
  },[close])
  return <div className="vision-lightbox-shade" onPointerDown={e=>{if(e.target===e.currentTarget)close()}}>
    <section className="vision-lightbox" role="dialog" aria-modal="true" aria-label={'Ampliar '+label}>
      <header className="split"><h2>{label}</h2><button type="button" onClick={close}>Fechar ×</button></header>
      <ImagePane url={url} view={view} adjust={setView}/>
    </section>
  </div>
}
function Card({name,round,s,deep,synced,adjust}:{name:string,round:Round,s:Settings,deep:boolean,synced?:View,adjust:(v:View)=>void}){
  const data=round.results[name],row=round.benchmark.find(r=>r.model===name)
  const [state,setState]=useState<Partial<ModelResult>>({}),[view,setView]=useState(defaultView),[focus,setFocus]=useState(0)
  const [lightbox,setLightbox]=useState<{url:string,label:string}|null>(null)
  const [cut,setCut]=useState(name===DEPTH?s.depth_threshold:s.threshold),[opacity,setOpacity]=useState(s.opacity),[error,setError]=useState('')
  const seq=useRef(0), previewTimer=useRef<number | undefined>(undefined)
  useEffect(()=>()=>{window.clearTimeout(previewTimer.current);seq.current++},[])
  const result={...data,...state,metrics:{...data.metrics,...state.metrics}}
  function updatePreview(t:number,o:number){
    setCut(t);setOpacity(o)
    const current=++seq.current
    window.clearTimeout(previewTimer.current)
    previewTimer.current=window.setTimeout(async()=>{
      try{
        const value=await api<ModelResult>('/preview',{run_id:round.run_id,model:name,threshold:name===DEPTH?s.threshold:t,opacity:o,depth_threshold:name===DEPTH?t:s.depth_threshold})
        if(seq.current===current){setState(value);setError('')}
      }catch(e){if(seq.current===current)setError(String(e))}
    },130)
  }
  async function candidate(index:number,choice?:number){
    if(!result.choices)return
    const choices=[...result.choices];if(choice!==undefined)choices[index]=choice
    const current=++seq.current
    try{const value=await api<ModelResult>('/sam/select-candidate',{run_id:round.run_id,choices,index});if(seq.current===current){setFocus(index);setState(value);setError('')}}catch(e){if(seq.current===current)setError(String(e))}
  }
  return <article className="card result-card">
    <header className="split"><h3>{name}</h3><span className="latency-badge">{fmt(row?.inference_ms)}</span></header>
    {data.error?<p className="error">{data.error}</p>:<>
      {(DINO.includes(name)||name===DEPTH)&&<details className="view-options"><summary>Ajustar visualização</summary>
        <label>Corte <input type="range" min={name===DEPTH?0:.02} max={name===DEPTH?1:.9} step=".01" value={cut} onChange={e=>void updatePreview(+e.target.value,opacity)}/><span className="mono">{cut.toFixed(2)}</span></label>
        {name!==DEPTH&&<label>Opacidade <input type="range" min="0" max="1" step=".05" value={opacity} onChange={e=>void updatePreview(cut,+e.target.value)}/><span>{opacity.toFixed(2)}</span></label>}
      </details>}
      {error&&<p className="error">{error}</p>}
      {result.overlay&&<><ImagePane url={result.overlay} view={synced||view} adjust={synced?adjust:setView}/><button className="image-expand" onClick={()=>setLightbox({url:result.overlay!,label:'Resultado · '+name})}>⤢ Ampliar em tela cheia</button></>}
      {name===SIGLIP&&<Bars values={(result.rows||[]).map(r=>[String(r.Estado),Number(r.Probabilidade),Boolean(r.Alerta)])}/>}
      <Metrics value={result} deep={deep}/>
      {deep&&<>
        {row&&<p className="mono muted">Carga {fmt(row.load_ms)} · {row.device} {row.resolution?`· ${row.resolution} px`:''}</p>}
        {name===SAM&&result.candidates&&<div className="toolbar">
          <label>Item <select value={focus} onChange={e=>void candidate(+e.target.value)}>{result.candidates.map((_,i)=><option value={i} key={i}>{i+1}</option>)}</select></label>
          <label>Candidata <select value={result.choices?.[focus]||0} onChange={e=>void candidate(focus,+e.target.value)}>{result.candidates[focus].map((v,i)=><option value={i} key={i}>#{i+1} · IoU {v.toFixed(3)}</option>)}</select></label>
        </div>}
        {result.details&&<div className="detail-images">{Object.entries(result.details).map(([key,url])=><figure key={key}><button type="button" className="detail-zoom" title="Ampliar imagem" onClick={()=>setLightbox({url,label:key})}><img src={url} alt={key}/></button><figcaption>{({relative:'Mapa relativo',original:'Original',mask:'Máscara',isolated:'Peça isolada'} as Record<string,string>)[key]||key} · clique para ampliar</figcaption></figure>)}</div>}
        <Table rows={result.rows||[]}/><Table rows={result.positions||[]}/>
        <div className="toolbar">{result.overlay&&<a className="button" href={result.overlay} download={name+'.png'}>Baixar PNG</a>}<button onClick={()=>save(name+'.json',JSON.stringify(result,null,2))}>Baixar JSON</button></div>
      </>}
    </>}
    {lightbox&&<ImageLightbox url={lightbox.url} label={lightbox.label} close={()=>setLightbox(null)}/>}
  </article>
}
export function Results({round,s,compact=false}:{round:Round,s:Settings,compact?:boolean}){
  const [deep,setDeep]=useState(false),[model,setModel]=useState(''),[sync,setSync]=useState(false),[view,setView]=useState(defaultView)
  const names=Object.keys(round.results)
  if(!names.length)return <div className="empty">Selecione pelo menos um modelo.</div>
  const picked=names.includes(model)?model:names[0]
  return <section className="results">
    {!compact&&<><Table rows={round.benchmark.map(row=>({'Modelo':row.model,'Inferência':fmt(row.inference_ms),'Dispositivo':row.device,
      'Métrica principal':Object.entries(round.results[row.model].metrics||{}).filter(([,v])=>!Array.isArray(v)).slice(0,1).map(([k,v])=>`${labels[k]||k} ${metric(v as number)}`).join('')||'—',Status:row.status==='ok'?'OK':'Erro'}))}/>
      <div className="split toolbar"><div className="segments"><button className={!deep?'selected':''} onClick={()=>setDeep(false)}>Lado a Lado</button><button className={deep?'selected':''} onClick={()=>setDeep(true)}>Inspeção Profunda</button></div>
      {deep?<select aria-label="Modelo em foco" value={picked} onChange={e=>setModel(e.target.value)}>{names.map(n=><option key={n}>{n}</option>)}</select>:<label className="check"><input type="checkbox" checked={sync} onChange={e=>{setSync(e.target.checked);setView(defaultView)}}/> Sincronizar zoom / pan</label>}</div></>}
    <div className={deep&&!compact?'deep-grid':'result-grid'}>{(deep&&!compact?[picked]:names).map(name=><Card key={round.run_id+name} name={name} round={round} s={s} deep={deep&&!compact} synced={sync&&!deep?view:undefined} adjust={setView}/>)}</div>
  </section>
}
