import {useEffect,useRef,useState} from 'react'
import {createRoot} from 'react-dom/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import version from '../../VERSION?raw'
import {Editor} from './canvas'
import {Controls} from './controls'
import {Results} from './results'
import {Files,Sku,Webcam} from './screens'
import {Guide} from './help'
import {api,DINO,DEPTH,GROUND,OCR,SAM,initial,upload,save,type Photo,type Round,type Settings,type System} from './api'
import './style.css'

const commandExamples = [
  ['Modelos e resolução', 'Ative o DINOv3 ConvNeXt e o SAM, use 512 px, ligue o filtro de máscara e compare por regiões.'],
  ['Filtro de profundidade', 'Ative o filtro de profundidade, ajuste o corte para 0,65 e o limiar para 0,30.'],
  ['OCR no Grounding', 'Ative o RapidOCR e o Grounding DINO. Use as ROIs e exija confiança mínima de 0,80.'],
] as const
const pages=['Comparador','Pipelines','Few-Shot / SKU','Webcam','Guia Industrial'] as const
type Page=(typeof pages)[number]
type Proposal={action:string,params:Partial<Settings>&{pipeline_id?:string},summary:string,warnings:string[],requires_confirmation:true}
const presets:Record<string,string[]>={grounding_sam_dino:[GROUND,SAM,DINO[1]],grounding_ocr:[GROUND,OCR],depth_sam:[DEPTH,SAM],sam_dino:[SAM,DINO[0]]}
const stageLabel=(id:string)=>presets[id]||[]
const parameterNames:Record<string,string>={models:'Modelos',resolution:'Resolução',threshold:'Limiar',precision:'Precisão',
  mode:'Modo',opacity:'Opacidade',use_sam_mask_filter:'Filtro SAM',use_depth_filter:'Filtro de profundidade',
  depth_threshold:'Corte de profundidade',grounding_prompt:'Objetos do Grounding',grounding_threshold:'Limiar da caixa',
  grounding_text_threshold:'Limiar do texto',siglip_states:'Estados SigLIP',siglip_threshold:'Limiar SigLIP',
  ocr_confidence:'Confiança do OCR',ocr_regex:'Regex',ocr_rois:'ROIs do OCR',px_per_mm:'px/mm',
  sam_cell_method:'Método por célula',pipeline_id:'Pipeline'}
const resetMarks={points:[],boxes:[],grid_rows:0,grid_cols:0}
const display=(value:unknown)=>value===true?'Ligado':value===false?'Desligado':Array.isArray(value)?value.join(', '):value==null?'—':String(value)

function App(){
  const [page,setPage]=useState<Page>('Comparador'),[system,setSystem]=useState<System>(),[s,setSettings]=useState(initial)
  const [sample,setSample]=useState<Photo>(),[refs,setRefs]=useState<Photo[]>([]),[examples,setExamples]=useState<Photo[]>([])
  const [round,setRound]=useState<Round>(),[compare,setCompare]=useState<Round>(),[pipelineRound,setPipelineRound]=useState<Round>()
  const [pipeline,setPipeline]=useState('grounding_ocr'),[entry,setEntry]=useState(true),[collapsed,setCollapsed]=useState(false),[editor,setEditor]=useState<Photo>()
  const [busy,setBusy]=useState(false),[message,setMessage]=useState(''),[configOpen,setConfigOpen]=useState(false),[assistant,setAssistant]=useState(false)
  const [runningModels,setRunningModels]=useState<string[]>([]),[toast,setToast]=useState<{text:string,error:boolean}>()
  const pending=useRef(false)
  useEffect(()=>{if(!toast)return;const id=window.setTimeout(()=>setToast(undefined),6000);return()=>window.clearTimeout(id)},[toast])
  const patch=(next:Partial<Settings>)=>setSettings(prev=>({...prev,...next}))
  const refresh=()=>api<System>('/system/status').then(setSystem)
  useEffect(()=>{refresh().then(()=>api<Photo[]>('/examples')).then(setExamples).catch(e=>setMessage(e.message))},[])
  function task(fn:()=>Promise<void>){
    if(pending.current)return
    pending.current=true;setBusy(true);setMessage('')
    void fn().catch(e=>setMessage(e instanceof Error?e.message:String(e))).finally(()=>{pending.current=false;setBusy(false)})
  }
  function pickPhoto(photo:Photo){setSample(photo);patch(resetMarks)}
  function preserveRefs(photos:Photo[]){
    setRefs(old=>[...new Map([...old,...photos].map(p=>[p.image_id,p])).values()].slice(0,10))
  }
  function pickExample(name:string){
    const isTray=name==='bandeja'
    const trial=examples.find(x=>x.name===(isTray?'bandeja_faltando.png':'painel_faltando_peca.png'))
    const reference=examples.find(x=>x.name===(isTray?'bandeja_normal.png':'painel_normal.png'))
    if(trial&&reference){setSample(trial);setRefs([reference]);patch({...resetMarks,...(isTray?{grid_rows:2,grid_cols:4}:{})})}
  }
  async function execute(id?:string){
    if(!sample)throw Error('Escolha uma imagem de teste.')
    setRunningModels(id?stageLabel(id):s.models)
    try {
    const result=await api<Round>(id?'/pipeline/run':'/analyze',{
      ...s,sample_id:sample.image_id,ref_ids:refs.map(p=>p.image_id),...(id?{pipeline_id:id}:{})})
    setRound(result);if(id)setPipelineRound(result);else setCompare(result)
    setEntry(false);void refresh().catch(()=>{})
    const errors=result.benchmark.filter(row=>row.status==='error')
    if(errors.length)setToast({text:'Erro em: '+errors.map(row=>row.model).join(', '),error:true})
    } finally { setRunningModels([]) }
  }
  async function exportRecipe(){
    const data=await api('/recipe/export',s)
    save('receita_ivanza.json',JSON.stringify(data,null,2))
    setToast({text:'Receita exportada.',error:false})
  }
  function choosePipeline(id:string){setPipeline(id);patch({models:stageLabel(id)})}
  function navigate(next:Page){
    setPage(next)
    if(next==='Pipelines')choosePipeline(pipeline)
  }
  async function confirm(p:Proposal){
    if(!p.requires_confirmation)return
    switch(p.action){
      case 'select_models': patch({models:p.params.models});break
      case 'update_params': patch(p.params);break
      case 'run_analysis': setPage('Comparador');await execute();break
      case 'run_pipeline': if(p.params.pipeline_id){setPipeline(p.params.pipeline_id);setPage('Pipelines');await execute(p.params.pipeline_id)}break
      case 'export_recipe': await exportRecipe();return
    }
    setToast(old=>old?.error?old:{text:'Comando aplicado.',error:false})
  }
  const input=<details open={entry} onToggle={e=>setEntry(e.currentTarget.open)} className="card input">
    <summary>Imagens de bancada <span className="muted mono">{sample?'1 teste':'Sem teste'} · {refs.length} ref.</span></summary>
    <div className="toolbar"><button onClick={()=>pickExample('painel')}>Exemplo: painel</button><button onClick={()=>pickExample('bandeja')}>Exemplo: bandeja</button>
      <Files label="Imagem de teste" change={files=>task(async()=>pickPhoto(await upload(files[0])))}/>
      <Files label="Referências" multiple change={files=>task(async()=>preserveRefs(await Promise.all(files.slice(0,10).map(upload))))}/>
      {sample&&(s.models.includes(SAM)||s.use_sam_mask_filter)&&<button onClick={()=>setEditor(sample)}>Editar marcações SAM</button>}
    </div>
    {(sample||refs.length>0)&&<div className="thumbnails">
      {sample&&<figure><img src={sample.url} alt="Imagem de teste"/><figcaption>Teste · {sample.width} × {sample.height}</figcaption></figure>}
      {refs.map((p,i)=><figure key={p.image_id}><button title="Remover referência" aria-label={'Remover referência '+(i+1)} onClick={()=>setRefs(old=>old.filter(x=>x.image_id!==p.image_id))}>×</button><img src={p.url} alt={'Referência '+(i+1)}/><figcaption>Referência {i+1}</figcaption></figure>)}
    </div>}
  </details>
  const configVisible=['Comparador','Pipelines','Webcam'].includes(page)
  return <div className="app">
    <header className="top"><div className="brand"><i>IV</i><div><strong>IVANZATECH</strong><span>VISION LAB <b>v{version.trim()}</b></span></div></div>
      <nav>{pages.map(p=><button key={p} className={page===p?'current':''} onClick={()=>navigate(p)}>{p==='Guia Industrial'?'Ajuda & Testes':p}</button>)}</nav>
      <div className="top-actions"><span className="hardware"><i/>{system?.device==='CUDA'?'CUDA · '+s.precision.toUpperCase():system?.device||'—'}</span>
        <button onClick={()=>setAssistant(!assistant)}>Assistente IA</button><button onClick={()=>setConfigOpen(true)}>Config / Tokens</button>
        <button title="Descarregar modelos" disabled={busy} onClick={()=>task(async()=>{await api('/system/vram/clear',{});await refresh();setToast({text:'VRAM e cache de modelos liberados.',error:false})})}>Limpar VRAM</button>
        <button disabled={busy} onClick={()=>task(exportRecipe)}>Exportar Receita</button>
      </div>
    </header>
    <div className="workspace">
      {configVisible&&<Controls s={s} patch={patch} run={()=>task(()=>execute(page==='Pipelines'?pipeline:undefined))} busy={busy} collapsed={collapsed} toggle={()=>setCollapsed(!collapsed)}/>}
      <main>
        {page!=='Guia Industrial'&&<div className="page-title"><div><h1>{page}</h1><p>{({Comparador:'Compare diagnóstico, imagem e latência na mesma rodada.',Pipelines:'Encadeie modelos sobre a mesma imagem.', 'Few-Shot / SKU':'Classificação por referências de cada classe.',Webcam:'Captura local e inferência assíncrona.'} as Record<string,string>)[page]}</p></div>{page==='Comparador'&&<span className="muted mono">{compare?.benchmark.length||0} modelos executados</span>}</div>}
        {message&&<div role="status" className={message.startsWith('Cache')?'notice':'error'}>{message}<button aria-label="Fechar aviso" onClick={()=>setMessage('')}>×</button></div>}
        {page==='Comparador'&&<>{input}{runningModels.length>0?<AnalysisSkeleton models={runningModels}/>:compare?<Results round={compare} s={s}/>:<div className="card empty"><span className="empty-target">⊕</span><h2>Palco de comparação</h2><p>Selecione uma imagem e execute os modelos.</p></div>}</>}
        {page==='Pipelines'&&<>{input}<div className="card toolbar"><label>Fluxo <select value={pipeline} onChange={e=>choosePipeline(e.target.value)}>{Object.entries(system?.pipelines||{grounding_ocr:'Grounding DINO → RapidOCR'}).map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label><button className="primary" disabled={busy||!sample} onClick={()=>task(()=>execute(pipeline))}>▶ Executar Pipeline</button></div>
          <div className="flow">{['Entrada',...stageLabel(pipeline),'Saída'].map((step,i)=><div className="card" key={step}><small>{i===0?'IMAGEM':i===stageLabel(pipeline).length+1?'RESULTADO':'ETAPA '+i}</small><strong>{step}</strong>{i===0&&sample&&<img src={sample.url} alt="Entrada do pipeline"/>}</div>)}</div>
          {runningModels.length>0?<AnalysisSkeleton models={runningModels}/>:pipelineRound&&pipelineRound.pipeline===pipeline&&<Results round={pipelineRound} s={s}/>}</>}
        <div hidden={page!=='Few-Shot / SKU'}><Sku settings={s} task={task}/></div>
        {page==='Webcam'&&<Webcam settings={s} refIds={refs.map(p=>p.image_id)} addRef={p=>preserveRefs([p])} edit={p=>setEditor(p)} task={task} onRound={setRound}/>}
        {page==='Guia Industrial'&&<Guide onDemo={demo=>{pickExample(demo);setPage('Comparador')}} onPage={navigate} onPipeline={id=>{choosePipeline(id);setPage('Pipelines')}}/>}
      </main>
    </div>
    {toast&&<div className={'toast '+(toast.error?'error':'notice')} role={toast.error?'alert':'status'}>{toast.text}<button aria-label="Fechar notificação" onClick={()=>setToast(undefined)}>×</button></div>}
    {editor&&<Editor photo={editor} settings={s} patch={patch} close={()=>setEditor(undefined)}/>}
    {configOpen&&<ConfigModal system={system} close={()=>setConfigOpen(false)} refresh={refresh}/>}
    {assistant&&<Assistant current={round} s={s} close={()=>setAssistant(false)} confirm={confirm} pipelines={system?.pipelines||{}}/>}
  </div>
}

function AnalysisSkeleton({models}:{models:string[]}){
  return <section className="result-grid" aria-live="polite" aria-label="Modelos em execução">
    {models.map(model=><article className="card result-card" key={model}><header className="split">
      <h3>{model}</h3><span className="latency-badge">Em andamento</span></header>
      <div className="skeleton-image"/><div className="skeleton-metrics"/>
    </article>)}
  </section>
}

function ConfigModal({system,close,refresh}:{system?:System,close:()=>void,refresh:()=>Promise<void>}){
  const [token,setToken]=useState(''),[checkpoint,setCheckpoint]=useState('vit')
  const [url,setUrl]=useState(system?.base_url||'https://openrouter.ai/api/v1'),[key,setKey]=useState(''),[model,setModel]=useState(system?.llm_model||'openrouter/free')
  const [status,setStatus]=useState(''),[failed,setFailed]=useState(false),[busy,setBusy]=useState(false)
  async function submit(action:()=>Promise<unknown>,success:string){
    setBusy(true);setFailed(false);setStatus('')
    try{await action();await refresh();setStatus(success)}catch(e){setFailed(true);setStatus(String(e))}finally{setBusy(false)}
  }
  const hf=(mode:string)=>submit(async()=>{await api('/auth/hf/'+mode,{checkpoint,...(token?{token}:{})});setToken('')},
                                mode==='save'?'Login salvo no perfil local.':'Acesso confirmado nesta sessão.')
  return <div className="modal-shade"><section className="modal config" role="dialog" aria-label="Configuração e credenciais">
    <div className="split"><h2>Config / Tokens</h2><button aria-label="Fechar" onClick={close}>×</button></div>
    <fieldset><legend>Hugging Face</legend><label>Checkpoint<select value={checkpoint} onChange={e=>setCheckpoint(e.target.value)}><option value="vit">DINOv3 ViT-S/16</option><option value="convnext">DINOv3 ConvNeXt Tiny</option></select></label>
      <label>Token de leitura<input type="password" autoComplete="off" value={token} onChange={e=>setToken(e.target.value)} placeholder={system?.hf_configured?'Credencial configurada':'hf_…'}/></label>
      <div className="toolbar"><button disabled={busy} onClick={()=>void hf('verify')}>Testar acesso</button><button disabled={busy} onClick={()=>void hf('save')}>Salvar no perfil local</button></div></fieldset>
    <fieldset><legend>Assistente OpenAI-compatible</legend><label>URL<input value={url} onChange={e=>setUrl(e.target.value)}/></label>
      <label>Modelo<input value={model} onChange={e=>setModel(e.target.value)}/></label>
      <label>Chave<input type="password" autoComplete="off" value={key} onChange={e=>setKey(e.target.value)} placeholder={system?.llm_configured?'Credencial configurada':'Opcional para motor local'}/></label>
      <button className="primary" disabled={busy} onClick={()=>void submit(async()=>{await api('/assistant/config',{base_url:url,model,...(key?{key}:{})});setKey('')},'Configuração aplicada nesta sessão.')}>Aplicar à sessão</button></fieldset>
    {status&&<p className={failed?'error':'good'}>{status}</p>}
  </section></div>
}
function Assistant({current,s,close,confirm,pipelines}:{current?:Round,s:Settings,close:()=>void,confirm:(p:Proposal)=>Promise<void>,pipelines:Record<string,string>}){
  const [tab,setTab]=useState<'report'|'command'>('report'),[focus,setFocus]=useState(''),[ocr,setOcr]=useState(false),[report,setReport]=useState('')
  const [instruction,setInstruction]=useState(''),[proposal,setProposal]=useState<Proposal>(),[error,setError]=useState(''),[busy,setBusy]=useState(false)
  async function act(fn:()=>Promise<void>){setBusy(true);setError('');try{await fn()}catch(e){setError(String(e))}finally{setBusy(false)}}
  const changes=proposal?.params||{}
  return <aside className="drawer" aria-label="Assistente IA">
    <header className="split"><h2>Assistente IA</h2><button aria-label="Fechar assistente" onClick={close}>×</button></header>
    <div className="segments"><button className={tab==='report'?'selected':''} onClick={()=>setTab('report')}>Relatório & Explicação</button><button className={tab==='command'?'selected':''} onClick={()=>setTab('command')}>Comandos (Beta)</button></div>
    {tab==='report'?<>
      <label>Foco do parecer<textarea maxLength={800} value={focus} placeholder="Ex.: comparar latência em CPU" onChange={e=>setFocus(e.target.value)}/></label>
      <label className="check"><input type="checkbox" checked={ocr} onChange={e=>setOcr(e.target.checked)}/> Enviar textos OCR ao provedor</label>
      <button className="primary" disabled={busy||!current?.run_id} onClick={()=>void act(async()=>setReport((await api<{markdown:string}>('/assistant/report',{run_id:current?.run_id,focus,allow_ocr:ocr})).markdown))}>Gerar Parecer da Rodada</button>
      {report&&<><article className="report"><ReactMarkdown remarkPlugins={[remarkGfm]} disallowedElements={['img','a']} unwrapDisallowed>{report}</ReactMarkdown></article>
      <div className="toolbar"><button onClick={()=>void act(()=>navigator.clipboard.writeText(report))}>Copiar</button><button onClick={()=>save('parecer_ivanza.md',report,'text/markdown')}>Baixar .md</button></div></>}
    </>:<>
      <div className="chips command-examples">{commandExamples.map(([label,example])=><button key={label}
        onClick={()=>{setInstruction(example);setProposal(undefined)}}>{label}</button>)}</div>
      <label>Instrução<textarea value={instruction} maxLength={1400} placeholder="Ex.: ativa o DINOv3 ConvNeXt em 512 px e ajusta o limiar para 0,30" onChange={e=>{setInstruction(e.target.value);setProposal(undefined)}}/></label>
      <button className="primary" disabled={busy||!instruction.trim()} onClick={()=>void act(async()=>{setProposal(undefined);setProposal(await api<Proposal>('/assistant/command',{instruction,settings:s}))})}>Preparar proposta</button>
      {proposal&&<article className="card proposal"><h3>{proposal.summary}</h3>
        {Object.entries(changes).map(([k,v])=><div className="change" key={k}><strong>{parameterNames[k]||k}</strong>
          <div className="diff"><code aria-label="Valor atual">{display((s as unknown as Record<string,unknown>)[k])}</code>
            <span aria-hidden="true">→</span><code aria-label="Novo valor">{k==='pipeline_id'?pipelines[String(v)]||String(v):display(v)}</code></div></div>)}
        {proposal.warnings.map((w,i)=><p className="muted" key={i}>{w}</p>)}
        <div className="toolbar"><button className="primary" disabled={busy} onClick={()=>void act(async()=>{await confirm(proposal);setProposal(undefined)})}>Confirmar e Aplicar</button><button onClick={()=>setProposal(undefined)}>Cancelar</button></div>
      </article>}
    </>}
    {busy&&<p role="status" className="muted">Aguardando resposta…</p>}{error&&<p role="alert" className="error">{error}</p>}
  </aside>
}
createRoot(document.getElementById('root')!).render(<App/>)
