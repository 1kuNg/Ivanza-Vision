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
type SkuClassEntry={name:string,photos:Photo[]}
type SkuDemoTest={name:string,expected:string,photo:Photo}
type SkuDemo={id:string,title:string,description:string,explanation:string,classes:SkuClassEntry[],tests:SkuDemoTest[]}

export function Sku({settings,task,requestedDemo,onHandled}:{settings:Settings,task:(fn:()=>Promise<void>)=>void,
  requestedDemo?:string,onHandled?:()=>void}){
  const [classes,setClasses]=useState<SkuClassEntry[]>([{name:'OK',photos:[]},{name:'NG',photos:[]}])
  const [model,setModel]=useState(DINO[0]),[active,setActive]=useState(false),[sample,setSample]=useState<Photo>()
  const [limit,setLimit]=useState(.75),[result,setResult]=useState<Prediction>()
  const [demo,setDemo]=useState<SkuDemo>(),[selectedTest,setSelectedTest]=useState(0)
  const [working,setWorking]=useState(false),[status,setStatus]=useState('')
  const [elapsed,setElapsed]=useState<number|undefined>()

  function perform(fn:()=>Promise<void>){
    task(async()=>{setWorking(true);try{await fn()}catch(error){
      setStatus('Não foi possível concluir o teste. Confira a mensagem de erro e a configuração do modelo.')
      throw error
    }finally{setWorking(false)}})
  }
  function edit(i:number,v:Partial<SkuClassEntry>){
    setDemo(undefined);setActive(false);setResult(undefined);setStatus('')
    setClasses(items=>items.map((item,j)=>i===j?{...item,...v}:item))
  }
  async function registerBank(name:string,entries:SkuClassEntry[]){
    await api('/sku/register',{model:name,classes:entries.map(item=>({name:item.name,image_ids:item.photos.map(p=>p.image_id)})),
      resolution:settings.resolution,precision:settings.precision})
    setActive(true)
  }
  async function classify(image:Photo,threshold:number){
    const start=performance.now()
    const prediction=await api<Prediction>('/sku/predict',{sample_id:image.image_id,threshold})
    setElapsed(performance.now()-start)
    setSample(image)
    setResult(prediction)
    return prediction
  }
  function runDemo(id:string,backbone:string){
    perform(async()=>{
      setStatus('Carregando imagens do exemplo...')
      setDemo(undefined);setActive(false);setResult(undefined);setElapsed(undefined)
      const data=await api<SkuDemo>('/sku/demos/'+encodeURIComponent(id)+'/load',{})
      setDemo(data);setClasses(data.classes);setModel(backbone);setSelectedTest(0);setSample(data.tests[0].photo)
      setLimit(.75)
      setStatus('Carregando '+backbone+' e registrando as referências reais...')
      await registerBank(backbone,data.classes)
      setStatus('Classificando a primeira imagem com '+backbone+'...')
      const predicted=await classify(data.tests[0].photo,.75)
      setStatus('Teste executado: '+predicted.label+'. Escolha outra imagem abaixo, inclusive a desconhecida.')
    })
  }
  useEffect(()=>{
    if(requestedDemo){
      onHandled?.()
      runDemo(requestedDemo,'DINOv2 Small')
    }
    // Este comando é disparado somente por uma seleção na ajuda.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[requestedDemo])

  function selectAndClassify(test:SkuDemoTest,index:number){
    setSelectedTest(index)
    setSample(test.photo);setResult(undefined)
    perform(async()=>{
      setStatus('Classificando '+test.name+'...')
      const output=await classify(test.photo,limit)
      setStatus('Resultado: '+output.label+' | esperado no exemplo: '+test.expected+'.')
    })
  }
  return <div className="sku-workspace">
    <section className="card sku-welcome">
      <div><span className="help-tag">TESTES GUIADOS · FEW-SHOT / SKU</span>
        <h2>Ensine a IA a reconhecer produtos com poucas imagens</h2>
        <p>O programa compara a foto com exemplos de cada classe, sem treinar um modelo do zero.
          Escolha um conjunto pronto para ver tudo funcionando ou cadastre suas próprias imagens.</p></div>
      <div className="sku-help-text"><strong>Como interpretar:</strong> similaridade maior significa mais parecido;
        margem maior significa mais diferença entre a melhor classe e a segunda.
        O limiar decide quando responder <strong>Desconhecido</strong>. Esses números não são porcentagens de certeza.</div>
    </section>

    <div className="sku-demo-grid">
      {([
        {id:'embalagens',title:'Frasco azul x caixa laranja',description:'Três referências de cada produto, duas imagens novas e uma lata desconhecida.'},
        {id:'fixadores',title:'Porca x arruela',description:'Três referências de cada peça, duas imagens novas e um parafuso desconhecido.'},
      ] as const).map(item=><article className="card sku-demo-card" key={item.id}>
        <div><span className="help-tag">Exemplo incluído · imagens sintéticas</span><h3>{item.title}</h3><p>{item.description}</p></div>
        <div className="sku-demo-actions">
          <button className="primary" disabled={working} onClick={()=>runDemo(item.id,'DINOv2 Small')}>Testar DINOv2</button>
          <button disabled={working} onClick={()=>runDemo(item.id,'DINOv3 ConvNeXt Tiny')}>Testar ConvNeXt Tiny</button>
        </div>
      </article>)}
    </div>
    <div className="card sku-note">
      <strong>O que é DINOv3 ConvNeXt Tiny?</strong> É outro tipo de extrator visual DINOv3, baseado em uma rede convolucional (CNN).
      Vale testar se reconhece as mesmas peças com boa velocidade e menos memória no seu computador.
      Para usar, aceite o acesso do checkpoint em <strong>Config / Tokens → DINOv3 ConvNeXt Tiny</strong>.
      O primeiro uso pode baixar os pesos. Os exemplos são ilustrações para experimentar — não substituem fotos industriais.
    </div>

    {status&&<div className="card sku-status" role="status">{working&&<span className="sku-activity" aria-hidden="true"/>}{status}</div>}

    {demo&&<section className="card sku-loaded">
      <div className="split"><div><h2>{demo.title} · imagens carregadas</h2><p>{demo.explanation}</p></div>
        <span className="good">Referências: {demo.classes.reduce((sum,c)=>sum+c.photos.length,0)} fotos</span></div>
      <div className="sku-reference-grid">{demo.classes.map(entry=><div key={entry.name} className="sku-reference-group">
        <h3>{entry.name} <small>· {entry.photos.length} fotos de referência</small></h3>
        <div className="sku-reference-photos">{entry.photos.map((p,i)=><img key={p.image_id} src={p.url} alt={entry.name+' referência '+(i+1)}/>)}</div>
      </div>)}</div>
      <h3>Experimente as três imagens de teste</h3>
      <div className="sku-trial-grid">{demo.tests.map((test,i)=><button key={test.photo.image_id}
        className={'sku-trial'+(selectedTest===i?' selected':'')} disabled={working}
        onClick={()=>active?selectAndClassify(test,i):(setSample(test.photo),setSelectedTest(i),setResult(undefined))}>
        <img src={test.photo.url} alt={'Teste '+test.name}/>
        <strong>{test.name}</strong>
        <small>{i===2?'Objeto não cadastrado':'Classe conhecida'} · {active?'Clique para classificar':'Selecione e registre o banco'}</small>
      </button>)}</div>
    </section>}

    <div className="split-panels">
      <section className="card"><h2>Cadastro de classes</h2>
        <p className="muted">Também pode substituir as imagens do exemplo por fotos reais (1 a 10 por classe).</p>
        <label>Modelo DINO<select value={model} disabled={working} onChange={e=>{setModel(e.target.value);setActive(false);setResult(undefined)}}>
          {DINO.map(m=><option key={m}>{m}</option>)}</select></label>
        {classes.map((entry,i)=><div key={i} className="sku-class-block">
          <div className="class-entry"><input aria-label={'Nome da classe '+(i+1)} value={entry.name}
            disabled={working} onChange={e=>edit(i,{name:e.target.value})}/>
            <Files label={'Fotos · '+entry.photos.length} multiple change={files=>perform(async()=>
              edit(i,{photos:await Promise.all(files.slice(0,10).map(upload))}))}/>
            {classes.length>2&&<button disabled={working} onClick={()=>{setClasses(cs=>cs.filter((_,j)=>j!==i));setActive(false);setDemo(undefined)}}>Remover</button>}
          </div>
          {entry.photos.length>0&&<div className="sku-mini-photos">{entry.photos.map(p=><img key={p.image_id} src={p.url} alt={'Foto de '+entry.name}/>)}</div>}
        </div>)}
        <div className="toolbar"><button disabled={working||classes.length>=6} onClick={()=>{setClasses([...classes,{name:'',photos:[]}]);setActive(false);setDemo(undefined)}}>+ Classe</button>
          <button className="primary" disabled={working||classes.some(c=>!c.name.trim()||!c.photos.length)}
            onClick={()=>perform(async()=>{setStatus('Registrando '+model+'...');await registerBank(model,classes);setResult(undefined);setStatus('Banco registrado. Agora classifique uma imagem.')})}>Registrar banco</button>
          <button disabled={working} onClick={()=>perform(async()=>{await api('/sku/clear',undefined,'DELETE');setActive(false);setResult(undefined);setStatus('Banco limpo.')})}>Limpar</button>
        </div>
      </section>
      <section className="card">
        <div className="split"><h2>Teste e similaridade</h2>{active&&<span className="good">Banco disponível · {model}</span>}</div>
        <p className="muted">Ajuste o limiar se um produto desconhecido for classificado como um produto conhecido.</p>
        <Range label="Limiar de desconhecido" value={limit} min={-1} max={1} set={v=>{setLimit(v);setResult(undefined)}}/>
        <div className="toolbar">
          <Files label="Imagem de teste" change={files=>perform(async()=>{setDemo(undefined);setSelectedTest(0);setSample(await upload(files[0]));setResult(undefined)})}/>
          <button className="primary" disabled={working||!sample||!active} onClick={()=>perform(async()=>{
            setStatus('Classificando imagem com '+model+'...');await classify(sample!,limit);setStatus('Classificação concluída.')
          })}>Classificar</button>
        </div>
        {!active&&<p className="muted">Primeiro registre um banco ou escolha um teste pronto acima.</p>}
        {sample&&<img className="sku-image" src={sample.url} alt="Amostra SKU"/>}
        {result&&<><ClassResult value={result}/><p className="sku-interpret">
          <strong>Classificação: {result.label}.</strong> Similaridade {result.similarity.toFixed(3)}, margem {result.margin.toFixed(3)}.
          {elapsed!==undefined&&<> Inferência aproximada {elapsed.toFixed(0)} ms (inclui chamada à API).</>}
          {demo&&<> Esperado neste teste: <strong>{demo.tests[selectedTest]?.expected||'—'}</strong>.</>}
          {result.label==='Desconhecido'
            ?' O resultado ficou abaixo do limiar definido. Não significa que o sistema provou que é uma nova classe.'
            :' Esta é a classe mais semelhante dentre as cadastradas, não uma certeza absoluta.'}
        </p></>}
      </section>
    </div>
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
