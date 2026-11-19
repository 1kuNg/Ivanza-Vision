import {DEPTH,DINO,GROUND,GROUPS,OCR,SAM,SIGLIP,type Patch,type Settings} from './api'
export function Range({label,value,set,min=0,max=1,step=.01}:{label:string,value:number,set:(x:number)=>void,min?:number,max?:number,step?:number}){
  return <label className="range"><span>{label}</span><b>{value.toFixed(2)}</b><input type="range" min={min} max={max} step={step} value={value} onChange={e=>set(+e.target.value)}/></label>
}
export function Controls({s,patch,run,busy,collapsed,toggle}:{s:Settings,patch:Patch,run:()=>void,busy:boolean,collapsed:boolean,toggle:()=>void}){
  if(collapsed)return <aside className="collapsed"><button title="Expandir configuração" onClick={toggle}>☰</button></aside>
  const has=(m:string)=>s.models.includes(m)
  const check=(key:'use_sam_mask_filter'|'use_depth_filter'|'ocr_rois',label:string)=><label className="check"><input type="checkbox" checked={s[key]} onChange={e=>patch({[key]:e.target.checked})}/>{label}</label>
  return <aside className="sidebar"><div className="split"><h2>Experimento</h2><button aria-label="Recolher configuração" title="Recolher configuração" onClick={toggle}>‹</button></div>
    <div className="control-scroll">
      {GROUPS.map(([label,models])=><fieldset key={label}><legend>{label}</legend><div className="chips">{models.map(m=><button aria-pressed={has(m)} className={has(m)?'selected':''} key={m} onClick={()=>patch({models:has(m)?s.models.filter(x=>x!==m):[...s.models,m]})}>{m}</button>)}</div></fieldset>)}
      {s.models.some(m=>DINO.includes(m))&&<details open><summary>DINO</summary><div className="parameters">
        <label>Resolução<select value={s.resolution} onChange={e=>patch({resolution:+e.target.value as Settings['resolution']})}><option>224</option><option>384</option><option>512</option></select></label>
        <label>Correspondência<select value={s.mode} onChange={e=>patch({mode:e.target.value as Settings['mode']})}><option value="posicao">Mesma posição</option><option value="regioes">Regiões semelhantes</option></select></label>
        <Range label="Limiar" value={s.threshold} min={.02} max={.90} set={threshold=>patch({threshold})}/>
        <Range label="Opacidade" value={s.opacity} set={opacity=>patch({opacity})}/>
        {check('use_sam_mask_filter','Filtrar fundo com SAM')}{has(DEPTH)&&check('use_depth_filter','Filtrar pela profundidade')}
      </div></details>}
      {has(SAM)&&<details><summary>SAM 2.1</summary><div className="parameters">
        <label>Calibração (px/mm)<input type="number" min="0" step=".1" value={s.px_per_mm} onChange={e=>patch({px_per_mm:+e.target.value})}/></label>
        <label>Método por célula<select value={s.sam_cell_method} onChange={e=>patch({sam_cell_method:e.target.value})}><option value="escala_de_cinza">Diferença em escala de cinza</option><option value="dino_por_celula">DINO por célula</option></select></label>
        {s.sam_cell_method==='dino_por_celula'&&<label>Backbone<select value={s.sam_cell_model} onChange={e=>patch({sam_cell_model:e.target.value})}>{DINO.map(m=><option key={m}>{m}</option>)}</select></label>}
        <label>Limiar por célula (%)<input type="number" min="0" max="100" step=".1" value={s.sam_cell_threshold} onChange={e=>patch({sam_cell_threshold:+e.target.value})}/></label>
      </div></details>}
      {has(GROUND)&&<details open><summary>Grounding DINO</summary><div className="parameters">
        <label>Objetos<input value={s.grounding_prompt} maxLength={320} onChange={e=>patch({grounding_prompt:e.target.value})}/></label>
        <Range label="Limiar da caixa" value={s.grounding_threshold} set={grounding_threshold=>patch({grounding_threshold})}/>
        <Range label="Limiar do texto" value={s.grounding_text_threshold} set={grounding_text_threshold=>patch({grounding_text_threshold})}/>
      </div></details>}
      {has(DEPTH)&&<details open><summary>Depth Anything V2</summary><div className="parameters"><Range label="Corte de profundidade" value={s.depth_threshold} set={depth_threshold=>patch({depth_threshold})}/></div></details>}
      {has(SIGLIP)&&<details open><summary>SigLIP 2</summary><div className="parameters">
        <label>Estados, um por linha<textarea defaultValue={s.siglip_states.join('\n')} onChange={e=>{const items=e.target.value.split('\n').map(s=>s.trim()).filter(Boolean);if(items.length<=16)patch({siglip_states:items})}}/></label>
        <Range label="Limiar de alerta" value={s.siglip_threshold} set={siglip_threshold=>patch({siglip_threshold})}/>
      </div></details>}
      {has(OCR)&&<details open><summary>RapidOCR</summary><div className="parameters">
        {check('ocr_rois','Ler ROIs do Grounding DINO')}<label>Regex<input value={s.ocr_regex} maxLength={250} onChange={e=>patch({ocr_regex:e.target.value})}/></label>
        <Range label="Confiança mínima" value={s.ocr_confidence} set={ocr_confidence=>patch({ocr_confidence})}/>
      </div></details>}
    </div>
    <div className="side-footer"><span>Precisão GPU</span><div className="segments">{(['fp16','fp32'] as const).map(p=><button key={p} className={s.precision===p?'selected':''} onClick={()=>patch({precision:p})}>{p.toUpperCase()}</button>)}</div>
      <button className="primary w-full" onClick={run} disabled={busy||!s.models.length}>{busy?'Executando…':'▶ Executar Comparação'}</button></div>
  </aside>
}
