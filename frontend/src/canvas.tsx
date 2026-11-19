import {useEffect,useRef,useState} from 'react'
import type {Box,Photo,Point,Settings,Patch} from './api'

type Mode='point'|'fix'|'grid'|'box'
type Marks=Pick<Settings,'points'|'boxes'|'grid_rows'|'grid_cols'|'grid_bounds'>
type Drag={kind:'pan'|'box'|'grid',start:[number,number], prior:Box, corner:number}
const clamp=(n:number)=>Math.max(0,Math.min(1,n))
const capture=(s:Settings):Marks=>({points:s.points,boxes:s.boxes,grid_rows:s.grid_rows,grid_cols:s.grid_cols,grid_bounds:s.grid_bounds})
export function Editor({photo,settings,patch,close}:{photo:Photo,settings:Settings,patch:Patch,close:()=>void}){
  const element=useRef<HTMLCanvasElement>(null), image=useRef<HTMLImageElement|null>(null), space=useRef(false), undo=useRef<Marks[]>([]), drag=useRef<Drag|null>(null)
  const [mode,setMode]=useState<Mode>(settings.grid_rows?'grid':settings.boxes.length?'box':'point'), [active,setActive]=useState(0),[sign,setSign]=useState<0|1>(1)
  const [view,setView]=useState({zoom:1,x:0,y:0}),[draft,setDraft]=useState<Box|null>(null),[tick,redraw]=useState(0)
  function change(next:Partial<Marks>){undo.current.push(capture(settings));patch(next)}
  function fit(c:HTMLCanvasElement){
    const fit=Math.min(c.clientWidth/photo.width,c.clientHeight/photo.height)*view.zoom
    const w=fit*photo.width,h=fit*photo.height
    return {w,h,l:(c.clientWidth-w)/2+view.x,t:(c.clientHeight-h)/2+view.y}
  }
  useEffect(()=>{
    const el=element.current!
    const im=new Image(); im.onload=()=>{image.current=im;redraw(n=>n+1)}; im.src=photo.url
    const observer=new ResizeObserver(()=>redraw(n=>n+1));observer.observe(el)
    return ()=>{observer.disconnect();im.onload=null}
  },[photo.url])
  useEffect(()=>{
    const c=element.current!,ctx=c.getContext('2d')!,ratio=window.devicePixelRatio||1
    c.width=c.clientWidth*ratio;c.height=c.clientHeight*ratio
    ctx.scale(ratio,ratio);ctx.fillStyle='#090d12';ctx.fillRect(0,0,c.clientWidth,c.clientHeight)
    const {w,h,l,t}=fit(c)
    if(image.current)ctx.drawImage(image.current,l,t,w,h)
    function rect(b:Box,color:string,label?:string){
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(l+b[0]*w,t+b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h)
      if(label){ctx.fillStyle=color;ctx.font='13px monospace';ctx.fillText(label,l+b[0]*w+5,t+b[1]*h+15)}
    }
    if(mode==='grid'&&settings.grid_rows){
      const b=draft||settings.grid_bounds
      for(let i=0;i<settings.grid_rows;i++)for(let j=0;j<settings.grid_cols;j++){
        const x=(b[2]-b[0])/settings.grid_cols,y=(b[3]-b[1])/settings.grid_rows
        rect([b[0]+j*x,b[1]+i*y,b[0]+(j+1)*x,b[1]+(i+1)*y],'#62b3e4',String(i*settings.grid_cols+j+1))
      }
      ctx.fillStyle='#f07126';for(const x of [b[0],b[2]])for(const y of [b[1],b[3]])ctx.fillRect(l+x*w-5,t+y*h-5,10,10)
    }else if(mode==='box'){
      settings.boxes.forEach((box,i)=>rect(box,i===active?'#f07126':'#62b3e4',String(i+1)))
      if(draft)rect(draft,'#f07126')
    }else settings.points.forEach((pts,i)=>pts.forEach(([x,y,positive])=>{
      ctx.beginPath();ctx.arc(l+w*x,t+h*y,i===active?7:5,0,Math.PI*2);ctx.fillStyle=positive?'#54cf92':'#f85149';ctx.fill()
      ctx.strokeStyle='#0b0f14';ctx.stroke();ctx.fillStyle='#fff';ctx.font='13px monospace';ctx.fillText(String(i+1),l+w*x+9,t+h*y)
    }))
  },[photo,settings,view,mode,active,draft,tick])
  useEffect(()=>{
    const c=element.current!
    const zoom=(e:WheelEvent)=>{e.preventDefault();setView(v=>{
      const zoom=Math.min(8,Math.max(1,v.zoom*Math.exp(-e.deltaY*.0017))),q=zoom/v.zoom,b=c.getBoundingClientRect()
      const x=e.clientX-b.left-c.clientWidth/2,y=e.clientY-b.top-c.clientHeight/2
      return {zoom,x:x-(x-v.x)*q,y:y-(y-v.y)*q}
    })}
    c.addEventListener('wheel',zoom,{passive:false});return ()=>c.removeEventListener('wheel',zoom)
  },[])
  function setTool(next:Mode){
    setMode(next);setActive(0)
    if(next==='grid')change({grid_rows:settings.grid_rows||2,grid_cols:settings.grid_cols||4,boxes:[],points:[]})
    else if(next==='box')change({grid_rows:0,grid_cols:0,points:[]})
    else if(settings.grid_rows||settings.boxes.length)change({grid_rows:0,grid_cols:0,boxes:[]})
  }
  function remove(){
    if(mode==='box')change({boxes:settings.boxes.filter((_,i)=>i!==active)})
    else if(mode==='grid')change({grid_rows:0,grid_cols:0})
    else change({points:settings.points.filter((_,i)=>i!==active)})
    setActive(0)
  }
  const norm=(e:React.PointerEvent<HTMLCanvasElement>):[number,number]=>{
    const c=e.currentTarget,b=c.getBoundingClientRect(),{w,h,l,t}=fit(c)
    return [(e.clientX-b.left-l)/w,(e.clientY-b.top-t)/h]
  }
  return <div className="modal-shade"><section className="modal editor" aria-label="Editor de marcações">
    <header className="split"><h2>Marcações SAM 2.1</h2><button onClick={close}>Concluir</button></header>
    <div className="toolbar">{([['point','1 peça por clique'],['fix','Corrigir (+ / −)'],['grid','Grade arrastável'],['box','Box prompt']] as const).map(([key,label])=><button className={mode===key?'selected':''} key={key} onClick={()=>setTool(key)}>{label}</button>)}
      {mode==='grid'?<><label>Linhas <input type="number" min="1" max="4" value={settings.grid_rows||2} onChange={e=>{const n=Math.min(4,Math.max(1,+e.target.value));change({grid_rows:n,grid_cols:Math.min(settings.grid_cols||4,Math.floor(12/n))})}}/></label><label>Colunas <input type="number" min="1" max={Math.min(6,Math.floor(12/(settings.grid_rows||2)))} value={settings.grid_cols||4} onChange={e=>change({grid_rows:settings.grid_rows||2,grid_cols:Math.max(1,Math.min(6,Math.floor(12/(settings.grid_rows||2)),+e.target.value))})}/></label></>:
      <select aria-label="Item ativo" value={active} onChange={e=>setActive(+e.target.value)}><option value={0}>Item 1</option>{Array.from({length:Math.max(0,(mode==='box'?settings.boxes:settings.points).length-1)},(_,i)=><option value={i+1} key={i}>{'Item '+(i+2)}</option>)}</select>}
      {mode==='fix'&&<select aria-label="Sinal" value={sign} onChange={e=>setSign(+e.target.value as 0|1)}><option value={1}>Positivo</option><option value={0}>Negativo</option></select>}
      <button onClick={()=>setView({zoom:1,x:0,y:0})}>Ajustar à tela</button><button onClick={remove}>Remover</button>
    </div>
    <canvas tabIndex={0} ref={element} onBlur={()=>{space.current=false}} onContextMenu={e=>e.preventDefault()}
      onKeyDown={e=>{if(e.code==='Space'){e.preventDefault();space.current=true}if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();const previous=undo.current.pop();if(previous){patch(previous);setMode(previous.grid_rows?'grid':previous.boxes.length?'box':'point')}}if(e.key==='Delete')remove()}}
      onKeyUp={e=>{if(e.code==='Space')space.current=false}}
      onPointerDown={e=>{
        e.currentTarget.focus()
        if(e.button===1||space.current){e.preventDefault();e.currentTarget.setPointerCapture(e.pointerId);drag.current={kind:'pan',start:[e.clientX,e.clientY],prior:[view.x,view.y,0,0],corner:0};return}
        const xy=norm(e)
        if(xy.some(v=>v<0||v>1)||e.button!==0)return
        if(mode==='box'&&settings.boxes.length<12){drag.current={kind:'box',start:xy,prior:[0,0,0,0],corner:0};e.currentTarget.setPointerCapture(e.pointerId)}
        if(mode==='grid'){
          const b=settings.grid_bounds, corners=[[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]]
          const {w,h}=fit(e.currentTarget)
          const idx=corners.findIndex(p=>Math.hypot((p[0]-xy[0])*w,(p[1]-xy[1])*h)<22)
          if(idx>=0){drag.current={kind:'grid',start:xy,prior:b,corner:idx};e.currentTarget.setPointerCapture(e.pointerId)}
        }
        const point:Point=[...xy,mode==='fix'?sign:1]
        if(mode==='point'&&settings.points.length<12){setActive(settings.points.length);change({points:[...settings.points,[point]]})}
        if(mode==='fix'&&settings.points[active]?.length<12)change({points:settings.points.map((pts,i)=>i===active?[...pts,point]:pts)})
      }}
      onPointerMove={e=>{
        const d=drag.current;if(!d)return
        if(d.kind==='pan'){setView(v=>({...v,x:d.prior[0]+e.clientX-d.start[0],y:d.prior[1]+e.clientY-d.start[1]}));return}
        const [x,y]=norm(e).map(clamp),[a,b]=d.start
        if(d.kind==='box')setDraft([Math.min(a,x),Math.min(b,y),Math.max(a,x),Math.max(b,y)])
        else{const p=[...d.prior] as Box;const ix=d.corner%2===0?0:2,iy=d.corner<2?1:3;p[ix]=ix===0?Math.min(x,p[2]-.02):Math.max(x,p[0]+.02);p[iy]=iy===1?Math.min(y,p[3]-.02):Math.max(y,p[1]+.02);setDraft(p)}
      }}
      onPointerUp={()=>{const d=drag.current;drag.current=null;if(d&&draft){if(d.kind==='grid')change({grid_bounds:draft});if(d.kind==='box'&&draft[2]-draft[0]>.005&&draft[3]-draft[1]>.005){setActive(settings.boxes.length);change({boxes:[...settings.boxes,draft]})}}setDraft(null)}}
      onPointerCancel={()=>{drag.current=null;setDraft(null)}}/>
    <small>Scroll para zoom · Espaço ou botão do meio para mover · Ctrl+Z para desfazer · Del para remover</small>
  </section></div>
}
