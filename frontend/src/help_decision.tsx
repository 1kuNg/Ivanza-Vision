import {useState} from 'react'

export type VisionTask='anomaly'|'sku'|'find'|'segment'|'read'|'state'|'depth'|'mixed'
type Navigation={
  onPage:(page:'Comparador'|'Pipelines'|'Few-Shot / SKU'|'Webcam')=>void
  onDemo:(demo:'painel'|'bandeja')=>void
  onSkuDemo:(demo:string)=>void
  onPipeline:(id:string)=>void
  onConvnextDemo:()=>void
  onSelectModel:(model:string)=>void
}

const tasks:{id:VisionTask,icon:string,title:string,subtitle:string}[]=[
  {id:'anomaly',icon:'🔎',title:'Há algo errado na peça?',subtitle:'Peça faltando, alteração, defeito visual'},
  {id:'sku',icon:'🏷️',title:'Qual produto é este?',subtitle:'Reconhecer SKU, tipo ou condição'},
  {id:'find',icon:'📦',title:'Onde estão os objetos?',subtitle:'Localizar vários itens na mesma imagem'},
  {id:'segment',icon:'✂️',title:'Quero separar uma peça do fundo',subtitle:'Máscara, recorte ou área da peça'},
  {id:'read',icon:'🔤',title:'Preciso ler uma etiqueta',subtitle:'Lote, data e identificação impressa'},
  {id:'state',icon:'💬',title:'Qual o estado geral da cena?',subtitle:'Comparar uma foto com descrições'},
  {id:'depth',icon:'📐',title:'O que parece mais perto?',subtitle:'Mapa de profundidade relativa'},
  {id:'mixed',icon:'🧩',title:'Quero combinar várias análises',subtitle:'Pipelines e limites entre etapas'},
]

const decisions:Record<VisionTask,{
  primary:string,secondary:string,why:string,prepare:string,observe:string,avoid:string,
  challenge:string,button:string,action:(nav:Navigation)=>void
}>={
  anomaly:{
    primary:'DINOv2 Small / DINOv3 Small',secondary:'SAM → DINO, quando o fundo atrapalha',
    why:'Compara a foto atual com uma ou mais fotos consideradas normais e destaca diferenças por região. Bom primeiro teste para ausência de componente e mudança de aparência.',
    prepare:'Foto de referência OK e foto de teste, com a peça em posição parecida, iluminação estável e defeitos que possam ser vistos pela câmera.',
    observe:'Mapa de diferenças, distância P95 e quantos exemplos normais também geram alertas.',
    avoid:'Não trata qualquer diferença como defeito. Uma sombra, rotação ou reflexo pode parecer anomalia; defeitos minúsculos podem não aparecer.',
    challenge:'Teste cinco amostras normais e cinco alteradas. Não escolha o limiar com apenas uma foto.',
    button:'Abrir painel com peça faltando',action:nav=>nav.onDemo('painel')
  },
  sku:{
    primary:'Few-Shot / SKU com DINOv2 ou DINOv3',secondary:'ConvNeXt Tiny para comparar outra arquitetura',
    why:'Recebe poucas referências de cada classe e procura o produto mais parecido. Pode usar classes como Produto A/B ou montagem OK/NOK, sem re-treinar os pesos do DINO.',
    prepare:'De 2 a 6 classes; 1 a 10 fotos por classe no Lab. Fotografe ângulos, posições e lotes reais diferentes. Reserve fotos novas que não entraram no cadastro.',
    observe:'Classe sugerida, similaridade, margem entre classes e o resultado Desconhecido conforme limiar.',
    avoid:'Classifica a imagem inteira: não separa automaticamente vários produtos. Desconhecidos podem ser aceitos como classes conhecidas; pontuações não são probabilidades.',
    challenge:'Cadastre duas classes e teste a terceira não cadastrada. Ajuste o limiar sem perder acertos nas classes conhecidas.',
    button:'Executar Few-Shot de embalagens',action:nav=>nav.onSkuDemo('embalagens')
  },
  find:{
    primary:'Grounding DINO Tiny',secondary:'Grounding → SAM para recortar cada região',
    why:'Você escreve nomes de objetos, geralmente em inglês, e recebe caixas em volta do que o modelo encontrou. Pode mostrar várias caixas na mesma foto.',
    prepare:'Uma imagem com objetos visíveis e nomes como bottle. cup. scissors. no campo Objetos.',
    observe:'Se cada caixa cobre o objeto correto, se há caixas repetidas e se algum objeto não foi encontrado.',
    avoid:'Não conhece necessariamente cada peça industrial específica; não garante contagem exata. Caixa não é máscara nem classificação de defeito.',
    challenge:'Coloque três objetos na mesa; retire um e observe se a contagem das caixas muda corretamente.',
    button:'Abrir Grounding DINO',action:nav=>nav.onSelectModel('Grounding DINO Tiny')
  },
  segment:{
    primary:'SAM 2.1 Tiny',secondary:'Grounding → SAM quando precisa localizar primeiro',
    why:'A partir de um ponto ou caixa, tenta contornar a peça e separar seus pixels do fundo.',
    prepare:'Uma foto nítida. Marque um ponto dentro da peça ou desenhe uma caixa no editor SAM; use pontos negativos para corrigir fundo.',
    observe:'Máscara, peça isolada e se a borda realmente acompanha todo o objeto.',
    avoid:'Segmentar não equivale a identificar o produto, aprovar qualidade ou detectar peça faltante. Pontos errados podem recortar o fundo.',
    challenge:'Recorte uma ferramenta, coloque outro objeto sobre ela e repita; veja quando a máscara falha.',
    button:'Abrir o Comparador com SAM',action:nav=>nav.onSelectModel('SAM 2.1 Tiny')
  },
  read:{
    primary:'RapidOCR',secondary:'Grounding DINO → RapidOCR para localizar a etiqueta primeiro',
    why:'Reconhece letras e números de uma imagem. Regex ajuda a conferir o formato esperado do texto.',
    prepare:'Etiquetas grandes, foco, boa iluminação e, se necessário, regex como LOTE-\\d{4}.',
    observe:'Texto reconhecido, confiança, correspondência com o texto impresso e resultado da regex.',
    avoid:'Regex não verifica se o lote existe. Leitura de código de barras/QR não é a mesma coisa que OCR textual; caracteres parecidos podem ser confundidos.',
    challenge:'Teste LOTE-1234, LOTE-ABC e uma etiqueta pouco iluminada para perceber erros de leitura.',
    button:'Abrir o Comparador com OCR',action:nav=>nav.onSelectModel('RapidOCR')
  },
  state:{
    primary:'SigLIP 2 Base',secondary:'Few-Shot quando há exemplos reais por classe',
    why:'Compara a foto com frases como a tidy workbench e a messy workbench. É útil para experimentar categorias gerais descritas por texto.',
    prepare:'Frases curtas e diferentes entre si, preferencialmente em inglês, representando estados visivelmente distintos.',
    observe:'Qual descrição recebeu maior pontuação em várias fotos de cada condição.',
    avoid:'As pontuações não são porcentagens calibradas nem certificação de segurança. Frases vagas e estados muito parecidos podem confundir.',
    challenge:'Fotografe uma mesa organizada e desorganizada e compare os mesmos prompts nas duas fotos.',
    button:'Abrir o Comparador com SigLIP',action:nav=>nav.onSelectModel('SigLIP 2 Base')
  },
  depth:{
    primary:'Depth Anything V2 Small',secondary:'Depth → SAM para tentar recortar um objeto próximo',
    why:'Estima a profundidade relativa a partir de uma imagem comum, destacando mudanças aparentes de perto e longe.',
    prepare:'Foto com objetos em planos de profundidade diferentes, mantendo a câmera em posição conhecida.',
    observe:'Mapa de profundidade e mudança entre as imagens ao aproximar um objeto.',
    avoid:'Não mede milímetros ou volume real. O valor do mapa não é uma distância física calibrada; superfícies reflexivas podem atrapalhar.',
    challenge:'Mova uma ferramenta para perto e longe da webcam e compare os mapas.',
    button:'Abrir o Comparador com profundidade',action:nav=>nav.onSelectModel('Depth Anything V2 Small')
  },
  mixed:{
    primary:'Um dos 4 pipelines já disponíveis',secondary:'Grounding + recorte + SKU por item seria uma evolução futura',
    why:'Uma etapa prepara a informação para a seguinte: encontrar → segmentar → comparar, encontrar → ler texto, profundidade → segmentar ou SAM → DINO.',
    prepare:'Confirme quais etapas pedem referência, ponto SAM, prompt de objeto e autorização Hugging Face.',
    observe:'Resultado de cada etapa, não apenas a saída final. Se a primeira etapa falhar, as seguintes podem estar erradas.',
    avoid:'O Lab não automatiza ainda Grounding → classificação Few-Shot de cada caixa. Pipelines disponíveis não significam supervisão industrial validada.',
    challenge:'Comece com Grounding → RapidOCR ou SAM → DINO e confira a saída de cada modelo.',
    button:'Abrir pipelines',action:nav=>nav.onPage('Pipelines')
  },
}

const matrix=[
  {model:'Few-Shot / SKU',answer:'Qual é a classe do produto?',output:'Classe + similaridade',needs:'Fotos por classe',not:'Não localiza cada objeto'},
  {model:'DINOv2 / DINOv3 ViT',answer:'O que mudou em relação ao normal?',output:'Mapa de diferenças',needs:'Foto(s) OK',not:'Não confirma defeito sozinho'},
  {model:'DINOv3 ConvNeXt Tiny',answer:'Reconhecer ou comparar com outra arquitetura',output:'Classe SKU ou mapa mais grosso',needs:'Fotos OK/classes + autorização HF',not:'Mapa menos detalhado neste Lab'},
  {model:'SAM 2.1 Tiny',answer:'Onde termina a peça?',output:'Máscara/recorte',needs:'Ponto, caixa ou grade',not:'Não decide OK/NOK'},
  {model:'Grounding DINO Tiny',answer:'Onde está o objeto citado?',output:'Caixas',needs:'Prompt com nomes de objetos',not:'Não garante contagem'},
  {model:'RapidOCR',answer:'O que está escrito?',output:'Texto e confiança',needs:'Texto legível',not:'Não valida cadastro ou QR'},
  {model:'SigLIP 2 Base',answer:'Qual frase combina mais?',output:'Pontuação por frase',needs:'Descrições de estados',not:'Não dá probabilidade calibrada'},
  {model:'Depth Anything V2 Small',answer:'O que está mais perto?',output:'Mapa relativo',needs:'Imagem com profundidade',not:'Não mede milímetros'},
]

const checks=[
  {name:'Dados',meaning:'Separe referências e imagens de teste. Não use a mesma foto para cadastrar e medir acerto.'},
  {name:'Variação normal',meaning:'Inclua luz, posição, lote, cor de fundo e operadores diferentes, desde que sejam condições permitidas.'},
  {name:'Falhas reais',meaning:'Monte uma amostra conhecida de defeitos e de peças aprovadas; registre falsos alarmes e falhas não detectadas.'},
  {name:'Rejeição',meaning:'No SKU, teste objetos não cadastrados. Escolha o limiar equilibrando falsas aceitações e rejeições.'},
  {name:'Desempenho',meaning:'Meça tempo de carregamento e inferência no PC de destino, com CPU/GPU e resolução reais.'},
  {name:'Produção',meaning:'O Lab é uma bancada de avaliação. Para uso contínuo, ainda são necessários câmera, iluminação, gatilho, integração com CLP e tratamento de erros.'},
]

function FeatureGrid({cols,highlight}:{cols:number,highlight:boolean}){
  return <div className="decision-grid-example" style={{gridTemplateColumns:'repeat('+cols+',1fr)'}}
      aria-label={'Exemplo esquemático de '+cols+' por '+cols+' posições'}>
    {Array.from({length:cols*cols},(_,i)=><span key={i} className={highlight && (Math.floor(i/cols)===Math.floor(cols*.48)) && Math.abs(i%cols-Math.floor(cols*.51))<=1?'hot':''}/>)}
  </div>
}

export function ChoosingGuide(nav:Navigation){
  const [chosen,setChosen]=useState<VisionTask>('anomaly')
  const [showAll,setShowAll]=useState(false)
  const d=decisions[chosen]
  return <div className="decision-guide">
    <div className="help-heading"><div><h2>Qual IA usar no meu problema?</h2>
      <p className="muted">Escolha o resultado que você precisa. A recomendação muda sem instalar nada.</p></div></div>
    <div className="decision-goals" role="group" aria-label="Objetivo da inspeção">
      {tasks.map(t=><button key={t.id} className={'decision-goal'+(chosen===t.id?' selected':'')}
        aria-pressed={chosen===t.id} onClick={()=>setChosen(t.id)}>
        <span className="decision-goal-icon">{t.icon}</span><strong>{t.title}</strong><small>{t.subtitle}</small>
      </button>)}
    </div>
    <section className="card decision-answer" aria-live="polite">
      <span className="help-tag">PRIMEIRA OPÇÃO PARA EXPERIMENTAR</span>
      <h3>{d.primary}</h3>
      <p>{d.why}</p>
      <div className="decision-details">
        <div><strong>O que preparar</strong><p>{d.prepare}</p></div>
        <div><strong>O que observar</strong><p>{d.observe}</p></div>
        <div><strong>Quando não confiar / limites</strong><p>{d.avoid}</p></div>
        <div><strong>Teste de validação</strong><p>{d.challenge}</p></div>
      </div>
      <p className="decision-alternative"><strong>Alternativa ou complemento:</strong> {d.secondary}</p>
      <button className="primary" onClick={()=>d.action(nav)}>{d.button}</button>
    </section>
    <div className="card help-advice">
      <h3>Uma imagem com vários produtos ao mesmo tempo</h3>
      <p>O <strong>Few-Shot atual classifica a foto inteira</strong>. Para identificar separadamente dez produtos em uma esteira, a estratégia seria localizar cada um com Grounding DINO (ou outro detector), recortar as caixas e classificar cada recorte por Few-Shot. <strong>Esse fluxo completo ainda não é automático nesta versão.</strong></p>
      <button onClick={()=>setChosen('find')}>Entender a etapa de localização</button>
    </div>
    <button onClick={()=>setShowAll(!showAll)} aria-expanded={showAll}>{showAll?'Ocultar':'Ver'} tabela resumida de todos os modelos</button>
    {showAll&&<div className="table-scroll"><table><thead><tr>
      <th>Modelo</th><th>Responde a quê?</th><th>Entrega</th><th>Precisa de</th><th>Limitação principal</th>
    </tr></thead><tbody>{matrix.map(m=><tr key={m.model}><td>{m.model}</td><td>{m.answer}</td><td>{m.output}</td><td>{m.needs}</td><td>{m.not}</td></tr>)}</tbody></table></div>}
  </div>
}

export function ArchitecturesGuide({onConvnextDemo,onSkuDemo}:Pick<Navigation,'onConvnextDemo'|'onSkuDemo'>){
  return <div className="decision-guide">
    <h2>ConvNeXt Tiny, Transformer e Few-Shot: qual a diferença?</h2>
    <p><strong>Few-Shot não é uma terceira arquitetura.</strong> É a maneira de classificar com poucas fotos usando as características produzidas por um backbone DINO. Esse backbone pode ser DINOv2 Small, DINOv3 Small (Transformer) ou DINOv3 ConvNeXt Tiny (CNN).</p>
    <div className="arch-compare">
      <section className="card arch-item">
        <span className="help-tag">CNN · REDE CONVOLUCIONAL</span>
        <h3>DINOv3 ConvNeXt Tiny</h3>
        <p>Combina padrões locais (bordas, formas e texturas) em camadas até formar uma representação da peça. Não é sinônimo de modelo que só enxerga pedaços: também reúne contexto.</p>
        <div className="arch-grid-label">Mapa de características a 512 px: 16 × 16</div>
        <FeatureGrid cols={16} highlight/>
        <p><strong>Quando experimentar:</strong> reconhecimento Few-Shot, testes de velocidade e peças com formas bem distintas.</p>
        <p><strong>Limite no Lab:</strong> a grade espacial é mais grossa (passo 32). Isso pode dificultar localizar defeitos muito pequenos no mapa, embora não determine sozinho a qualidade da classificação.</p>
      </section>
      <section className="card arch-item">
        <span className="help-tag">VIT · VISION TRANSFORMER</span>
        <h3>DINOv3 Small</h3>
        <p>Divide a imagem em blocos e usa atenção para relacionar informações entre regiões. Também aprende contornos, formas e texturas; não significa que entenda automaticamente a causa de um defeito.</p>
        <div className="arch-grid-label">Mapa de características a 512 px: 32 × 32</div>
        <FeatureGrid cols={32} highlight/>
        <p><strong>Quando experimentar:</strong> comparação de referência, localização visual de alterações e classificação Few-Shot.</p>
        <p><strong>Limite no Lab:</strong> mais posições no mapa não garantem melhor resultado em todos os defeitos. Compare tempo, memória e falsos alarmes.</p>
      </section>
    </div>
    <p className="muted arch-caption">Os quadrados representam a resolução espacial dos mapas que o código do Lab usa (512/32 e 512/16); não são regiões exatas de atenção nem uma fotografia produzida pelos modelos. O DINOv2 Small usa outra configuração, com passo 14 e tamanhos 224, 336 e 518 px.</p>
    <div className="card help-advice">
      <h3>Como comparar de forma justa</h3>
      <p>Para a mesma peça, use imagens de referência e teste iguais, condições de iluminação iguais e o mesmo computador. Compare se a falha foi encontrada, quantas imagens normais geraram alertas e o tempo de inferência. Não compare apenas o brilho do mapa, pois os escores dos backbones não são diretamente equivalentes.</p>
      <div className="toolbar">
        <button className="primary" onClick={onConvnextDemo}>Abrir painel: DINOv2 x ConvNeXt</button>
        <button onClick={()=>onSkuDemo('fixadores')}>Abrir Few-Shot de fixadores</button>
      </div>
      <p><strong>Acesso:</strong> DINOv3 Small e ConvNeXt Tiny usam checkpoints no Hugging Face e podem exigir autorização e download inicial. No Lab, a primeira execução é mais lenta que uma rodada com modelo já carregado.</p>
    </div>
    <div className="card help-advice">
      <h3>O que é Few-Shot, de verdade?</h3>
      <p>Você cadastra algumas fotos de cada classe. O DINO extrai vetores dessas imagens; uma foto nova é comparada com esses vetores, e a classe com maior similaridade é sugerida. <strong>Os pesos do DINO não são retreinados.</strong></p>
      <p>O Lab aceita <strong>de 2 a 6 classes e de 1 a 10 fotos por classe</strong>. Poucas fotos ajudam a iniciar testes, mas não garantem generalização para novos lotes, câmeras e iluminação. A opção Desconhecido depende de um limiar configurado, não de detecção perfeita de novidade.</p>
    </div>
  </div>
}

export function ValidationGuide(){
  return <div className="decision-guide">
    <h2>Como saber se a IA está pronta para o seu processo?</h2>
    <p>Um resultado bonito na tela não prova desempenho industrial. Use este roteiro antes de escolher qualquer modelo.</p>
    <div className="decision-checks">{checks.map((c,i)=><article className="card" key={c.name}>
      <span className="help-tag">PASSO {i+1}</span><h3>{c.name}</h3><p>{c.meaning}</p>
    </article>)}</div>
    <div className="card help-advice">
      <h3>Como interpretar erros importantes</h3>
      <p><strong>Falso positivo:</strong> a IA aponta defeito em um produto bom. <strong>Falso negativo:</strong> um defeito real passa despercebido. <strong>Falsa aceitação no SKU:</strong> um item não cadastrado recebe nome de uma classe conhecida. Os três merecem medição com exemplos reais.</p>
      <p><strong>Em resumo:</strong> primeiro faça funcionar na bancada; depois meça com novas fotos; só então pense em alarme, rejeito automático ou integração CLP.</p>
    </div>
  </div>
}
