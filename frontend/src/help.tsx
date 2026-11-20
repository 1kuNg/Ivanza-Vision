import {useState} from 'react'
import {Guide as TechnicalGuide} from './screens'
import {ChoosingGuide,ArchitecturesGuide,ValidationGuide} from './help_decision'

type Page = 'Comparador' | 'Pipelines' | 'Few-Shot / SKU' | 'Webcam'
type Demo = 'painel' | 'bandeja'
type Props = {
  onDemo:(demo:Demo)=>void
  onPage:(page:Page)=>void
  onPipeline:(id:string)=>void
  onSkuDemo:(id:string)=>void
  onConvnextDemo:()=>void
}

type Test = {
  title:string
  level:string
  category:string
  model:string
  materials:string
  steps:string[]
  expected:string
  interpret:string
  page:Page
  demo?:Demo
  pipeline?:string
  skuDemo?:string
  convnext?:boolean
}
const MODELS = [
  {name:'DINOv2', nickname:'O detector de diferenças', symbol:'🔎',
   does:'Compara a imagem com fotos normais e destaca onde algo ficou visualmente diferente. É ótimo para peças faltando, montagem e aparência.',
   try:'Fotografe uma garrafa com tampa, tire a tampa sem mexer na câmera e compare com a foto normal.',
   read:'Olhe o mapa de diferenças e a distância P95. Valores maiores significam mais diferença, não necessariamente defeito.',
   note:'Ele não identifica sozinho a causa da diferença. Luz e ângulo também afetam o resultado.'},
  {name:'DINOv3', nickname:'Uma segunda opinião para inspeção', symbol:'🔬',
   does:'Também compara imagens e extrai características visuais. Você pode verificar qual versão encontra melhor os detalhes da sua aplicação.',
   try:'Analise o mesmo painel com DINOv2 e DINOv3, mantendo a mesma resolução.',
   read:'Compare o mapa de diferenças, a localização da falha e os milissegundos de inferência.',
   note:'Os checkpoints DINOv3 precisam de autorização no Hugging Face e podem ser mais pesados.'},
  {name:'DINOv3 ConvNeXt Tiny', nickname:'Um reconhecedor com outra arquitetura', symbol:'⚙️',
   does:'É um DINOv3 baseado em rede convolucional (CNN), em vez do Transformer visual da versão Small. Pode ser interessante para reconhecimento de produtos e para comparar velocidade e consumo de memória.',
   try:'Abra o exemplo do painel com DINOv2 e ConvNeXt Tiny ou experimente o classificador SKU com as duas embalagens prontas.',
   read:'Veja se reconheceu a classe correta no Few-Shot e compare tempos e mapas de diferença no Comparador.',
   note:'No mapa de diferenças, cada célula cobre uma área maior que no DINOv3 ViT. Para detalhes pequenos, compare com outros modelos. Precisa liberar o checkpoint em Config / Tokens.'},
  {name:'SAM 2.1', nickname:'A tesoura inteligente', symbol:'✂️',
   does:'Recorta um objeto do fundo por meio de pontos, caixas ou uma grade. O resultado é uma máscara que acompanha o formato da peça.',
   try:'Fotografe um alicate, abra Editar marcações SAM e clique no centro da ferramenta.',
   read:'Veja Máscara e Peça isolada. Confira se o objeto inteiro foi incluído sem muito fundo.',
   note:'O SAM não sabe automaticamente se uma peça está aprovada ou defeituosa.'},
  {name:'Grounding DINO', nickname:'Encontra objetos pelo nome', symbol:'📦',
   does:'Você digita nomes em inglês, como bottle. cup. scissors., e ele tenta desenhar caixas em volta desses objetos.',
   try:'Espalhe três objetos na mesa e peça para localizar cada um.',
   read:'Conte as caixas e veja se cada retângulo está na posição correta.',
   note:'Nem sempre encontra tudo. Pode marcar coisas erradas; não é um contador garantido.'},
  {name:'SigLIP 2', nickname:'Compara a foto com frases', symbol:'🏷️',
   does:'Compara a imagem com descrições, como a tidy workbench e a messy workbench. Ajuda a explorar o estado geral de uma cena.',
   try:'Fotografe uma bancada organizada e depois a mesma bancada bagunçada.',
   read:'Compare a pontuação das frases nas duas fotos, em vez de olhar um número isolado.',
   note:'As pontuações não são probabilidades de segurança ou de defeito calibradas.'},
  {name:'Depth Anything V2', nickname:'Uma noção de perto e longe', symbol:'📏',
   does:'Cria um mapa de profundidade relativa usando uma câmera comum. Ajuda a destacar o primeiro plano e perceber mudanças de ocupação.',
   try:'Fotografe uma mão na frente da mesa e depois mova a mão para mais longe.',
   read:'Veja como o mapa de profundidade muda com a posição.',
   note:'Não mede centímetros, volume ou altura real sem calibração e outros sensores.'},
  {name:'RapidOCR', nickname:'Lê etiquetas e códigos impressos', symbol:'🔤',
   does:'Extrai letras e números de uma imagem. Você pode conferir lote, data, identificação e formato do texto.',
   try:'Escreva LOTE-1234 em um papel grande, fotografe e execute RapidOCR.',
   read:'Confira o texto reconhecido, a confiança e a validação por regex, se configurada.',
   note:'Um texto legível mas incorreto pode passar; regex valida formato, não consulta um cadastro.'},
]

const PIPELINES = [
  {id:'grounding_sam_dino', title:'Grounding DINO → SAM → DINOv3', nickname:'Encontrar, recortar e inspecionar',
   stages:['Grounding DINO procura a peça pelo nome em inglês e desenha a caixa.','SAM usa essa caixa para separar a peça do fundo.','DINOv3 compara os detalhes da peça com fotos de referência normais.'],
   uses:'Inspeção de peças que mudam um pouco de posição, conferência de montagem e comparação de produtos sobre fundos diferentes.',
   try:'Fotografe uma garrafa com tampa e outra sem tampa, em boa iluminação. Use a normal como referência, selecione este pipeline e procure bottle.',
   watch:'Confira primeiro se a caixa encontrou a peça; depois se a máscara está correta; por fim, o mapa de diferenças. Se a detecção falhar, as etapas seguintes também falham.',
   warning:'Requer referência normal e DINOv3 autorizado. Uma máscara boa não confirma que a peça está aprovada.'},
  {id:'grounding_ocr', title:'Grounding DINO → RapidOCR', nickname:'Encontrar a etiqueta e ler',
   stages:['Grounding DINO procura a região do rótulo ou etiqueta.','RapidOCR lê os caracteres apenas na região encontrada.'],
   uses:'Ler lote e validade em embalagens, verificar texto de etiquetas e facilitar a localização de códigos em fotos com vários elementos.',
   try:'Fotografe uma embalagem com etiqueta grande e legível. Use o prompt label. e execute este pipeline.',
   watch:'Veja se existe uma caixa sobre a etiqueta e se o texto lido corresponde ao impresso.',
   warning:'Se a etiqueta não for localizada, este pipeline não tem região para o OCR ler. Para texto simples, RapidOCR sozinho pode ser melhor.'},
  {id:'depth_sam', title:'Depth Anything → SAM', nickname:'Destacar o que parece mais próximo',
   stages:['Depth Anything estima quais regiões parecem mais próximas da câmera.','SAM recebe um ponto encontrado no mapa e tenta recortar o objeto dessa região.'],
   uses:'Experiências com objetos à frente de uma bancada, separação aproximada de primeiro plano e exploração de bandejas e cavidades.',
   try:'Coloque uma ferramenta à frente de uma mesa com fundo distante. Tire uma foto e execute o pipeline.',
   watch:'Veja se o ponto mais próximo cai na ferramenta e se a máscara cobre a peça inteira.',
   warning:'A profundidade é relativa, e o objeto escolhido pode não ser o desejado. Não é medição 3D nem solução pronta para robô.'},
  {id:'sam_dino', title:'SAM → DINO', nickname:'Inspecionar a peça sem o fundo',
   stages:['Você marca a peça com pontos ou caixa, e o SAM cria a máscara.','O DINO compara as características na região selecionada, reduzindo a influência do fundo.'],
   uses:'Inspecionar uma peça em uma mesa, reduzir falsas diferenças causadas pelo cenário e comparar acabamento ou montagem.',
   try:'Fotografe o mesmo objeto normal e alterado. Selecione este pipeline, marque o objeto no editor SAM e carregue a foto normal como referência.',
   watch:'Confira a máscara antes de interpretar o mapa. Compare o resultado com e sem filtro do fundo.',
   warning:'Precisa de referência normal e de uma seleção SAM correta nas duas imagens.'},
]

const TESTS:Test[] = [
  {title:'Painel com peça faltando', level:'Muito fácil',category:'Inspeção',model:'DINOv2',materials:'Nenhum — imagens incluídas no Lab',demo:'painel',page:'Comparador',
   steps:['Clique em Abrir exemplo pronto.','A referência é o painel completo; a imagem de teste tem uma peça faltando.','Execute o DINOv2 Small e observe o mapa.','Repita com resolução 512 para comparar os detalhes.'],
   expected:'O local da peça ausente deve se destacar das regiões normais.',interpret:'Compare com uma imagem normal. Se toda a cena aparecer diferente, pode ser alteração de luz ou enquadramento.'},
  {title:'Bandeja com posição vazia',level:'Muito fácil',category:'Inspeção',model:'DINOv2 + SAM',materials:'Nenhum — imagens incluídas no Lab',demo:'bandeja',page:'Comparador',
   steps:['Clique em Abrir exemplo pronto.','Confira a grade 2 × 4 que separa as posições.','Ative DINOv2 e SAM 2.1 e execute.','Abra Inspeção Profunda e examine as células.'],
   expected:'A posição vazia deve apresentar diferenças em relação à bandeja normal.',interpret:'Confira se a grade está alinhada. Um quadrado fora da peça pode dar resultado enganoso.'},
  {title:'Garrafa sem tampa',level:'Muito fácil',category:'Inspeção',model:'DINOv2',materials:'Uma garrafa e câmera',page:'Comparador',
   steps:['Fotografe a garrafa com tampa e carregue como Referências.','Sem mexer na garrafa ou câmera, retire a tampa e fotografe.','Carregue a nova foto como Imagem de teste.','Selecione DINOv2 Small e execute.'],
   expected:'A região da tampa pode ser destacada no mapa de diferenças.',interpret:'Fotografe também outra garrafa normal para descobrir a variação natural do teste.'},
  {title:'Few-Shot pronto: reconhecer duas embalagens',level:'Muito fácil',category:'Reconhecimento',model:'DINOv2 ou ConvNeXt Tiny',materials:'Nenhum — 9 imagens de demonstração carregadas automaticamente',page:'Few-Shot / SKU',skuDemo:'embalagens',
   steps:['Clique em Abrir exemplo pronto e deixe o Lab registrar as duas classes de embalagem.','Observe as 3 referências por classe e o primeiro resultado real.','Clique na imagem de teste Caixa laranja e depois Lata verde (desconhecida).','Compare classificação, similaridade, margem e limiar de desconhecido.','Troque para ConvNeXt Tiny na própria tela e repita o teste.'],
   expected:'O modelo deve reconhecer as classes cadastradas quando as características forem suficientes; a lata pode ser rejeitada ajustando o limiar.',
   interpret:'A previsão é calculada pelo modelo real. Um item desconhecido pode ser confundido com classes conhecidas se o limiar estiver baixo. Não são porcentagens de certeza.'},
  {title:'Few-Shot pronto: identificar porca, arruela e parafuso',level:'Muito fácil',category:'Reconhecimento',model:'DINOv2 ou ConvNeXt Tiny',materials:'Nenhum — exemplos sintéticos incluídos',page:'Few-Shot / SKU',skuDemo:'fixadores',
   steps:['Abra o exemplo para cadastrar 3 imagens de Porca sextavada e 3 de Arruela.','Observe a primeira classificação executada pelo DINO.','Teste a imagem de Arruela e a de Parafuso, que não estava no cadastro.','Ajuste o limiar e compare as respostas.','Repita com ConvNeXt Tiny para conferir qual arquitetura funciona melhor.'],
   expected:'Classificações e similaridades reais para formatos de fixadores diferentes.',
   interpret:'Os exemplos são ilustrações controladas. Valide com fotografias verdadeiras antes de aplicar a uma linha industrial.'},
  {title:'Encontrar vários objetos ao mesmo tempo',level:'Fácil',category:'Localização',model:'Grounding DINO',materials:'Garrafa, xícara e tesoura',page:'Comparador',
   steps:['Fotografe os objetos afastados entre si.','Ative Grounding DINO Tiny.','No campo Objetos escreva bottle. cup. scissors.','Execute e compare as caixas com os objetos reais.'],
   expected:'Caixas em volta dos objetos encontrados.',interpret:'Remova um objeto e repita. Confira falsos positivos e objetos não encontrados.'},
  {title:'Separar uma ferramenta do fundo',level:'Fácil',category:'Segmentação',model:'SAM 2.1',materials:'Alicate ou chave de fenda',page:'Comparador',
   steps:['Carregue uma foto da ferramenta.','Ative SAM 2.1 Tiny e abra Editar marcações SAM.','Clique no objeto com 1 peça por clique.','Execute e abra Máscara e Peça isolada em Inspeção Profunda.'],
   expected:'Uma máscara seguindo o contorno da ferramenta.',interpret:'Se o fundo entrar na máscara, use marcações negativas no editor.'},
  {title:'Ler lote e validar formato',level:'Fácil',category:'Texto',model:'RapidOCR',materials:'Um papel com LOTE-1234',page:'Comparador',
   steps:['Fotografe de perto o papel com LOTE-1234.','Ative somente RapidOCR e deixe Ler ROIs desmarcado.','Configure Regex com LOTE-\\d{4} e execute.','Mude a inscrição para LOTE-ABC e repita.'],
   expected:'OCR lê o lote e a coluna Regex permite conferir o padrão.',interpret:'Se o OCR ler errado, uma regex não corrige a leitura. Teste foco e iluminação.'},
  {title:'Testar perto e longe com câmera comum',level:'Fácil',category:'Profundidade',model:'Depth Anything V2',materials:'Mão e uma mesa',page:'Comparador',
   steps:['Fotografe sua mão próxima à câmera, com a mesa ao fundo.','Selecione Depth Anything V2 Small e execute.','Afaste a mão e faça outra foto.','Compare os mapas de profundidade relativa.'],
   expected:'Regiões do mapa mudam conforme a profundidade aparente.',interpret:'As cores não são uma régua de centímetros; não use para medições industriais sem calibração.'},
  {title:'Mesa organizada ou bagunçada',level:'Fácil',category:'Estado visual',model:'SigLIP 2',materials:'Ferramentas sobre uma mesa',page:'Comparador',
   steps:['Fotografe uma bancada organizada e uma desorganizada.','Ative SigLIP 2 Base.','Adicione as frases a tidy workbench e a messy workbench.','Compare os resultados de cada fotografia.'],
   expected:'A pontuação das descrições tende a acompanhar a cena.',interpret:'As frases competem visualmente, mas os escores não significam porcentagem de certeza.'},
  {title:'Encontrar e ler uma etiqueta',level:'Intermediário',category:'Pipeline',model:'Grounding → OCR',materials:'Embalagem com rótulo legível',page:'Pipelines',pipeline:'grounding_ocr',
   steps:['Fotografe uma embalagem com etiqueta grande.','Clique em Abrir pipeline e configure Objetos como label.','Carregue a imagem de teste e execute.','Veja se Grounding encontrou a etiqueta antes de conferir o texto do OCR.'],
   expected:'O detector localiza uma região e o OCR tenta ler seu conteúdo.',interpret:'Se o primeiro modelo não encontrou a etiqueta, aproxime a câmera ou altere o prompt.'},
  {title:'Comparar DINOv2 com DINOv3',level:'Intermediário',category:'Comparação',model:'DINOv2 + DINOv3',materials:'Painel do exemplo + DINOv3 autorizado',demo:'painel',page:'Comparador',
   steps:['Abra o exemplo do painel.','Ative DINOv2 Small e DINOv3 Small.','Autorize o checkpoint DINOv3 em Config / Tokens, se necessário.','Execute e compare mapa, inferência e resolução.'],
   expected:'Dois mapas de diferenças e tempos de análise para a mesma imagem.',interpret:'O melhor para sua aplicação é o que localiza falhas com menos falsos alarmes.'},
  {title:'Comparação pronta: DINOv2 x DINOv3 ConvNeXt Tiny',level:'Fácil com acesso liberado',category:'Comparação',model:'ConvNeXt Tiny',materials:'Painel de exemplo e autorização do checkpoint no Hugging Face',page:'Comparador',convnext:true,
   steps:['Abra este teste: o Lab coloca o painel completo e o painel com peça faltando.','Deixa DINOv2 Small e DINOv3 ConvNeXt Tiny selecionados, em resolução 384.','Em Config / Tokens, escolha DINOv3 ConvNeXt Tiny e verifique a autorização.','Execute a comparação e analise os dois mapas de diferenças e os tempos.'],
   expected:'Resultados separados dos dois modelos reais para a mesma peça.',
   interpret:'O ConvNeXt trabalha com grade mais grossa (stride 32); compare detecção de defeitos e velocidade, não apenas a cor do mapa.'},
  {title:'Detectar mudança de cor ou etiqueta',level:'Fácil',category:'Inspeção',model:'DINOv2',materials:'Duas embalagens parecidas',page:'Comparador',
   steps:['Fotografe uma embalagem e salve como referência.','Troque a etiqueta ou o produto por outro muito parecido.','Carregue a imagem nova como teste.','Compare os resultados em 224 e 512.'],
   expected:'Partes alteradas podem gerar maior distância visual.',interpret:'Alterações muito pequenas podem não aparecer; compare sempre com exemplos reais de defeitos.'},
  {title:'Ver como luz e sombra causam falsos alarmes',level:'Fácil',category:'Qualidade',model:'DINOv2',materials:'Um objeto e uma luminária',page:'Comparador',
   steps:['Fotografe um objeto normal e guarde como referência.','Faça outra foto normal, mantendo a mesma luz.','Mude apenas a iluminação e compare.','Depois faça uma alteração real na peça com a luz original.'],
   expected:'Você verá que sombra também muda a aparência medida pelo modelo.',interpret:'Um bom sistema precisa separar variações normais de defeitos com calibração e imagens de validação.'},
  {title:'Inspecionar objetos ao vivo na webcam',level:'Intermediário',category:'Webcam',model:'DINOv2 / Few-Shot',materials:'Webcam e um objeto',page:'Webcam',
   steps:['Prepare referência no Comparador ou classes no Few-Shot.','Abra Webcam, selecione a câmera e clique em Iniciar.','Capture a referência, se necessário, e clique em Analisar quadro.','Experimente Análise contínua com intervalo maior em CPU.'],
   expected:'Resultado visual e tempo de inferência de cada quadro analisado.',interpret:'A visualização da câmera não implica análise de todos os quadros; o processamento pode ser mais lento.'},
  {title:'Recortar objeto parcialmente escondido',level:'Intermediário',category:'Segmentação',model:'SAM 2.1',materials:'Duas ferramentas que possam ser sobrepostas',page:'Comparador',
   steps:['Fotografe uma ferramenta sozinha.','Faça nova foto com outra ferramenta cobrindo parte dela.','Marque a parte visível com pontos SAM.','Compare as máscaras e ajuste com pontos negativos.'],
   expected:'Você conseguirá ver até onde a segmentação se mantém confiável.',interpret:'O teste é de imagem parada. Este Lab não implementa automaticamente rastreamento de vídeo pelo SAM.'},
  {title:'Localizar e inspecionar com pipeline completo',level:'Intermediário',category:'Pipeline',model:'Grounding → SAM → DINOv3',materials:'Objeto normal, alterado e DINOv3 autorizado',page:'Pipelines',pipeline:'grounding_sam_dino',
   steps:['Fotografe uma garrafa com e sem tampa.','Abra o pipeline, carregue a foto normal como Referências.','Configure Objetos como bottle. e use a imagem sem tampa como teste.','Execute e confira as três etapas na ordem.'],
   expected:'Uma caixa, uma máscara e um mapa de diferenças da peça.',interpret:'Se a localização ou a máscara falhar, não confie na análise final.'},
]

const TERMS = [
  ['Foto de referência','É a foto de um produto considerado normal. DINO precisa dela para saber com o que comparar.'],
  ['Foto de teste','É a nova foto que você quer analisar. Pode estar normal ou ter alguma alteração.'],
  ['Mapa de diferenças','A região mais destacada apresenta mudança visual. Cor forte não significa automaticamente peça defeituosa.'],
  ['Similaridade global','Resume quanto a imagem inteira parece com a referência. Maior normalmente indica mais semelhança.'],
  ['Distância média','Diferença visual média dos pequenos pedaços da imagem. Maior indica mais mudança.'],
  ['Distância P95','Valor acima do qual ficam apenas cerca de 5% dos pedaços comparados. Útil para chamar atenção para mudanças localizadas.'],
  ['Distância máxima','O pedaço mais diferente. Pode ser uma falha, um reflexo ou ruído. Não avalie sozinho.'],
  ['Limiar','Limite que você ajusta usando várias fotos normais e alteradas. Não existe um único valor correto para tudo.'],
  ['Máscara SAM','A parte da imagem que o SAM selecionou como objeto. Confira se o contorno ficou correto.'],
  ['IoU do SAM','Estimativa interna da qualidade da máscara; não é aprovação de qualidade da peça.'],
  ['Caixas / detecções','Retângulos onde Grounding DINO acredita que há um objeto solicitado. Confira a posição e a quantidade.'],
  ['Score / confiança','É uma pontuação do modelo. Não representa automaticamente chance real de defeito.'],
  ['Regex','Regra simples para conferir formato de textos. Exemplo: LOTE-1234, mas não LOTE-ABC.'],
  ['Inferência em ms','Tempo de análise do modelo. Em CPU pode ser muito mais lento do que em GPU.'],
  ['224 / 384 / 512 px','Resolução usada na análise DINO. Maior pode mostrar detalhes menores, mas gasta mais tempo e memória.'],
  ['Mesma posição / Regiões semelhantes','Mesma posição ajuda quando a câmera é fixa. Regiões semelhantes tolera deslocamentos, mas pode deixar passar erros de posição.'],
]

const normalize=(value:string)=>value.normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase()
const tabs = [
  ['inicio','Comece aqui'],['decidir','Qual modelo escolher?'],['modelos','O que cada IA faz'],['arquiteturas','CNN x Transformer'],['pipelines','Pipelines explicados'],
  ['testes','Testes para experimentar'],['interpretar','Como ler os resultados'],['validar','Limites & validação'],['tecnico','Guia técnico'],
] as const
type Tab = typeof tabs[number][0]

export function Guide({onDemo,onPage,onPipeline,onSkuDemo,onConvnextDemo}:Props){
  const [tab,setTab]=useState<Tab>('inicio')
  const [search,setSearch]=useState('')
  const [category,setCategory]=useState('Todos')
  const categories=['Todos',...Array.from(new Set(TESTS.map(item=>item.category)))]
  const shownTests=TESTS.filter(t=>(category==='Todos'||category===t.category)&&normalize([t.title,t.model,t.materials,t.expected].join(' ')).includes(normalize(search)))
  const shownModels=MODELS.filter(m=>normalize([m.name,m.nickname,m.does,m.try].join(' ')).includes(normalize(search)))
  const shownPipelines=PIPELINES.filter(p=>normalize([p.title,p.nickname,p.uses,p.try].join(' ')).includes(normalize(search)))
  function openTest(t:Test){if(t.skuDemo)onSkuDemo(t.skuDemo);else if(t.convnext)onConvnextDemo();else if(t.demo)onDemo(t.demo);else if(t.pipeline)onPipeline(t.pipeline);else onPage(t.page)}
  return <section className="easy-guide">
    <header className="help-hero">
      <div><span className="help-eyebrow">IVANZA VISION LAB · AJUDA PRÁTICA</span><h1>Descubra o que a IA enxerga.</h1>
        <p>Explicações simples, testes com objetos comuns e dicas para entender cada resultado. Comece pelo exemplo pronto ou escolha uma experiência.</p></div>
      <button className="primary" onClick={()=>{setTab('testes');setSearch('');setCategory('Todos')}}>Explorar {TESTS.length} testes</button>
    </header>
    <nav className="help-tabs" aria-label="Seções da ajuda">
      {tabs.map(([id,label])=><button key={id} className={tab===id?'selected':''} onClick={()=>{setTab(id);setSearch('')}}>{label}</button>)}
    </nav>

    {tab==='inicio'&&<div className="help-content">
      <h2>Primeiros 5 minutos</h2>
      <p className="muted">Você não precisa configurar nenhum modelo avançado para começar. Escolha um dos exemplos abaixo.</p>
      <div className="help-cards">
        <article className="card help-card"><span className="help-tag">1 · Pronto para usar</span><h3>Uma peça desapareceu</h3>
          <p>Compare um painel completo com outro que perdeu uma peça. O DINOv2 marca as diferenças.</p>
          <button className="primary" onClick={()=>onDemo('painel')}>Abrir exemplo: painel</button></article>
        <article className="card help-card"><span className="help-tag">2 · Pronto para usar</span><h3>Uma bandeja está incompleta</h3>
          <p>Veja uma bandeja com posição vazia. Use o DINOv2 e a grade do SAM para comparar as posições.</p>
          <button className="primary" onClick={()=>onDemo('bandeja')}>Abrir exemplo: bandeja</button></article>
        <article className="card help-card"><span className="help-tag">3 · Com objetos reais</span><h3>Ensine duas classes em minutos</h3>
          <p>Cadastre fotos de dois produtos e teste se o sistema reconhece novas fotos ou um produto desconhecido.</p>
          <button className="primary" onClick={()=>onSkuDemo('embalagens')}>Executar exemplo Few-Shot pronto</button></article>
      </div>
      <div className="card help-advice"><h3>Qual modelo usar?</h3>
        <p><strong>Mudou alguma coisa?</strong> DINO. <strong>Quero separar o objeto do fundo?</strong> SAM. <strong>Quero encontrar uma peça pelo nome?</strong> Grounding DINO. <strong>Quero ler uma etiqueta?</strong> RapidOCR.</p>
        <p><strong>Quero comparar a foto com uma frase?</strong> SigLIP 2. <strong>Quero ver perto e longe?</strong> Depth Anything. <strong>Quero combinar várias etapas?</strong> Abra Pipelines explicados.</p>
        <div className="toolbar"><button className="primary" onClick={()=>setTab('decidir')}>Me ajude a escolher</button><button onClick={()=>setTab('modelos')}>Entender todos os modelos</button></div>
      </div>
      <div className="card help-advice"><h3>Antes de confiar no resultado</h3>
        <p>Fotografe com luz estável, procure manter o mesmo enquadramento e teste <strong>fotos normais e fotos com defeito</strong>. Uma diferença visual não é necessariamente uma falha de qualidade.</p>
        <div className="toolbar"><button onClick={()=>setTab('interpretar')}>Aprender a interpretar números e mapas</button><button onClick={()=>setTab('validar')}>Entender os limites para uso industrial</button></div>
      </div>
    </div>}

    {tab==='decidir'&&<ChoosingGuide onPage={onPage} onDemo={onDemo} onSkuDemo={onSkuDemo} onPipeline={onPipeline} onConvnextDemo={onConvnextDemo}/>}
    {tab==='arquiteturas'&&<ArchitecturesGuide onConvnextDemo={onConvnextDemo} onSkuDemo={onSkuDemo}/>}
    {tab==='validar'&&<ValidationGuide/>}

    {tab==='modelos'&&<div className="help-content">
      <div className="help-heading"><div><h2>O que cada inteligência artificial faz?</h2><p className="muted">Pense em cada modelo como uma ferramenta diferente.</p></div>
        <input aria-label="Buscar modelos" type="search" placeholder="Buscar modelo ou função" value={search} onChange={e=>setSearch(e.target.value)}/></div>
      <div className="help-cards">{shownModels.map(model=><article className="card help-card" key={model.name}>
        <div className="help-model-head"><span className="help-symbol">{model.symbol}</span><div><h3>{model.name}</h3><strong className="help-subtitle">{model.nickname}</strong></div></div>
        <p>{model.does}</p>
        <div className="help-card-limit"><strong>Limitação importante:</strong> {model.note}</div><details><summary>Como testar e interpretar</summary><p><strong>Teste rápido:</strong> {model.try}</p><p><strong>O que olhar:</strong> {model.read}</p></details>
        <button onClick={()=>model.name==='DINOv3 ConvNeXt Tiny'?onConvnextDemo():onPage('Comparador')}>{model.name==='DINOv3 ConvNeXt Tiny'?'Abrir comparação pronta':'Abrir Comparador'}</button>
      </article>)}</div>
      {!shownModels.length&&<p>Nenhum modelo encontrado. Tente outra palavra.</p>}
    </div>}

    {tab==='pipelines'&&<div className="help-content">
      <div className="help-heading"><div><h2>O que cada pipeline faz?</h2><p className="muted">Pipeline é quando a saída de um modelo ajuda o próximo. Você não precisa executar cada etapa manualmente.</p></div>
        <input aria-label="Buscar pipelines" type="search" placeholder="Buscar pipeline ou uso" value={search} onChange={e=>setSearch(e.target.value)}/></div>
      {shownPipelines.map(p=><article className="card help-pipeline" key={p.id}>
        <div className="help-pipeline-top"><div><span className="help-tag">Pipeline disponível</span><h3>{p.title}</h3><p className="help-subtitle">{p.nickname}</p></div><button className="primary" onClick={()=>onPipeline(p.id)}>Abrir pipeline</button></div>
        <div className="help-stages">{p.stages.map((step,i)=><div key={step}><span className="help-step">{i+1}</span><p>{step}</p></div>)}</div>
        <p><strong>Onde usar na indústria:</strong> {p.uses}</p><p className="help-pipeline-warning"><strong>Quando não usar:</strong> {p.warning}</p>
        <details><summary>Teste simples, como interpretar e limites</summary>
          <p><strong>Experimente:</strong> {p.try}</p><p><strong>O que observar:</strong> {p.watch}</p><p><strong>Importante:</strong> {p.warning}</p>
        </details>
      </article>)}
      {!shownPipelines.length&&<p>Nenhum pipeline encontrado. Tente outro termo.</p>}
    </div>}

    {tab==='testes'&&<div className="help-content">
      <div className="help-heading"><div><h2>{TESTS.length} ideias para testar na prática</h2><p className="muted">Abra uma experiência para ver o material, os passos e o resultado esperado.</p></div>
        <input aria-label="Buscar testes" type="search" placeholder="Buscar por peça, modelo ou uso" value={search} onChange={e=>setSearch(e.target.value)}/></div>
      <div className="help-filters">{categories.map(c=><button key={c} className={category===c?'selected':''} onClick={()=>setCategory(c)}>{c}</button>)}</div>
      <div className="help-test-list">{shownTests.map((t,i)=><details className="card help-test" key={t.title}>
        <summary><span className="help-test-summary"><span><span className="help-tag">{t.level} · {t.category}</span><strong>{t.title}</strong></span><small>{t.model}</small></span></summary>
        <div className="help-test-body">
          <p><strong>Você precisa de:</strong> {t.materials}</p>
          <ol>{t.steps.map(step=><li key={step}>{step}</li>)}</ol>
          <p><strong>O que deve acontecer:</strong> {t.expected}</p>
          <p><strong>Como interpretar:</strong> {t.interpret}</p>
          <button className="primary" onClick={()=>openTest(t)}>{t.skuDemo?'Executar Few-Shot pronto':t.convnext?'Comparar DINOv2 x ConvNeXt':t.demo?'Abrir exemplo pronto':t.pipeline?'Abrir pipeline':'Ir para '+t.page}</button>
        </div>
      </details>)}</div>
      {!shownTests.length&&<p>Nenhum teste encontrado. Limpe a busca ou escolha Todos.</p>}
    </div>}

    {tab==='interpretar'&&<div className="help-content">
      <h2>Como interpretar os resultados sem complicação</h2>
      <p>O programa pode mostrar pontuações, mapas e máscaras. Eles ajudam você a decidir onde olhar, mas não substituem um teste com fotos normais e defeituosas.</p>
      <div className="card help-advice"><h3>Exemplo real: uma tampa desapareceu</h3>
        <p><strong>Imagem de referência:</strong> garrafa com tampa. <strong>Imagem de teste:</strong> a mesma garrafa sem tampa. O DINO tende a aumentar a diferença na região da tampa.</p>
        <p><strong>O que fazer depois:</strong> fotografe cinco garrafas normais e cinco sem tampa, compare os resultados e ajuste seu limiar. Assim você descobre se a diferença é repetível.</p>
      </div>
      <div className="help-terms">{TERMS.map(([term,meaning])=><details className="card" key={term}><summary>{term}</summary><p>{meaning}</p></details>)}</div>
      <div className="card help-advice"><h3>E quando a imagem muda inteira?</h3><p>Geralmente vale conferir luz, foco, sombras, posição da câmera e rotação do produto antes de culpar a peça. No DINO, use <strong>Mesma posição</strong> para câmera fixa e teste <strong>Regiões semelhantes</strong> quando a posição variar, lembrando que isso pode esconder erros de montagem.</p></div>
    </div>}

    {tab==='tecnico'&&<div className="help-content"><div className="card help-advice"><h2>Detalhes para quem quiser avançar</h2><p>Este é o guia técnico anterior, preservado para consultar parâmetros, limites, licenças e estudos industriais. Para começar, use as outras abas.</p></div><TechnicalGuide/></div>}
  </section>
}
