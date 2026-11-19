"""Conteúdo técnico da navegação Guia Industrial."""
from __future__ import annotations



MATRIX = """
| Modelo / backbone | Escolha quando | Licença |
|---|---|---|
| DINOv2 ViT-S/14 reg4 | Você quer uma referência para SKU, Poka-Yoke ou anomalias por patch. Use câmera fixa e compare latência em 224 e 336 px. | [Apache 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| DINOv3 ViT-S/16 | A área de interesse é pequena e você pode testar o custo de 384/512 px. Compare com o DINOv2 no mesmo conjunto. | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| DINOv3 ConvNeXt-Tiny | Você quer testar um backbone convolucional e comparar desempenho em CPU/GPU. Cada célula da saída final tem stride 32. | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| SAM 2.1 Hiera-Tiny / Small | Você precisa de máscara por peça, recorte, contorno, medida ou propagação temporal. Comece com Tiny e compare a estabilidade de bordas com Small. | [Apache 2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) |
| Grounding DINO Tiny | Localize caixas por descrição em inglês. CPU serve para bancada. Priorize GPU quando houver várias imagens ou uma meta de cadência. | [Apache 2.0](https://huggingface.co/IDEA-Research/grounding-dino-tiny) |
| Depth Anything V2 Small | Estime profundidade relativa, recorte o primeiro plano e compare as células de uma cavidade. Comece em CPU e meça a latência em GPU se o uso for contínuo. | [Apache 2.0](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| SigLIP 2 Base | Avalie estados visuais independentes descritos em texto. CPU serve para ensaios esporádicos. GPU reduz o custo por imagem quando há monitoramento contínuo. | [Apache 2.0](https://huggingface.co/google/siglip2-base-patch16-224) |
| RapidOCR | Leia texto e valide lote, validade e etiquetas em CPU via ONNX. Recortes bem enquadrados reduzem a área a processar e evitam texto fora da etiqueta. | [Apache 2.0](https://pypi.org/project/rapidocr-onnxruntime/) |
"""

CASES = [
    ('Poka-Yoke visual · presença e montagem', 'DINOv2 ViT-S/14 reg4', 'Baixa',
     'Câmera fixa, ROI por ponto de montagem, 10 fotos OK e 10 NG como ponto de partida.',
     [
         'Recorte sempre a mesma ROI e extraia o vetor global `[CLS]` normalizado.',
         'Treine LogisticRegression ou LinearSVC, ou compare por cosseno com os exemplos OK e NG.',
         'Separe fotos de validação por turno ou lote. Ajuste o limiar para suprimir falsos aceites e contabilize falsos rejeites.',
     ]),
    ('Reconhecimento de SKU high-mix · sem retreinamento', 'DINOv2 ou DINOv3', 'Baixa',
     'Três a dez imagens por SKU, com poses e variações que ocorrem na linha, e exemplos de produtos desconhecidos.',
     [
         'Normalize os embeddings globais e salve a matriz em `.npy` com a lista de SKUs e as configurações de extração.',
         'Na inferência, calcule `embedding @ banco.T`. Use a maior similaridade por SKU e registre a margem entre os dois melhores.',
         'Ajuste o limiar de Desconhecido em SKUs fora do cadastro. Para incluir um SKU, acrescente embeddings e versione o banco.',
     ]),
    ('Sensor virtual de processo · drift, entupimento, espuma e desalinhamento', 'DINOv2 / DINOv3', 'Média',
     'Vídeo normal de turnos e regimes diferentes, eventos de referência e amostra sincronizada com o processo.',
     [
         'Extraia `[CLS]` por quadro e ajuste centroide com distância de Mahalanobis regularizada ou treine um Isolation Forest.',
         'Calibre a escala do score com dados retidos. Suavize com média móvel e use persistência mínima para disparar eventos.',
         'Publique score de 0,0 a 1,0 com timestamp e qualidade do dado via MQTT, OPC UA ou Modbus. Trate câmera offline como perda de qualidade.',
     ]),
    ('Defeitos superficiais few-shot · AnomalyDINO / PatchCore', 'DINOv2 reg4 ou DINOv3 ViT a 512 px', 'Média a alta',
     'Cinco a 30 fotos apenas OK para montar o banco, iluminação estável e defeitos reais para validação.',
     [
         'Extraia patch tokens e filtre fundo com PCA ou máscara SAM. Preserve a geometria do letterbox para reprojeção.',
         'Reduza o banco por coreset subsampling. Faça busca k-NN, por exemplo em FAISS/GPU, e versione distância e índice.',
         'Use a média do 1% de patches com maior distância como score de bancada. Ajuste com defeitos reais e compare por tipo e tamanho.',
     ]),
    ('Localização zero-shot de referência · ROI dinâmica', 'DINOv2 ou DINOv3 ViT', 'Média',
     'Uma peça-padrão fotografada, ponto de referência anotado e faixa aceitável de deslocamento.',
     [
         'Escolha o patch do ponto-chave na peça-padrão e guarde seu vetor normalizado.',
         'Calcule similaridade cosseno com a grade do quadro novo. Rejeite máximos fracos ou empates e reprojete o melhor ponto em pixels.',
         'Refine a posição em uma janela local e use o ponto para posicionar a ROI da inspeção seguinte.',
     ]),
    ('Curadoria e pré-rotulagem offline de datasets', 'DINOv2 ou DINOv3', 'Média',
     'Vídeo bruto de produtos, turnos e falhas, com timestamp e identificação de lote.',
     [
         'Amostre quadros e extraia `[CLS]` offline. Agrupe com K-Means ou HDBSCAN.',
         'Comece rotulando cerca de 5% do acervo, escolhendo centros, exemplos diversos e outliers relevantes. Revise cobertura de classes raras.',
         'Organize divisões por vídeo/lote e treine YOLO ou RT-DETR. Expanda a amostra a partir dos erros de validação.',
     ]),
    ('Fábrica de datasets com vídeo', 'SAM 2.1 Video Predictor', 'Média',
     'Vídeo de 2–5 minutos e o primeiro quadro da faixa de interesse com a peça visível.',
     [
         'Clique no primeiro quadro e propague a máscara pelos quadros com o Video Predictor.',
         'Revise entrada, saída e oclusões. Corrija prompts quando houver perda da peça.',
         'Exporte polígonos e treine um modelo leve de segmentação para a borda, como YOLO11-Seg. Separe vídeos inteiros entre treino e teste.',
     ]),
    ('Metrologia 2D · granulometria e área', 'SAM 2.1 Tiny / Small', 'Média',
     'Câmera e altura fixas, gabarito no plano da peça e prompts fixos ou caixas geradas por um detector.',
     [
         'Calibre px/mm e verifique distorção de lente e perspectiva em toda a ROI.',
         'Segmente cada objeto. Calcule a área da máscara ou use `cv2.contourArea` e `cv2.minAreaRect` para dimensões orientadas.',
         'Divida área por (px/mm)² e comprimentos por px/mm. Para granulometria, agregue distribuição de tamanhos e trate grãos sobrepostos.',
     ]),
    ('Pipeline híbrido · SAM 2.1 + DINOv3', 'SAM 2.1 com DINOv3 ViT', 'Alta',
     'Referências OK, imagens de teste, prompts ou detector de peças e escala no plano de medição.',
     [
         'Fluxo A. Segmente a peça nas referências e no teste. Projete as máscaras para as grades DINO e restrinja score e heatmap aos patches da peça.',
         'Fluxo B. Converta o máximo do heatmap DINO em um ponto na imagem original e use-o como prompt do SAM.',
         'Valide se a máscara do fluxo B recorta o defeito e ajuste com pontos negativos ou uma caixa local. Depois converta a área em mm² e aplique a regra calibrada.',
     ]),
    ('Tracking com oclusão · memória de vídeo', 'SAM 2.1 Video Predictor / memória de streaming', 'Alta',
     'Vídeo contínuo, ROI de entrada e saída, detecção ou prompt inicial e sequência de oclusões para teste.',
     [
         'Inicialize o ID na zona de entrada e propague a máscara quadro a quadro com memória limitada.',
         'Para operar 24/7, descarte IDs que saem da ROI e programe reset e novo prompt periódicos. Reinicialize também após perda persistente.',
         'Monitore drift, latência, memória e recuperação após oclusão em ensaio prolongado antes de habilitar o fluxo contínuo.',
     ]),
    ('Bin picking · peças variadas com RGB-D', 'SAM 2.1 + câmera RGB-D', 'Alta',
     'RGB e profundidade alinhados, câmera calibrada em relação ao robô e geometria da garra.',
     [
         'Remova o plano do recipiente e encontre um pico de altura válido. Projete esse ponto na imagem RGB para o prompt SAM.',
         'Use profundidade sob a máscara para calcular centroide 3D e orientação candidata via eixos principais.',
         'Avalie alcance, colisão e abertura da garra na célula validada. Teste peças empilhadas, reflexos e ausência de profundidade.',
     ]),
    ('Deploy · do laboratório para a linha', 'DINO, SAM 2.1 e CNN leve conforme a cadência', 'Alta',
     'Conjunto de referência com OK/NG, resolução e normalização fixas, meta de latência p95 e hardware final.',
     [
         'Exporte para ONNX. Compile em TensorRT FP16 para NVIDIA/Jetson ou OpenVINO para IPC Intel x86. Valide score, máscaras e latência ponta a ponta no equipamento.',
         'Em ViTs, evite PTQ INT8 ingênua em LayerNorm/Softmax. Outliers podem mudar a ordem dos vizinhos k-NN. Para testar INT8, priorize DINOv3 ConvNeXt-Tiny ou precisão mista e calibre com imagens reais.',
         'Separe Image Encoder e Mask Decoder do SAM. Calcule o encoder uma vez por quadro e reutilize o embedding para múltiplos prompts. Na linha, use prompts dirigidos e mantenha AutomaticMaskGenerator no fluxo offline.',
         'Para uma meta inferior a 10 ms por quadro, use DINO como professor offline por destilação de features ou pseudorrótulos. Treine ConvNeXt-Tiny, MobileNetV4 ou ResNet34 e valide a CNN leve na linha.',
     ]),
]


EXTRA_CASES = [
    ('Remoção de fundo 2.5D, nível e cavidade', 'Depth Anything V2 Small com DINO ou SAM 2.1', 'Média',
     'Câmera fixa, fotos de posições vazias e ocupadas e grade coerente com o gabarito.',
     [
         'Gere o mapa relativo e ajuste o corte de primeiro plano. Profundidade estimada pode reduzir a influência de cor e sombra em relação a um corte por intensidade. Teste reflexos e variações de luz.',
         'Compare a proximidade média de cada célula com amostras vazias e cheias para sinalizar cavidades ou mudança de nível. A escala é normalizada por imagem e depende da composição da cena.',
         'Para explorar bin picking, envie o ponto mais próximo para o SAM e revise a máscara resultante. Para obter pose ou altura em unidades físicas, use profundidade calibrada e valide a geometria da célula.',
     ]),
    ('Auditoria 5S, segurança e estado de cena', 'SigLIP 2 Base', 'Baixa a média',
     'Vista estável e descrições como open electrical panel, wet floor e empty conveyor.',
     [
         'Escreva um estado por linha e execute comparações independentes com sigmoide. Estados simultâneos podem ter escores altos.',
         'Avalie alertas em fotos do local, incluindo iluminação adversa e objetos que confundem o modelo. Defina limites a partir desses exemplos, sem precisar treinar um classificador próprio.',
         'Publique o evento quando o estado exceder o limiar pelo tempo exigido. Use-o como indicação para revisão e reserve funções de proteção às salvaguardas da instalação.',
     ]),
    ('OCR industrial de lote, validade e etiquetas', 'RapidOCR em CPU, com Grounding DINO opcional', 'Baixa a média',
     'Imagem legível da etiqueta, formato esperado por SKU e exemplos de impressão boa e ruim.',
     [
         'Localize a etiqueta com uma ROI fixa ou Grounding DINO usando label ou date code.',
         'Leia a imagem inteira ou as ROIs em CPU via ONNX. Inspecione a confiança e a posição das palavras.',
         r'Valide a string inteira com uma regex como `LOTE-\d+` e ajuste a confiança mínima com fotos reais. Para datas, valide também o calendário e o prazo no sistema consumidor.',
     ]),
    ('Pipeline Grounding DINO → SAM 2.1 → DINOv3 / RapidOCR',
     'Grounding DINO Tiny, SAM 2.1, DINOv3 ou RapidOCR', 'Média a alta',
     'Fotos com as peças desejadas, termos em inglês que as descrevam e referências normais para inspeção DINOv3.',
     [
         'Detecte componentes e etiquetas com prompts de texto. Ajuste os limiares e confirme posição e contagem no conjunto de referência.',
         'Envie as caixas ao SAM como prompts de caixa ou use os centros como os pontos positivos disponíveis no laboratório. Isso automatiza o posicionamento inicial sem cliques nem grade fixa.',
         'Use a máscara do SAM para restringir as regiões comparadas pelo DINOv3. Para etiquetas, leia a ROI com RapidOCR. Use a máscara como pré-filtro apenas se conservar todo o texto.',
         'Valide separadamente correspondência, segmentação e leitura. Registre detecções ausentes e baixa confiança para revisão, com limites por SKU.',
     ]),
]


