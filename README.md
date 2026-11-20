# Ivanza Vision Lab v0.5.1

Bancada local de visão industrial com FastAPI, React, TypeScript, Vite e Tailwind. O servidor Python serve a interface compilada e a API no mesmo endereço.

## Instalação

No Windows, extraia o pacote e execute `iniciar_windows.bat`. Use Python 3.10 a 3.12 x64 (3.11 ou 3.12 recomendado). O inicializador abre o navegador assim que a API responder. Endereço manual: [http://127.0.0.1:8501](http://127.0.0.1:8501). O pacote inclui o frontend compilado. Node.js é necessário apenas para recompilar a interface.

Ao atualizar uma instalação que usava Streamlit na mesma porta, feche as abas antigas. Se o navegador ainda mostrar a interface anterior, abra o endereço em uma janela anônima ou recarregue com Ctrl+F5. O cliente antigo pode continuar consultando rotas `/_stcore/` até a aba ser fechada.

No Linux, execute `bash iniciar_linux.sh`. O ambiente Python é isolado em `.venv`. A primeira instalação e os primeiros downloads de modelos requerem conexão. Os pesos são os checkpoints oficiais e o Hugging Face guarda o cache local.

## Telas

- **Comparador**. Carregue imagem e referências pelo seletor ou escolha os exemplos. Selecione os modelos e execute. Use o editor SAM para pontos, marcações negativas, grade e caixas. Compare saídas com zoom sincronizado ou abra a inspeção profunda.
- **Pipelines**. Selecione Grounding → SAM → DINOv3, Grounding → RapidOCR, Depth → SAM ou SAM → DINO. Os fluxos DINO usam imagens normais de referência.
- **Few-Shot / SKU**. Registre de duas a seis classes com uma a dez fotos por classe. O vizinho mais próximo usa os vetores globais do DINO. Ajuste o limiar de desconhecido para o conjunto em uso.
  - **Exemplos prontos (sem webcam):** na própria tela, escolha **Embalagens: frasco x caixa** ou **Fixadores: porca x arruela** e clique em **Testar DINOv2** ou **Testar ConvNeXt Tiny**. O Lab cria três imagens de referência por classe, registra o banco de vetores e executa uma classificação real. Depois selecione as três imagens de teste, incluindo um objeto não cadastrado.
  - As imagens são geradas sinteticamente para demonstração. Nenhum score ou resultado é simulado: as inferências usam o checkpoint selecionado. A rejeição de um desconhecido depende do limiar e não é garantida. Para testes industriais, substitua por fotos reais.
  - O DINOv3 ConvNeXt Tiny requer acesso ao próprio checkpoint em **Config / Tokens → DINOv3 ConvNeXt Tiny**; a primeira execução pode baixar os pesos.
  - Na **Ajuda & Testes** também há um botão para comparar **DINOv2 x ConvNeXt Tiny** no mesmo painel com peça faltando.
- **Webcam**. A câmera pertence ao computador que executa o servidor. Capture referências, analise o quadro mais recente ou ative o intervalo contínuo. A captura MJPEG é independente da inferência.
- **Ajuda & Testes (para iniciantes)**. Na aba **Qual modelo escolher?**, selecione um problema (anomalia, SKU, localização de objetos, segmentação, leitura de texto, estado visual, profundidade ou pipeline). O programa explica o modelo recomendado, o que preparar, como avaliar, limites, um teste e o botão para abrir a bancada. Em **CNN x Transformer**, compare DINOv3 ConvNeXt Tiny e DINOv3 Small visualmente, com as resoluções do mapa e diferenças do Few-Shot. Em **Limites & validação**, siga o roteiro de amostras reais, falsos positivos/negativos, desconhecidos, latência e integração industrial.
- **Guia técnico**. A antiga matriz de modelos, licenças e casos industriais continua disponível, mas separada das explicações simplificadas.

Para DINOv3, aceite os termos do checkpoint no Hugging Face e teste um token de leitura em **Config / Tokens**. O teste mantém a credencial na sessão. **Salvar no perfil local** usa o armazenamento oficial do Hugging Face.

## CPU, GPU NVIDIA e troca dinâmica

O indicador no topo mostra **o dispositivo ativo do PyTorch**, não apenas a existência física de uma placa NVIDIA.

- **Auto:** usa CUDA quando `torch.cuda.is_available()` é verdadeiro; senão utiliza CPU.
- **CPU:** força execução na CPU, mesmo que exista GPU compatível.
- **GPU NVIDIA:** força CUDA. Esta opção só fica disponível quando o PyTorch instalado na pasta `.venv` consegue acessar a GPU.
- A troca **não reinicia o servidor**. Ela espera uma análise ativa terminar, libera os modelos e os recarrega no novo dispositivo na análise seguinte. As fotos e resultados já exibidos permanecem. A primeira análise após a mudança pode demorar mais.
- **RapidOCR** usa ONNX Runtime CPU nesta versão; a seleção da GPU se aplica aos modelos PyTorch (DINO, SAM, Grounding DINO, Depth Anything e SigLIP).
- **Diagnóstico GPU**, no topo, informa placa, versão do driver quando disponível, compilação CUDA do PyTorch e o motivo mais provável de não haver CUDA.

### Preparar PyTorch CUDA no Windows

Se o computador tem placa NVIDIA, mas o Lab mostra apenas CPU, o PyTorch dentro da `.venv` pode estar instalado **sem CUDA**. A instalação de outro Python ou de ferramentas CUDA no sistema **não altera automaticamente** a versão de PyTorch da `.venv`.

1. Instale os drivers NVIDIA adequados à GPU e execute `nvidia-smi` no terminal para verificar se o driver responde.
2. Feche o Ivanza Vision Lab.
3. Na pasta do programa, execute **`habilitar_cuda_windows.cmd`**. Ele oferece a instalação do PyTorch CUDA 12.6 apenas na `.venv` e pede confirmação, sem modificar o Python global.
4. Se o driver e a placa exigirem outra variante CUDA, o comando `habilitar_cuda_windows.cmd -Cuda cu128` seleciona CUDA 12.8. Confira a compatibilidade no [guia oficial PyTorch](https://pytorch.org/get-started/locally/).
5. Depois da instalação, abra `iniciar_windows.bat` e selecione **Auto** ou **GPU NVIDIA** no topo.

A instalação CUDA pode baixar vários gigabytes e exige internet. Não execute o instalador enquanto o Lab estiver aberto. Se CUDA continuar indisponível, confira o diagnóstico antes de reinstalar.

Para testar manualmente o ambiente isolado:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Assistente IA

O padrão é `https://openrouter.ai/api/v1` com `openrouter/free`. Configure `LLM_BASE_URL`, `LLM_MODEL` e `LLM_API_KEY` (ou `OPENROUTER_API_KEY`) no ambiente ou use **Config / Tokens** para a sessão. Para Ollama ou llama.cpp use a URL OpenAI-compatible local, o nome do modelo e a chave quando exigida.

O assistente envia instruções e perguntas em linguagem natural, parâmetros da bancada e telemetria selecionada ao provedor configurado. Credenciais conhecidas são mascaradas e imagens ficam no servidor. O texto OCR só é incluído quando você ativa a opção no relatório. O OpenRouter aceita uma lista de modelos separados por vírgula para fallback. As propostas passam por validação e só são aplicadas após confirmação.

## Desenvolvimento e testes

```sh
python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8501 --workers 1
```

Rode `python -m pytest -q` com pytest instalado. A API é documentada em `/docs`. Os modelos, arquivos, cache e preferências de sessão são mantidos em memória conforme aplicável. Use um único processo para compartilhar o cache e o dispositivo de captura.
