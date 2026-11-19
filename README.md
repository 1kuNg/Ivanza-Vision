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
- **Webcam**. A câmera pertence ao computador que executa o servidor. Capture referências, analise o quadro mais recente ou ative o intervalo contínuo. A captura MJPEG é independente da inferência.
- **Guia Industrial**. Matriz comparativa, licenças e casos técnicos com busca.

Para DINOv3, aceite os termos do checkpoint no Hugging Face e teste um token de leitura em **Config / Tokens**. O teste mantém a credencial na sessão. **Salvar no perfil local** usa o armazenamento oficial do Hugging Face.

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
