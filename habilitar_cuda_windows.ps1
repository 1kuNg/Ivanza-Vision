# Instala uma versão CUDA do PyTorch somente no ambiente isolado deste Lab.
# Execute com o servidor desligado. Não altera o Python global nem o driver NVIDIA.
param([ValidateSet('cu126','cu128')][string]$Cuda='cu126')
$ErrorActionPreference = 'Stop'
$homeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $homeDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Ambiente .venv nao encontrado. Rode iniciar_windows.bat pelo menos uma vez antes de instalar CUDA.'
}
Write-Host '===================================================='
Write-Host 'Ivanza Vision Lab - Habilitar GPU NVIDIA (PyTorch CUDA)'
Write-Host '===================================================='
Write-Host ('Ambiente: ' + $python)
Write-Host ('Indice oficial PyTorch: ' + $Cuda)
Write-Host 'Mantenha o Ivanza Vision Lab fechado durante esta instalacao.' -ForegroundColor Yellow
Write-Host 'Os downloads do PyTorch com CUDA podem ser grandes.' -ForegroundColor Yellow
try {
    & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    if ($LASTEXITCODE -ne 0) { Write-Host '[AVISO] nvidia-smi retornou erro. Confirme o driver NVIDIA.' -ForegroundColor Yellow }
} catch {
    Write-Host '[AVISO] nvidia-smi nao encontrado no PATH. Verifique o driver NVIDIA.' -ForegroundColor Yellow
}
Write-Host ''
& $python -c 'import sys,torch;print("Python:",sys.version.split()[0],"| PyTorch:",torch.__version__,"| CUDA:",torch.version.cuda,"| Disponivel:",torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'O ambiente Python atual esta quebrado. Corrija-o antes da instalacao.' }
Write-Host ''
$answer = Read-Host 'Deseja instalar/reinstalar o PyTorch CUDA apenas na .venv? (S/N)'
if ($answer -notin @('s','S','sim','SIM','Sim')) {
    Write-Host 'Nenhuma alteracao realizada.'
    exit 0
}
$url = 'https://download.pytorch.org/whl/' + $Cuda
Write-Host ('[INSTALANDO] PyTorch e torchvision de ' + $url) -ForegroundColor Cyan
& $python -m pip install --upgrade --force-reinstall 'torch>=2.5.1,<3' 'torchvision>=0.20.1,<1' --index-url $url --disable-pip-version-check
if ($LASTEXITCODE -ne 0) {
    throw 'Falha ao instalar PyTorch CUDA. Confira internet, espaco no disco e compatibilidade do Python.'
}
Write-Host ''
Write-Host '[TESTE] PyTorch, torchvision e acesso a GPU:' -ForegroundColor Cyan
& $python -c 'import torch,torchvision;print("PyTorch:",torch.__version__);print("torchvision:",torchvision.__version__);print("CUDA compilado:",torch.version.cuda);print("CUDA disponivel:",torch.cuda.is_available());print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nao acessivel")'
if ($LASTEXITCODE -ne 0) { throw 'Instalacao concluida, mas as bibliotecas nao passaram no teste. Veja os erros acima.' }
& $python -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 2)'
if ($LASTEXITCODE -ne 0) {
    throw 'PyTorch CUDA instalado, mas a GPU nao esta acessivel. Atualize o driver ou confira se a placa suporta a versao CUDA escolhida.'
}
Write-Host ''
Write-Host '[OK] GPU NVIDIA disponivel! Abra iniciar_windows.bat e selecione GPU NVIDIA no topo.' -ForegroundColor Green
