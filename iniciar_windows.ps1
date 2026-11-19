# Ivanza Vision Lab v0.5.0; compatibilidade com Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $projectDir 'logs'
try { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
catch {
    $logDir = Join-Path $env:TEMP 'Ivanzatech-Vision-Lab-logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir ('inicio_' + $stamp + '.log')
$started = $false
try { Start-Transcript -Path $logFile | Out-Null; $started = $true }
catch { Write-Host 'AVISO: Nao foi possivel iniciar a transcricao.' -ForegroundColor Yellow }

function Stage([string]$message) {
    Write-Host ''
    Write-Host ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $message) -ForegroundColor Cyan
}

# Captura stdout + stderr em arquivo e os mostra no terminal. Isso preserva
# o traceback de importacao que antes nao aparecia no log do PowerShell.
$script:commandNumber = 0
function Invoke-Py([string]$exe, [string[]]$argsList, [bool]$streamOutput = $false) {
    $script:commandNumber += 1
    $capture = Join-Path $logDir ('comando_' + $stamp + '_' + $script:commandNumber + '.txt')
    $old = $ErrorActionPreference
    $code = 1
    try {
        $ErrorActionPreference = 'Continue'
        if ($streamOutput) {
            # Preserva a saida ao vivo e grava stderr como texto, sem cabecalho de erro do PowerShell.
            & $exe @argsList 2>&1 | ForEach-Object { [string]$_ } | Tee-Object -FilePath $capture
        } else {
            & $exe @argsList *> $capture
        }
        $code = $LASTEXITCODE
    } catch {
        Add-Content -LiteralPath $capture -Value $_.Exception.ToString()
    } finally {
        $ErrorActionPreference = $old
    }
    if (-not $streamOutput -and (Test-Path -LiteralPath $capture)) {
        Get-Content -LiteralPath $capture | ForEach-Object { Write-Host $_ }
    }
    if ($code -ne 0) {
        throw ('Python retornou codigo {0}. Detalhes no arquivo {1}.' -f $code, $capture)
    }
}

function Find-Python {
    # Prioriza python no PATH; py.exe pode existir sem runtime instalado.
    $candidates = @(
        @{ Ex = 'python'; Args = @() },
        @{ Ex = 'python3'; Args = @() },
        @{ Ex = 'py'; Args = @('-3.11') },
        @{ Ex = 'py'; Args = @('-3.12') },
        @{ Ex = 'py'; Args = @('-3.10') }
    )
    foreach ($item in $candidates) {
        if (-not (Get-Command $item.Ex -ErrorAction SilentlyContinue)) { continue }
        $old = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $cmdArgs = @($item.Args) + @('-c', 'import sys; print(sys.version_info[0],sys.version_info[1])')
            $output = & $item.Ex @cmdArgs 2>$null
            $code = $LASTEXITCODE
            if ($code -eq 0 -and (@($output | Where-Object { ([string]$_) -match '^3 (10|11|12)$' })).Count -gt 0) {
                return $item
            }
        } catch { } finally { $ErrorActionPreference = $old }
    }
    throw 'Python 3.10 a 3.12 x64 nao encontrado. Instale Python 3.11 x64 com Add Python to PATH.'
}

$success = $false
try {
    Stage '[1/4] Verificando o Python existente...'
    Write-Host ('Projeto: ' + $projectDir)
    Write-Host ('Log: ' + $logFile)
    $venvDir = Join-Path $projectDir '.venv'
    $venvPy = Join-Path $venvDir 'Scripts\python.exe'
    $needNewVenv = -not (Test-Path -LiteralPath $venvPy)
    if (-not $needNewVenv) {
        try { Invoke-Py $venvPy @('-c', 'import sys; print(sys.version); sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)') }
        catch {
            Write-Host '[AVISO] Ambiente existente indisponivel ou incompativel com RapidOCR.' -ForegroundColor Yellow
            $needNewVenv = $true
        }
    }

    Stage '[2/4] Preparando ambiente isolado...'
    if ($needNewVenv) {
        $python = Find-Python
        if (Test-Path -LiteralPath $venvDir) {
            $backup = Join-Path $projectDir ('.venv_invalido_' + $stamp)
            Rename-Item -LiteralPath $venvDir -NewName (Split-Path $backup -Leaf)
            Write-Host ('Ambiente anterior preservado em ' + $backup)
        }
        Invoke-Py $python.Ex (@($python.Args) + @('-m', 'venv', $venvDir))
        if (-not (Test-Path -LiteralPath $venvPy)) { throw 'Falha ao criar .venv. Verifique permissoes do Windows.' }
    } else { Write-Host '[OK] .venv existente preservada: nao precisa criar Python novamente.' -ForegroundColor Green }

    Stage '[3/4] Diagnosticando bibliotecas...'
    $diag = Join-Path $projectDir 'diagnostico_ambiente.py'
    $depsOK = $true
    try { Invoke-Py $venvPy @($diag) }
    catch {
        $depsOK = $false
        Write-Host '[AVISO] Dependencias basicas com erro. Tentando instalar apenas o que faltar.' -ForegroundColor Yellow
    }
    if (-not $depsOK) {
        # Nao usa --upgrade: bibliotecas ja instaladas nao sao baixadas novamente.
        Invoke-Py $venvPy @('-m', 'pip', 'install', '-r', (Join-Path $projectDir 'requirements.txt'), '--disable-pip-version-check')
        # Se houver erro de importacao de binarios, mostrar o traceback e nao destruir .venv.
        Invoke-Py $venvPy @($diag)
    }
    Stage '[4/4] Abrindo Ivanza Vision Lab...'
    Write-Host 'Endereco local: http://127.0.0.1:8501'
    Write-Host 'Mantenha esta janela aberta. Para encerrar: Ctrl+C.'
    if (-not (Test-Path -LiteralPath (Join-Path $projectDir 'frontend/dist/index.html'))) {
        throw 'Frontend compilado ausente. Extraia o pacote completo ou execute npm ci e npm run build em frontend.'
    }
    Set-Location -LiteralPath $projectDir
    Invoke-Py $venvPy @('-m', 'uvicorn', 'backend.api:app', '--host', '127.0.0.1', '--port', '8501', '--workers', '1', '--timeout-graceful-shutdown', '5') $true
    $success = $true
} catch {
    Write-Host ''
    Write-Host '[FALHA] Ivanza Vision Lab nao iniciou.' -ForegroundColor Red
    Write-Host ('Detalhes: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('Logs completos: ' + $logDir) -ForegroundColor Yellow
} finally {
    if ($started) { try { Stop-Transcript | Out-Null } catch { } }
}
if ($success) { exit 0 }
exit 1
