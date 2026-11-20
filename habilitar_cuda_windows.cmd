@echo off
setlocal
cd /d "%~dp0"
echo.
echo === Ivanza Vision Lab - Preparar GPU NVIDIA ===
echo A instalacao usa apenas a pasta .venv deste programa.
echo Feche a janela do Ivanza Vision Lab antes de continuar.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0habilitar_cuda_windows.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
echo.
if not "%EXITCODE%"=="0" echo [AVISO] Falha no preparo CUDA. Leia as mensagens acima.
pause
exit /b %EXITCODE%
