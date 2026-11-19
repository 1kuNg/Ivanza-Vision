@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Ivanza Vision Lab - Inicializacao

REM Usar um unico processo de CMD, sem chamada aninhada: preserva as aspas do -File.
REM %~dp0 e a pasta deste BAT, incluindo a barra final.
set "RESULT=1"
set "SCRIPT=%~dp0iniciar_windows.ps1"

cd /d "%~dp0"
if errorlevel 1 goto :erro_pasta
if not exist "%SCRIPT%" goto :erro_script
where powershell.exe >nul 2>&1
if errorlevel 1 goto :erro_powershell

echo.
echo ============================================================
set "IVANZA_VISION_VERSION="
if exist "%~dp0VERSION" set /p IVANZA_VISION_VERSION=<"%~dp0VERSION"
echo  IVANZATECH - Vision Lab %IVANZA_VISION_VERSION% - Windows
echo ============================================================
echo.
echo Iniciando o PowerShell com o caminho correto...
echo.

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" goto :erro_execucao

echo.
echo O Ivanza Vision Lab foi encerrado normalmente.
goto :final

:erro_pasta
echo [FALHA] Nao foi possivel abrir a pasta do aplicativo.
goto :final

:erro_script
echo [FALHA] Arquivo iniciar_windows.ps1 nao encontrado.
echo Extraia TODO o conteudo do ZIP antes de iniciar.
goto :final

:erro_powershell
echo [FALHA] powershell.exe nao foi encontrado no Windows.
goto :final

:erro_execucao
echo.
echo [FALHA] A inicializacao terminou com codigo %RESULT%.
echo Se o PowerShell iniciou, consulte o arquivo mais recente da pasta logs.

:final
echo.
echo Este terminal permanecera aberto para mostrar o resultado.
pause
exit /b %RESULT%
