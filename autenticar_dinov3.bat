@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Ivanza Vision Lab - Login Hugging Face DINOv3
cd /d "%~dp0"
echo.
echo ==================================================
echo   IVANZA VISION LAB - AUTENTICAR DINOV3
 echo ==================================================
echo.
if not exist ".venv\Scripts\hf.exe" (
  echo [FALHA] Executavel hf.exe nao encontrado na pasta .venv.
  echo Primeiro inicie o Ivanza Vision Lab com iniciar_windows.bat.
  goto :fim
)
echo 1. Entre no Hugging Face e aceite os termos do modelo DINOv3.
echo 2. Na etapa seguinte, faca login no navegador ou use um token Read.
echo 3. Nunca envie o token por mensagem nem o registre em arquivos de log.
echo.
echo Abrindo a pagina oficial do modelo...
start "" "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m"
echo.
echo Aceite os termos no navegador antes de prosseguir.
pause
".venv\Scripts\hf.exe" auth login
if errorlevel 1 (
  echo.
  echo [FALHA] Nao foi possivel autenticar. Verifique a conexao e tente novamente.
  goto :fim
)
echo.
echo Verificando a conta autenticada...
".venv\Scripts\hf.exe" auth whoami
if errorlevel 1 (
  echo [AVISO] Nao foi possivel verificar o login. Tente novamente.
) else (
  echo.
  echo Conta autenticada. Volte ao Ivanza Vision Lab para verificar o acesso e executar DINOv3.
)
:fim
echo.
pause
endlocal
