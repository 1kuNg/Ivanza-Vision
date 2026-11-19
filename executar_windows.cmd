@echo off
REM Atalho de compatibilidade. A logica de inicializacao fica somente no BAT.
call "%~dp0iniciar_windows.bat"
exit /b %ERRORLEVEL%
