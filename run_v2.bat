@echo off
REM Odysseus V2 Bridge Server — Início rápido (Windows)
REM
REM Uso:
REM   run_v2.bat                    - Inicia na porta 8001
REM   run_v2.bat --port 8001        - Porta personalizada
REM   run_v2.bat --model qwen3:8b    - Modelo padrão
REM

cd /d "%~dp0"

echo.
echo ========================================
echo  Odysseus V2 Bridge Server
echo ========================================
echo.

REM Instalar dependências se necessário
pip install fastapi uvicorn httpx 2>nul

python v2_server.py %*

pause
