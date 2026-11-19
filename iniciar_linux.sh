#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/python -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else "Use Python 3.10 a 3.12 com RapidOCR ONNX.")'
.venv/bin/python -m pip install -r requirements.txt
test -f frontend/dist/index.html || { echo "Execute npm run build em frontend."; exit 1; }
exec .venv/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port 8501 --workers 1 --timeout-graceful-shutdown 5
