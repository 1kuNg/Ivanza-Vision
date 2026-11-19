"""Diagnostico offline do ambiente; nao baixa nem carrega pesos de modelos.

Retorna 0 se as bibliotecas basicas funcionam. Imports de modelos opcionais
sao informativos e nao impedem a inicializacao do laboratorio.
"""
from __future__ import annotations

import importlib
import platform
import sys
import traceback

from importlib import metadata


REQUIRED = (
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("httpx", "httpx"),
    ("python_multipart", "python-multipart"),
    ("cv2", "opencv-python"),
    ("safetensors", "safetensors"),
    ("rapidocr_onnxruntime", "rapidocr-onnxruntime"),
)
OPTIONAL = (
    ("transformers", "Sam2Model"),
    ("transformers", "Sam2Processor"),
    ("transformers", "Dinov2WithRegistersModel"),
    ("transformers", "DINOv3ViTModel"),
    ("transformers", "DINOv3ConvNextModel"),
    ("transformers", "AutoImageProcessor"),
    ("transformers", "AutoModel"),
    ("transformers", "AutoModelForDepthEstimation"),
    ("transformers", "AutoModelForZeroShotObjectDetection"),
)


def main():
    print("=== Diagnostico de dependencias Ivanza Vision Lab v0.5.0 ===", flush=True)
    print("Python:", sys.version.replace("\n", " "), flush=True)
    print("Sistema:", platform.platform(), flush=True)
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        print('[FALHA] Use Python 3.10 a 3.12 para instalar RapidOCR ONNX.', flush=True)
        return 1
    errors = []
    for module, package in REQUIRED:
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = "nao instalado"
        print(f"[TESTE] import {module} (pip: {package} {installed})", flush=True)
        try:
            importlib.import_module(module)
            if module == 'transformers':
                from packaging.version import Version
                minimum = '4.57'
                major = '5'
                if not Version(minimum) <= Version(installed) < Version(major):
                    raise RuntimeError(f'{package} {installed} fora do intervalo >= {minimum}, < {major}.')
            print(f"[OK] {module}", flush=True)
        except Exception as exc:
            errors.append(module)
            print(f"[ERRO] {module}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

    print("=== Recursos opcionais (nao impedem o programa de iniciar) ===", flush=True)
    for module, name in OPTIONAL:
        if module in errors:
            print(f"[PULADO] from {module} import {name}: modulo basico indisponivel", flush=True)
            continue
        try:
            getattr(importlib.import_module(module), name)
            print(f"[OK] {name}", flush=True)
        except Exception as exc:
            print(f"[AVISO] {name}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
    if errors:
        print("[FALHA] Modulos necessarios com erro: " + ", ".join(errors), flush=True)
        return 1
    print("[OK] Bibliotecas basicas prontas; cada modelo sera carregado quando escolhido.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
