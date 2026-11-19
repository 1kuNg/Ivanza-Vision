"""Autenticação explícita do DINOv3; nunca logar ou persistir tokens implicitamente."""
from __future__ import annotations

import re

from vision_core import DINOV3_ID


def verify_dinov3_access(token: str | None = None, *, model_id: str = DINOV3_ID) -> str:
    """Confirma permissão de leitura nos pesos via cabeçalho HTTP, sem baixá-los."""
    from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata
    token = token.strip() if token else None
    account = HfApi().whoami(token=token)
    url = hf_hub_url(repo_id=model_id, filename='model.safetensors')
    get_hf_file_metadata(url, token=token)
    return account.get('name') or 'conta autenticada'


def save_dinov3_login(token: str, *, model_id: str = DINOV3_ID) -> str:
    """Persistência apenas sob ação explícita do usuário."""
    from huggingface_hub import login
    from inspect import signature
    token = token.strip()
    if not token:
        raise ValueError('Token ausente')
    username = verify_dinov3_access(token, model_id=model_id)
    options = {'token': token, 'add_to_git_credential': False}
    args = signature(login).parameters
    if 'skip_if_logged_in' in args:
        options['skip_if_logged_in'] = False
    elif 'new_session' in args:
        options['new_session'] = True
    login(**options)
    return username


def redact_secret(message: str, token: str | None) -> str:
    """Não expor o token mesmo que uma dependência inclua credenciais em erros."""
    if token and token.strip():
        message = message.replace(token.strip(), '[token oculto]')
    return re.sub(r'\bhf_[A-Za-z0-9]{10,}', '[token oculto]', message)
