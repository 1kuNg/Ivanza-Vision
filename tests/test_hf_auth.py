"""Testes isolados de autenticação, sem usar credenciais ou internet."""
import sys
from types import SimpleNamespace

import pytest


def test_verify_explicit_token_and_existing_login(monkeypatch):
    from hf_auth import verify_dinov3_access
    seen = []
    class FakeAPI:
        def whoami(self, token=None):
            seen.append(('whoami',token))
            return {'name':'usuario-teste'}
    def fake_url(**kwargs):
        seen.append(('url',kwargs))
        return 'https://huggingface.co/fake/model.safetensors'
    def fake_metadata(url, token):
        seen.append(('metadata',url,token))
        return object()
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(HfApi=FakeAPI,hf_hub_url=fake_url,get_hf_file_metadata=fake_metadata))
    assert verify_dinov3_access(' hf_exemplo ')=='usuario-teste'
    assert seen[0] == ('whoami','hf_exemplo')
    assert seen[1][1]['filename'] == 'model.safetensors'
    assert seen[2][2] == 'hf_exemplo'
    seen.clear()
    assert verify_dinov3_access()=='usuario-teste'
    assert seen[0] == ('whoami',None)


def test_save_login_only_after_validation(monkeypatch):
    import hf_auth
    saved=[]
    monkeypatch.setattr(hf_auth,'verify_dinov3_access', lambda token, *, model_id: 'autorizado')
    def fake_login(token, add_to_git_credential=False, skip_if_logged_in=True):
        saved.append((token,add_to_git_credential,skip_if_logged_in))
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(login=fake_login))
    assert hf_auth.save_dinov3_login(' hf_exemplo ')=='autorizado'
    assert saved == [('hf_exemplo',False,False)]
    with pytest.raises(ValueError,match='ausente'):
        hf_auth.save_dinov3_login(' ')


def test_dinov3_passes_token_to_both_processors(monkeypatch):
    from vision_core import DinoV3Extractor
    calls=[]
    class FakeLoaded:
        config=SimpleNamespace(patch_size=16,num_register_tokens=4)
        def to(self,device):
            return self
        def eval(self):
            return self
    class FakeLoader:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id,kwargs))
            return FakeLoaded()
    class FakeCuda:
        @staticmethod
        def is_available():return False
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=FakeCuda()))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoModel=FakeLoader,AutoImageProcessor=FakeLoader))
    extractor=DinoV3Extractor(device='cpu',token='hf_exemplo')
    assert extractor.device=='cpu'
    assert len(calls)==2
    assert all(kwargs['token']=='hf_exemplo' for _,kwargs in calls)
    assert calls[1][1]['use_safetensors'] is True


def test_error_redacts_token():
    from hf_auth import redact_secret
    assert redact_secret('Falha em hf_privado: Acesso negado', 'hf_privado') == 'Falha em [token oculto]: Acesso negado'
    assert redact_secret('Erro comum', None) == 'Erro comum'
