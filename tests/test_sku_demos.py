"""Demonstrações Few-Shot reproduzíveis sem baixar pesos gigantes para o CI."""
from hashlib import sha256
from io import BytesIO

import pytest
from PIL import Image

from sku_demos import SKU_DEMOS, render_demo_image, demo_payload


@pytest.mark.parametrize('demo', ['embalagens','fixadores'])
def test_demo_includes_two_known_classes_and_unknown(demo):
    seen = []

    def upload(data):
        image = Image.open(BytesIO(data))
        image.verify()
        digest = sha256(data).hexdigest()
        seen.append(digest)
        return dict(image_id=digest,width=640,height=420,url=f'/api/images/{digest}')

    payload = demo_payload(demo, upload)
    assert payload['id'] == demo
    assert len(payload['classes']) == 2
    assert len(payload['tests']) == 3
    assert [len(c['photos']) for c in payload['classes']] == [3,3]
    assert payload['tests'][0]['expected'] == payload['classes'][0]['name']
    assert payload['tests'][1]['expected'] == payload['classes'][1]['name']
    assert payload['tests'][2]['expected'] == 'Desconhecido'
    assert payload['tests'][2]['name'] == SKU_DEMOS[demo]['unknown']
    assert len(set(seen)) == len(seen) == 9
    assert set(c['name'] for c in payload['classes']) == set(SKU_DEMOS[demo]['classes'])


def test_demos_are_reproducible_and_validated():
    for demo,item in [('embalagens','frasco'),('fixadores','arruela')]:
        first=render_demo_image(demo,item,0)
        second=render_demo_image(demo,item,0)
        assert first == second and first.startswith(b'\x89PNG\r\n\x1a\n')
        with Image.open(BytesIO(first)) as img:
            assert img.size == (640,420) and img.mode == 'RGB'
    with pytest.raises(KeyError):
        render_demo_image('inexistente','frasco',0)
    with pytest.raises(ValueError):
        render_demo_image('fixadores','frasco',0)
