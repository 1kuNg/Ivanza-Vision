from collections import OrderedDict
from hashlib import sha256
from threading import Lock
from uuid import uuid4

from dinov2_core import open_image, png_bytes


class LRU:
    def __init__(self, capacity, max_bytes=None):
        self.capacity, self.max_bytes = capacity, max_bytes
        self.items = OrderedDict()
        self.total = 0
        self.lock = Lock()

    def put(self, key, value, size=0):
        with self.lock:
            if self.max_bytes is not None and size > self.max_bytes:
                raise ValueError('Arquivo acima da capacidade do armazenamento.')
            prior = self.items.pop(key, None)
            if prior is not None:
                self.total -= prior[1]
            self.items[key] = (value, size)
            self.total += size
            while len(self.items) > self.capacity or self.max_bytes is not None and self.total > self.max_bytes:
                self.total -= self.items.popitem(last=False)[1][1]
        return key

    def get(self, key):
        with self.lock:
            value = self.items.get(key)
            if value is None:
                return None
            self.items.move_to_end(key)
            return value[0]


class Store:
    def __init__(self):
        self.images = LRU(50, 256 * 1024**2)
        self.artifacts = LRU(240, 128 * 1024**2)
        self.rounds = LRU(20)
        self.views = LRU(35, 160 * 1024**2)
        self.sessions = LRU(24)

    def upload(self, data):
        digest = sha256(data).hexdigest()
        image = self.images.get(digest)
        if image is None:
            image = open_image(data)
            if image.width * image.height > 12_000_000:
                raise ValueError('Use uma imagem com até 12 megapixels.')
            served = png_bytes(image)
            self.images.put(digest, (image, served), image.width * image.height * 3 + len(served))
        else:
            image = image[0]
        return dict(image_id=digest, width=image.width, height=image.height, url=f'/api/images/{digest}')

    def image(self, key):
        entry = self.images.get(key)
        if entry is None:
            raise ValueError('A imagem expirou ou não existe. Envie novamente.')
        return entry[0]

    def artifact(self, image):
        data = png_bytes(image)
        ident = self.artifacts.put(sha256(data).hexdigest(), data, len(data))
        return f'/api/artifacts/{ident}'

    def session(self, key):
        if key:
            value = self.sessions.get(key)
            if value is not None:
                return key, value
        key = uuid4().hex
        value = {}
        self.sessions.put(key, value)
        return key, value
