"""Extração DINOv2 e inspeção por distância cosseno de patches."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import io
from typing import Iterable, Literal

import numpy as np
from PIL import Image, ImageOps

MODEL_ID = "facebook/dinov2-with-registers-small"
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
Mode = Literal["posicao", "regioes"]


@dataclass(frozen=True)
class Letterbox:
    size: int
    box: tuple[int, int, int, int]
    original_size: tuple[int, int]

    def valid_patch_mask(self, grid: tuple[int, int], foreground=None) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        rows, cols = grid
        xs = np.arange(cols + 1) * self.size // cols
        ys = np.arange(rows + 1) * self.size // rows
        if foreground is None:
            valid_x = (xs[:-1] < x1) & (xs[1:] > x0)
            valid_y = (ys[:-1] < y1) & (ys[1:] > y0)
            return valid_y[:, None] & valid_x[None, :]
        foreground = np.asarray(foreground, dtype=bool)
        if foreground.shape != self.original_size[::-1]:
            raise ValueError("A máscara SAM precisa ter a resolução da imagem original.")
        # BOX preserva cobertura de regiões finas ao reduzir uma máscara em ponto flutuante.
        mask = Image.fromarray(foreground.astype(np.float32)).resize(
            (x1 - x0, y1 - y0), Image.Resampling.BOX)
        canvas = np.zeros((self.size, self.size), dtype=bool)
        canvas[y0:y1, x0:x1] = np.asarray(mask) > 0
        by_row = np.logical_or.reduceat(canvas, ys[:-1], axis=0)
        return np.logical_or.reduceat(by_row, xs[:-1], axis=1)


@dataclass
class Features:
    global_vector: np.ndarray
    patches: np.ndarray
    grid: tuple[int, int]
    geometry: Letterbox | None = None

    def valid_patch_mask(self, foreground=None) -> np.ndarray:
        if self.geometry is not None:
            return self.geometry.valid_patch_mask(self.grid, foreground)
        if foreground is not None:
            raise ValueError("A geometria do extrator é necessária para usar a máscara SAM.")
        return np.ones(self.grid, dtype=bool)


@dataclass
class Analysis:
    global_similarity: float
    patch_distances: np.ndarray
    mean_distance: float
    p95_distance: float
    max_distance: float
    mode: Mode
    count_references: int
    geometry: Letterbox | None = None
    foreground: np.ndarray | None = None


def open_image(data: bytes) -> Image.Image:
    """Valida o tamanho, aplica EXIF e retorna RGB na orientação de exibição."""
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > 45_000_000:
            raise ValueError("Imagem muito grande. Use até 45 megapixels.")
        return ImageOps.exif_transpose(source).convert("RGB")


def letterbox_geometry(image: Image.Image, size: int) -> Letterbox:
    scale = size / max(image.size)
    width, height = (max(1, min(size, round(side * scale))) for side in image.size)
    left, top = (size - width) // 2, (size - height) // 2
    return Letterbox(size, (left, top, left + width, top + height), image.size)


def prepare_image(image: Image.Image, size: int = IMAGE_SIZE, multiple: int = 14,
                  *, with_geometry: bool = False):
    """Preserva o campo de visão e permite reutilizar a transformação exata."""
    if multiple <= 0 or size <= 0 or size % multiple:
        raise ValueError(f"A resolução deve ser um múltiplo positivo de {multiple}.")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    geometry = letterbox_geometry(rgb, size)
    x0, y0, x1, y1 = geometry.box
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    padded.paste(rgb.resize((x1 - x0, y1 - y0), Image.Resampling.BICUBIC), (x0, y0))
    return (padded, geometry) if with_geometry else padded


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows / np.maximum(norms, 1e-8)


def inference_precision(device: str, precision: str) -> str:
    if precision not in ("FP16", "FP32"):
        raise ValueError("Selecione FP16 ou FP32.")
    return precision if device == "cuda" else "FP32"


def inference_autocast(torch, device: str, precision: str):
    if inference_precision(device, precision) == "FP16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def vit_features(tokens: np.ndarray, geometry: Letterbox, patch_size: int,
                 num_register_tokens: int) -> Features:
    grid_side = geometry.size // patch_size
    start = 1 + num_register_tokens
    expected = start + grid_side ** 2
    if tokens.ndim != 2 or tokens.shape[0] != expected:
        raise ValueError(f"Tokens DINO com formato inesperado {tokens.shape}. Esperados {expected}.")
    return Features(_normalize_rows(tokens[0]), _normalize_rows(tokens[start:]),
                    (grid_side, grid_side), geometry)


class DinoExtractor:
    model_id = MODEL_ID

    def __init__(self, device: str | None = None):
        import torch
        from transformers import Dinov2WithRegistersModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Dinov2WithRegistersModel.from_pretrained(MODEL_ID, use_safetensors=True)
        self.model = self.model.to(self.device).eval()
        self.patch_size = int(self.model.config.patch_size)
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 4))

    def extract(self, image: Image.Image, *, size: int = IMAGE_SIZE, precision: str = "FP32") -> Features:
        torch = self.torch
        img, geometry = prepare_image(image, size, self.patch_size, with_geometry=True)
        rgb = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        pixels = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        with torch.inference_mode(), inference_autocast(torch, self.device, precision):
            output = self.model(pixel_values=pixels)
        tokens = output.last_hidden_state[0].float().cpu().numpy()
        return vit_features(tokens, geometry, self.patch_size, self.num_register_tokens)


def inspect(reference_features: Iterable[Features], sample: Features, mode: Mode = "posicao",
            *, sample_foreground=None, reference_patch_masks=None) -> Analysis:
    refs = list(reference_features)
    if not refs:
        raise ValueError("Adicione pelo menos uma imagem de referência normal.")
    if mode not in ("posicao", "regioes"):
        raise ValueError("Modo de comparação desconhecido.")
    if any(ref.grid != sample.grid or ref.patches.shape != sample.patches.shape for ref in refs):
        raise ValueError("As grades de patches das imagens não são compatíveis.")

    sample_global = _normalize_rows(sample.global_vector)
    reference_globals = _normalize_rows(np.stack([ref.global_vector for ref in refs]))
    global_similarity = float(np.clip(np.max(reference_globals @ sample_global), -1.0, 1.0))
    valid = sample.valid_patch_mask(sample_foreground).ravel()
    ref_valid = np.stack([ref.valid_patch_mask().ravel() for ref in refs])
    if reference_patch_masks is not None:
        supplied = np.stack([np.asarray(mask, dtype=bool).ravel() for mask in reference_patch_masks])
        if supplied.shape != ref_valid.shape:
            raise ValueError("As máscaras de referência não correspondem à grade de patches.")
        ref_valid &= supplied
    q = _normalize_rows(sample.patches)
    r = _normalize_rows(np.stack([ref.patches for ref in refs]))
    if mode == "posicao":
        valid &= ref_valid.any(axis=0)
        similarities = np.einsum("pd,rpd->rp", q[valid], r[:, valid])
        nearest = np.where(ref_valid[:, valid], similarities, -np.inf).max(axis=0)
    else:
        bank = r[ref_valid]
        if not len(bank):
            raise ValueError("Nenhum patch válido nas imagens de referência. Revise os pontos SAM.")
        # Limita a memória temporária em grades de alta resolução com várias referências.
        queries = q[valid]
        nearest = np.empty(len(queries), dtype=np.float32)
        for offset in range(0, len(queries), 128):
            block = queries[offset:offset + 128]
            nearest[offset:offset + len(block)] = np.max(block @ bank.T, axis=1)
    if not valid.any():
        raise ValueError("Nenhum patch válido comparável. Confira enquadramento, modo e pontos do SAM.")

    distances = np.full(len(q), np.nan, dtype=np.float32)
    scored = np.clip(1.0 - nearest, 0.0, 2.0)
    distances[valid] = scored
    return Analysis(global_similarity, distances.reshape(sample.grid), float(scored.mean()),
                    float(np.quantile(scored, 0.95)), float(scored.max()), mode, len(refs),
                    sample.geometry, sample_foreground)


def heatmap_images(sample_image: Image.Image, distances: np.ndarray, threshold: float = 0.25,
                   opacity: float = 0.70, *, geometry: Letterbox | None = None,
                   foreground: np.ndarray | None = None) -> tuple[Image.Image, Image.Image]:
    """Reprojeta os patches sobre a imagem, com peso zero fora da área comparável."""
    if distances.ndim != 2 or 0 in distances.shape:
        raise ValueError("O mapa de distâncias deve ser bidimensional e não vazio.")
    sample = ImageOps.exif_transpose(sample_image).convert("RGB")
    geometry = geometry or letterbox_geometry(sample, max(distances.shape))
    if sample.size != geometry.original_size:
        raise ValueError("A geometria do mapa não corresponde à imagem.")
    sample.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    values = np.asarray(distances, dtype=np.float32)
    valid = np.isfinite(values)
    side = (geometry.size, geometry.size)

    def project(array, resample):
        # Amplia com a mesma transformação normalizada e desfaz o letterbox.
        return np.asarray(Image.fromarray(array.astype(np.float32)).resize(side, resample)
                          .resize(sample.size, resample, box=geometry.box))

    weights = project(valid, Image.Resampling.BILINEAR)
    expanded = project(np.where(valid, values, 0), Image.Resampling.BILINEAR)
    expanded = expanded / np.maximum(weights, 1e-6)
    support = project(valid, Image.Resampling.NEAREST) > 0.5
    if foreground is not None:
        mask = Image.fromarray(np.asarray(foreground, dtype=bool))
        support &= np.asarray(mask.resize(sample.size, Image.Resampling.NEAREST))
    if not support.any():
        return sample.copy(), Image.new("RGB", sample.size, (32, 32, 32))
    low, high = np.quantile(expanded[support], [0.05, 0.95])
    scaled = np.clip((expanded - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    relative_rgb = np.stack([np.full_like(scaled, 255), 220 - 178 * scaled, 45 - 7 * scaled], axis=2)
    relative_rgb[~support] = 32
    relative_map = Image.fromarray(relative_rgb.astype(np.uint8))

    span = max(0.06, float(np.quantile(values[valid], 0.98)) - threshold)
    strength = np.clip((expanded - threshold) / span, 0.0, 1.0)
    alpha = (255 * strength * support * np.clip(opacity, 0.0, 1.0)).astype(np.uint8)
    color = Image.new("RGB", sample.size, (255, 65, 30))
    return Image.composite(color, sample, Image.fromarray(alpha)), relative_map


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def csv_patch_scores(distances: np.ndarray) -> bytes:
    lines = ["linha,coluna,distancia_cosseno"]
    for (row, column), value in np.ndenumerate(distances):
        field = f"{float(value):.6f}" if np.isfinite(value) else ""
        lines.append(f"{row + 1},{column + 1},{field}")
    return ("\ufeff" + "\n".join(lines) + "\n").encode("utf-8")
