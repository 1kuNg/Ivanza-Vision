"""Classificação de SKU e receita de bancada, independentes da interface."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dinov2_core import _normalize_rows
from extra_models_core import EXTRA_ENGINES
from vision_core import DINO_STRIDES, MODELS, SAM_NAME, dino_image_size

from version import VERSION


@dataclass
class SkuBank:
    labels: tuple[str, ...]
    vectors: np.ndarray
    classes: np.ndarray

    @classmethod
    def from_examples(cls, examples) -> SkuBank:
        labels = tuple(name.strip() for name, _ in examples)
        if not 2 <= len(labels) <= 6:
            raise ValueError('Cadastre de duas a seis classes.')
        if any(not name or name.casefold() == 'desconhecido' for name in labels):
            raise ValueError('Informe o nome de cada classe. “Desconhecido” é reservado para rejeição.')
        if len({name.casefold() for name in labels}) != len(labels):
            raise ValueError('Use nomes diferentes para cada classe.')
        counts = [len(vectors) for _, vectors in examples]
        if not all(1 <= count <= 10 for count in counts):
            raise ValueError('Adicione de uma a dez fotos em cada classe.')
        vectors = np.stack([vector for _, samples in examples for vector in samples])
        if vectors.ndim != 2 or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError('O banco de SKUs contém um vetor inválido.')
        return cls(labels, _normalize_rows(vectors), np.repeat(np.arange(len(labels)), counts))

    def classify(self, vector: np.ndarray, threshold: float) -> SkuPrediction:
        vector = np.asarray(vector, dtype=np.float32)
        if (vector.shape != self.vectors.shape[1:] or not np.isfinite(vector).all()
                or not np.linalg.norm(vector) or not -1 <= threshold <= 1):
            raise ValueError('Vetor de teste ou limiar incompatível com o banco de SKUs.')
        similarities = np.clip(self.vectors @ _normalize_rows(vector), -1, 1)
        scores = np.full(len(self.labels), -1.0, dtype=np.float32)
        np.maximum.at(scores, self.classes, similarities)
        order = np.argsort(-scores, kind='stable')
        best, second = order[:2]
        similarity = float(scores[best])
        label = self.labels[best] if similarity >= threshold else 'Desconhecido'
        return SkuPrediction(label, similarity, float(scores[best] - scores[second]),
                             {self.labels[i]: float(scores[i]) for i in order})


@dataclass
class SkuPrediction:
    label: str
    similarity: float
    margin: float
    by_class: dict[str, float]


def inspection_recipe(model, size_level, precision, mode, threshold, regions, points, px_per_mm,
                      *, hybrid=False, cell_method=None, cell_model=None, cell_threshold=None, extra_settings=None):
    recipe = {
        'schema_version': 1,
        'versao_lab': VERSION,
        'modelo': model,
        'model_id': MODELS[model],
        'resolucao': dino_image_size(model, size_level) if model in DINO_STRIDES else 1024,
        'precisao': precision,
        'precisao_cpu': 'FP32',
        'modo_comparacao': mode,
        'limiar_calibrado': {
            'metrica': 'distancia_cosseno_patch', 'valor': threshold,
        } if model != SAM_NAME else (
            {'metrica': cell_method, 'valor': cell_threshold} if cell_method else None
        ),
        'coordenadas': 'normalizadas_na_imagem_original',
        'rois': [{'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
                 for x0, y0, x1, y1 in regions or []],
        'prompts_sam': points or [],
        'px_por_mm': px_per_mm if px_per_mm > 0 else None,
        'segmentacao_sam': {
            'model_id': MODELS[SAM_NAME], 'filtrar_patches_dino': hybrid,
        } if hybrid or model == SAM_NAME else None,
        'comparacao_grade': {
            'metodo': cell_method,
            'limiar_pct': cell_threshold,
            'model_id': MODELS[cell_model] if cell_model else None,
            'resolucao': dino_image_size(cell_model, size_level) if cell_model else None,
        } if regions and cell_method else None,
    }

    return recipe_with_extras(extra_settings, recipe)


def recipe_with_extras(settings, base=None):
    recipe = dict(base) if base is not None else {'versao_lab': VERSION}
    if settings:
        recipe['schema_version'] = 2
        recipe['modelos_extras'] = {
            key: {**values, 'model_id': EXTRA_ENGINES[key][0]} for key, values in settings.items()
        }
    return recipe


@dataclass(frozen=True)
class InspectionSettings:
    size_level: int
    precision: str
    mode: str
    hybrid: bool
    cell_method: str
    cell_model: str
    cell_threshold: float
    px_per_mm: float = field(compare=False)
