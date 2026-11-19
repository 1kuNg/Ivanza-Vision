"""Backends e visualizacoes do Ivanza Vision Lab.

Nunca substitua os checkpoints oficiais por dados simulados. Todos os erros de
acesso/instalacao sao propagados ao usuario com contexto.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dinov2_core import (
    DinoExtractor, Features, IMAGE_SIZE, MODEL_ID as DINOV2_ID,
    _normalize_rows, prepare_image, inspect, vit_features, inference_autocast, inference_precision,
)

DINOV3_ID = 'facebook/dinov3-vits16-pretrain-lvd1689m'
DINOV3_CONVNEXT_ID = 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'
SAM_NAME = 'SAM 2.1 Tiny'
SAM2_ID = 'facebook/sam2.1-hiera-tiny'
DINO_STRIDES = {
    'DINOv2 Small': 14,
    'DINOv3 Small': 16,
    'DINOv3 ConvNeXt Tiny': 32,
}
MODELS = {
    'DINOv2 Small': DINOV2_ID,
    'DINOv3 Small': DINOV3_ID,
    'DINOv3 ConvNeXt Tiny': DINOV3_CONVNEXT_ID,
    SAM_NAME: SAM2_ID,
}
DINO_SIZES = {
    14: (224, 336, 518),
    16: (224, 384, 512),
    32: (224, 384, 512),
}


def dino_image_size(model: str, level: int) -> int:
    return DINO_SIZES[DINO_STRIDES[model]][level]


class DinoV3Extractor:
    """DINOv3 ViT ou ConvNeXt com a geometria definida pelo laboratório."""
    def __init__(self, device: str | None = None, token: str | None = None,
                 model_id: str = DINOV3_ID):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if model_id not in (DINOV3_ID, DINOV3_CONVNEXT_ID):
            raise ValueError('Checkpoint DINOv3 desconhecido.')
        self.torch = torch
        self.model_id = model_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_convnext = model_id == DINOV3_CONVNEXT_ID
        auth = {'token': token} if token else {}
        self.processor = AutoImageProcessor.from_pretrained(model_id, **auth)
        self.model = AutoModel.from_pretrained(model_id, use_safetensors=True, **auth).to(self.device)
        self.model.eval()
        self.patch_size = 32 if self.is_convnext else int(self.model.config.patch_size)
        self.num_register_tokens = int(getattr(self.model.config, 'num_register_tokens', 4))

    def extract(self, image: Image.Image, *, size: int = IMAGE_SIZE, precision: str = 'FP32') -> Features:
        img, geometry = prepare_image(image, size, self.patch_size, with_geometry=True)
        inputs = self.processor(images=img, return_tensors='pt', do_resize=False, do_center_crop=False)
        pixels = inputs['pixel_values'].to(self.device)
        if tuple(pixels.shape[-2:]) != (size, size):
            raise ValueError('O processador DINOv3 alterou a geometria da imagem.')
        torch = self.torch
        with torch.inference_mode(), inference_autocast(torch, self.device, precision):
            outputs = self.model(pixel_values=pixels)
        spatial = outputs.last_hidden_state[0].float().cpu().numpy()
        if self.is_convnext:
            side = size // self.patch_size
            if spatial.ndim == 2 and spatial.shape[0] == 1 + side ** 2:
                # Transformers 4.57 concatena o vetor global aos patches da última etapa.
                patches = spatial[1:]
            elif spatial.ndim == 3 and spatial.shape[-2:] == (side, side):
                patches = spatial.reshape(spatial.shape[0], side ** 2).T
            else:
                raise ValueError(f'Mapa ConvNeXt com formato inesperado {spatial.shape}.')
            global_vector = outputs.pooler_output[0].float().cpu().numpy()
            return Features(_normalize_rows(global_vector), _normalize_rows(patches),
                            (side, side), geometry)
        return vit_features(spatial, geometry, self.patch_size, self.num_register_tokens)


def clean_sam_objects(objects, max_objects: int = 12, max_per_object: int = 12):
    """Cada item tem pontos (x,y,classe), +1 interior e 0 exterior.

    Compatibilidade: lista antiga de centros [(x,y), ...] vira um item por ponto.
    """
    if not objects:
        return []
    if isinstance(objects[0], (tuple, list)) and len(objects[0]) == 2 and all(
        isinstance(v, (int, float, np.number)) for v in objects[0]
    ):
        objects = [[(x, y, 1)] for x, y in objects]
    if len(objects) > max_objects:
        raise ValueError(f'Marcar no máximo {max_objects} peças por análise.')
    normalized = []
    for idx, points in enumerate(objects):
        if not points:
            # Item em edição sem pontos ainda não entra na inferência.
            continue
        if len(points) > max_per_object:
            raise ValueError(f'O item {idx+1} tem mais de {max_per_object} pontos.')
        clean = []
        for point in points:
            if len(point) != 3:
                raise ValueError('Cada ponto precisa de X, Y e tipo (+ ou -).')
            x, y, label = point
            if not np.isfinite(x) or not np.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
                raise ValueError('Coordenadas devem estar entre 0 e 1.')
            if int(label) not in (0, 1):
                raise ValueError('Tipo de ponto inválido: use 1 (dentro) ou 0 (fora).')
            clean.append((float(x), float(y), int(label)))
        if not any(label == 1 for _, _, label in clean):
            raise ValueError(f'O item {idx+1} precisa de um ponto dentro da peça (+).')
        normalized.append(clean)
    return normalized


def sam2_prompt_arrays(objects, width: int, height: int):
    """Formato oficial: [imagem][objeto][ponto][xy]; rótulos 1/0, padding pelo processor."""
    normalized = clean_sam_objects(objects)
    if not normalized:
        return [[]], [[]]
    input_points = [[[
        [float(x * (width-1)), float(y * (height-1))]
        for x, y, _ in obj
    ] for obj in normalized]]
    input_labels = [[[label for _, _, label in obj] for obj in normalized]]
    return input_points, input_labels



def sam_grid_regions(rows: int, columns: int,
                     bounds: tuple[float, float, float, float] = (.10, .16, .90, .86)) -> list[tuple[float, float, float, float]]:
    """Regiões fixas por posição, em ordem de leitura (linha e coluna).

    As coordenadas são normalizadas, permitindo repetir o teste em resoluções distintas.
    """
    if not 1 <= rows <= 4 or not 1 <= columns <= 6 or rows * columns > 12:
        raise ValueError('A grade deve conter de 1 a 12 posições (até 4 linhas e 6 colunas).')
    x0, y0, x1, y1 = (float(v) for v in bounds)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('Os limites da grade devem formar um retângulo válido.')
    dx, dy = (x1-x0)/columns, (y1-y0)/rows
    return [(x0 + c*dx, y0 + r*dy, x0 + (c+1)*dx, y0 + (r+1)*dy)
            for r in range(rows) for c in range(columns)]


def sam_grid_objects(regions: Sequence[tuple[float, float, float, float]]):
    """Cada célula constitui um objeto SAM diferente — nunca oito pontos do mesmo objeto."""
    return [[((x0+x1)/2, (y0+y1)/2, 1)] for x0, y0, x1, y1 in regions]


def sam_grid_preview(image: Image.Image, regions, statuses=None) -> Image.Image:
    """Grade e numeração legíveis sobre a foto. Não representa máscaras SAM."""
    output = image.convert('RGB').copy()
    w, h = output.size
    pen = ImageDraw.Draw(output)
    width = max(2, round(min(w,h)/210))
    for i, (x0,y0,x1,y1) in enumerate(regions):
        box = (round(x0*(w-1)), round(y0*(h-1)), round(x1*(w-1)), round(y1*(h-1)))
        flagged = statuses is not None and i < len(statuses) and statuses[i]
        color = (240, 109, 39) if flagged else (58, 207, 182)
        pen.rectangle(box, outline=color, width=width)
        pen.text((box[0]+5, box[1]+5), f'#{i+1}', fill='white',
                 stroke_fill=(25, 33, 40), stroke_width=2)
        cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
        pen.ellipse((cx-4,cy-4,cx+4,cy+4), fill=color, outline='white', width=2)
    return output


def sam_cell_box(image: Image.Image, region):
    """Janela interna da célula, com margem para excluir linhas da grade."""
    w, h = image.size
    x0, y0, x1, y1 = region
    margin_x = (x1-x0)*.12
    margin_y = (y1-y0)*.12
    left = max(0, int((x0+margin_x)*w))
    right = min(w, max(left+1, int((x1-margin_x)*w)))
    top = max(0, int((y0+margin_y)*h))
    bottom = min(h, max(top+1, int((y1-margin_y)*h)))
    return left, top, right, bottom


def sam_position_comparison(image: Image.Image, reference: Image.Image, regions,
                            threshold_pct: float = 7.0, *, dino_vectors=None):
    """Diferença *visual* por posição, não classificador treinado de presença.

    Exige geometria aproximadamente igual entre a referência e a foto de teste.
    Mantém a comparação independente da máscara SAM (que pode incluir uma cavidade vazia).
    """
    if not 0 <= threshold_pct <= 100:
        raise ValueError('Limiar deve ficar entre 0 e 100%.')
    im = image.convert('RGB')
    ref = reference.convert('RGB')
    if abs(im.width/im.height - ref.width/ref.height) > 0.025:
        raise ValueError('A imagem normal deve ter proporções semelhantes à foto de teste; alinhe a câmera.')
    if dino_vectors is not None and len(dino_vectors) != len(regions):
        raise ValueError('Os vetores DINO devem incluir uma comparação por célula.')
    if dino_vectors is None:
        if im.size != ref.size:
            ref = ref.resize(im.size, Image.Resampling.BILINEAR)
        a = np.asarray(ImageOps.grayscale(im), dtype=np.float32)
        b = np.asarray(ImageOps.grayscale(ref), dtype=np.float32)
    rows = []
    for i, region in enumerate(regions):
        if dino_vectors is None:
            left, top, right, bottom = sam_cell_box(im, region)
            diff_pct = float(np.mean(np.abs(a[top:bottom,left:right] - b[top:bottom,left:right])) / 255 * 100)
        else:
            query, normal = dino_vectors[i]
            similarity = np.dot(_normalize_rows(query), _normalize_rows(normal))
            diff_pct = 50 * (1 - float(np.clip(similarity, -1, 1)))
        changed = diff_pct >= threshold_pct
        rows.append({'Posição': f'#{i+1}', 'Diferença visual (%)': round(diff_pct, 2),
                     'Comparação com a referência': ('Alteração — conferir presença/peça' if changed
                                                    else 'Sem alteração relevante'),
                     'suspeita': changed})
    return rows


def sam_clip_mask(mask: np.ndarray, region: tuple[float,float,float,float]):
    """Restringe os pixels de uma peça à sua própria região da bandeja."""
    h, w = mask.shape
    x0,y0,x1,y1 = region
    l = max(0, min(w-1, round(x0*w)))
    r = max(l+1, min(w, round(x1*w)))
    t = max(0, min(h-1, round(y0*h)))
    b = max(t+1, min(h, round(y1*h)))
    result = np.zeros_like(mask, dtype=bool)
    result[t:b, l:r] = np.asarray(mask[t:b, l:r], dtype=bool)
    return result


@dataclass
class MaskResult:
    masks: list[np.ndarray]
    labels: list[str]
    areas: list[float]
    boxes: list[tuple[int, int, int, int]]
    image: Image.Image
    elapsed: float
    # Todas as candidatas são guardadas para trocar sem inferência adicional.
    candidates: list[list[np.ndarray]] | None = None
    candidate_scores: list[list[float]] | None = None
    chosen_indices: list[int] | None = None
    prompts: list[list[tuple[float, float, int]]] | None = None
    regions: list[tuple[float, float, float, float]] | None = None


def select_mask_candidates(image: Image.Image, seg: MaskResult,
                           choices: Sequence[int]) -> MaskResult:
    """Troca a máscara de cada objeto sem executar o modelo novamente."""
    if seg.candidates is None or len(choices) != len(seg.candidates):
        raise ValueError('As candidatas ou seleções são incompatíveis.')
    masks = []
    for choices_i, candidates_i in zip(choices, seg.candidates):
        if not 0 <= choices_i < len(candidates_i):
            raise ValueError('Número de candidata inválido.')
        masks.append(candidates_i[choices_i])
    overlay, areas, boxes = draw_masks(image, masks, seg.prompts)
    if seg.regions:
        overlay = sam_grid_preview(overlay, seg.regions)
    return MaskResult(masks, seg.labels, areas, boxes, overlay, seg.elapsed,
                      seg.candidates, seg.candidate_scores, list(choices), seg.prompts, seg.regions)


def sam_mask_diagnostics(mask: np.ndarray, points: Sequence[tuple[float, float, int]] | None = None,
                         score: float | None = None, px_per_mm: float = 0) -> dict:
    """Alertas geométricos indicativos; não significam aprovação/reprovação."""
    points = points if points is not None else ()
    h, w = mask.shape
    area_pct = float(np.asarray(mask, dtype=bool).mean()*100)
    warnings = []
    inside = outside = 0
    for x, y, label in points:
        value = bool(mask[round(y*(h-1)), round(x*(w-1))])
        if label == 1:
            inside += int(value)
        else:
            outside += int(not value)
    ninside = sum(1 for *_, lbl in points if lbl == 1)
    noutside = len(points)-ninside
    if area_pct < 0.06:
        warnings.append('Máscara vazia ou muito pequena')
    if area_pct > 65:
        warnings.append('Máscara cobre grande parte da imagem')
    if inside < ninside:
        warnings.append('Ponto positivo fora da máscara')
    if outside < noutside:
        warnings.append('Ponto negativo dentro da máscara')
    result = {'area_pct': area_pct, 'positive_ok': f'{inside}/{ninside}',
            'negative_ok': f'{outside}/{noutside}',
            'score': score, 'warnings': warnings,
            'status': 'Revisar' if warnings else 'Verificar visualmente'}

    if px_per_mm > 0 and np.isfinite(px_per_mm):
        occupied_rows = np.flatnonzero(np.any(mask, axis=1))
        occupied_cols = np.flatnonzero(np.any(mask, axis=0))
        result.update(area_mm2=float(np.count_nonzero(mask) / px_per_mm ** 2),
                      width_mm=float((np.ptp(occupied_cols) + 1) / px_per_mm) if occupied_cols.size else 0.0,
                      height_mm=float((np.ptp(occupied_rows) + 1) / px_per_mm) if occupied_rows.size else 0.0)
    return result


def sam_detail_images(image: Image.Image, mask: np.ndarray, padding: int = 18,
                      region: tuple[float,float,float,float] | None = None):
    """Original com contexto, máscara monocromática e objeto isolado, mesma janela."""
    base = image.convert('RGB')
    arr = np.asarray(base)
    h, w = mask.shape
    ys, xs = np.where(mask)
    if region is not None:
        x0,y0,x1,y1 = region
        window = (max(0,round(x0*w)), max(0,round(y0*h)),
                  min(w,round(x1*w)), min(h,round(y1*h)))
        if window[2] <= window[0] or window[3] <= window[1]:
            raise ValueError('Região de recorte inválida.')
        original = base.crop(window)
        binary = Image.fromarray(mask.astype(np.uint8)*255, mode='L').crop(window).convert('RGB')
        isolated = np.full_like(arr, 238)
        isolated[mask] = arr[mask]
        return original, binary, Image.fromarray(isolated, 'RGB').crop(window)
    if len(xs) == 0:
        # Nunca devolver recorte enganoso quando o modelo não encontrou o objeto.
        empty = Image.new('RGB', (160, 160), (70, 70, 70))
        return empty, Image.new('RGB', (160, 160), 'black'), empty.copy()
    pad = max(padding, int(max(xs.max()-xs.min(), ys.max()-ys.min())*0.12))
    x0, y0 = max(0, int(xs.min())-pad), max(0, int(ys.min())-pad)
    x1, y1 = min(w, int(xs.max())+pad+1), min(h, int(ys.max())+pad+1)
    window = (x0, y0, x1, y1)
    original = base.crop(window)
    binary = Image.fromarray((mask.astype(np.uint8)*255), mode='L').crop(window).convert('RGB')
    isolated = np.full_like(arr, 238)
    isolated[mask] = arr[mask]
    cropped = Image.fromarray(isolated, 'RGB').crop(window)
    return original, binary, cropped


class Sam2Segmenter:
    """SAM2 Transformers: várias peças e múltiplos pontos por peça numa inferência."""
    def __init__(self, device: str | None = None):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Sam2Processor.from_pretrained(SAM2_ID)
        self.model = Sam2Model.from_pretrained(SAM2_ID, use_safetensors=True).to(self.device)
        self.model.eval()

    def segment(self, image: Image.Image, objects=None, multiple_candidates: bool = True,
                regions: list[tuple[float,float,float,float]] | None = None, *,
                precision: str = 'FP32',
                boxes: Sequence[tuple[float, float, float, float]] | None = None) -> MaskResult:
        prompts = clean_sam_objects(objects)
        original = image.convert('RGB')
        width, height = original.size
        processor_inputs = {}
        if prompts:
            processor_inputs['input_points'], processor_inputs['input_labels'] = sam2_prompt_arrays(
                prompts, width, height)
        if boxes:
            if len(boxes) > 12 or (prompts and len(prompts) != len(boxes)):
                raise ValueError('Use até 12 caixas e uma caixa por grupo de pontos.')
            pixel_boxes = []
            for x0, y0, x1, y1 in boxes:
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ValueError('Caixa SAM inválida. Use coordenadas normalizadas.')
                pixel_boxes.append([x0 * width, y0 * height, x1 * width, y1 * height])
            processor_inputs['input_boxes'] = [pixel_boxes]
            if not prompts:
                prompts = [[] for _ in boxes]
        if not prompts:
            raise ValueError('Marque um ponto positivo ou desenhe uma caixa.')
        if regions is not None and len(regions) != len(prompts):
            raise ValueError('Número de regiões diferente do número de peças.')
        start = perf_counter()
        inputs = self.processor(images=original, **processor_inputs, return_tensors='pt').to(self.device)
        original_sizes = inputs['original_sizes']
        with self.torch.inference_mode(), inference_autocast(self.torch, self.device, precision):
            outputs = self.model(**inputs, multimask_output=multiple_candidates)
        masks_tensor = self.processor.post_process_masks(
            outputs.pred_masks.float().cpu(), original_sizes.cpu() if hasattr(original_sizes, 'cpu') else original_sizes
        )[0]
        arrays = masks_tensor.detach().cpu().numpy()
        if arrays.ndim == 2 and len(prompts) == 1:
            arrays = arrays[None, None, ...]
        elif arrays.ndim == 3:
            arrays = arrays[:, None, ...]
        if arrays.ndim != 4 or arrays.shape[0] != len(prompts) or arrays.shape[-2:] != (height, width):
            raise ValueError(f'SAM 2 devolveu dimensões inesperadas {arrays.shape}.')
        scores = outputs.iou_scores.detach().cpu().float().numpy()
        if scores.ndim == 3:
            scores = scores[0]
        if scores.ndim != 2 or scores.shape != arrays.shape[:2]:
            raise ValueError(f'Pontuações de máscaras inesperadas {scores.shape}.')
        candidates = [[np.asarray(channel > 0, dtype=bool) for channel in item] for item in arrays]
        if regions is not None:
            candidates = [[sam_clip_mask(mask, regions[i]) for mask in candidate]
                          for i, candidate in enumerate(candidates)]
        choices = []
        for item_scores, item_masks in zip(scores, candidates):
            nonempty = [i for i, candidate in enumerate(item_masks) if candidate.any()]
            pool = nonempty if nonempty else list(range(len(item_masks)))
            choices.append(max(pool, key=lambda i: float(item_scores[i])))
        masks = [candidates[i][selected] for i, selected in enumerate(choices)]
        overlay, areas, boxes = draw_masks(original, masks, prompts)
        if regions is not None:
            overlay = sam_grid_preview(overlay, regions)
        return MaskResult(masks, [f'Item {i+1}' for i in range(len(masks))],
                          areas, boxes, overlay, perf_counter()-start,
                          candidates, scores.tolist(), choices, prompts, regions)


COLORS = [(234, 107, 33), (33, 146, 202), (96, 180, 117),
          (199, 87, 161), (219, 175, 57), (110, 108, 205)]


def draw_masks(image: Image.Image, masks: Sequence[np.ndarray], points=None):
    base = np.asarray(image.convert('RGB')).copy()
    h, w = base.shape[:2]
    areas: list[float] = []
    boxes: list[tuple[int, int, int, int]] = []
    for idx, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (h, w):
            raise ValueError('A máscara deve ter a mesma largura e altura da imagem.')
        ys, xs = np.where(mask)
        areas.append(float(len(xs) / (w * h)))
        if len(xs) == 0:
            boxes.append((0, 0, 0, 0))
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        boxes.append(bbox)
        color = np.asarray(COLORS[idx % len(COLORS)], dtype=np.float32)
        base[mask] = np.uint8(np.clip(base[mask].astype(np.float32) * 0.55 + color * 0.45, 0, 255))
    overlay = Image.fromarray(base)
    d = ImageDraw.Draw(overlay)
    for idx, box in enumerate(boxes):
        if not areas[idx]:
            continue
        color = COLORS[idx % len(COLORS)]
        d.rectangle(box, outline=color, width=max(2, w // 250))
        label_pos = (box[0]+3, max(0, box[1]-17))
        d.text(label_pos, f'#{idx+1}', fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    if points:
        # Compatibilidade com centros antigos e com vários pontos +/- por peça.
        if points and len(points[0]) == 2 and isinstance(points[0][0], (float, int)):
            points = [[(x,y,1)] for x,y in points]
        for i, item in enumerate(points):
            for x, y, label in item:
                px, py = int(x * (w-1)), int(y * (h-1))
                if label == 1:
                    d.ellipse((px-5, py-5, px+5, py+5), fill=COLORS[i % len(COLORS)], outline='white', width=2)
                else:
                    d.line((px-6, py-6, px+6, py+6), fill=(255,255,255), width=3)
                    d.line((px-6, py+6, px+6, py-6), fill=(255,255,255), width=3)
    return overlay, areas, boxes


def object_crops(image: Image.Image, result: MaskResult) -> list[Image.Image]:
    crops = []
    for mask in result.masks:
        if np.any(mask):
            _, _, crop = sam_detail_images(image, mask)
            crops.append(crop)
    return crops


def display_sam_prompts(image: Image.Image, objects, active: int = 0) -> Image.Image:
    output = image.convert('RGB').copy()
    d = ImageDraw.Draw(output)
    w, h = output.size
    for i, prompts in enumerate(objects):
        col = COLORS[i % len(COLORS)]
        for x, y, label in prompts:
            xx, yy = round(x * (w-1)), round(y * (h-1))
            radius = 9 if i == active else 6
            if label == 1:
                d.ellipse((xx-radius, yy-radius, xx+radius, yy+radius), fill=col,
                          outline='white' if i == active else 'black', width=2)
                d.text((xx+11, yy-10), str(i+1), fill=col, stroke_fill='white', stroke_width=2)
            else:
                d.line((xx-7, yy-7, xx+7, yy+7), fill='white', width=4)
                d.line((xx-7, yy+7, xx+7, yy-7), fill='white', width=4)
                d.line((xx-7, yy-7, xx+7, yy+7), fill=col, width=2)
                d.line((xx-7, yy+7, xx+7, yy-7), fill=col, width=2)
    return output


def display_points(image: Image.Image, points: Sequence[tuple[float, float]]) -> Image.Image:
    return display_sam_prompts(image, [[(x,y,1)] for x,y in points])


def clean_points(points: Sequence[tuple[float, float]], max_points: int = 12):
    if len(points) > max_points:
        raise ValueError(f'Marcar no máximo {max_points} peças por análise.')
    result = []
    for x, y in points:
        if not (np.isfinite(x) and np.isfinite(y)) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError('Pontos devem estar entre 0 e 1 em ambos os eixos.')
        result.append((float(x), float(y)))
    return result


def feature_cache_key(extractor, image_size: int, precision: str, ref_hash: str):
    return (extractor.model_id, image_size, inference_precision(extractor.device, precision), ref_hash)


def cached_reference_feature(extractor, reference: Image.Image, ref_hash: str,
                             feature_cache: dict, image_size: int, precision: str):
    key = feature_cache_key(extractor, image_size, precision, ref_hash)
    if key in feature_cache:
        return feature_cache[key], True
    feature = extractor.extract(reference, size=image_size, precision=precision)
    feature_cache[key] = feature
    while len(feature_cache) > 64:
        feature_cache.pop(next(iter(feature_cache)))
    return feature, False


def benchmark_dino(extractor, refs: Sequence[Image.Image], sample: Image.Image, mode: str, feature_cache: dict,
                   *, ref_hashes: Sequence[str] | None = None, image_size: int = IMAGE_SIZE,
                   precision: str = 'FP32', sample_mask=None, reference_masks=None,
                   reference_mask_provider=None):
    """Mede preparação das referências, extração da amostra e busca separadamente."""
    if ref_hashes is None:
        ref_hashes = [sha256(ref.tobytes() + str((ref.mode, ref.size)).encode()).hexdigest() for ref in refs]
    if len(ref_hashes) != len(refs):
        raise ValueError('Informe um hash por referência.')
    start = perf_counter()
    features = []
    cache_hits = 0
    for ref, ref_hash in zip(refs, ref_hashes):
        feature, hit = cached_reference_feature(extractor, ref, ref_hash, feature_cache, image_size, precision)
        features.append(feature)
        cache_hits += int(hit)
    projected = None
    if reference_mask_provider is not None:
        projected = [reference_mask_provider(ref, ref_hash, feature) for ref, ref_hash, feature in
                     zip(refs, ref_hashes, features)]
    elif reference_masks is not None:
        if len(reference_masks) != len(refs):
            raise ValueError('Informe uma máscara SAM por referência.')
        projected = [feature.valid_patch_mask(mask) for feature, mask in zip(features, reference_masks)]
    ref_sec = perf_counter() - start
    start = perf_counter()
    test_feature = extractor.extract(sample, size=image_size, precision=precision)
    infer_sec = perf_counter() - start
    start = perf_counter()
    result = inspect(features, test_feature, mode=mode, sample_foreground=sample_mask,
                     reference_patch_masks=projected)
    compare_sec = perf_counter() - start
    return result, {'referencias_s': ref_sec, 'imagem_s': infer_sec,
                    'comparacao_s': compare_sec, 'cache_hits': cache_hits}
