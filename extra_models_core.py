"""Motores opcionais e pós-processamento independente de Streamlit."""
from __future__ import annotations

from dataclasses import dataclass, replace
from inspect import signature
from threading import Lock
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from dinov2_core import inference_autocast

GROUNDING_ID = 'IDEA-Research/grounding-dino-tiny'
DEPTH_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
SIGLIP_ID = 'google/siglip2-base-patch16-224'
OCR_ID = 'rapidocr-onnxruntime'


def normalize_grounding_prompt(text: str) -> str:
    labels = dict.fromkeys(' '.join(part.lower().split()) for part in re.split(r'[,.]', text))
    return ' '.join(f'{label}.' for label in labels if label)


def scene_states(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(' '.join(line.lower().split()) for line in text.splitlines() if line.strip()))


def pixel_box(box, size):
    w, h = size
    x0, y0, x1, y1 = box
    return (max(0, min(w, int(np.floor(x0)))), max(0, min(h, int(np.floor(y0)))),
            max(0, min(w, int(np.ceil(x1)))), max(0, min(h, int(np.ceil(y1)))))


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]


def sam_center_prompts(detections, size):
    w, h = size
    return [[((x0 + x1 - 1) / (2 * max(w - 1, 1)), (y0 + y1 - 1) / (2 * max(h - 1, 1)), 1)]
            for x0, y0, x1, y1 in (item.box for item in detections)]


class GroundingDetector:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(GROUNDING_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_ID, use_safetensors=True).to(self.device).eval()

    def detect(self, image, text, threshold=0.30, text_threshold=0.25, *, precision='FP16'):
        prompt = normalize_grounding_prompt(text)
        if not prompt:
            raise ValueError('Informe pelo menos uma classe para localizar.')
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(self.device)
        torch = self.torch
        with torch.inference_mode(), inference_autocast(torch, self.device, precision):
            outputs = self.model(**inputs)
        post_process = self.processor.post_process_grounded_object_detection
        parameters = signature(post_process).parameters
        box_argument = 'threshold' if 'threshold' in parameters else 'box_threshold'
        args = {box_argument: threshold, 'text_threshold': text_threshold,
                'target_sizes': [(image.height, image.width)]}
        if 'input_ids' in parameters:
            args['input_ids'] = inputs.input_ids
        with torch.inference_mode():
            outputs.logits = outputs.logits.float()
            outputs.pred_boxes = outputs.pred_boxes.float()
            result = post_process(outputs, **args)[0]
        labels = result['text_labels'] if 'text_labels' in result else result['labels']
        detections = []
        for label, score, box in zip(labels, result['scores'].tolist(), result['boxes'].tolist()):
            x0, y0, x1, y1 = pixel_box(box, image.size)
            if x1 > x0 and y1 > y0:
                detections.append(Detection(str(label).strip(' .') or '(sem rótulo)',
                                            float(score), (x0, y0, x1, y1)))
        if not detections:
            return []
        from torchvision.ops import nms

        boxes = torch.tensor([detection.box for detection in detections], dtype=torch.float32)
        scores = torch.tensor([detection.score for detection in detections], dtype=torch.float32)
        return [detections[index] for index in nms(boxes, scores, iou_threshold=0.50).tolist()]


def normalize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all():
        raise ValueError('O mapa de profundidade precisa ser bidimensional e conter valores finitos.')
    low, high = float(depth.min()), float(depth.max())
    return (depth - low) / (high - low) if high > low else np.zeros_like(depth)


def depth_mask(depth, threshold: float):
    if not 0 <= threshold <= 1:
        raise ValueError('O corte de profundidade deve ficar entre 0 e 1.')
    return np.asarray(depth) >= threshold


def depth_cell_means(depth, regions):
    h, w = depth.shape
    means = []
    for left, top, right, bottom in regions:
        x0, y0, x1, y1 = pixel_box((left*w, top*h, right*w, bottom*h), (w, h))
        x0, y0 = min(x0, w-1), min(y0, h-1)
        cell = depth[y0:max(y0+1, y1), x0:max(x0+1, x1)]
        finite = cell[np.isfinite(cell)]
        means.append(float(finite.mean()) if finite.size else 0.0)
    return means


def peak_depth_point(depth) -> tuple[float, float]:
    import cv2

    h, w = depth.shape
    finite = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
    # A mediana remove picos isolados. Mapas menores que a janela mantêm o pico original.
    filtered = cv2.medianBlur(finite, 3) if min(h, w) >= 3 else finite
    if filtered.max() == filtered.min():
        return ((.5 if w > 1 else 0.), (.5 if h > 1 else 0.))
    y, x = np.unravel_index(filtered.argmax(), filtered.shape)
    return float(x / max(w - 1, 1)), float(y / max(h - 1, 1))


def depth_preview(depth) -> Image.Image:
    return ImageOps.colorize(Image.fromarray(np.uint8(np.clip(depth, 0, 1) * 255)),
                             '#102348', '#fee966', mid='#17a7a4')


class DepthEstimator:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(DEPTH_ID, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            DEPTH_ID, use_safetensors=True).to(self.device).eval()

    def estimate(self, image, *, precision='FP16'):
        torch = self.torch
        inputs = self.processor(images=image, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            with inference_autocast(torch, self.device, precision):
                output = self.model(**inputs).predicted_depth
            output = torch.nn.functional.interpolate(
                output[:, None].float(), size=(image.height, image.width),
                mode='bicubic', align_corners=False)[0, 0]
        return normalize_depth(output.cpu().numpy())


def sigmoid_probabilities(logits):
    import torch
    return torch.sigmoid(logits.float())


def scene_alerts(states, probabilities, threshold):
    return [{'Estado': state, 'Probabilidade': float(prob), 'Alerta': bool(prob >= threshold)}
            for state, prob in zip(states, probabilities)]


class SiglipClassifier:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoProcessor.from_pretrained(SIGLIP_ID, use_fast=True)
        self.model = AutoModel.from_pretrained(SIGLIP_ID, use_safetensors=True).to(self.device).eval()

    def classify(self, image, states, *, precision='FP16'):
        if not states:
            raise ValueError('Informe pelo menos um estado, um por linha.')
        torch = self.torch
        inputs = self.processor(images=image, text=list(states), padding='max_length',
                                max_length=64, truncation=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            with inference_autocast(torch, self.device, precision):
                outputs = self.model(**inputs)
            return sigmoid_probabilities(outputs.logits_per_image)[0].cpu().tolist()


@dataclass(frozen=True)
class OcrReading:
    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]
    roi: str | None = None
    regex_ok: bool | None = None


def compile_ocr_regex(pattern):
    try:
        return re.compile(pattern) if pattern.strip() else None
    except (re.error, OverflowError, RecursionError) as error:
        raise ValueError(f'Regex inválido. {error}') from error


def filter_ocr(readings, min_confidence=0.50, pattern=''):
    regex = compile_ocr_regex(pattern)
    return [replace(item, regex_ok=bool(regex.search(item.text.strip())) if regex else None)
            for item in readings if item.confidence >= min_confidence]


class RapidOcrEngine:
    device = 'cpu'

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()
        # O pacote muda configurações internas a cada chamada.
        self.lock = Lock()

    def read(self, image, detections=None):
        areas = [(None, (0, 0, *image.size))] if detections is None else [
            (f'#{i} {d.label}', d.box) for i, d in enumerate(detections, 1)]
        readings = []
        with self.lock:
            for name, box in areas:
                x0, y0, _, _ = box
                rgb = np.asarray(image.crop(box).convert('RGB'))
                bgr = np.ascontiguousarray(rgb[..., ::-1])
                results, _ = self.engine(bgr, text_score=0.0)
                for polygon, text, confidence in results or []:
                    absolute = tuple((float(x + x0), float(y + y0)) for x, y in polygon)
                    readings.append(OcrReading(text, float(confidence), absolute, name))
        return readings


def draw_detections(image, detections):
    output = image.convert('RGB').copy()
    draw = ImageDraw.Draw(output)
    width = max(2, image.width // 300)
    for item in detections:
        x0, y0, x1, y1 = item.box
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline='#ff9f35', width=width)
        draw.text((x0 + 2, y0 + 2), f'{item.label}  {item.score:.2f}', fill='white',
                  stroke_width=2, stroke_fill='#121820')
    return output


def draw_ocr(image, readings):
    output = image.convert('RGB').copy()
    draw = ImageDraw.Draw(output)
    for item in readings:
        color = '#ea692f' if item.regex_ok is False else '#39e0bc'
        draw.line([*item.polygon, item.polygon[0]], fill=color, width=max(2, image.width // 300))
        draw.text(item.polygon[0], item.text, fill='white', stroke_width=2, stroke_fill='#121820')
    return output


EXTRA_ENGINES = {
    'grounding_dino': (GROUNDING_ID, GroundingDetector),
    'depth_anything': (DEPTH_ID, DepthEstimator),
    'siglip2': (SIGLIP_ID, SiglipClassifier),
    'rapidocr': (OCR_ID, RapidOcrEngine),
}
