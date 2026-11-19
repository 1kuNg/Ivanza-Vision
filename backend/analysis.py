"""Orquestra os núcleos originais e serializa resultados para a bancada."""
from dataclasses import asdict
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from dinov2_core import heatmap_images
from extra_models_core import (depth_mask, depth_preview, depth_cell_means, draw_detections, draw_ocr,
    filter_ocr, peak_depth_point, scene_alerts, scene_states)
from vision_core import (DINO_STRIDES, SAM_NAME, benchmark_dino, cached_reference_feature, dino_image_size,
    sam_grid_regions, sam_grid_objects, sam_mask_diagnostics, sam_detail_images, select_mask_candidates,
    sam_cell_box, sam_position_comparison)

GROUND = 'Grounding DINO Tiny'
DEPTH = 'Depth Anything V2 Small'
SIGLIP = 'SigLIP 2 Base'
OCR = 'RapidOCR'


def size_for(model, setting):
    return dino_image_size(model, (224, 384, 512).index(setting))


class BoxProcessor:
    def __init__(self, processor, boxes):
        self.processor = processor
        self.boxes = boxes

    def __call__(self, *, images, **kwargs):
        w, h = images.size
        return self.processor(images=images,
            input_boxes=[[[x0*(w-1), y0*(h-1), x1*(w-1), y1*(h-1)] for x0,y0,x1,y1 in self.boxes]], **kwargs)

    def post_process_masks(self, *args, **kwargs):
        return self.processor.post_process_masks(*args, **kwargs)


def segment(model, image, points, boxes, regions, precision):
    if not boxes:
        return model.segment(image, points, regions=regions, precision=precision)
    original = model.processor
    model.processor = BoxProcessor(original, boxes)
    try:
        return model.segment(image, points, regions=regions, precision=precision)
    finally:
        model.processor = original


def centers(boxes):
    return [[((x0+x1)/2, (y0+y1)/2, 1)] for x0,y0,x1,y1 in boxes]


def sam_output(store, sample, result, px, index):
    detail = []
    for i, mask in enumerate(result.masks):
        score = result.candidate_scores[i][result.chosen_indices[i]]
        diag = sam_mask_diagnostics(mask, result.prompts[i], score, px)
        detail.append(dict(item=i+1, area_px=int(mask.sum()), **diag))
    parts = sam_detail_images(sample, result.masks[index],
                              region=result.regions[index] if result.regions else None)
    return dict(overlay=store.artifact(result.image), metrics=dict(mask_count=len(detail), iou_scores=[
                    row['score'] for row in detail], area_mm2=[row['area_mm2'] for row in detail if 'area_mm2' in row]),
                rows=detail, choices=result.chosen_indices, candidates=result.candidate_scores,
                details=dict(zip(('original','mask','isolated'), map(store.artifact, parts))))


def heatmap_output(store, sample, result, threshold, opacity):
    overlay, relative = heatmap_images(sample, result.patch_distances, threshold, opacity,
                                       geometry=result.geometry, foreground=result.foreground)
    return dict(overlay=store.artifact(overlay), details=dict(relative=store.artifact(relative)))


def analyze(manager, store, settings, session, redact, pipeline=None):
    sample = store.image(settings.sample_id)
    refs = [store.image(key) for key in settings.ref_ids]
    p = settings.precision.upper()
    s = settings
    models = list(s.models)
    if pipeline == 'grounding_sam_dino':
        dino = next((m for m in models if m.startswith('DINOv3')), 'DINOv3 Small')
        models = [GROUND, SAM_NAME, dino]
    elif pipeline == 'grounding_ocr':
        models = [GROUND, OCR]
    elif pipeline == 'depth_sam':
        models = [DEPTH, SAM_NAME]
    elif pipeline == 'sam_dino':
        models = [SAM_NAME, *([m for m in models if m in DINO_STRIDES] or ['DINOv2 Small'])]
    hybrid = s.use_sam_mask_filter or pipeline in ('grounding_sam_dino','sam_dino')
    if hybrid and any(m in DINO_STRIDES for m in models) and SAM_NAME not in models:
        models.insert(0, SAM_NAME)
    if s.ocr_rois and OCR in models and GROUND not in models:
        models.insert(0, GROUND)
    order = sorted(models, key=lambda m: 0 if m in (GROUND, DEPTH) else 1 if m == SAM_NAME else 2)
    regions = sam_grid_regions(s.grid_rows, s.grid_cols, s.grid_bounds) if s.grid_rows else None
    boxes = s.boxes
    points = sam_grid_objects(regions) if regions else centers(boxes) if boxes else s.points
    run_id = uuid4().hex
    data, times = {}, {}
    detections = foreground = depth = None
    reference_masks = None
    with manager.lock:
        for name in order:
            stage = perf_counter()
            loaded = stage
            try:
                if name in DINO_STRIDES and not refs:
                    raise ValueError('Adicione uma referência normal para o DINO.')
                model = manager.get(name, session.get('hf'))
                loaded = perf_counter()
                details = {}
                resolution = None
                if name == GROUND:
                    detections = model.detect(sample, s.grounding_prompt, s.grounding_threshold,
                                              s.grounding_text_threshold, precision=p)
                    infer = perf_counter() - loaded
                    details = dict(overlay=store.artifact(draw_detections(sample, detections)),
                                   rows=[asdict(d) for d in detections], metrics=dict(box_count=len(detections)))
                elif name == DEPTH:
                    depth = model.estimate(sample, precision=p)
                    infer = perf_counter() - loaded
                    details = depth_output(store, depth, s.depth_threshold, regions)
                    store.views.put((run_id,name), (depth,regions), depth.nbytes)
                elif name == SIGLIP:
                    states = scene_states('\n'.join(s.siglip_states))
                    probs = model.classify(sample, states, precision=p)
                    infer = perf_counter() - loaded
                    details = dict(rows=scene_alerts(states, probs, s.siglip_threshold),
                                   metrics=dict(probabilities=probs, alert_count=sum(x>=s.siglip_threshold for x in probs)))
                elif name == OCR:
                    use_rois = s.ocr_rois or pipeline == 'grounding_ocr'
                    if use_rois and not detections:
                        raise ValueError('O Grounding DINO não encontrou ROIs para leitura.')
                    readings = filter_ocr(model.read(sample, detections if use_rois else None),
                                          s.ocr_confidence, s.ocr_regex)
                    infer = perf_counter() - loaded
                    details = dict(overlay=store.artifact(draw_ocr(sample, readings)),
                                   rows=[asdict(r) for r in readings],
                                   metrics=dict(read_count=len(readings),
                                                regex_ok=[r.regex_ok for r in readings if r.regex_ok is not None]))
                elif name == SAM_NAME:
                    if pipeline == 'grounding_sam_dino':
                        if not detections:
                            raise ValueError('O Grounding DINO não encontrou caixas para segmentar.')
                        w,h = sample.size
                        boxes = [(x0/w,y0/h,x1/w,y1/h) for x0,y0,x1,y1 in (d.box for d in detections[:12])]
                        points, regions = centers(boxes), None
                    if pipeline == 'depth_sam':
                        if depth is None or not depth.any():
                            raise ValueError('Não há pico de profundidade para segmentar.')
                        points, boxes, regions = [[(*peak_depth_point(depth),1)]], [], None
                    result = segment(model, sample, points, boxes if not regions else [], regions, p)
                    infer = result.elapsed
                    foreground = np.logical_or.reduce(result.masks)
                    details = sam_output(store, sample, result, s.px_per_mm, 0)
                    if regions and refs:
                        vectors = None
                        if s.sam_cell_method == 'dino_por_celula':
                            engine = manager.get(s.sam_cell_model, session.get('hf'))
                            side = size_for(s.sam_cell_model,s.resolution)
                            vectors = []
                            for roi in regions:
                                ref = refs[0].crop(sam_cell_box(refs[0], roi))
                                query = sample.crop(sam_cell_box(sample, roi))
                                value,_ = cached_reference_feature(engine, ref, f'{s.ref_ids[0]}:{roi}', manager.features,side,p)
                                vectors.append((engine.extract(query,size=side,precision=p).global_vector,value.global_vector))
                        details['positions'] = sam_position_comparison(sample,refs[0],regions,s.sam_cell_threshold,
                                                                     dino_vectors=vectors)
                    footprint = sum(mask.nbytes for group in result.candidates for mask in group)
                    if footprint <= store.views.max_bytes:
                        store.views.put((run_id,name), (s.sample_id,result,s.px_per_mm), footprint)
                else:
                    mask = foreground if hybrid else None
                    if hybrid and mask is None:
                        raise ValueError('O filtro depende de uma máscara SAM válida.')
                    if reference_masks is None and (hybrid or s.use_depth_filter):
                        reference_masks = []
                        for ref in refs:
                            items = []
                            if hybrid:
                                sam = manager.get(SAM_NAME,session.get('hf'))
                                items.append(np.logical_or.reduce(segment(sam,ref,points,boxes if not regions else [],regions,p).masks))
                            if s.use_depth_filter:
                                items.append(depth_mask(manager.get(DEPTH).estimate(ref,precision=p),s.depth_threshold))
                            reference_masks.append(np.logical_and.reduce(items))
                    if s.use_depth_filter:
                        if depth is None:
                            depth = manager.get(DEPTH).estimate(sample,precision=p)
                        cut = depth_mask(depth,s.depth_threshold)
                        mask = cut if mask is None else cut & mask
                    resolution = size_for(name,s.resolution)
                    result,timing = benchmark_dino(model,refs,sample,s.mode,manager.features,ref_hashes=s.ref_ids,
                        image_size=resolution,precision=p,sample_mask=mask,reference_masks=reference_masks)
                    infer = timing['imagem_s']
                    details = heatmap_output(store,sample,result,s.threshold,s.opacity)
                    details['metrics'] = {f:getattr(result,f) for f in ('global_similarity','mean_distance','p95_distance','max_distance')}
                    details['timing_ms'] = {key:round(value*1e3,2) for key,value in timing.items() if key.endswith('_s')}
                    store.views.put((run_id,name),(s.sample_id,result),result.patch_distances.nbytes)
                times[name] = dict(model=name, inference_ms=round(infer*1e3,2), load_ms=round((loaded-stage)*1e3,2),
                                   device=model.device.upper(), resolution=resolution,status='ok')
                data[name] = details
            except Exception as exc:
                times[name] = dict(model=name,inference_ms=None,device=None,status='error')
                data[name] = dict(error=redact(exc,session))
    output = dict(run_id=run_id, sample_id=s.sample_id, benchmark=[times[m] for m in models],
                  results={m:data[m] for m in models}, pipeline=pipeline)
    store.rounds.put(run_id,(output,s))
    return output


def depth_output(store, depth, threshold, regions):
    mask = depth_mask(depth,threshold)
    return dict(overlay=store.artifact(depth_preview(depth)),
        details=dict(mask=store.artifact(Image.fromarray(mask.astype('uint8')*255))),
        rows=[dict(cell=i+1, mean_depth=v, below_threshold=v<threshold)
              for i,v in enumerate(depth_cell_means(depth,regions or []))],
        metrics=dict(foreground_ratio=float(mask.mean()), peak_depth=float(depth.max())))
