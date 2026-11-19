from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from .model_manager import ALL_MODELS
from vision_core import DINO_STRIDES

PIPELINES = {
    'grounding_sam_dino': 'Grounding DINO → SAM 2.1 → DINOv3',
    'grounding_ocr': 'Grounding DINO → RapidOCR',
    'depth_sam': 'Depth Anything V2 → SAM 2.1',
    'sam_dino': 'SAM 2.1 → filtro de fundo DINO',
}
Hash = Annotated[str, StringConstraints(pattern=r'^[a-f0-9]{64}$')]
Unit = Annotated[float, Field(ge=0, le=1)]
Box = tuple[Unit, Unit, Unit, Unit]
Point = tuple[Unit, Unit, Literal[0, 1]]
Precision = Literal['fp16', 'fp32']
Resolution = Literal[224, 384, 512]


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, hide_input_in_errors=True)


class Settings(Strict):
    models: list[str] = Field(default_factory=lambda: ['DINOv2 Small'], max_length=8)
    precision: Precision = 'fp16'
    resolution: Resolution = 224
    threshold: float = Field(.30, ge=.02, le=.90)
    opacity: Unit = .70
    mode: Literal['posicao', 'regioes'] = 'posicao'
    use_sam_mask_filter: bool = False
    px_per_mm: float = Field(0, ge=0, le=100000)
    points: list[Annotated[list[Point], Field(min_length=1, max_length=12)]] = Field(default_factory=list, max_length=12)
    boxes: list[Box] = Field(default_factory=list, max_length=12)
    grid_rows: int = Field(0, ge=0, le=4)
    grid_cols: int = Field(0, ge=0, le=6)
    grid_bounds: Box = (.10,.16,.90,.86)
    sam_cell_method: Literal['dino_por_celula', 'escala_de_cinza'] = 'escala_de_cinza'
    sam_cell_model: str = 'DINOv2 Small'
    sam_cell_threshold: float = Field(7, ge=0, le=100)
    grounding_prompt: str = Field('screw. label. cap.', max_length=320)
    grounding_threshold: Unit = .30
    grounding_text_threshold: Unit = .25
    depth_threshold: Unit = .50
    use_depth_filter: bool = False
    siglip_states: list[Annotated[str, StringConstraints(min_length=1, max_length=120)]] = Field(
        default_factory=lambda: ['open electrical panel','closed electrical panel','wet floor','empty conveyor'],
        min_length=1, max_length=16)
    siglip_threshold: Unit = .50
    ocr_confidence: Unit = .50
    ocr_regex: str = Field('', max_length=250)
    ocr_rois: bool = False

    @field_validator('models')
    @classmethod
    def model_names(cls, names):
        if any(name not in ALL_MODELS for name in names) or len(set(names)) != len(names):
            raise ValueError('Modelos inválidos ou repetidos.')
        return names

    @model_validator(mode='after')
    def geometry(self):
        if bool(self.grid_rows) != bool(self.grid_cols) or self.grid_rows*self.grid_cols > 12:
            raise ValueError('Grade inválida.')
        if self.sam_cell_model not in DINO_STRIDES:
            raise ValueError('Modelo de célula inválido.')
        for x0,y0,x1,y1 in [self.grid_bounds,*self.boxes]:
            if not (x0 < x1 and y0 < y1):
                raise ValueError('Caixa sem área.')
        return self


class AnalysisInput(Settings):
    sample_id: Hash
    ref_ids: list[Hash] = Field(default_factory=list,max_length=10)


class PipelineInput(AnalysisInput):
    pipeline_id: Literal['grounding_sam_dino','grounding_ocr','depth_sam','sam_dino']
