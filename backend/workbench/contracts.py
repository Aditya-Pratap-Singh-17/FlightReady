from typing import Literal
from pydantic import Field
from backend.context import Model, DatasetContext


class ContextRequest(Model):
    text: str = Field(default='', max_length=12000)
    tags: dict[str, list[str]] = Field(default_factory=dict)
    enable_remote: bool = False


class RegisterRequest(Model):
    path: str
    name: str = Field(default='', max_length=160)
    metadata: dict = Field(default_factory=dict)


class RenameRequest(Model):
    name: str = Field(min_length=1, max_length=160)


class SplitRequest(Model):
    purpose: Literal['train_val_test', 'training_only'] = 'train_val_test'
    seed: int = 42
    train_fraction: float = Field(default=.7, gt=0, lt=1)
    val_fraction: float = Field(default=.15, gt=0, lt=1)
    group_column: str | None = None
    acknowledge_missing_groups: bool = False
    near_distance: int = Field(default=4, ge=1, le=16)


class ReadinessRequest(Model):
    dataset_id: str
    context: DatasetContext
    split_id: str | None = None


class TrainRequest(Model):
    dataset_id: str
    split_id: str
    readiness_id: str
    aggregation_id: str | None = None
    validation_dataset_id: str | None = None
    validation_split_id: str | None = None
    validation_class_mapping: dict[str, str] = Field(default_factory=dict)
    acknowledge_validation_groups: bool = False
    model_type: Literal['candidate', 'reference', 'combined', 'controlled'] = 'candidate'
    reference_dataset_id: str | None = None
    reference_split_id: str | None = None
    class_mapping: dict[str, str] = Field(default_factory=dict)
    architecture: Literal['yolo11n', 'yolo11s'] = 'yolo11n'
    initialization: Literal['scratch', 'pretrained'] = 'scratch'
    weights_path: str | None = None
    epochs: int = Field(default=10, ge=1, le=10000)
    batch: int = Field(default=4, ge=1, le=256)
    image_size: int = Field(default=512, ge=64, le=2048, multiple_of=32)
    learning_rate: float = Field(default=.001, gt=0, le=1)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    device: Literal['cpu', '0', 'auto'] = 'auto'
    experiment_group: str = Field(default='', max_length=160)


class TestRequest(Model):
    model_ids: list[str] = Field(min_length=1, max_length=10)
    dataset_id: str
    split_id: str | None = None
    class_mapping: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=.25, gt=0, lt=1)
    iou: float = Field(default=.7, gt=0, lt=1)
    match_iou: float = Field(default=.5, gt=0, lt=1)
    device: Literal['cpu', '0', 'auto'] = 'auto'
    acknowledge_missing_groups: bool = False


class VariantRequest(Model):
    dataset_id: str
    split_id: str
    seed: int = 42
    intervention: Literal['control', 'blur', 'resolution', 'size', 'imbalance', 'label_omission', 'box_perturbation', 'environment', 'wrong_class', 'noise', 'exposure', 'small_object_reduction', 'negative_coverage']
    severity: float = Field(default=.25, ge=0, le=1)
    classes: list[str] = Field(default_factory=list)
    environment_column: str | None = None
    allowed_values: list[str] = Field(default_factory=list, max_length=100)
    maximum_native_area: float = Field(default=1024, gt=0, le=10**9)
    hard_negative_column: str | None = None


class RecordResponse(Model):
    id: str
    kind: str
    created_at: str
    payload: dict


class ScorePreviewRequest(Model):
    # Values remain uncoerced until aggregate() checks booleans, NaN and negatives.
    weights: dict[str, object] = Field(default_factory=dict, max_length=64)
