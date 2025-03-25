from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ltxv_trainer.model_loader import LtxvModelVersion
from ltxv_trainer.quantization import QuantizationOptions


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model and training mode"""

    model_source: str | Path | LtxvModelVersion = Field(
        default=LtxvModelVersion.latest(),
        description="Model source - can be a HuggingFace repo ID, local path, or LtxvModelVersion",
    )

    training_mode: Literal["lora", "full"] = Field(
        default="lora",
        description="Training mode - either LoRA fine-tuning or full model fine-tuning",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    # noinspection PyNestedDecorators
    @field_validator("model_source", mode="before")
    @classmethod
    def validate_model_source(cls, v):  # noqa: ANN001, ANN206
        """Try to convert model source to LtxvModelVersion if possible."""
        if isinstance(v, (str, LtxvModelVersion)):
            try:
                return LtxvModelVersion(v)
            except ValueError:
                return v
        return v


class LoraConfig(ConfigBaseModel):
    """Configuration for LoRA fine-tuning"""

    rank: int = Field(
        default=64,
        description="Rank of LoRA adaptation",
        ge=2,
    )

    alpha: int = Field(
        default=64,
        description="Alpha scaling factor for LoRA",
        ge=1,
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        le=1.0,
    )

    target_modules: list[str] = Field(
        default=("to_k", "to_q", "to_v", "to_out.0"),
        description="List of modules to target with LoRA",
    )


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters"""

    learning_rate: float = Field(
        default=5e-4,
        description="Learning rate for optimization",
    )

    steps: int = Field(
        default=3000,
        description="Number of training steps",
    )

    batch_size: int = Field(
        default=2,
        description="Batch size for training",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of steps to accumulate gradients",
    )

    max_grad_norm: float = Field(
        default=1.0,
        description="Maximum gradient norm for clipping",
    )

    optimizer_type: Literal["adamw", "adamw8bit"] = Field(
        default="adamw",
        description="Type of optimizer to use for training",
    )

    scheduler_type: Literal[
        "constant",
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
    ] = Field(
        default="linear",
        description="Type of scheduler to use for training",
    )

    scheduler_params: dict = Field(
        default_factory=dict,
        description="Parameters for the scheduler",
    )

    enable_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory at the cost of slower training",
    )

    first_frame_conditioning_p: float = Field(
        default=0.1,
        description="Probability of conditioning on the first frame during training",
        ge=0.0,
        le=1.0,
    )


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization"""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(
        default="bf16",
        description="Mixed precision training mode",
    )

    quantization: QuantizationOptions | None = Field(
        default=None,
        description="Quantization precision to use",
    )

    load_text_encoder_in_8bit: bool = Field(
        default=False,
        description="Whether to load the text encoder in 8-bit precision to save memory",
    )

    compile_with_inductor: bool = Field(
        default=True,
        description="Compile the model with Torch Inductor",
    )

    compilation_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="reduce-overhead",
        description="Compilation mode for Torch Inductor",
    )


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing"""

    preprocessed_data_root: str = Field(
        description="Path to folder containing preprocessed training data",
    )

    num_dataloader_workers: int = Field(
        default=2,
        description="Number of background processes for data loading (0 means synchronous loading)",
        ge=0,
    )


class ValidationConfig(ConfigBaseModel):
    """Configuration for validation during training"""

    prompts: list[str] = Field(
        default_factory=list,
        description="List of prompts to use for validation",
    )

    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt to use for validation examples",
    )

    images: list[str] | None = Field(
        default=None,
        description="List of image paths to use for validation. "
        "One image path must be provided for each validation prompt",
    )

    video_dims: tuple[int, int, int] = Field(
        default=(704, 480, 161),
        description="Dimensions of validation videos (width, height, frames)",
    )

    seed: int = Field(
        default=42,
        description="Random seed used when sampling validation videos",
    )

    inference_steps: int = Field(
        default=50,
        description="Number of inference steps for validation",
        gt=0,
    )

    interval: int | None = Field(
        default=100,
        description="Number of steps between validation runs. If None, validation is disabled.",
        gt=0,
    )

    videos_per_prompt: int = Field(
        default=1,
        description="Number of videos to generate per validation prompt",
        gt=0,
    )

    guidance_scale: float = Field(
        default=3.5,
        description="Guidance scale to use during validation",
        ge=1.0,
    )


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training"""

    interval: int | None = Field(
        default=None,
        description="Number of steps between checkpoint saves. If None, intermediate checkpoints are disabled.",
        gt=0,
    )

    keep_last_n: int = Field(
        default=1,
        description="Number of most recent checkpoints to keep. Set to -1 to keep all checkpoints.",
        ge=-1,
    )


class HubConfig(ConfigBaseModel):
    """Configuration for Hugging Face Hub integration"""

    push_to_hub: bool = Field(default=False, description="Whether to push the model weights to the Hugging Face Hub")
    hub_model_id: str | None = Field(
        default=None, description="Hugging Face Hub repository ID (e.g., 'username/repo-name')"
    )

    @model_validator(mode="after")
    def validate_hub_config(self) -> "HubConfig":
        """Validate that hub_model_id is not None when push_to_hub is True."""
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("hub_model_id must be specified when push_to_hub is True")
        return self


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training"""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(
        default="shifted_logit_normal",
        description="Mode to use for timestep sampling",
    )

    timestep_sampling_params: dict = Field(
        default_factory=dict,
        description="Parameters for timestep sampling",
    )


class LtxvTrainerConfig(ConfigBaseModel):
    """Unified configuration for LTXV training"""

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)

    # General configuration
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    output_dir: str = Field(
        default="outputs",
        description="Directory to save model outputs",
    )

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        """Expand user home directory in output path."""
        return str(Path(v).expanduser().resolve())
