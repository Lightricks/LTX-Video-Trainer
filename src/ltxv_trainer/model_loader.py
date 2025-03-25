import json
import tempfile
from enum import Enum
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import torch
from diffusers import (
    AutoencoderKLLTXVideo,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    LTXVideoTransformer3DModel,
)
from pydantic import BaseModel, ConfigDict
from transformers import T5EncoderModel, T5Tokenizer

# The main HF repo to load scheduler, tokenizer, and text encoder from
HF_MAIN_REPO = "Lightricks/LTX-Video"


class LtxvModelVersion(str, Enum):
    """Available LTXV model versions."""

    LTXV_2B_090 = "LTXV_2B_0.9.0"
    LTXV_2B_091 = "LTXV_2B_0.9.1"
    LTXV_2B_095 = "LTXV_2B_0.9.5"
    LTXV_2B_096_DEV = "LTXV_2B_0.9.6_DEV"
    LTXV_2B_096_DISTILLED = "LTXV_2B_0.9.6_DISTILLED"
    LTXV_13B_097_DEV = "LTXV_13B_097_DEV"
    LTXV_13B_097_DISTILLED = "LTXV_13B_097_DISTILLED"

    def __str__(self) -> str:
        """Return the version string."""
        return self.value

    @classmethod
    def latest(cls) -> "LtxvModelVersion":
        """Get the latest available version."""
        return cls.LTXV_2B_095

    @property
    def hf_repo(self) -> str:
        """Get the HuggingFace repo for this version."""
        match self:
            case LtxvModelVersion.LTXV_2B_090:
                return "Lightricks/LTX-Video"
            case LtxvModelVersion.LTXV_2B_091:
                return "Lightricks/LTX-Video-0.9.1"
            case LtxvModelVersion.LTXV_2B_095:
                return "Lightricks/LTX-Video-0.9.5"
            case LtxvModelVersion.LTXV_2B_096_DEV:
                raise ValueError("LTXV_2B_096_DEV does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_2B_096_DISTILLED:
                raise ValueError("LTXV_2B_096_DISTILLED does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_13B_097_DEV:
                raise ValueError("LTXV_13B_097_DEV does not have a HuggingFace repo")
            case LtxvModelVersion.LTXV_13B_097_DISTILLED:
                raise ValueError("LTXV_13B_097_DISTILLED does not have a HuggingFace repo")
        raise ValueError(f"Unknown version: {self}")

    @property
    def safetensors_url(self) -> str:  # noqa: PLR0911
        """Get the safetensors URL for this version."""
        match self:
            case LtxvModelVersion.LTXV_2B_090:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors"
            case LtxvModelVersion.LTXV_2B_091:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors"
            case LtxvModelVersion.LTXV_2B_095:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.5.safetensors"
            case LtxvModelVersion.LTXV_2B_096_DEV:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-dev-04-25.safetensors"
            case LtxvModelVersion.LTXV_2B_096_DISTILLED:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-2b-0.9.6-distilled-04-25.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DEV:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev.safetensors"
            case LtxvModelVersion.LTXV_13B_097_DISTILLED:
                return "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-distilled.safetensors"
        raise ValueError(f"Unknown version: {self}")


# Type for model sources - can be:
# 1. HuggingFace repo ID (str)
# 2. Local path (str or Path)
# 3. Direct version specification (LtxvModelVersion)
ModelSource = Union[str, Path, LtxvModelVersion]


class LtxvModelComponents(BaseModel):
    """Container for all LTXV model components."""

    scheduler: FlowMatchEulerDiscreteScheduler
    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel
    vae: AutoencoderKLLTXVideo
    transformer: LTXVideoTransformer3DModel

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """
    Load the Flow Matching scheduler component from the main HF repo.

    Returns:
        Loaded scheduler
    """
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        HF_MAIN_REPO,
        subfolder="scheduler",
    )


def load_tokenizer() -> T5Tokenizer:
    """
    Load the T5 tokenizer component from the main HF repo.

    Returns:
        Loaded tokenizer
    """
    return T5Tokenizer.from_pretrained(
        HF_MAIN_REPO,
        subfolder="tokenizer",
    )


def load_text_encoder(*, load_in_8bit: bool = False) -> T5EncoderModel:
    """
    Load the T5 text encoder component from the main HF repo.

    Args:
        load_in_8bit: Whether to load in 8-bit precision

    Returns:
        Loaded text encoder
    """
    kwargs = (
        {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        if load_in_8bit
        else {"torch_dtype": torch.bfloat16}
    )
    return T5EncoderModel.from_pretrained(HF_MAIN_REPO, subfolder="text_encoder", **kwargs)


def load_vae(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderKLLTXVideo:
    """
    Load the VAE component.

    Args:
        source: Model source (HF repo, local path, or version)
        dtype: Data type for the VAE

    Returns:
        Loaded VAE
    """
    if isinstance(source, str):  # noqa: SIM102
        # Try to parse as version first
        if version := try_parse_version(source):
            source = version

    if isinstance(source, LtxvModelVersion):
        # NOTE: LTXV_2B_095's VAE must be loaded from the Diffusers folder-format instead of safetensors
        # This is a special case also for LTXV_2B_096_DEV and LTXV_13B_097_* which
        # don't have standalone HuggingFace repos, but share the same VAE as LTXV_2B_095.
        # Remove this once Diffusers properly supports loading from the safetensors file.
        if source in (
            LtxvModelVersion.LTXV_2B_095,
            LtxvModelVersion.LTXV_2B_096_DEV,
            LtxvModelVersion.LTXV_2B_096_DISTILLED,
            LtxvModelVersion.LTXV_13B_097_DEV,
            LtxvModelVersion.LTXV_13B_097_DISTILLED,
        ):
            return AutoencoderKLLTXVideo.from_pretrained(
                LtxvModelVersion.LTXV_2B_095.hf_repo,
                subfolder="vae",
                torch_dtype=dtype,
            )
        return AutoencoderKLLTXVideo.from_single_file(
            source.safetensors_url,
            torch_dtype=dtype,
        )
    elif isinstance(source, (str, Path)):
        if _is_huggingface_repo(source):
            return AutoencoderKLLTXVideo.from_pretrained(
                source,
                subfolder="vae",
                torch_dtype=dtype,
            )
        elif _is_safetensors_url(source):
            return AutoencoderKLLTXVideo.from_single_file(
                source,
                torch_dtype=dtype,
            )

    raise ValueError(f"Invalid model source: {source}")


def load_transformer(
    source: ModelSource,
    *,
    dtype: torch.dtype = torch.float32,
) -> LTXVideoTransformer3DModel:
    """
    Load the transformer component.

    Args:
        source: Model source (HF repo, local path, or version)
        dtype: Data type for the transformer

    Returns:
        Loaded transformer
    """
    if isinstance(source, str):  # noqa: SIM102
        # Try to parse as version first
        if version := try_parse_version(source):
            source = version

    if isinstance(source, LtxvModelVersion):
        # Special case for LTXV-13B which doesn't yet have a Diffusers config
        if source in (
            LtxvModelVersion.LTXV_13B_097_DEV,
            LtxvModelVersion.LTXV_13B_097_DISTILLED,
        ):
            return _load_ltxv_13b_transformer(source.safetensors_url, dtype=dtype)

        return LTXVideoTransformer3DModel.from_single_file(
            source.safetensors_url,
            torch_dtype=dtype,
        )
    elif isinstance(source, (str, Path)):
        if _is_huggingface_repo(source):
            return LTXVideoTransformer3DModel.from_pretrained(
                source,
                subfolder="transformer",
                torch_dtype=dtype,
            )
        elif _is_safetensors_url(source):
            return LTXVideoTransformer3DModel.from_single_file(
                source,
                torch_dtype=dtype,
            )

    raise ValueError(f"Invalid model source: {source}")


def load_ltxv_components(
    model_source: ModelSource | None = None,
    *,
    load_text_encoder_in_8bit: bool = False,
    transformer_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.bfloat16,
) -> LtxvModelComponents:
    """
    Load all components of the LTXV model from a specified source.
    Note: scheduler, tokenizer, and text encoder are always loaded from the main HF repo.

    Args:
        model_source: Source to load the VAE and transformer from. Can be:
            - HuggingFace repo ID (e.g. "Lightricks/LTX-Video")
            - Local path to model files (str or Path)
            - LtxvModelVersion enum value
            - None (will use the latest version)
        load_text_encoder_in_8bit: Whether to load text encoder in 8-bit precision
        transformer_dtype: Data type for transformer model
        vae_dtype: Data type for VAE model

    Returns:
        LtxvModelComponents containing all loaded model components
    """

    if model_source is None:
        model_source = LtxvModelVersion.latest()

    return LtxvModelComponents(
        scheduler=load_scheduler(),
        tokenizer=load_tokenizer(),
        text_encoder=load_text_encoder(load_in_8bit=load_text_encoder_in_8bit),
        vae=load_vae(model_source, dtype=vae_dtype),
        transformer=load_transformer(model_source, dtype=transformer_dtype),
    )


def try_parse_version(source: str | Path) -> LtxvModelVersion | None:
    """
    Try to parse a string as an LtxvModelVersion.

    Args:
        source: String to parse

    Returns:
        LtxvModelVersion if successful, None otherwise
    """
    try:
        return LtxvModelVersion(str(source))
    except ValueError:
        return None


def _is_huggingface_repo(source: str | Path) -> bool:
    """
    Check if a string is a valid HuggingFace repo ID.

    Args:
        source: String or Path to check

    Returns:
        True if the string looks like a HF repo ID
    """
    # Basic check: contains slash, no URL components
    return "/" in source and not urlparse(source).scheme


def _is_safetensors_url(source: str | Path) -> bool:
    """
    Check if a string is a valid safetensors URL.
    """
    return source.endswith(".safetensors")


def _load_ltxv_13b_transformer(safetensors_url: str, *, dtype: torch.dtype) -> LTXVideoTransformer3DModel:
    """A specific loader for LTXV-13B's transformer which doesn't yet have a Diffusers config"""
    transformer_13b_config = {
        "_class_name": "LTXVideoTransformer3DModel",
        "_diffusers_version": "0.33.0.dev0",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_out_bias": True,
        "caption_channels": 4096,
        "cross_attention_dim": 4096,
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "num_attention_heads": 32,
        "num_layers": 48,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "qk_norm": "rms_norm_across_heads",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
        json.dump(transformer_13b_config, f)
        f.flush()
        return LTXVideoTransformer3DModel.from_single_file(
            safetensors_url,
            config=f.name,
            torch_dtype=dtype,
        )
