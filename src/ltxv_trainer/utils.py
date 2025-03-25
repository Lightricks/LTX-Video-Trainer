import io
import subprocess
from pathlib import Path

import torch
from PIL import ExifTags, Image, ImageCms, ImageOps
from PIL.Image import Image as PilImage

from ltxv_trainer import logger


def get_gpu_memory_gb(device: torch.device) -> float:
    """Get current GPU memory usage in GB using nvidia-smi"""
    try:
        device_id = device.index if device.index is not None else 0
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
                "-i",
                str(device_id),
            ],
            encoding="utf-8",
        )
        return float(result.strip()) / 1024  # Convert MB to GB
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to get GPU memory from nvidia-smi: {e}")
        # Fallback to torch
        return torch.cuda.memory_allocated(device) / 1024**3


def open_image_as_srgb(image_path: str | Path | io.BytesIO) -> PilImage:
    """
    Opens an image file, applies rotation (if it's set in metadata) and converts it
    to the sRGB color space respecting the original image color space .
    """
    with Image.open(image_path) as img_raw:
        img = ImageOps.exif_transpose(img_raw)

    input_icc_profile = img.info.get("icc_profile")

    # Try to convert to sRGB if the image has ICC profile metadata
    srgb_profile = ImageCms.createProfile(colorSpace="sRGB")
    if input_icc_profile is not None:
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc_profile))
        srgb_img = ImageCms.profileToProfile(img, input_profile, srgb_profile, outputMode="RGB")
    else:
        # Try fall back to checking EXIF
        exif_data = img.getexif()
        if exif_data is not None:
            color_space_value = exif_data.get(ExifTags.Base.ColorSpace.value)
            EXIF_COLORSPACE_SRGB = 1  # noqa: N806
            if color_space_value is None:
                logger.info(
                    f"Opening image file '{image_path}' that has no ICC profile and EXIF has no"
                    " ColorSpace tag, assuming sRGB",
                )
            elif color_space_value != EXIF_COLORSPACE_SRGB:
                raise ValueError(
                    "Image has colorspace tag in EXIF but it isn't set to sRGB,"
                    " conversion is not supported."
                    f" EXIF ColorSpace tag value is {color_space_value}",
                )

        srgb_img = img.convert("RGB")

        # Set sRGB profile in metadata since now the image is assumed to be in sRGB.
        srgb_profile_data = ImageCms.ImageCmsProfile(srgb_profile).tobytes()
        srgb_img.info["icc_profile"] = srgb_profile_data

    return srgb_img
