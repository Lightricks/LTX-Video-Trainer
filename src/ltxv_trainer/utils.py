import io
import subprocess
import os
from typing import List, Union
from pathlib import Path

import torch
from loguru import logger
from PIL import ExifTags, Image, ImageCms, ImageOps
from PIL.Image import Image as PilImage
import numpy as np
from diffusers.utils import export_to_video
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card



from PIL import Image

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


def save_model_card(
    output_dir: str,
    repo_id: str,
    pretrained_model_name_or_path: str,
    videos: Union[List[str], Union[List[PilImage.Image], List[np.ndarray]]],
    validation_prompts: List[str],
    fps: int = 30,
) -> None:
    widget_dict = []
    if videos is not None and len(videos) > 0:
        for i, (video, validation_prompt) in enumerate(zip(videos, validation_prompts)):
            if not isinstance(video, str):
                export_to_video(video, os.path.join(output_dir, f"final_video_{i}.mp4"), fps=fps)
            widget_dict.append(
                {
                    "text": validation_prompt if validation_prompt else " ",
                    "output": {"url": video if isinstance(video, str) else f"final_video_{i}.mp4"},
                }
            )
    if pretrained_model_name_or_path not in ["Lightricks/LTX-Video", "Lightricks/LTX-Video-0.9.5"]:
        pretrained_model_name_or_path = "Lightricks/LTX-Video"

    model_description = f"""
# LoRA Finetune

<Gallery />

## Model description

This is a lora finetune of model: `{pretrained_model_name_or_path}`.

The model was trained using [`LTX-Video Community Trainer`](https://github.com/Lightricks/LTX-Video-Trainer).

## Download model

[Download LoRA]({repo_id}/tree/main) in the Files & Versions tab.

## Usage

### Using Trained LoRAs with `diffusers`:
Requires the [🧨 Diffusers library](https://github.com/huggingface/diffusers) installed.

Text-to-Video generation using the trained LoRA:
```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("{repo_id}", adapter_name="ltxv-lora")
pipe.set_adapters(["ltxv-lora"], [0.75])
pipe.to("cuda")

prompt = "{validation_prompts[0]}"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

For Image-to-Video:
```python
import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("{repo_id}", adapter_name="ltxv-lora")
pipe.set_adapters(["ltxv-lora"], [0.75])
pipe.to("cuda")

image = load_image(
    "url_to_your_image",
)
prompt = "{validation_prompts[0]}"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=704,
    height=480,
    num_inference_steps=50,
).frames[0]
export_to_video(video, "output.mp4", fps=24)
```

### 🔌 Using Trained LoRAs in ComfyUI

After training your LoRA, you can use it in ComfyUI by following these steps:

1. Copy your trained LoRA weights (`.safetensors` file) to the `models/loras` folder in your ComfyUI installation.

2. Install the ComfyUI-LTXVideoLoRA custom node:

   ```bash
   # In the root folder of your ComfyUI installation
   cd custom_nodes
   git clone https://github.com/dorpxam/ComfyUI-LTXVideoLoRA
   pip install -r ComfyUI-LTXVideoLoRA/requirements.txt
   ```

3. In your ComfyUI workflow:
   - Add the "LTXV LoRA Selector" node to choose your LoRA file
   - Connect it to the "LTXV LoRA Loader" node to apply the LoRA to your generation

You can find reference Text-to-Video (T2V) and Image-to-Video (I2V) workflows in the [official LTXV ComfyUI repository](https://github.com/Lightricks/ComfyUI-LTXVideo).

```py
TODO
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters) on loading LoRAs in diffusers.
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        base_model=pretrained_model_name_or_path,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-video",
        "image-to-video",
        "ltx-video"
        "diffusers",
        "lora",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(output_dir, "README.md"))