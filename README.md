<div align="center">

# LTX-Video Community Trainer

[Official GitHub Repo](https://github.com/Lightricks/LTX-Video) |
[Website](https://www.lightricks.com/ltxv) |
[Model](https://huggingface.co/Lightricks/LTX-Video) |
[Demo](https://app.ltx.studio/ltx-video) |
[Paper](https://arxiv.org/abs/2501.00103)

</div>

This repository provides tools and scripts for training and fine-tuning Lightricks' LTX Video model.
It allows training LoRAs on top of LTXV, as well as fine-tuning the entire model on custom datasets.
The repository also includes auxiliary utilities for preprocessing datasets, captioning videos, splitting scenes, etc.

---

## 📚 Table of Contents

- [Getting Started](#-getting-started)
- [Dataset Preparation](#-dataset-preparation)
  - [Split Scenes](#1-split-scenes-split_scenespy)
  - [Caption Videos](#2-caption-videos-caption_videospy)
  - [Preprocess Dataset](#3-dataset-preprocessing-preprocess_datasetpy)
- [Training Configuration](#-training-configuration)
  - [Example Configurations](#example-configurations)
- [Running the Trainer](#-running-the-trainer)
- [Using Trained LoRAs in ComfyUI](#-using-trained-loras-in-comfyui)
- [Example LoRAs](#-example-loras)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)

---

## 🚀 Getting Started

To get started, clone this repository and install the required dependencies:

First, install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.
Then clone the repository and install the dependencies:

```bash
git clone https://github.com/LightricksResearch/ltxv-community-trainer
cd ltxv-community-trainer
uv sync
source .venv/bin/activate
```

Follow the steps below to prepare your dataset and configure your training job.

---

### 🎬 Dataset Preparation

This section describes the workflow for preparing and preprocessing your dataset for training. The general flow is:

1. (Optional) Split long videos into scenes using `split_scenes.py`
2. (Optional) Generate captions for your videos using `caption_videos.py`
3. Preprocess your dataset using `preprocess_dataset.py` to compute and cache video latents and text embeddings
4. Run the trainer with your preprocessed dataset

#### 1. Split Scenes (`split_scenes.py`)

If you're starting with long-form videos (e.g., downloaded from YouTube), you should first split them into shorter, coherent scenes:

```bash
# Split a long video into scenes
python scripts/split_scenes.py input.mp4 scenes_output_dir/ \
    --filter-shorter-than 5s
```

This will create multiple video clips in `scenes_output_dir`.
These clips will be the input for the captioning step, if you choose to use it.

#### 2. Caption Videos (`caption_videos.py`)

If your dataset doesn't include captions, you can automatically generate them using vision-language models.
Use the directory containing your video clips (either from step 1, or your own clips):

```bash
# Generate captions for all videos in the scenes directory
python scripts/caption_videos.py scenes_output_dir/ \
    --output scenes_output_dir/captions.json \
    --captioner-type llava_next_7b
```

This will create a captions.json file which contains video paths and their captions
This JSON file will be used as input for the data preprocessing step.

#### 3. Dataset Preprocessing (`preprocess_dataset.py`)

This step preprocesses your video dataset by:

1. Resizing and cropping videos to fit specified resolution buckets
2. Computing and caching video latent representations
3. Computing and caching text embeddings for captions

Using the captions.json file generated in step 2:

```bash
# Preprocess the dataset using the generated captions.json
python scripts/preprocess_dataset.py scenes_output_dir/captions.json \
    --resolution-buckets "768x768x25" \
    --caption-column "caption" \
    --video-column "media_path"
```

The preprocessing significantly accelerates training and reduces GPU memory usage.

##### Resolution Buckets

Videos are organized into "buckets" of specific dimensions (width × height × frames). Each video is assigned to the nearest matching bucket.
Currently, the trainer only supports using a single resolution bucket.

The dimensions of each bucket must follow these constraints due to LTXV's VAE architecture:

- Spatial dimensions (width and height) must be multiples of 32
- Number of frames must be a multiple of 8 plus 1 (e.g., 25, 33, 41, 49, etc.)

Guidelines for choosing training resolution:

- For high-quality, detailed videos: use larger spatial dimensions (e.g. 768x448) with fewer frames (e.g. 89)
- For longer, motion-focused videos: use smaller spatial dimensions (512×512) with more frames (121)
- Memory usage increases with both spatial and temporal dimensions

Example usage:

```bash
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "768x768x25"
```

This creates a bucket with:

- 768×768 resolution
- 25 frames

Videos are processed as follows:

1. Videos are resized maintaining aspect ratio until either width or height matches the target (768 in this example)
2. The larger dimension is center cropped to match the bucket's dimensions
3. Frames are sampled uniformly to match the bucket's frame count (25 in this example)

> [!NOTE]
> The sequence length processed by the transformer model can be calculated as:
>
> ```
> sequence_length = (H/32) * (W/32) * ((F-1)/8 + 1)
> ```
>
> Where:
>
> - H = Height of video's latent
> - W = Width of video's latent
> - F = Number of latent frames
> - 32 = VAE's spatial downsampling factor
> - 8 = VAE's temporal downsampling factor
>
> For example, a 768×448×89 video would have sequence length:
>
> ```
> (768/32) * (448/32) * ((89-1)/8 + 1) = 24 * 14 * 12 = 4,032
> ```
>
> Keep this in mind when choosing video dimensions, as longer sequences require more memory and computation power.

> [!WARNING]
> While the preprocessing script supports multiple buckets, the trainer currently only works with a single resolution bucket.
> Please ensure you specify just one bucket in your preprocessing command.

##### Dataset Format

The script supports these input formats:

1. Directory with text files:

```
dataset/
├── captions.txt      # One caption per line
└── video_paths.txt   # One video path per line
```

```bash
python scripts/preprocess_dataset.py dataset/ \
    --caption-column captions \
    --video-column video_paths
```

2. Single metadata file:

```bash
# Using CSV/JSON/JSONL, e.g.
python scripts/preprocess_dataset.py dataset.json \
    --caption-column "caption" \
    --video-column "video_path" \
    --model-source "LTXV_2B_0.9.5"  # Optional: specify a specific version, defaults to latest
```

##### Output Structure

The preprocessed data is saved in a `.precomputed` directory:

```
dataset/
└── .precomputed/
    ├── latents/     # Cached video latents
    └── conditions/  # Cached text embeddings
```

##### LoRA Trigger Words

When training a LoRA, you can specify a trigger token that will be prepended to all captions:

```bash
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "1024x576x65" \
    --id-token "<TOK>"
```

This acts as a trigger word that activates the LoRA during inference when you include the same token in your prompts.

##### Decoding videos

By providing the `--decode-videos` flag, the script will also VAE-decode the precomputed latents and save the resulting videos under `.precomputed/decoded_videos` so you can look at and evaluate the data in the latents. This is useful for debugging and ensuring that your dataset is being processed correctly.

```bash
# Preprocess dataset and decode videos for verification
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "768x768x25" \
    --decode-videos
```

For single-frame images, they are saved as PNG files instead of MP4.

## ⚙️ Training Configuration

The trainer uses structured Pydantic models for configuration, making it easy to customize training parameters.
The main configuration class is [`LtxvTrainerConfig`](src/ltxv_trainer/config.py), which includes:

- **ModelConfig**: Base model and training mode settings
- **LoraConfig**: LoRA fine-tuning parameters
- **OptimizationConfig**: Learning rates, batch sizes, and scheduler settings
- **AccelerationConfig**: Mixed precision and optimization settings
- **DataConfig**: Data loading parameters
- **ValidationConfig**: Validation and inference settings
- **CheckpointsConfig**: Checkpoint saving frequency and retention settings
- **FlowMatchingConfig**: Timestep sampling parameters

### Example Configurations

Check out our example configurations in the `configs` directory. You can use these as templates
for your training runs:

- 📄 [LoRA Fine-tuning Example](configs/ltxv_2b_lora.yaml)
- 📄 [LoRA Fine-tuning Example (Low VRAM)](configs/ltxv_2b_lora_low_vram.yaml)
- 📄 [Full Model Fine-tuning Example](configs/ltxv_2b_full.yaml)

---

## ⚡ Running the Trainer

After preprocessing your dataset and preparing a configuration file, you can start training using the trainer script:

```bash
# Train a LoRA
python scripts/train.py configs/lora_example.yaml

# Fine-tune the full model
python scripts/train.py configs/full_example.yaml
```

The trainer loads your configuration, initializes models, applies optimizations, runs the training loop with progress tracking, generates validation videos (if configured), and saves the trained weights.

For LoRA training, the weights will be saved as `lora_weights.safetensors` in your output directory.
For full model fine-tuning, the weights will be saved as `model_weights.safetensors`.

---

## 🔌 Using Trained LoRAs in ComfyUI

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

---

## 🍰 Example LoRAs

Here are some example LoRAs trained using this trainer, along with their training datasets:

### Cakeify Effect

The [Cakeify LoRA](https://huggingface.co/Lightricks/LTX-Video-Cakeify-LoRA) transforms videos to make objects appear as if they're made of cake.
The effect was trained on the [Cakeify Dataset](https://huggingface.co/datasets/Lightricks/Cakeify-Dataset).

### Squish Effect

The [Squish LoRA](https://huggingface.co/Lightricks/LTX-Video-Squish-LoRA) creates a playful squishing effect on objects in videos.
It was trained on the [Squish Dataset](https://huggingface.co/datasets/Lightricks/Squish-Dataset), which contains just 5 example videos.

These examples demonstrate how you can train specialized video effects using this trainer.
Feel free to use these datasets as references for preparing your own training data.

---

## ️🔧 Utility Scripts

### LoRA Format Convertor

Using `scripts/convert_checkpoint.py` you can convert your LoRA saved file from `diffusers` library format to `ComfyUI` format.

```bash
# Convert from diffusers to ComfyUI format
python scripts/convert_checkpoint.py input.safetensors --to-comfy --output_path output.safetensors

# Convert from ComfyUI to diffusers format
python scripts/convert_checkpoint.py input.safetensors --output_path output.safetensors
```

If no output path is specified, the script will automatically generate one by adding `_comfy` or `_diffusers` suffix to the input filename.

### Latents Decoding Script

Using `scripts/decode_latents.py` you can decode precomputed video latents back into video files.
This is useful for verifying the quality of your preprocessed dataset or debugging the preprocessing pipeline.

```bash
# Basic usage
python scripts/decode_latents.py /path/to/latents/dir --output-dir /path/to/output
```

The script will:

1. Load the VAE model from the specified path
2. Process all `.pt` latent files in the input directory
3. Decode each latent back into a video using the VAE
4. Save the resulting videos as MP4 files in the output directory

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

- **Share Your Work**: If you've trained interesting LoRAs or achieved cool results, please share them with the community.
- **Report Issues**: Found a bug or have a suggestion? Open an issue on GitHub.
- **Submit PRs**: Help improve the codebase with bug fixes or general improvements.
- **Feature Requests**: Have ideas for new features? Let us know through GitHub issues.

---

## 🫶 Acknowledgements

Parts of this project are inspired by and incorporate ideas from several awesome open-source projects:

- [a-r-r-o-w/finetrainers](https://github.com/a-r-r-o-w/finetrainers)
- [bghira/SimpleTuner](https://github.com/bghira/SimpleTuner)

---

Happy training! 🎉
