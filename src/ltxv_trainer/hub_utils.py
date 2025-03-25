import tempfile
from pathlib import Path
from typing import List, Union

import imageio
from huggingface_hub import HfApi, create_repo
from loguru import logger

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.model_loader import try_parse_version
from scripts.convert_checkpoint import convert_checkpoint


def convert_video_to_gif(video_path: Path, output_path: Path) -> None:
    """Convert a video file to GIF format."""
    try:
        # Read the video file
        reader = imageio.get_reader(str(video_path))
        fps = reader.get_meta_data()["fps"]

        # Write GIF file with infinite loop
        writer = imageio.get_writer(
            str(output_path),
            fps=min(fps, 15),  # Cap FPS at 15 for reasonable file size
            loop=0,  # 0 means infinite loop
        )

        for frame in reader:
            writer.append_data(frame)

        writer.close()
        reader.close()
    except Exception as e:
        logger.error(f"Failed to convert video to GIF: {e}")
        return None


def create_model_card(
    output_dir: Union[str, Path],
    videos: List[Path],
    config: LtxvTrainerConfig,
) -> Path:
    """Generate and save a model card for the trained model."""

    repo_id = config.hub.hub_model_id
    pretrained_model_name_or_path = config.model.model_source
    validation_prompts = config.validation.prompts
    output_dir = Path(output_dir)
    template_path = Path(__file__).parent.parent.parent / "templates" / "model_card.md"

    if not template_path.exists():
        logger.warning("⚠️ Model card template not found, using default template")
        return

    # Read the template
    template = template_path.read_text()

    # Get model name from repo_id
    model_name = repo_id.split("/")[-1]

    # Get base model information
    version = try_parse_version(pretrained_model_name_or_path)
    if version:
        base_model_link = version.safetensors_url
        base_model_name = str(version)
    else:
        base_model_link = f"https://huggingface.co/{pretrained_model_name_or_path}"
        base_model_name = pretrained_model_name_or_path

    # Format validation prompts and create grid layout
    prompts_text = ""
    sample_grid = []

    if validation_prompts and videos:
        prompts_text = "Example prompts used during validation:\n\n"

        # Create samples directory
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True, parents=True)

        # Process videos and create cells
        cells = []
        for i, (prompt, video) in enumerate(zip(validation_prompts, videos, strict=False)):
            if video.exists():
                # Add prompt to text section
                prompts_text += f"- `{prompt}`\n"

                # Convert video to GIF
                gif_path = samples_dir / f"sample_{i}.gif"
                try:
                    convert_video_to_gif(video, gif_path)

                    # Create grid cell with collapsible description
                    cell = (
                        f"![example{i + 1}](./samples/sample_{i}.gif)"
                        "<br>"
                        '<details style="max-width: 300px; margin: auto;">'
                        f"<summary>Prompt</summary>"
                        f"{prompt}"
                        "</details>"
                    )
                    cells.append(cell)
                except Exception as e:
                    logger.error(f"Failed to process video {video}: {e}")

        # Calculate optimal grid dimensions
        num_cells = len(cells)
        if num_cells > 0:
            # Aim for a roughly square grid, with max 4 columns
            num_cols = min(4, num_cells)
            num_rows = (num_cells + num_cols - 1) // num_cols  # Ceiling division

            # Create grid rows
            for row in range(num_rows):
                start_idx = row * num_cols
                end_idx = min(start_idx + num_cols, num_cells)
                row_cells = cells[start_idx:end_idx]
                # Properly format the row with table markers and exact number of cells
                formatted_row = "| " + " | ".join(row_cells) + " |"
                sample_grid.append(formatted_row)

    # Join grid rows with just the content, no headers needed
    grid_text = "\n".join(sample_grid) if sample_grid else ""

    # Fill in the template
    model_card_content = template.format(
        base_model=base_model_name,
        base_model_link=base_model_link,
        model_name=model_name,
        training_type="LoRA fine-tuning" if config.model.training_mode == "lora" else "Full model fine-tuning",
        training_steps=config.optimization.steps,
        learning_rate=config.optimization.learning_rate,
        batch_size=config.optimization.batch_size,
        validation_prompts=prompts_text,
        sample_grid=grid_text,
    )

    # Save the model card directly
    model_card_path = output_dir / "README.md"
    model_card_path.write_text(model_card_content)

    return model_card_path


def push_to_hub(weights_path: Path, sampled_videos_paths: List[Path], config: LtxvTrainerConfig) -> None:
    """Push the trained LoRA weights to HuggingFace Hub."""
    if not config.hub.push_to_hub:
        return

    if not config.hub.hub_model_id:
        logger.warning("⚠️ HuggingFace hub_model_id not specified, skipping push to hub")
        return

    api = HfApi()

    # Try to create repo if it doesn't exist
    try:
        create_repo(
            repo_id=config.hub.hub_model_id,
            repo_type="model",
            exist_ok=True,  # Don't raise error if repo exists
        )
    except Exception as e:
        logger.error(f"❌ Failed to create repository: {e}")
        return

    # Upload the original weights file
    try:
        api.upload_file(
            path_or_fileobj=str(weights_path),
            path_in_repo=weights_path.name,
            repo_id=config.hub.hub_model_id,
            repo_type="model",
        )
    except Exception as e:
        logger.error(f"❌ Failed to push {weights_path.name} to HuggingFace Hub: {e}")
    # Create a temporary directory for the files we want to upload
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Save model card and copy videos to temp directory
            create_model_card(
                output_dir=temp_path,
                videos=sampled_videos_paths,
                config=config,
            )

            # Upload the model card and samples directory
            api.upload_folder(
                folder_path=str(temp_path),  # Convert to string for compatibility
                repo_id=config.hub.hub_model_id,
                repo_type="model",
            )

            logger.info(f"✅ Successfully uploaded model card and sample videos to {config.hub.hub_model_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save/upload model card and videos: {e}")

    logger.info(f"✅ Successfully pushed original LoRA weights to {config.hub.hub_model_id}")

    # Convert and upload ComfyUI version
    try:
        # Create a temporary directory for the converted file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert the weights to ComfyUI format
            comfy_path = Path(temp_dir) / f"{weights_path.stem}_comfy{weights_path.suffix}"

            convert_checkpoint(
                input_path=str(weights_path),
                to_comfy=True,
                output_path=str(comfy_path),
            )

            # Find the converted file
            converted_files = list(Path(temp_dir).glob("*.safetensors"))
            if not converted_files:
                logger.warning("⚠️ No converted ComfyUI weights found")
                return

            converted_file = converted_files[0]
            comfy_filename = f"comfyui_{weights_path.name}"

            # Upload the converted file
            api.upload_file(
                path_or_fileobj=str(converted_file),
                path_in_repo=comfy_filename,
                repo_id=config.hub.hub_model_id,
                repo_type="model",
            )
            logger.info(f"✅ Successfully pushed ComfyUI LoRA weights to {config.hub.hub_model_id}")

    except Exception as e:
        logger.error(f"❌ Failed to convert and push ComfyUI version: {e}")
