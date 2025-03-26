#!/usr/bin/env python3

"""
Master script to run the complete LTXV LoRA training pipeline.

This script orchestrates the entire pipeline:
1. Scene splitting (if raw videos exist)
2. Video captioning (if scenes exist)
3. Dataset preprocessing
4. Model training

Usage:
    run_pipeline.py basename --resolution-buckets 768x768x49 --config-template configs/ltxv_lora_config.yaml
"""

import inspect

# Add the parent directory to the Python path so we can import from scripts
import sys
from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from typer.models import OptionInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.caption_videos import VIDEO_EXTENSIONS
from scripts.caption_videos import main as caption_videos
from scripts.convert_checkpoint import main as convert_checkpoint
from scripts.preprocess_dataset import main as preprocess_dataset
from scripts.split_scenes import main as split_scenes
from scripts.train import main as train


def typer_unpacker(f: Callable) -> Callable:
    def wrapper(*args, **kwargs) -> None:
        # Get the default function argument that aren't passed in kwargs via the
        # inspect module: https://stackoverflow.com/a/12627202
        missing_default_values = {
            k: v.default
            for k, v in inspect.signature(f).parameters.items()
            if v.default is not inspect.Parameter.empty and k not in kwargs
        }

        for name, func_default in missing_default_values.items():
            # If the default value is a typer.Option or typer.Argument, we have to
            # pull either the .default attribute and pass it in the function
            # invocation, or call it first.
            if isinstance(func_default, OptionInfo):
                if callable(func_default.default):
                    kwargs[name] = func_default.default()
                else:
                    kwargs[name] = func_default.default

        # Call the wrapped function with the defaults injected if not specified.
        return f(*args, **kwargs)

    return wrapper


console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    help="Run the complete LTXV training pipeline.",
)


def process_raw_videos(raw_dir: Path, scenes_dir: Path) -> None:
    """Process raw videos by splitting them into scenes.

    Args:
        raw_dir: Directory containing raw videos
        scenes_dir: Directory to save split scenes
    """
    # Get all video files
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(list(raw_dir.glob(f"*.{ext}")) + list(raw_dir.glob(f"*.{ext.upper()}")))

    if not video_files:
        console.print("[bold yellow]No video files found in raw directory.[/]")
        return

    console.print(f"Found [bold]{len(video_files)}[/] video files to process.")

    # Create scenes directory
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # Get the main function from the registered commands
    split_func = typer_unpacker(split_scenes)

    # Process each video
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Splitting videos into scenes", total=len(video_files))

        for video_file in video_files:
            # Split the video into scenes directly in the scenes directory
            console.print(f"Splitting video: {video_file}")
            split_func(
                video_path=str(video_file),
                output_dir=str(scenes_dir),
                detector="content",  # Use default content-based detection
            )

            progress.advance(task)


def process_scenes(scenes_dir: Path) -> None:
    """Process scenes by generating captions.

    Args:
        scenes_dir: Directory containing split scenes
    """
    # Check if scenes directory exists and contains subdirectories
    if not scenes_dir.exists() or not any(scenes_dir.iterdir()):
        console.print("[bold yellow]No scenes directory found or empty.[/]")
        return

    # Get the main function from the registered commands
    caption_func = typer_unpacker(caption_videos)

    caption_func(
        input_path=str(scenes_dir),  # Use current directory (scenes_dir)
        output=str(scenes_dir / "captions.json"),  # Save in current directory
        captioner_type="llava_next_7b",  # Use default captioner
    )


def preprocess_data(scenes_dir: Path, resolution_buckets: str, id_token: str | None = None) -> None:
    """Preprocess the dataset using the provided resolution buckets.

    Args:
        scenes_dir: Directory containing split scenes and captions
        resolution_buckets: Resolution buckets string (e.g. "768x768x49")
        id_token: Optional token to prepend to each caption (acts as a trigger word when training a LoRA)
    """
    if not scenes_dir.exists():
        console.print("[bold yellow]Scenes directory not found.[/]")
        return

    # Check for captions file
    captions_file = scenes_dir / "captions.json"
    if not captions_file.exists():
        console.print("[bold yellow]Captions file not found.[/]")
        return

    # Create preprocessed data directory
    preprocessed_dir = scenes_dir / ".precomputed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Get the main function from the registered commands
    preprocess_func = typer_unpacker(preprocess_dataset)

    # Preprocess the dataset
    preprocess_func(
        dataset_path=str(captions_file),
        resolution_buckets=resolution_buckets,
        caption_column="caption",
        video_column="media_path",
        output_dir=str(preprocessed_dir),
        id_token=id_token,
        decode_videos=True,  # Enable video decoding for verification
    )


def prepare_and_run_training(
    basename: str,
    config_template: Path,
    scenes_dir: Path,
    rank: int,
) -> None:
    """Prepare training configuration and run training.

    Args:
        basename: Base name for the project
        config_template: Path to the configuration template file
        scenes_dir: Directory containing preprocessed data
        rank: LoRA rank to use for training
    """
    if not config_template.exists():
        console.print(f"[bold red]Configuration template not found: {config_template}[/]")
        return

    # Read template and replace placeholders
    config_content = config_template.read_text()
    config_content = config_content.replace("[BASENAME]", basename)
    config_content = config_content.replace("[RANK]", str(rank))

    # Parse the config content to get the output directory
    import json

    import yaml

    config_data = yaml.safe_load(config_content)
    output_dir = Path(config_data.get("output_dir", "outputs"))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read prompts from captions.json if available
    captions_file = scenes_dir / "captions.json"
    if captions_file.exists():
        with open(captions_file) as f:
            captions_data = json.load(f)
            # Get up to 3 prompts from the captions
            prompts = [item["caption"] for item in captions_data[:3]]
            if prompts:
                # Replace validation.prompts in the config
                config_data["validation"]["prompts"] = prompts
                # Convert back to YAML string
                config_content = yaml.dump(config_data)

    # Save instantiated configuration
    config_path = output_dir / "config.yaml"
    config_path.write_text(config_content)

    # Get the main function from the registered commands
    train_func = typer_unpacker(train)

    # Run training
    train_func(config_path=str(config_path))

    # Convert LoRA to ComfyUI format
    console.print("[bold blue]Converting LoRA to ComfyUI format...[/]")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Find the latest checkpoint in the output directory
    lora_checkpoints = list(checkpoint_dir.glob("lora_weights_step_*.safetensors"))
    if not lora_checkpoints:
        console.print("[bold yellow]No LoRA checkpoints found in output directory.[/]")
        lora_path = None
    else:
        # Sort by step number (extracted from filename)
        lora_checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
        lora_path = lora_checkpoints[-1]  # Get the latest checkpoint
        console.print(f"[bold blue]Found latest checkpoint: {lora_path.name}[/]")

    if lora_path and lora_path.exists():
        convert_func = typer_unpacker(convert_checkpoint)
        convert_func(
            input_path=str(lora_path),
            to_comfy=True,
        )
        console.print("[bold green]LoRA conversion complete![/]")
    else:
        console.print(f"[bold yellow]No LoRA weights found at {lora_path} to convert.[/]")


@app.command()
def main(
    basename: str = typer.Argument(..., help="Base name for the project"),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF" (e.g. "768x768x49")',
    ),
    config_template: Path = typer.Option(  # noqa: B008
        ...,
        help="Path to the configuration template file",
        exists=True,
        dir_okay=False,
    ),
    id_token: str | None = typer.Option(
        default=None,
        help="Optional token to prepend to each caption (acts as a trigger word when training a LoRA)",
    ),
    rank: int = typer.Option(
        ...,
        help="LoRA rank to use for training",
        min=1,
        max=128,
    ),
) -> None:
    """Run the complete LTXV training pipeline."""
    # Define directories
    raw_dir = Path(f"{basename}_raw")
    scenes_dir = Path(f"{basename}_scenes")

    # Step 1: Process raw videos if they exist
    if raw_dir.exists() and any(raw_dir.iterdir()):
        console.print("[bold blue]Step 1: Processing raw videos...[/]")
        process_raw_videos(raw_dir, scenes_dir)
    else:
        console.print("[bold yellow]Raw videos directory not found or empty. Skipping scene splitting.[/]")

    # Step 2: Generate captions if scenes exist
    if scenes_dir.exists() and any(scenes_dir.iterdir()):
        console.print("[bold blue]Step 2: Generating captions...[/]")
        process_scenes(scenes_dir)
    else:
        console.print("[bold yellow]Scenes directory not found or empty. Skipping captioning.[/]")

    # Step 3: Preprocess dataset
    console.print("[bold blue]Step 3: Preprocessing dataset...[/]")
    preprocess_data(scenes_dir, resolution_buckets, id_token)

    # Step 4: Run training
    console.print("[bold blue]Step 4: Running training...[/]")
    prepare_and_run_training(basename, config_template, scenes_dir, rank)

    console.print("[bold green]Pipeline completed successfully![/]")


if __name__ == "__main__":
    app()
