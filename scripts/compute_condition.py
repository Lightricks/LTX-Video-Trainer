"""
Compute condition on videos.

This script provides a command-line interface for generating condition videos to be used as a reference for IC-LoRA.
Note that it reads and write to the same file, the output of caption_video.py,
where it adds the "reference_path" field to the json.

Basic usage:
    # Compute condition for all videos in a directory
    compute_condition.py videos_dir/ --output videos_dir/captions.json
"""

# Standard library imports
import json
from pathlib import Path
from typing import Dict

# Third-party imports
import cv2
import torch
import torchvision.transforms.functional as TF  # noqa: N812
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers.utils.logging import disable_progress_bar

# Local imports
from ltxv_trainer.video_utils import read_video, save_video

# Initialize console and disable progress bars
console = Console()
disable_progress_bar()


def compute_condition(
    images: torch.Tensor,
) -> torch.Tensor:
    """Compute Canny edge detection on a batch of images.

    Args:
        images: Batch of images tensor of shape [B, C, H, W]

    Returns:
        Binary edge masks tensor of shape [B, H, W]
    """
    # Convert to grayscale if needed
    if images.shape[1] == 3:
        images = TF.rgb_to_grayscale(images)

    # Ensure images are in [0, 1] range
    if images.max() > 1.0:
        images = images / 255.0

    # Compute Canny edges
    edge_masks = []
    for image in images:
        # Convert to numpy for OpenCV
        image_np = (image.squeeze().cpu().numpy() * 255).astype("uint8")

        # Apply Canny edge detection
        edges = cv2.Canny(
            image_np,
            threshold1=100,
            threshold2=200,
        )

        # Convert back to tensor
        edge_mask = torch.from_numpy(edges).float()
        edge_masks.append(edge_mask)

    edges = torch.stack(edge_masks)
    edges = torch.stack([edges] * 3, dim=1)  # Convert to 3-channel
    return edges


def _get_meta_data(
    output_path: Path,
) -> Dict[str, str]:
    """Get set of existing reference video paths without loading the actual files.

    Args:
        output_path: Path to the reference video paths file

    Returns:
        Dictionary mapping media paths to reference video paths
    """
    if not output_path.exists():
        return {}

    console.print(f"[bold blue]Reading meta data from [cyan]{output_path}[/]...[/]")

    try:
        with output_path.open("r", encoding="utf-8") as f:
            json_data = json.load(f)
        return json_data

    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not check meta data: {e}[/]")
        return {}


def _save_dataset_json(
    reference_paths: Dict[str, str],
    output_path: Path,
) -> None:
    """Save dataset json with reference video paths.

    Args:
        reference_paths: Dictionary mapping media paths to reference video paths
        output_path: Path to save the output file
    """

    with output_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)
        new_json_data = json_data.copy()
        for i, item in enumerate(json_data):
            media_path = item["media_path"]
            reference_path = reference_paths[media_path]
            new_json_data[i]["reference_path"] = reference_path

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(new_json_data, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]✓[/] Reference video paths saved to [cyan]{output_path}[/]")
    console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
    console.print("  reference_column='[cyan]reference_path[/]'")
    console.print("  video_column='[cyan]media_path[/]'")


def process_media(
    input_path: Path,
    output_path: Path,
    override: bool,
    batch_size: int = 100,
) -> None:
    """Process videos and images to compute condition on videos.

    Args:
        input_path: Path to input video/image file or directory
        output_path: Path to output reference video file
        override: Whether to override existing reference video files
    """
    if not output_path.exists():
        raise FileNotFoundError(
            f"Output file does not exist: {output_path}. This is also the input file for the dataset."
        )

    # Check for existing reference video files
    meta_data = _get_meta_data(output_path)

    base_dir = input_path.resolve()
    console.print(f"Using [bold blue]{base_dir}[/] as base directory for relative paths")

    # Filter media files
    media_to_process = []
    skipped_media = []

    def media_path_to_reference_path(media_file: Path) -> Path:
        return media_file.parent / (media_file.stem + "_reference" + media_file.suffix)

    media_files = [base_dir / Path(sample["media_path"]) for sample in meta_data]
    for media_file in media_files:
        reference_path = media_path_to_reference_path(media_file)
        media_to_process.append(media_file)

    console.print(f"Processing [bold]{len(media_to_process)}[/] media.")

    # Initialize progress tracking
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    # Process media files
    media_paths = [item["media_path"] for item in meta_data]
    reference_paths = {rel_path: str(media_path_to_reference_path(Path(rel_path))) for rel_path in media_paths}

    with progress:
        task = progress.add_task("Computing condition on videos", total=len(media_to_process))

        for media_file in media_to_process:
            progress.update(task, description=f"Processing [bold blue]{media_file.name}[/]")

            rel_path = str(media_file.resolve().relative_to(base_dir))
            reference_path = media_path_to_reference_path(media_file)
            reference_paths[rel_path] = str(reference_path.relative_to(base_dir))

            if not reference_path.resolve().exists() or override:
                try:
                    video, fps = read_video(media_file)

                    # Process frames in batches
                    condition_frames = []

                    for i in range(0, len(video), batch_size):
                        batch = video[i : i + batch_size]
                        condition_batch = compute_condition(batch)
                        condition_frames.append(condition_batch)

                    # Concatenate all edge frames
                    all_condition = torch.cat(condition_frames, dim=0)

                    # Save the edge video
                    save_video(all_condition, reference_path.resolve(), fps=fps)

                except Exception as e:
                    console.print(f"[bold red]Error processing [bold blue]{media_file}[/]: {e}[/]")
                    reference_paths.pop(rel_path)
            else:
                skipped_media.append(media_file)

            progress.advance(task)

    # Save results
    _save_dataset_json(reference_paths, output_path)

    # Print summary
    total_to_process = len(media_files) - len(skipped_media)
    console.print(
        f"[bold green]✓[/] Processed [bold]{total_to_process}/{len(media_files)}[/] media successfully.",
    )


app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Compute condition on videos.",
)


@app.command()
def main(
    input_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to input video/image file or directory containing media files",
        exists=True,
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Path to json output file for reference video paths. "
        "This is also the input file for the dataset, the output of compute_captions.py.",
    ),
    override: bool = typer.Option(
        False,
        "--override",
        help="Whether to override existing reference video files",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        help="Batch size for processing videos",
    ),
) -> None:
    """Compute condition on videos.

    This script generates condition videos for given videos.
    The paths in the output file will be relative to the output file's directory.

    Examples:
        # Process all videos in a directory
        compute_condition.py videos_dir/ -o videos_dir/captions.json
    """

    # Ensure output path is absolute
    output = Path(output).resolve()
    console.print(f"Output will be saved to [bold blue]{output}[/]")

    # Verify output path exists
    if not output.exists():
        raise FileNotFoundError(f"Output file does not exist: {output}. This is also the input file for the dataset.")

    # Process media files
    process_media(
        input_path=input_path,
        output_path=output,
        override=override,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()
