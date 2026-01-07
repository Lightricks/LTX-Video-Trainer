"""Gradio interface for LTX Video Trainer."""

import datetime
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import gradio as gr
import torch
import yaml
from huggingface_hub import login

from ltxv_trainer.captioning import (
    DEFAULT_VLM_CAPTION_INSTRUCTION,
    CaptionerType,
    create_captioner,
)
from ltxv_trainer.hf_hub_utils import convert_video_to_gif
from ltxv_trainer.model_loader import (
    LtxvModelVersion,
)
from ltxv_trainer.trainer import LtxvTrainer, LtxvTrainerConfig
from scripts.dataset_manager import DatasetManager
from scripts.jobs.database import JobDatabase, JobStatus
from scripts.preprocess_dataset import preprocess_dataset
from scripts.process_videos import parse_resolution_buckets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

torch.cuda.empty_cache()
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
DATASETS_DIR = PROJECT_ROOT / "datasets"
VALIDATION_SAMPLES_DIR = OUTPUTS_DIR / "validation_samples"

OUTPUTS_DIR.mkdir(exist_ok=True)
TRAINING_DATA_DIR.mkdir(exist_ok=True)
VALIDATION_SAMPLES_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingConfigParams:
    """Parameters for generating training configuration."""

    model_source: str
    learning_rate: float
    steps: int
    lora_rank: int
    batch_size: int
    validation_prompt: str
    video_dims: tuple[int, int, int]
    validation_interval: int = 100
    push_to_hub: bool = False
    hub_model_id: str | None = None


@dataclass
class TrainingState:
    """State for tracking training progress."""

    status: str | None = None
    progress: str | None = None
    validation: str | None = None
    download: str | None = None
    error: str | None = None
    hf_repo: str | None = None
    checkpoint_path: str | None = None

    def reset(self) -> None:
        """Reset state to initial values."""
        self.status = "running"
        self.progress = None
        self.validation = None
        self.download = None
        self.error = None
        self.hf_repo = None
        self.checkpoint_path = None

    def update(self, **kwargs: str | int | None) -> None:
        """Update state with provided values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class TrainingParams:
    dataset_name: str
    validation_prompt: str
    learning_rate: float
    steps: int
    lora_rank: int
    batch_size: int
    model_source: str
    width: int
    height: int
    num_frames: int
    push_to_hub: bool
    hf_model_id: str
    hf_token: str | None = None
    id_token: str | None = None
    validation_interval: int = 100


def _handle_validation_sample(step: int, video_path: Path) -> str | None:
    """Handle validation sample conversion and storage.

    Args:
        step: Current training step
        video_path: Path to the validation video

    Returns:
        Path to the GIF file if successful, None otherwise
    """
    gif_path = VALIDATION_SAMPLES_DIR / f"sample_step_{step}.gif"
    try:
        convert_video_to_gif(video_path, gif_path)
        logger.info(f"New validation sample converted to GIF at step {step}: {gif_path}")
        return str(gif_path)
    except Exception as e:
        logger.error(f"Failed to convert validation video to GIF: {e}")
        return None


def generate_training_config(params: TrainingConfigParams, training_data_dir: str) -> dict:
    """Generate training configuration from parameters.

    Args:
        params: Training configuration parameters
        training_data_dir: Directory containing training data

    Returns:
        Dictionary containing the complete training configuration
    """
    template_path = Path(__file__).parent.parent / "configs" / "ltxv_13b_lora_template.yaml"
    with open(template_path) as f:
        config = yaml.safe_load(f)

    config["model"]["model_source"] = params.model_source
    config["lora"]["rank"] = params.lora_rank
    config["lora"]["alpha"] = params.lora_rank
    config["optimization"]["learning_rate"] = params.learning_rate
    config["optimization"]["steps"] = params.steps
    config["optimization"]["batch_size"] = params.batch_size
    config["data"]["preprocessed_data_root"] = str(training_data_dir)
    config["output_dir"] = str(OUTPUTS_DIR / f"lora_r{params.lora_rank}")

    config["hub"] = {
        "push_to_hub": params.push_to_hub,
        "hub_model_id": params.hub_model_id if params.push_to_hub else None,
    }

    width, height, num_frames = params.video_dims
    config["validation"] = {
        "prompts": [params.validation_prompt],
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "video_dims": [width, height, num_frames],
        "seed": 42,
        "inference_steps": 30,
        "interval": params.validation_interval,
        "videos_per_prompt": 1,
        "guidance_scale": 3.5,
    }

    if "validation" in config and "images" not in config["validation"]:
        config["validation"]["images"] = None

    return config


class GradioUI:
    """Class to manage Gradio UI components and state."""

    def __init__(self) -> None:
        self.training_state = TrainingState()
        self.dataset_manager = DatasetManager(datasets_root=DATASETS_DIR)
        self.current_dataset = None

        db_path = PROJECT_ROOT / "jobs.db"
        self.job_db = JobDatabase(db_path)
        self.current_job_id = None

        self.validation_prompt = None
        self.status_output = None
        self.progress_output = None
        self.download_btn = None
        self.hf_repo_link = None

    def reset_interface(self) -> dict:
        """Reset the interface and clean up all training data.

        Returns:
            Dictionary of Gradio component updates
        """
        self.training_state.reset()
        self.current_job_id = None

        if TRAINING_DATA_DIR.exists():
            shutil.rmtree(TRAINING_DATA_DIR)
        TRAINING_DATA_DIR.mkdir(exist_ok=True)

        if OUTPUTS_DIR.exists():
            shutil.rmtree(OUTPUTS_DIR)
        OUTPUTS_DIR.mkdir(exist_ok=True)
        VALIDATION_SAMPLES_DIR.mkdir(exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return {
            self.validation_prompt: gr.update(
                value="a professional portrait video of a person with blurry bokeh background",
                info="Include the LoRA ID token (e.g., &lt;lora&gt;) in this prompt if desired.",
            ),
            self.status_output: gr.update(value=""),
            self.progress_output: gr.update(value=""),
            self.download_btn: gr.update(visible=False),
            self.hf_repo_link: gr.update(visible=False, value=""),
        }

    def get_model_path(self) -> str | None:
        """Get the path to the trained model file."""
        if self.training_state.download and Path(self.training_state.download).exists():
            return self.training_state.download
        return None

    def update_progress(self) -> tuple[str, str, gr.update, str, gr.update]:
        """Update the UI with current training progress."""
        # Default return values
        job_display = ""
        progress_msg = ""
        file_update = gr.update(visible=False)
        hf_html = ""
        hf_update = gr.update(visible=False)

        # Handle current job from queue
        if self.current_job_id:
            job = self.job_db.get_job(self.current_job_id)
            if job:
                job_display = f"**Current Job:** #{self.current_job_id} | Dataset: `{job['dataset_name']}`"
                status = job["status"]
                progress = job.get("progress", "")

                status_messages = {
                    JobStatus.PENDING: "Waiting in queue...",
                    JobStatus.RUNNING: progress if progress else "Training in progress...",
                    JobStatus.COMPLETED: "Training completed!",
                    JobStatus.FAILED: "Training failed",
                    JobStatus.CANCELLED: "Job cancelled",
                }
                progress_msg = status_messages.get(status, "")

                # Handle completed job artifacts
                if status == JobStatus.COMPLETED:
                    hf_repo = job.get("hf_repo_url", "")
                    checkpoint_path = job.get("checkpoint_path", "")

                    if hf_repo:
                        hf_html = f'<a href="{hf_repo}" target="_blank">View model on HuggingFace Hub</a>'
                        hf_update = gr.update(visible=True)
                    elif checkpoint_path and Path(checkpoint_path).exists():
                        file_update = gr.update(
                            value=checkpoint_path, visible=True, label=f"Download {Path(checkpoint_path).name}"
                        )

                return (job_display, progress_msg, file_update, hf_html, hf_update)

        # Handle legacy direct training mode
        if self.training_state.status is not None:
            job_display = "**Direct Training Mode** (Legacy)"
            progress_msg = self.training_state.progress
            status = self.training_state.status

            if status == "complete":
                if self.training_state.hf_repo:
                    hf_html = (
                        f'<a href="{self.training_state.hf_repo}" target="_blank">View model on HuggingFace Hub</a>'
                    )
                    hf_update = gr.update(visible=True)
                elif self.training_state.download and Path(self.training_state.download).exists():
                    file_update = gr.update(
                        value=self.training_state.download,
                        visible=True,
                        label=f"Download {Path(self.training_state.download).name}",
                    )

            return (job_display, progress_msg, file_update, hf_html, hf_update)

        # No active training
        return (job_display, progress_msg, file_update, hf_html, hf_update)

    def _save_checkpoint(self, saved_path: Path, trainer_config: LtxvTrainerConfig) -> tuple[Path, str | None]:
        """Save and copy the checkpoint to a permanent location.

        Args:
            saved_path: Path where the checkpoint was initially saved
            trainer_config: Training configuration

        Returns:
            Tuple of (permanent checkpoint path, HF repo URL if applicable)
        """
        permanent_checkpoint_dir = OUTPUTS_DIR / "checkpoints"
        permanent_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"comfy_lora_checkpoint_{timestamp}.safetensors"
        permanent_checkpoint_path = permanent_checkpoint_dir / checkpoint_filename

        try:
            shutil.copy2(saved_path, permanent_checkpoint_path)
            logger.info(f"Checkpoint copied to permanent location: {permanent_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to copy checkpoint: {e}")
            permanent_checkpoint_path = saved_path

        hf_repo = (
            f"https://huggingface.co/{trainer_config.hub.hub_model_id}" if trainer_config.hub.hub_model_id else None
        )

        return permanent_checkpoint_path, hf_repo

    def _preprocess_dataset(
        self,
        dataset_file: Path,
        model_source: str,
        width: int,
        height: int,
        num_frames: int,
        id_token: str | None = None,
    ) -> tuple[bool, str | None]:
        """Preprocess the dataset by computing video latents and text embeddings.

        Args:
            dataset_file: Path to the dataset.json file
            model_source: Model source identifier
            width: Video width
            height: Video height
            num_frames: Number of frames
            id_token: Optional token to prepend to captions (for LoRA training)

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            precomputed_dir = TRAINING_DATA_DIR / ".precomputed"
            if precomputed_dir.exists():
                shutil.rmtree(precomputed_dir)

            resolution_buckets = f"{width}x{height}x{num_frames}"
            parsed_buckets = parse_resolution_buckets(resolution_buckets)

            preprocess_dataset(
                dataset_file=str(dataset_file),
                caption_column="caption",
                video_column="media_path",
                resolution_buckets=parsed_buckets,
                batch_size=1,
                output_dir=None,
                id_token=id_token,
                vae_tiling=False,
                decode_videos=True,
                model_source=model_source,
                device="cuda" if torch.cuda.is_available() else "cpu",
                load_text_encoder_in_8bit=False,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True, None

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False, f"Error preprocessing dataset: {e!s}"

    def _should_preprocess_data(
        self,
        width: int,
        height: int,
        num_frames: int,
        videos: list[str],
    ) -> bool:
        """Check if data needs to be preprocessed based on resolution changes.

        Args:
            width: Video width
            height: Video height
            num_frames: Number of frames
            videos: List of video file paths

        Returns:
            True if preprocessing is needed, False otherwise
        """
        resolution_file = TRAINING_DATA_DIR / ".resolution_config"
        current_resolution = f"{width}x{height}x{num_frames}"
        needs_to_copy = False
        for video in videos:
            if Path(video).exists():
                needs_to_copy = True
        if needs_to_copy:
            logger.info("Videos provided, will copy them to training directory.")
            return True, needs_to_copy

        if not resolution_file.exists() or not (TRAINING_DATA_DIR / "captions.json").exists():
            return True, needs_to_copy

        try:
            with open(resolution_file) as f:
                previous_resolution = f.read().strip()
            return previous_resolution != current_resolution, needs_to_copy
        except Exception:
            return True, needs_to_copy

    def _save_resolution_config(
        self,
        width: int,
        height: int,
        num_frames: int,
    ) -> None:
        """Save current resolution configuration.

        Args:
            width: Video width
            height: Video height
            num_frames: Number of frames
        """
        resolution_file = TRAINING_DATA_DIR / ".resolution_config"
        current_resolution = f"{width}x{height}x{num_frames}"

        with open(resolution_file, "w") as f:
            f.write(current_resolution)

    def _sync_captions_from_ui(
        self, params: TrainingParams, training_captions_file: Path
    ) -> tuple[dict[str, str] | None, str | None]:
        """Sync captions from the UI to captions.json. Returns (captions_data, error_message)."""
        if params.captions_json:
            try:
                dataset = json.loads(params.captions_json)
                # Convert list of dicts to captions_data dict
                captions_data = {item["media_path"]: item["caption"] for item in dataset}
                # Save to captions.json (overwrite every time)
                with open(training_captions_file, "w") as f:
                    json.dump(captions_data, f, indent=2)
                return captions_data, None
            except Exception as e:
                return None, f"Invalid captions JSON: {e!s}"
        else:
            return None, "No captions found in the UI. Please process videos first."

    # ruff: noqa: PLR0912
    def start_training(
        self,
        params: TrainingParams,
    ) -> tuple[str, gr.update]:
        """Queue a training job."""
        # Validate dataset exists
        if not params.dataset_name:
            return "Please select a dataset from the Datasets tab", gr.update(interactive=True)

        managed_dataset_dir = self.dataset_manager.datasets_root / params.dataset_name
        managed_dataset_json = managed_dataset_dir / "dataset.json"

        if not managed_dataset_json.exists():
            return f"Dataset '{params.dataset_name}' not found or has no videos", gr.update(interactive=True)

        # Check if dataset is preprocessed
        precomputed_dir = managed_dataset_dir / ".precomputed"
        if not precomputed_dir.exists():
            return (
                f"Dataset '{params.dataset_name}' is not preprocessed. Please preprocess it in the Datasets tab first.",
                gr.update(interactive=True),
            )

        # Create job parameters
        job_params = {
            "model_source": params.model_source,
            "learning_rate": params.learning_rate,
            "steps": params.steps,
            "lora_rank": params.lora_rank,
            "batch_size": params.batch_size,
            "width": params.width,
            "height": params.height,
            "num_frames": params.num_frames,
            "id_token": params.id_token or "",
            "validation_prompt": params.validation_prompt,
            "validation_interval": params.validation_interval,
            "push_to_hub": params.push_to_hub,
            "hf_model_id": params.hf_model_id if params.push_to_hub else "",
            "hf_token": params.hf_token if params.push_to_hub else "",
        }

        # Create job in database
        try:
            job_id = self.job_db.create_job(params.dataset_name, job_params)
            self.current_job_id = job_id

            return (
                f"Job #{job_id} created and queued for training on dataset '{params.dataset_name}'.\n"
                f"The worker will start training automatically. Check the Queue tab for status.",
                gr.update(interactive=True),
            )
        except Exception as e:
            return f"Failed to create training job: {e}", gr.update(interactive=True)

    def start_training_direct(
        self,
        params: TrainingParams,
    ) -> tuple[str, gr.update]:
        """Start the training process directly (legacy method)."""
        if params.hf_token:
            login(token=params.hf_token)

        try:
            # Clear any existing CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Set training status
            self.training_state.reset()  # This sets status to "running"

            # Prepare data directory
            data_dir = TRAINING_DATA_DIR
            data_dir.mkdir(exist_ok=True)

            # Validate dataset exists
            if not params.dataset_name:
                return "Please select a dataset from the Datasets tab", gr.update(interactive=True)

            # Use managed dataset
            managed_dataset_dir = self.dataset_manager.datasets_root / params.dataset_name
            managed_dataset_json = managed_dataset_dir / "dataset.json"

            if not managed_dataset_json.exists():
                return f"Dataset '{params.dataset_name}' not found or has no videos", gr.update(interactive=True)

            # Copy dataset.json to training directory
            training_captions_file = data_dir / "captions.json"
            shutil.copy2(managed_dataset_json, training_captions_file)

            # Load the dataset to get video paths
            with open(training_captions_file) as f:
                dataset = json.load(f)

            # Copy videos from managed dataset to training directory
            for item in dataset:
                src_video = managed_dataset_dir / item["media_path"]
                dest_video = data_dir / Path(item["media_path"]).name

                if not dest_video.exists() and src_video.exists():
                    shutil.copy2(src_video, dest_video)

                # Update the media_path to be relative to data_dir
                item["media_path"] = Path(item["media_path"]).name

            # Save updated dataset with corrected paths
            with open(training_captions_file, "w") as f:
                json.dump(dataset, f, indent=2)

            # Check if preprocessing is needed
            needs_preprocessing = self._should_preprocess_data(params.width, params.height, params.num_frames, [])[0]

            # Preprocess if needed (first time or resolution changed)
            if needs_preprocessing:
                # Clean up existing precomputed data
                precomputed_dir = TRAINING_DATA_DIR / ".precomputed"
                if precomputed_dir.exists():
                    shutil.rmtree(precomputed_dir)

                success, error_msg = self._preprocess_dataset(
                    dataset_file=training_captions_file,
                    model_source=params.model_source,
                    width=params.width,
                    height=params.height,
                    num_frames=params.num_frames,
                    id_token=params.id_token,
                )
                if not success:
                    return error_msg, gr.update(interactive=True)

                # Save current resolution config after successful preprocessing
                self._save_resolution_config(params.width, params.height, params.num_frames)

            # Generate training config
            config_params = TrainingConfigParams(
                model_source=params.model_source,
                learning_rate=params.learning_rate,
                steps=params.steps,
                lora_rank=params.lora_rank,
                batch_size=params.batch_size,
                validation_prompt=params.validation_prompt,
                video_dims=(params.width, params.height, params.num_frames),
                validation_interval=params.validation_interval,
                push_to_hub=params.push_to_hub,
                hub_model_id=params.hf_model_id if params.push_to_hub else None,
            )

            config = generate_training_config(config_params, str(data_dir))
            config_path = OUTPUTS_DIR / "train_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f, indent=4)

            # Run training
            self.run_training(config_path)

            return "Training completed!", gr.update(interactive=True)

        except Exception as e:
            return f"Error during training: {e!s}", gr.update(interactive=True)
        finally:
            # Clean up CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    def run_training(self, config_path: Path) -> None:
        """Run the training process and update progress."""
        # Reset training state at the start
        self.training_state.reset()

        try:
            # Load config from YAML
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            # Initialize trainer config and trainer
            trainer_config = LtxvTrainerConfig(**config_dict)
            trainer = LtxvTrainer(trainer_config)

            def training_callback(step: int, total_steps: int, sampled_videos: list[Path] | None = None) -> None:
                """Callback function to update training progress and show samples."""
                # Update progress
                progress_pct = (step / total_steps) * 100
                self.training_state.update(progress=f"Step {step}/{total_steps} ({progress_pct:.1f}%)")

                # Update validation video at validation intervals
                if step % trainer_config.validation.interval == 0 and sampled_videos:
                    # Convert the first sample to GIF
                    gif_path = _handle_validation_sample(step, sampled_videos[0])
                    if gif_path:
                        self.training_state.update(validation=gif_path)

            logger.info("Starting training...")

            # Start training with callback
            saved_path, stats = trainer.train(disable_progress_bars=False, step_callback=training_callback)

            # Save checkpoint and get paths
            permanent_checkpoint_path, hf_repo = self._save_checkpoint(saved_path, trainer_config)

            # Update training outputs with completion status
            self.training_state.update(
                status="complete",
                download=str(permanent_checkpoint_path),
                hf_repo=hf_repo,
                checkpoint_path=str(permanent_checkpoint_path),
            )

            logger.info(f"Training completed. Model saved to {permanent_checkpoint_path}")
            logger.info(f"Training stats: {stats}")

        except Exception as e:
            logger.error(f"Training failed: {e!s}", exc_info=True)
            self.training_state.update(status="failed", error=str(e))
            raise
        finally:
            # Don't reset current_job here - let the UI handle it
            if self.training_state.status == "running":
                self.training_state.update(status="failed")

    def create_new_dataset(self, name: str) -> tuple[str, gr.Dropdown, list, dict, None, str, str, gr.Dropdown]:
        """Create a new dataset.

        Args:
            name: Name of the dataset

        Returns:
            Tuple of (status message, updated dropdown, empty gallery, empty stats, cleared fields, training dropdown)
        """
        if not name or not name.strip():
            return "Please enter a dataset name", gr.update(), gr.update(), {}, None, "", "", gr.update()

        try:
            self.dataset_manager.create_dataset(name)
            datasets = self.dataset_manager.list_datasets()
            # Return updated dropdown with new dataset selected, and clear all other fields
            return (
                f"Created dataset: {name}",
                gr.update(choices=datasets, value=name),
                [],  # Empty gallery
                {"name": name, "total_videos": 0, "captioned": 0, "uncaptioned": 0, "preprocessed": False},  # Stats
                None,  # Clear selected video
                "",  # Clear video name
                "",  # Clear caption editor
                gr.update(choices=datasets),  # Update training tab dropdown
            )
        except Exception as e:
            return f"Error: {e}", gr.update(), gr.update(), {}, None, "", "", gr.update()

    def load_dataset(self, dataset_name: str) -> tuple[str | None, list, dict, None | str, str, str]:
        """Load a dataset and display its contents.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Tuple of (dataset name, gallery items, statistics, cleared video/name/caption)
        """
        if not dataset_name:
            return None, [], {}, None, "", ""

        self.current_dataset = dataset_name
        items = self.dataset_manager.get_dataset_items(dataset_name)
        stats = self.dataset_manager.get_dataset_stats(dataset_name)

        # Prepare gallery items (thumbnails with captions)
        gallery_items = [
            (
                item["thumbnail"],
                item["caption"][:50] + "..."
                if len(item.get("caption", "")) > 50
                else item.get("caption", "") or "No caption",
            )
            for item in items
            if item["thumbnail"]
        ]

        # Clear the video preview and caption editor when switching datasets
        return dataset_name, gallery_items, stats, None, "", ""

    def upload_videos_to_dataset(self, files: list, dataset_name: str) -> tuple[str, gr.Gallery, dict]:
        """Upload videos to a dataset.

        Args:
            files: List of video file paths
            dataset_name: Name of the dataset

        Returns:
            Tuple of (status message, updated gallery, updated stats)
        """
        if not dataset_name:
            return "Please select a dataset first", gr.update(), gr.update()

        if not files:
            return "No files selected", gr.update(), gr.update()

        try:
            result = self.dataset_manager.add_videos(dataset_name, files)

            message = f"Added {len(result['added'])} videos"
            if result["failed"]:
                message += f", {len(result['failed'])} failed"
                for failed in result["failed"][:3]:  # Show first 3 failures
                    message += f"\n- {failed['video']}: {failed['error']}"

            # Refresh gallery and stats
            items = self.dataset_manager.get_dataset_items(dataset_name)
            gallery_items = [
                (
                    i["thumbnail"],
                    (
                        i.get("caption", "")[:50] + "..."
                        if len(i.get("caption", "")) > 50
                        else i.get("caption", "") or "No caption"
                    ),
                )
                for i in items
                if i["thumbnail"]
            ]
            stats = self.dataset_manager.get_dataset_stats(dataset_name)

            return message, gr.update(value=gallery_items), stats
        except Exception as e:
            return f"Error: {e}", gr.update(), gr.update()

    def upload_videos_with_references(
        self, video_files: list, reference_files: list, dataset_name: str
    ) -> tuple[str, gr.Gallery, dict]:
        """Upload videos with reference videos to a dataset for IC-LoRA training.

        Args:
            video_files: List of target video file paths
            reference_files: List of reference video file paths
            dataset_name: Name of the dataset

        Returns:
            Tuple of (status message, updated gallery, updated stats)
        """
        if not dataset_name:
            return "Please select a dataset first", gr.update(), gr.update()

        if not video_files:
            return "No target videos selected", gr.update(), gr.update()

        if not reference_files:
            return "No reference videos selected", gr.update(), gr.update()

        try:
            result = self.dataset_manager.add_videos(dataset_name, video_files, reference_files)

            message = f"Added {len(result['added'])} video pairs (target + reference)"
            if result["failed"]:
                message += f", {len(result['failed'])} failed"
                for failed in result["failed"][:3]:  # Show first 3 failures
                    message += f"\n- {failed['video']}: {failed['error']}"

            # Refresh gallery and stats
            items = self.dataset_manager.get_dataset_items(dataset_name)
            gallery_items = [
                (
                    i["thumbnail"],
                    (
                        i.get("caption", "")[:50] + "..."
                        if len(i.get("caption", "")) > 50
                        else i.get("caption", "") or "No caption"
                    ),
                )
                for i in items
                if i["thumbnail"]
            ]
            stats = self.dataset_manager.get_dataset_stats(dataset_name)

            return message, gr.update(value=gallery_items), stats
        except Exception as e:
            return f"Error: {e}", gr.update(), gr.update()

    def select_video_from_gallery(self, evt: gr.SelectData, dataset_name: str) -> tuple[str | None, str, str]:
        """Handle video selection from gallery.

        Args:
            evt: Selection event data
            dataset_name: Name of the dataset

        Returns:
            Tuple of (video path, video name, caption)
        """
        if not dataset_name:
            return None, "", ""

        items = self.dataset_manager.get_dataset_items(dataset_name)
        if evt.index >= len(items):
            return None, "", ""

        selected_item = items[evt.index]
        video_path = selected_item["full_video_path"]
        video_name = Path(selected_item["media_path"]).name
        caption = selected_item.get("caption", "")

        return video_path, video_name, caption

    def save_caption_edit(self, dataset_name: str, video_name: str, caption: str) -> tuple[str, dict]:
        """Save edited caption for a video.

        Args:
            dataset_name: Name of the dataset
            video_name: Name of the video file
            caption: New caption text

        Returns:
            Tuple of (status message, updated stats)
        """
        if not dataset_name or not video_name:
            return "No video selected", gr.update()

        try:
            self.dataset_manager.update_caption(dataset_name, video_name, caption)
            stats = self.dataset_manager.get_dataset_stats(dataset_name)
            return f"Caption saved for {video_name}", stats
        except Exception as e:
            return f"Error: {e}", gr.update()

    def auto_caption_single(
        self, dataset_name: str, video_name: str, vlm_instruction: str, use_8bit: bool
    ) -> tuple[str, str]:
        """Auto-generate caption for a single video.

        Args:
            dataset_name: Name of the dataset
            video_name: Name of the video file
            vlm_instruction: Custom VLM instruction for captioning
            use_8bit: Whether to use 8-bit quantization

        Returns:
            Tuple of (generated caption, status message)
        """
        if not dataset_name or not video_name:
            return "", "No video selected"

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Use custom instruction if provided, otherwise use default
            instruction = (
                vlm_instruction if vlm_instruction and vlm_instruction.strip() else DEFAULT_VLM_CAPTION_INSTRUCTION
            )

            captioner = create_captioner(
                captioner_type=CaptionerType.QWEN_25_VL,
                use_8bit=use_8bit,
                vlm_instruction=instruction,
                device=device,
            )

            video_path = Path(self.dataset_manager.datasets_root) / dataset_name / "videos" / video_name
            caption = captioner.caption(str(video_path))

            self.dataset_manager.update_caption(dataset_name, video_name, caption)
            return caption, f"Caption generated for {video_name}"
        except Exception as e:
            return "", f"Error: {e}"

    def auto_caption_all_uncaptioned(self, dataset_name: str, vlm_instruction: str, use_8bit: bool) -> tuple[str, dict]:
        """Auto-generate captions for all uncaptioned videos.

        Args:
            dataset_name: Name of the dataset
            vlm_instruction: Custom VLM instruction for captioning
            use_8bit: Whether to use 8-bit quantization

        Returns:
            Tuple of (status message, updated stats)
        """
        if not dataset_name:
            return "Please select a dataset first", gr.update()

        try:
            items = self.dataset_manager.get_dataset_items(dataset_name)
            uncaptioned = [i for i in items if not i.get("caption") or not i.get("caption").strip()]

            if not uncaptioned:
                stats = self.dataset_manager.get_dataset_stats(dataset_name)
                return "All videos already have captions", stats

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Use custom instruction if provided, otherwise use default
            instruction = (
                vlm_instruction if vlm_instruction and vlm_instruction.strip() else DEFAULT_VLM_CAPTION_INSTRUCTION
            )

            captioner = create_captioner(
                captioner_type=CaptionerType.QWEN_25_VL,
                use_8bit=use_8bit,
                vlm_instruction=instruction,
                device=device,
            )

            for _idx, item in enumerate(uncaptioned):
                video_path = Path(self.dataset_manager.datasets_root) / dataset_name / item["media_path"]
                caption = captioner.caption(str(video_path))
                video_name = Path(item["media_path"]).name
                self.dataset_manager.update_caption(dataset_name, video_name, caption)

            stats = self.dataset_manager.get_dataset_stats(dataset_name)
            return f"Generated captions for {len(uncaptioned)} videos", stats
        except Exception as e:
            return f"Error: {e}", gr.update()

    def validate_dataset_ui(self, dataset_name: str) -> dict:
        """Validate a dataset and return issues.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Validation results dictionary
        """
        if not dataset_name:
            return {"error": "Please select a dataset first"}

        try:
            return self.dataset_manager.validate_dataset(dataset_name)
        except Exception as e:
            return {"error": str(e)}

    def preprocess_dataset_ui(
        self,
        dataset_name: str,
        width: int,
        height: int,
        num_frames: int,
        id_token: str,
        decode_videos: bool,
        load_text_encoder_8bit: bool,
    ) -> tuple[str, dict]:
        """Preprocess a dataset from the UI.

        Args:
            dataset_name: Name of the dataset
            width: Video width
            height: Video height
            num_frames: Number of frames
            id_token: ID token to prepend to captions
            decode_videos: Whether to decode videos for verification
            load_text_encoder_8bit: Whether to load text encoder in 8-bit mode

        Returns:
            Tuple of (status message, updated stats)
        """
        if not dataset_name:
            return "Please select a dataset first", gr.update()

        try:
            dataset_dir = self.dataset_manager.datasets_root / dataset_name
            dataset_json = dataset_dir / "dataset.json"

            if not dataset_json.exists():
                return "Dataset JSON not found", gr.update()

            # Check if dataset has reference videos
            with open(dataset_json) as f:
                dataset_items = json.load(f)

            has_references = any(item.get("reference_path") for item in dataset_items)

            resolution_buckets = f"{width}x{height}x{num_frames}"
            parsed_buckets = parse_resolution_buckets(resolution_buckets)

            preprocess_dataset(
                dataset_file=str(dataset_json),
                caption_column="caption",
                video_column="media_path",
                resolution_buckets=parsed_buckets,
                batch_size=1,
                output_dir=None,  # Will use default .precomputed
                id_token=id_token if id_token and id_token.strip() else None,
                vae_tiling=False,
                decode_videos=decode_videos,
                model_source=LtxvModelVersion.latest(),
                device="cuda" if torch.cuda.is_available() else "cpu",
                load_text_encoder_in_8bit=load_text_encoder_8bit,
                reference_column="reference_path" if has_references else None,
            )

            stats = self.dataset_manager.get_dataset_stats(dataset_name)
            message = "Preprocessing complete!"
            if has_references:
                message += " (IC-LoRA reference videos processed)"
            if decode_videos:
                decoded_dir = dataset_dir / ".precomputed" / "decoded_videos"
                message += f"\nDecoded videos saved to: {decoded_dir}"
            return message, stats
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            return f"Preprocessing failed: {e}", gr.update()

    def delete_video_from_dataset(
        self, dataset_name: str, video_name: str
    ) -> tuple[str, gr.Gallery, None, str, str, dict]:
        """Delete a video from a dataset.

        Args:
            dataset_name: Name of the dataset
            video_name: Name of the video to delete

        Returns:
            Tuple of (status message, updated gallery, cleared video/name/caption, updated stats)
        """
        if not dataset_name or not video_name:
            return "No video selected", gr.update(), None, "", "", gr.update()

        try:
            self.dataset_manager.delete_video(dataset_name, video_name)

            # Refresh gallery
            items = self.dataset_manager.get_dataset_items(dataset_name)
            gallery_items = [
                (
                    i["thumbnail"],
                    (
                        i.get("caption", "")[:50] + "..."
                        if len(i.get("caption", "")) > 50
                        else i.get("caption", "") or "No caption"
                    ),
                )
                for i in items
                if i["thumbnail"]
            ]
            stats = self.dataset_manager.get_dataset_stats(dataset_name)

            return f"Deleted {video_name}", gr.update(value=gallery_items), None, "", "", stats
        except Exception as e:
            return f"Error: {e}", gr.update(), None, "", "", gr.update()

    def split_scenes_and_add(
        self,
        dataset_name: str,
        video_file: str,
        detector_type: str,
        min_scene_length: int | None,
        threshold: float | None,
        filter_shorter: str | None,
        max_scenes: int | None,
    ) -> tuple[str, gr.Gallery, dict]:
        """Split a video into scenes and add them to the dataset.

        Args:
            dataset_name: Name of the dataset
            video_file: Path to the video file
            detector_type: Type of scene detector to use
            min_scene_length: Minimum scene length in frames
            threshold: Detection threshold
            filter_shorter: Filter scenes shorter than this duration
            max_scenes: Maximum number of scenes (0 for unlimited)

        Returns:
            Tuple of (status message, updated gallery, updated stats)
        """
        if not dataset_name:
            return "Please select a dataset first", gr.update(), gr.update()

        if not video_file:
            return "Please upload a video file", gr.update(), gr.update()

        try:
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                video_path = Path(video_file)

                scene_files = self.dataset_manager.split_video_scenes(
                    video_path=video_path,
                    output_dir=temp_path,
                    detector_type=detector_type,
                    min_scene_length=int(min_scene_length) if min_scene_length else None,
                    threshold=float(threshold) if threshold else None,
                    filter_shorter_than=filter_shorter if filter_shorter and filter_shorter.strip() else None,
                    max_scenes=int(max_scenes) if max_scenes and max_scenes > 0 else None,
                )

                if not scene_files:
                    return "No scenes detected", gr.update(), gr.update()

                result = self.dataset_manager.add_videos(dataset_name, [str(f) for f in scene_files])

                message = (
                    f"Split into {len(scene_files)} scenes using {detector_type} detector. "
                    f"Added {len(result['added'])} scenes to dataset."
                )
                if result["failed"]:
                    message += f" {len(result['failed'])} failed."

                items = self.dataset_manager.get_dataset_items(dataset_name)
                gallery_items = [
                    (
                        i["thumbnail"],
                        (
                            i.get("caption", "")[:50] + "..."
                            if len(i.get("caption", "")) > 50
                            else i.get("caption", "") or "No caption"
                        ),
                    )
                    for i in items
                    if i["thumbnail"]
                ]
                stats = self.dataset_manager.get_dataset_stats(dataset_name)

                return message, gr.update(value=gallery_items), stats

        except Exception as e:
            logger.error(f"Scene splitting failed: {e}", exc_info=True)
            return f"Error: {e}", gr.update(), gr.update()

    def filter_dataset_gallery(self, dataset_name: str, search_text: str) -> list:
        """Filter gallery by caption search.

        Args:
            dataset_name: Name of the dataset
            search_text: Text to search for in captions

        Returns:
            Filtered gallery items
        """
        if not dataset_name:
            return []

        items = self.dataset_manager.get_dataset_items(dataset_name)

        if search_text and search_text.strip():
            search_lower = search_text.lower()
            items = [i for i in items if search_lower in i.get("caption", "").lower()]

        gallery_items = [
            (
                i["thumbnail"],
                (
                    i.get("caption", "")[:50] + "..."
                    if len(i.get("caption", "")) > 50
                    else i.get("caption", "") or "No caption"
                ),
            )
            for i in items
            if i["thumbnail"]
        ]

        return gallery_items

    def refresh_job_list(self, show_all: bool = False) -> list[list]:
        """Get formatted list of jobs for display.

        Args:
            show_all: If True, show all jobs. If False, show only latest job.

        Returns:
            List of job rows for dataframe display
        """
        jobs = self.job_db.get_all_jobs()

        # If not showing all, only show the most recent job
        if not show_all and jobs:
            jobs = [jobs[0]]  # Jobs are already ordered by ID desc (most recent first)

        # Format for dataframe
        rows = []
        for job in jobs:
            rows.append(
                [
                    job["id"],
                    job["status"],
                    job["dataset_name"],
                    job["created_at"][:19] if job.get("created_at") else "",
                    job.get("progress", "")[:50] if job.get("progress") else "",
                ]
            )

        return rows

    def get_latest_job_id(self) -> int:
        """Get the ID of the latest job.

        Returns:
            Latest job ID or 0 if no jobs
        """
        jobs = self.job_db.get_all_jobs()
        return jobs[0]["id"] if jobs else 0

    def get_running_job(self) -> dict | None:
        """Get the currently running job if any.

        Returns:
            Running job dict or None
        """
        jobs = self.job_db.get_all_jobs()
        for job in jobs:
            if job["status"] == JobStatus.RUNNING:
                return job
        return None

    def get_current_job_display(self) -> tuple[str, str, str | None, str, str]:
        """Get formatted display for the current running job.

        Returns:
            Tuple of (status_html, job_info, validation_sample, validation_prompt, logs)
        """
        running_job = self.get_running_job()

        if not running_job:
            return (
                '<div style="text-align: center; padding: 20px; color: #666;">No job currently running</div>',
                "",
                None,
                "",
                "",
            )

        # Format job info
        job_info = f"""**Job #{running_job["id"]}** - {running_job["dataset_name"]}
**Status:** {running_job["status"]}
**Progress:** {running_job.get("progress", "Starting...")}
**Started:** {running_job.get("started_at", "N/A")[:19] if running_job.get("started_at") else "N/A"}
"""

        # Get validation sample
        validation_sample = running_job.get("validation_sample")
        if validation_sample and not Path(validation_sample).is_absolute():
            validation_sample = str(PROJECT_ROOT / validation_sample)

        # Get validation prompt from job params
        validation_prompt = running_job.get("params", {}).get("validation_prompt", "")

        # Get logs
        logs = running_job.get("logs", "") or "Waiting for logs..."

        # Status HTML with color coding
        status_color = "#2563eb"  # blue for running
        status_html = (
            f'<div style="background: {status_color}15; border-left: 4px solid {status_color}; '
            f'padding: 12px; margin: 8px 0; border-radius: 4px;">'
            f'<strong style="color: {status_color};">▶️ Training in Progress - Job #{running_job["id"]}</strong>'
            f"</div>"
        )

        return status_html, job_info, validation_sample, validation_prompt, logs

    def view_job_details(self, job_id: int | float) -> tuple[int, dict, str | None, str, gr.Accordion]:
        """View job details by ID.

        Args:
            job_id: The job ID to view

        Returns:
            Tuple of (job_id, job_details, validation_sample, logs, accordion_update)
        """
        try:
            if not job_id or job_id == 0:
                return 0, {}, None, "Please enter a valid Job ID", gr.Accordion(open=False)

            job = self.job_db.get_job(int(job_id))
            if job:
                logs = job.get("logs", "") or "No logs available yet"
                validation_sample = job.get("validation_sample")
                # Convert path to absolute if it's a relative path
                if validation_sample and not Path(validation_sample).is_absolute():
                    validation_sample = str(PROJECT_ROOT / validation_sample)
                return int(job_id), job, validation_sample, logs, gr.Accordion(open=True)
            else:
                return int(job_id), {}, None, f"Job #{int(job_id)} not found", gr.Accordion(open=True)
        except (TypeError, ValueError) as e:
            logger.error(f"Error viewing job details: {e}")
            return 0, {}, None, f"Error: {e}", gr.Accordion(open=False)

    def clear_completed_jobs(self) -> tuple[str, list[list]]:
        """Clear all completed, failed, and cancelled jobs.

        Returns:
            Tuple of (status message, updated job list)
        """
        self.job_db.clear_completed_jobs()
        return "✅ Cleared all completed jobs", self.refresh_job_list(show_all=True)

    def stop_worker(self) -> tuple[str, list[list]]:
        """Stop the worker and cancel the current running job.

        Returns:
            Tuple of (status message, updated job list)
        """
        try:
            # First, cancel any running jobs
            jobs = self.job_db.get_all_jobs()
            cancelled_jobs = []
            for job in jobs:
                if job["status"] == JobStatus.RUNNING:
                    self.job_db.update_job_status(
                        job["id"], JobStatus.CANCELLED, error_message="Worker stopped by user"
                    )
                    cancelled_jobs.append(job["id"])

            # Send shutdown signal to worker
            db_path = PROJECT_ROOT / "jobs.db"
            shutdown_signal_file = db_path.parent / ".worker_shutdown_signal"
            shutdown_signal_file.touch()

            if cancelled_jobs:
                msg = f"✅ Cancelled job(s) #{', #'.join(map(str, cancelled_jobs))} and sent shutdown signal to worker."
            else:
                msg = "✅ Shutdown signal sent to worker. No running jobs to cancel."

            return msg, self.refresh_job_list(show_all=True)
        except Exception as e:
            logger.error(f"Error stopping worker: {e}")
            return f"❌ Error: {e}", self.refresh_job_list(show_all=True)

    def start_worker(self) -> str:
        """Start the worker process.

        Returns:
            Status message
        """
        try:
            import subprocess
            import sys

            # Check if already running
            status = self.check_worker_status()
            if "running" in status.lower():
                return f"⚠️ Worker is already running\n{status}"

            # Remove shutdown signal if it exists
            shutdown_signal_file = PROJECT_ROOT / ".worker_shutdown_signal"
            if shutdown_signal_file.exists():
                shutdown_signal_file.unlink()

            # Start worker process
            worker_script = PROJECT_ROOT / "scripts" / "jobs" / "run_worker.py"
            python_exe = sys.executable

            subprocess.Popen(
                [python_exe, str(worker_script)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent
            )

            return "✅ Worker started successfully"
        except Exception as e:
            logger.error(f"Error starting worker: {e}")
            return f"❌ Error starting worker: {e}"

    def check_worker_status(self) -> str:
        """Check if worker is running.

        Returns:
            Worker status message
        """
        try:
            import psutil

            # Look for worker process
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline", [])
                    if cmdline and "run_worker.py" in " ".join(cmdline):
                        return f"✅ Worker is running (PID: {proc.info['pid']})"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return "⚠️ Worker is not running. Start it with: python scripts/jobs/run_worker.py"
        except ImportError:
            return "💡 Install psutil to check worker status: pip install psutil"
        except Exception as e:
            return f"❓ Could not determine worker status: {e}"

    def create_ui(self) -> gr.Blocks:  # noqa: PLR0915
        """Create the Gradio UI."""
        with gr.Blocks() as blocks:
            gr.Markdown("# LTX-Video Trainer")

            with gr.Tab("Training"):
                gr.Markdown("# 🎬 Train LTX-Video LoRA")

                # Dataset Selection
                with gr.Group():
                    gr.Markdown("### 📁 Dataset")
                    dataset_for_training = gr.Dropdown(
                        choices=self.dataset_manager.list_datasets(),
                        label="Select Dataset",
                        interactive=True,
                        info="Choose a dataset you created in the Datasets tab",
                    )

                # Training Configuration
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### ⚙️ Training Parameters")
                            model_source = gr.Dropdown(
                                choices=[str(v) for v in LtxvModelVersion],
                                value=str(LtxvModelVersion.latest()),
                                label="Model Version",
                                info="Select the model version to use for training",
                            )

                            with gr.Row():
                                lr = gr.Number(value=2e-4, label="Learning Rate", scale=1)
                                steps = gr.Number(
                                    value=1500,
                                    label="Training Steps",
                                    precision=0,
                                    info="Total training steps",
                                    scale=1,
                                )

                            with gr.Row():
                                lora_rank = gr.Dropdown(
                                    choices=list(range(8, 257, 8)),
                                    value=128,
                                    label="LoRA Rank",
                                    info="Higher = more capacity",
                                    scale=1,
                                )
                                batch_size = gr.Number(value=1, label="Batch Size", precision=0, scale=1)

                            validation_interval = gr.Number(
                                value=100,
                                label="Validation Interval",
                                precision=0,
                                info="Steps between validation samples",
                                minimum=1,
                            )

                        with gr.Group():
                            gr.Markdown("### 🎞️ Video Settings")
                            id_token = gr.Textbox(
                                label="LoRA ID Token",
                                placeholder="Optional: e.g., <lora> or <mytoken>",
                                value="",
                                info="Prepended to training captions",
                            )

                            with gr.Row():
                                width = gr.Dropdown(
                                    choices=list(range(256, 1025, 32)),
                                    value=768,
                                    label="Width",
                                    info="Multiple of 32",
                                )
                                height = gr.Dropdown(
                                    choices=list(range(256, 1025, 32)),
                                    value=768,
                                    label="Height",
                                    info="Multiple of 32",
                                )
                                num_frames = gr.Dropdown(
                                    choices=list(range(9, 129, 8)),
                                    value=25,
                                    label="Frames",
                                    info="Multiple of 8",
                                )

                    with gr.Column(scale=1), gr.Group():
                        # Training Monitor
                        gr.Markdown("### 📊 Training Monitor")

                        with gr.Row():
                            train_btn = gr.Button("▶️ Start Training", variant="primary", size="lg", scale=2)
                            reset_btn = gr.Button("🔄 Reset", variant="secondary", size="lg", scale=1)

                        current_job_display = gr.Markdown(value="", visible=True, label="Current Job")

                        self.status_output = gr.Textbox(label="Status", interactive=False)
                        self.progress_output = gr.Textbox(label="Progress", interactive=False)

                        gr.Markdown("**Validation Sample**")
                        self.validation_prompt = gr.Textbox(
                            label="Validation Prompt",
                            placeholder="Enter the prompt to use for validation samples",
                            value="a professional portrait video of a person with blurry bokeh background",
                            interactive=True,
                            info="Prompt used to generate sample videos during training (view in Jobs tab)",
                            lines=2,
                        )

                # Advanced Settings (Collapsible)
                with gr.Accordion("🚀 HuggingFace Hub (Optional)", open=False):
                    push_to_hub = gr.Checkbox(
                        label="Push to HuggingFace Hub",
                        value=False,
                        info="Automatically upload trained model to HuggingFace Hub",
                    )
                    with gr.Row():
                        hf_token = gr.Textbox(
                            label="HuggingFace Token",
                            type="password",
                            placeholder="hf_...",
                            info="Your HuggingFace API token",
                        )
                        hf_model_id = gr.Textbox(
                            label="Model ID",
                            placeholder="username/model-name",
                            info="Repository name on HuggingFace",
                        )

                # Results Section
                with gr.Group():
                    gr.Markdown("### ✅ Training Results")
                    with gr.Row():
                        self.download_btn = gr.DownloadButton(
                            label="📥 Download LoRA Weights",
                            visible=False,
                            interactive=True,
                            size="lg",
                        )
                        self.hf_repo_link = gr.HTML(
                            value="",
                            visible=False,
                            label="HuggingFace Hub",
                        )

                # Jobs Queue Link
                gr.Markdown("---")  # Separator
                gr.Markdown("### 📋 Training Queue Status")
                gr.Markdown(
                    "💡 **Tip:** Switch to the **Jobs** tab to monitor training progress, "
                    "view logs, and manage the queue."
                )

            # Jobs Tab
            with gr.Tab("Jobs"):
                gr.Markdown("# 📋 Training Jobs")

                # Worker Status Section
                with gr.Group():
                    gr.Markdown("## 🔧 Worker Status")
                    with gr.Row():
                        check_worker_btn = gr.Button("🔍 Check Status", size="sm", scale=1)
                        stop_worker_btn = gr.Button("🛑 Stop Worker", variant="stop", size="sm", scale=1)

                    queue_status = gr.Textbox(
                        label="",
                        interactive=False,
                        show_label=False,
                        container=True,
                        placeholder="Worker status will appear here...",
                    )

                gr.Markdown("---")

                # Current Running Job Section
                with gr.Group():
                    gr.Markdown("## ▶️ Current Training Job")

                    current_job_status = gr.HTML(
                        value=(
                            '<div style="text-align: center; padding: 20px; color: #666;">'
                            "No job currently running</div>"
                        )
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            current_job_info = gr.Markdown(value="")

                            gr.Markdown("#### 🎬 Latest Validation Sample")
                            current_job_sample = gr.Video(
                                label="",
                                show_label=False,
                                interactive=False,
                                autoplay=True,
                                loop=True,
                                height=300,
                            )

                            gr.Markdown("#### 💬 Validation Prompt")
                            current_job_prompt = gr.Textbox(
                                label="",
                                show_label=False,
                                interactive=False,
                                lines=2,
                                placeholder="Validation prompt will appear here...",
                            )

                        with gr.Column(scale=3):
                            gr.Markdown("#### 📝 Training Logs (Live)")
                            current_job_logs = gr.Textbox(
                                label="",
                                lines=25,
                                max_lines=30,
                                interactive=False,
                                show_copy_button=True,
                                placeholder="Logs will appear here when training starts...",
                                autoscroll=True,
                            )

                gr.Markdown("---")

                # Job History Section
                with gr.Group():
                    gr.Markdown("## 📚 Job History")

                    with gr.Row():
                        refresh_jobs_btn = gr.Button("🔄 Refresh Jobs", size="sm")
                        clear_completed_btn = gr.Button("🗑️ Clear Completed", size="sm", variant="secondary")

                    job_table = gr.Dataframe(
                        headers=["ID", "Status", "Dataset", "Created", "Progress"],
                        datatype=["number", "str", "str", "str", "str"],
                        label="All Jobs",
                        interactive=False,
                        wrap=True,
                        row_count=10,
                    )

                    # Manual job selection for viewing older jobs
                    with gr.Accordion("🔍 View Specific Job", open=False):
                        with gr.Row():
                            selected_job_id = gr.Number(
                                label="Job ID",
                                precision=0,
                                value=0,
                                interactive=True,
                                scale=1,
                            )
                            view_job_btn = gr.Button("View", size="sm", scale=1)

                        selected_job_details = gr.JSON(label="Job Configuration", value={})

                        gr.Markdown("#### Validation Sample")
                        selected_job_sample = gr.Video(
                            label="",
                            show_label=False,
                            interactive=False,
                            autoplay=True,
                            loop=True,
                            height=250,
                        )

                        gr.Markdown("#### Logs")
                        selected_job_logs = gr.Textbox(
                            label="",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                        )

                with gr.Accordion("💡 Worker Information", open=False):
                    gr.Markdown(
                        "### Worker Management\n\n"
                        "The worker processes training jobs from the queue.\n\n"
                        "**To start:** Click the '▶️ Start Worker' button above, or run:\n"
                        "```bash\n"
                        "python scripts/jobs/run_worker.py\n"
                        "```\n\n"
                        "**To stop:** Click the '🛑 Stop Worker' button above.\n\n"
                        "**To check status:** Click the '🔍 Check Worker Status' button."
                    )

                    gr.Textbox(
                        label="Worker Status",
                        interactive=False,
                        placeholder="Click 'Check Worker Status' to see if worker is running...",
                    )

                # Event handlers

                # Auto-refresh current job on timer
                def auto_refresh_current_job() -> tuple[str, str, str | None, str, str, list, str]:
                    """Auto-refresh the current running job display and check worker status."""
                    status_html, job_info, validation_sample, validation_prompt, logs = self.get_current_job_display()
                    job_list = self.refresh_job_list(show_all=True)
                    worker_status = self.check_worker_status()
                    return status_html, job_info, validation_sample, validation_prompt, logs, job_list, worker_status

                job_refresh_timer = gr.Timer(value=5)
                job_refresh_timer.tick(
                    fn=auto_refresh_current_job,
                    outputs=[
                        current_job_status,
                        current_job_info,
                        current_job_sample,
                        current_job_prompt,
                        current_job_logs,
                        job_table,
                        queue_status,
                    ],
                    show_progress=False,
                )

                # Manual refresh
                refresh_jobs_btn.click(
                    auto_refresh_current_job,
                    outputs=[
                        current_job_status,
                        current_job_info,
                        current_job_sample,
                        current_job_prompt,
                        current_job_logs,
                        job_table,
                        queue_status,
                    ],
                )

                # Clear completed jobs
                clear_completed_btn.click(
                    self.clear_completed_jobs,
                    outputs=[queue_status, job_table],
                )

                # Worker control
                check_worker_btn.click(
                    self.check_worker_status,
                    outputs=[queue_status],
                )

                stop_worker_btn.click(
                    self.stop_worker,
                    outputs=[queue_status, job_table],
                )

                # View specific job
                def view_specific_job(job_id: int | float) -> tuple[dict, str | None, str]:
                    if not job_id or job_id == 0:
                        return {}, None, "Please enter a valid Job ID"
                    job_id_int, details, sample, logs, _ = self.view_job_details(job_id)
                    return details, sample, logs

                view_job_btn.click(
                    view_specific_job,
                    inputs=[selected_job_id],
                    outputs=[selected_job_details, selected_job_sample, selected_job_logs],
                )

                selected_job_id.submit(
                    view_specific_job,
                    inputs=[selected_job_id],
                    outputs=[selected_job_details, selected_job_sample, selected_job_logs],
                )

            # Datasets Tab
            with gr.Tab("Datasets"), gr.Row():
                with gr.Column(scale=1):
                    # Dataset selection/creation
                    gr.Markdown("## Manage Datasets")

                    dataset_dropdown = gr.Dropdown(
                        choices=self.dataset_manager.list_datasets(),
                        label="Select Dataset",
                        interactive=True,
                    )

                    with gr.Row():
                        new_dataset_name = gr.Textbox(label="New Dataset Name", placeholder="my_dataset")
                        create_dataset_btn = gr.Button("Create Dataset", variant="primary")

                    # Dataset statistics
                    stats_box = gr.JSON(label="Dataset Statistics", value={})

                    # Captioning settings
                    gr.Markdown("### Caption Settings")

                    with gr.Accordion("🎨 VLM Captioning Options", open=False):
                        vlm_instruction = gr.Textbox(
                            label="Custom VLM Instruction",
                            placeholder="Leave empty to use default instruction",
                            value="",
                            lines=3,
                            info="Custom instruction for the vision-language model when generating captions",
                        )
                        vlm_use_8bit = gr.Checkbox(
                            label="Use 8-bit Quantization",
                            value=True,
                            info="Reduces VRAM usage during captioning",
                        )

                    # Batch operations
                    gr.Markdown("### Batch Operations")

                    with gr.Row():
                        auto_caption_btn = gr.Button("Auto-Caption All Uncaptioned", size="sm")
                        validate_btn = gr.Button("Validate Dataset", size="sm")

                    validation_result = gr.JSON(label="Validation Results", value={})

                    # Preprocessing options
                    gr.Markdown("### Preprocessing Options")

                    with gr.Accordion("⚙️ Advanced Preprocessing", open=False):
                        decode_videos_check = gr.Checkbox(
                            label="Decode Videos for Verification",
                            value=False,
                            info="Decode preprocessed latents to verify quality (slower, uses more disk space)",
                        )
                        load_text_encoder_8bit = gr.Checkbox(
                            label="Load Text Encoder in 8-bit",
                            value=False,
                            info="Reduces VRAM usage during text embedding computation",
                        )

                    with gr.Row():
                        preprocess_btn = gr.Button("Preprocess Dataset", variant="primary")

                    preprocess_status = gr.Textbox(label="Preprocessing Status", interactive=False)

                with gr.Column(scale=3):
                    # Video upload area
                    gr.Markdown("## Upload Videos")

                    video_uploader = gr.File(
                        label="Drag and drop videos here",
                        file_count="multiple",
                        file_types=["video"],
                        type="filepath",
                    )

                    upload_btn = gr.Button("Add to Dataset")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)

                    # IC-LoRA reference videos
                    with gr.Accordion("🎯 IC-LoRA: Add Reference Videos (Optional)", open=False):
                        gr.Markdown(
                            "**IC-LoRA Training:** Upload reference videos paired with target videos "
                            "for advanced video-to-video transformations. "
                            "Reference videos must match target videos in count, resolution, and length."
                        )
                        reference_uploader = gr.File(
                            label="Reference Videos (for IC-LoRA) - Upload in same order as target videos",
                            file_count="multiple",
                            file_types=["video"],
                            type="filepath",
                        )
                        upload_with_ref_btn = gr.Button("Add Videos with References", variant="secondary")

                    # Scene splitting
                    with gr.Accordion("✂️ Split Long Videos into Scenes", open=False):
                        gr.Markdown(
                            "Upload a long video and automatically split it into individual scenes "
                            "using advanced detection algorithms"
                        )
                        scene_video_uploader = gr.File(
                            label="Video to Split",
                            file_count="single",
                            file_types=["video"],
                            type="filepath",
                        )
                        with gr.Row():
                            scene_detector_type = gr.Dropdown(
                                choices=["content", "adaptive", "threshold", "histogram"],
                                value="content",
                                label="Detection Algorithm",
                                info=(
                                    "Content: HSV color changes | Adaptive: Two-phase cuts | "
                                    "Threshold: Fades | Histogram: YUV changes"
                                ),
                            )
                            scene_threshold = gr.Number(
                                label="Detection Threshold",
                                value=27.0,
                                info="Lower = more sensitive",
                            )
                        with gr.Row():
                            scene_min_length = gr.Number(
                                label="Min Scene Length (frames)",
                                value=30,
                                precision=0,
                                info="Minimum scene duration during detection",
                            )
                            scene_filter_shorter = gr.Textbox(
                                label="Filter Shorter Than",
                                value="2s",
                                info="Filter scenes shorter than duration (e.g., '2s', '60' frames)",
                            )
                            scene_max_scenes = gr.Number(
                                label="Max Scenes (0=unlimited)",
                                value=0,
                                precision=0,
                                info="Maximum number of scenes to produce",
                            )
                        split_scenes_btn = gr.Button("Split Scenes and Add to Dataset", variant="primary")

                    # Visual dataset browser
                    gr.Markdown("## Dataset Browser")

                    with gr.Row():
                        search_box = gr.Textbox(
                            label="Search captions", placeholder="Filter by caption text...", scale=4
                        )
                        refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=1)

                    # Video gallery with captions
                    dataset_gallery = gr.Gallery(
                        label="Videos",
                        columns=4,
                        height="auto",
                        object_fit="contain",
                        allow_preview=True,
                    )

                    # Selected video editor
                    gr.Markdown("### Edit Selected Video")

                    with gr.Row():
                        selected_video = gr.Video(label="Preview", interactive=False)

                        with gr.Column():
                            selected_video_name = gr.Textbox(label="Video Name", interactive=False)

                            caption_editor = gr.Textbox(
                                label="Caption",
                                placeholder="Enter caption for this video...",
                                lines=3,
                                interactive=True,
                            )

                            with gr.Row():
                                save_caption_btn = gr.Button("Save Caption", variant="primary")
                                generate_caption_btn = gr.Button("Auto-Generate")
                                delete_video_btn = gr.Button("Delete Video", variant="stop")

            # Event handlers
            # Update HF fields visibility based on push_to_hub checkbox
            push_to_hub.change(
                lambda x: {
                    hf_token: gr.update(visible=x),
                    hf_model_id: gr.update(visible=x),
                },
                inputs=[push_to_hub],
                outputs=[hf_token, hf_model_id],
            )

            train_btn.click(
                lambda dataset_name,
                validation_prompt,
                lr,
                steps,
                lora_rank,
                batch_size,
                model_source,
                width,
                height,
                num_frames,
                push_to_hub,
                hf_model_id,
                hf_token,
                id_token,
                validation_interval: self.start_training(
                    TrainingParams(
                        dataset_name=dataset_name,
                        validation_prompt=validation_prompt,
                        learning_rate=lr,
                        steps=steps,
                        lora_rank=lora_rank,
                        batch_size=batch_size,
                        model_source=model_source,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        push_to_hub=push_to_hub,
                        hf_model_id=hf_model_id,
                        hf_token=hf_token,
                        id_token=id_token,
                        validation_interval=validation_interval,
                    )
                ),
                inputs=[
                    dataset_for_training,
                    self.validation_prompt,
                    lr,
                    steps,
                    lora_rank,
                    batch_size,
                    model_source,
                    width,
                    height,
                    num_frames,
                    push_to_hub,
                    hf_model_id,
                    hf_token,
                    id_token,
                    validation_interval,
                ],
                outputs=[self.status_output, train_btn],
            )

            # Update timer to use class method
            timer = gr.Timer(value=10)  # 10 second interval
            timer.tick(
                fn=self.update_progress,
                inputs=None,
                outputs=[
                    current_job_display,
                    self.status_output,
                    self.download_btn,
                    self.hf_repo_link,
                    self.hf_repo_link,
                ],
                show_progress=False,
            )

            # Handle download button click
            self.download_btn.click(self.get_model_path, inputs=None, outputs=[self.download_btn])

            # Handle reset button click
            reset_btn.click(
                self.reset_interface,
                inputs=None,
                outputs=[
                    self.validation_prompt,
                    self.status_output,
                    self.progress_output,
                    self.download_btn,
                    self.hf_repo_link,
                ],
            )

            # Dataset event handlers
            create_dataset_btn.click(
                self.create_new_dataset,
                inputs=[new_dataset_name],
                outputs=[
                    upload_status,
                    dataset_dropdown,
                    dataset_gallery,
                    stats_box,
                    selected_video,
                    selected_video_name,
                    caption_editor,
                    dataset_for_training,  # Update training tab dropdown
                ],
            )

            dataset_dropdown.change(
                self.load_dataset,
                inputs=[dataset_dropdown],
                outputs=[
                    dataset_dropdown,
                    dataset_gallery,
                    stats_box,
                    selected_video,
                    selected_video_name,
                    caption_editor,
                ],
            )

            upload_btn.click(
                self.upload_videos_to_dataset,
                inputs=[video_uploader, dataset_dropdown],
                outputs=[upload_status, dataset_gallery, stats_box],
            )

            upload_with_ref_btn.click(
                self.upload_videos_with_references,
                inputs=[video_uploader, reference_uploader, dataset_dropdown],
                outputs=[upload_status, dataset_gallery, stats_box],
            )

            split_scenes_btn.click(
                self.split_scenes_and_add,
                inputs=[
                    dataset_dropdown,
                    scene_video_uploader,
                    scene_detector_type,
                    scene_min_length,
                    scene_threshold,
                    scene_filter_shorter,
                    scene_max_scenes,
                ],
                outputs=[upload_status, dataset_gallery, stats_box],
            )

            dataset_gallery.select(
                self.select_video_from_gallery,
                inputs=[dataset_dropdown],
                outputs=[selected_video, selected_video_name, caption_editor],
            )

            save_caption_btn.click(
                self.save_caption_edit,
                inputs=[dataset_dropdown, selected_video_name, caption_editor],
                outputs=[upload_status, stats_box],
            )

            generate_caption_btn.click(
                self.auto_caption_single,
                inputs=[dataset_dropdown, selected_video_name, vlm_instruction, vlm_use_8bit],
                outputs=[caption_editor, upload_status],
            )

            auto_caption_btn.click(
                self.auto_caption_all_uncaptioned,
                inputs=[dataset_dropdown, vlm_instruction, vlm_use_8bit],
                outputs=[preprocess_status, stats_box],
            )

            validate_btn.click(self.validate_dataset_ui, inputs=[dataset_dropdown], outputs=[validation_result])

            preprocess_btn.click(
                self.preprocess_dataset_ui,
                inputs=[
                    dataset_dropdown,
                    width,
                    height,
                    num_frames,
                    id_token,
                    decode_videos_check,
                    load_text_encoder_8bit,
                ],
                outputs=[preprocess_status, stats_box],
            )

            refresh_btn.click(
                self.load_dataset,
                inputs=[dataset_dropdown],
                outputs=[
                    dataset_dropdown,
                    dataset_gallery,
                    stats_box,
                    selected_video,
                    selected_video_name,
                    caption_editor,
                ],
            )

            delete_video_btn.click(
                self.delete_video_from_dataset,
                inputs=[dataset_dropdown, selected_video_name],
                outputs=[
                    upload_status,
                    dataset_gallery,
                    selected_video,
                    selected_video_name,
                    caption_editor,
                    stats_box,
                ],
            )

            search_box.change(
                self.filter_dataset_gallery, inputs=[dataset_dropdown, search_box], outputs=[dataset_gallery]
            )

        return blocks


def main() -> None:
    """Main entry point."""
    ui = GradioUI()
    demo = ui.create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
