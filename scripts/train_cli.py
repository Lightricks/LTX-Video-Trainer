#!/usr/bin/env python3
"""
Command-line training script that integrates with the job queue system.

This script can be called directly with parameters or via the queue worker.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from huggingface_hub import login

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.trainer import LtxvTrainer
from scripts.jobs.database import JobDatabase, JobStatus


def generate_config(args: argparse.Namespace, output_dir: Path) -> dict:
    """Generate training configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments
        output_dir: Output directory for training

    Returns:
        Configuration dictionary
    """
    config = {
        "model": {
            "model_source": args.model_version,
            "training_mode": "lora",
            "load_checkpoint": None,
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_rank,  # Usually alpha = rank
            "dropout": 0.0,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "conditioning": {
            "mode": "none",
            "first_frame_conditioning_p": 0.1,
        },
        "optimization": {
            "learning_rate": args.learning_rate,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw",
            "scheduler_type": "linear",
            "scheduler_params": {},
            "enable_gradient_checkpointing": False,
        },
        "acceleration": {
            "mixed_precision_mode": "bf16",
            "quantization": None,
            "load_text_encoder_in_8bit": True,
            "compile_with_inductor": False,
            "compilation_mode": "reduce-overhead",
        },
        "data": {
            "preprocessed_data_root": str(args.dataset_dir),
            "num_dataloader_workers": 2,
        },
        "validation": {
            "prompts": [args.validation_prompt],
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
            "images": None,
            "reference_videos": None,
            "video_dims": [args.width, args.height, args.num_frames],
            "seed": 42,
            "inference_steps": 30,
            "interval": args.validation_interval,
            "videos_per_prompt": 1,
            "guidance_scale": 3.5,
            "skip_initial_validation": False,
        },
        "checkpoints": {
            "interval": None,
            "keep_last_n": 1,
        },
        "hub": {
            "push_to_hub": args.push_to_hub,
            "hub_model_id": args.hf_model_id if args.push_to_hub else None,
        },
        "flow_matching": {
            "timestep_sampling_mode": "shifted_logit_normal",
            "timestep_sampling_params": {},
        },
        "wandb": {
            "enabled": False,
            "project": "ltxv-trainer",
            "entity": None,
            "tags": [],
            "log_validation_videos": True,
        },
        "seed": 42,
        "output_dir": str(output_dir),
    }

    return config


def setup_dataset(dataset_name: str, datasets_root: Path, training_data_dir: Path) -> None:
    """Set up the training dataset by copying from managed datasets.

    Args:
        dataset_name: Name of the managed dataset
        datasets_root: Root directory of managed datasets
        training_data_dir: Temporary training data directory
    """
    managed_dataset_dir = datasets_root / dataset_name
    managed_dataset_json = managed_dataset_dir / "dataset.json"

    if not managed_dataset_json.exists():
        raise ValueError(f"Dataset '{dataset_name}' not found")

    if training_data_dir.exists():
        shutil.rmtree(training_data_dir)
    training_data_dir.mkdir(parents=True)

    shutil.copy2(managed_dataset_json, training_data_dir / "dataset.json")

    precomputed_dir = managed_dataset_dir / ".precomputed"
    if precomputed_dir.exists():
        shutil.copytree(precomputed_dir, training_data_dir / ".precomputed")


def train_with_progress_callback(trainer: LtxvTrainer, db: JobDatabase, job_id: int) -> None:
    """Train with progress updates to database.

    Args:
        trainer: Trainer instance
        db: Job database
        job_id: Current job ID
    """

    def progress_callback(step: int, total_steps: int, sampled_videos: list[Path] | None = None) -> None:
        """Update job progress in database."""
        progress_pct = (step / total_steps) * 100
        progress_msg = f"Step {step}/{total_steps} ({progress_pct:.1f}%)"

        validation_sample = None
        if sampled_videos:
            validation_sample = str(sampled_videos[0])

        db.update_job_status(job_id, JobStatus.RUNNING, progress=progress_msg, validation_sample=validation_sample)

    trainer.train(step_callback=progress_callback)


def main() -> int | None:
    """Main entry point for CLI training."""
    parser = argparse.ArgumentParser(description="Train LTX-Video LoRA model")

    parser.add_argument("--job-id", type=int, help="Job ID for queue tracking")

    parser.add_argument("--dataset", required=True, help="Dataset name")

    parser.add_argument("--model-version", default="0.9.1", help="Model version")

    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=1500, help="Training steps")
    parser.add_argument("--lora-rank", type=int, default=128, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    parser.add_argument("--width", type=int, default=768, help="Video width")
    parser.add_argument("--height", type=int, default=768, help="Video height")
    parser.add_argument("--num-frames", type=int, default=25, help="Number of frames")
    parser.add_argument("--id-token", type=str, default="", help="LoRA ID token")

    parser.add_argument(
        "--validation-prompt",
        default="a professional portrait video of a person",
        help="Validation prompt",
    )
    parser.add_argument("--validation-interval", type=int, default=100, help="Validation interval")

    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hf-model-id", type=str, help="HuggingFace model ID")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    datasets_root = project_root / "datasets"
    training_data_dir = (
        project_root / "training_data" / f"job_{args.job_id}" if args.job_id else project_root / "training_data"
    )
    output_dir = (
        project_root / "outputs" / f"lora_r{args.lora_rank}_job{args.job_id}"
        if args.job_id
        else project_root / "outputs" / f"lora_r{args.lora_rank}"
    )

    db = None
    if args.job_id:
        db_path = project_root / "jobs.db"
        db = JobDatabase(db_path)

    try:
        if "HF_TOKEN" in os.environ:
            login(token=os.environ["HF_TOKEN"])

        args.dataset_dir = training_data_dir
        setup_dataset(args.dataset, datasets_root, training_data_dir)

        config = generate_config(args, output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, indent=2)

        trainer_config = LtxvTrainerConfig(**config)
        trainer = LtxvTrainer(trainer_config)

        if db and args.job_id:
            train_with_progress_callback(trainer, db, args.job_id)
        else:
            trainer.train()

        return 0

    except Exception as e:
        if db and args.job_id:
            db.update_job_status(args.job_id, JobStatus.FAILED, error_message=str(e))
        raise

    finally:
        if db:
            db.close()


if __name__ == "__main__":
    sys.exit(main())
