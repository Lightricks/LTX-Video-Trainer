import itertools
import os
import random
import time
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import MagicMock

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
from huggingface_hub import create_repo, upload_folder
from loguru import logger
from peft import LoraConfig, get_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Group,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from ltxv_trainer.config import LtxvTrainerConfig
from ltxv_trainer.datasets import PrecomputedDataset
from ltxv_trainer.model_loader import load_ltxv_components
from ltxv_trainer.quantization import quantize_model
from ltxv_trainer.timestep_samplers import SAMPLERS
from ltxv_trainer.utils import get_gpu_memory_gb

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

COMPILE_WARMUP_STEPS = 5


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    compilation_time_seconds: Optional[float]  # Only if compile_with_inductor=True
    training_time: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float


class LtxvTrainer:
    def __init__(self, trainer_config: LtxvTrainerConfig) -> None:
        self._config = trainer_config
        self._console = Console()
        self._print_config(trainer_config)
        self._setup_accelerator()
        self._load_models()
        self._compile_transformer()
        self._collect_trainable_params()
        self._global_step = -1
        self._checkpoint_paths = []

    def train(  # noqa: PLR0912, PLR0915
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback = None,
    ) -> tuple[Path, TrainingStats]:
        """
        Start the training process.
        Returns:
            Tuple of (saved_model_path, training_stats)
        """
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()
        set_seed(cfg.seed)

        if cfg.model.training_mode == "lora":
            self._init_lora_weights()

        self._init_optimizer()
        self._init_dataloader()
        self._init_timestep_sampler()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("🚀 Starting training...")

        # Create progress columns with simplified styling
        if disable_progress_bars:
            train_progress = MagicMock()
            sample_progress = MagicMock()
            logger.warning("Progress bars disabled. Status messages will be printed occasionally instead.")
        else:
            train_progress = Progress(
                TextColumn("Training Step"),
                MofNCompleteColumn(),
                BarColumn(bar_width=40, style="blue"),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TextColumn("Time/Step: {task.fields[step_time]:.2f}s"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(compact=True),
            )

            # Create a separate progress instance for sampling
            sample_progress = Progress(
                TextColumn("Sampling validation videos"),
                MofNCompleteColumn(),
                BarColumn(bar_width=40, style="blue"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(compact=True),
            )

        self._transformer.train()
        self._global_step = 0

        # For tracking compilation time
        compilation_time = None
        peak_mem_during_training = start_mem

        # Track when actual training starts (after compilation)
        actual_training_start = None

        with Live(Panel(Group(train_progress, sample_progress)), refresh_per_second=2):
            task = train_progress.add_task(
                "Training",
                total=cfg.optimization.steps,
                loss=0.0,
                lr=cfg.optimization.learning_rate,
                step_time=0.0,
            )

            if cfg.validation.interval:
                self._sample_videos(sample_progress)

            for step, batch in enumerate(self._data_loader):
                # Measure compilation time (first COMPILE_WARMUP_STEPS steps)
                if step == COMPILE_WARMUP_STEPS and cfg.acceleration.compile_with_inductor:
                    compilation_time = time.time() - train_start_time
                    actual_training_start = time.time()
                elif step == COMPILE_WARMUP_STEPS and not cfg.acceleration.compile_with_inductor:
                    actual_training_start = train_start_time

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % cfg.optimization.gradient_accumulation_steps == 0
                    if is_optimization_step:
                        self._global_step += 1

                    loss = self._training_step(batch)
                    self._accelerator.backward(loss)

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                        and self._accelerator.is_main_process
                    ):
                        sampled_videos_paths = self._sample_videos(sample_progress)

                    # Save checkpoint if needed
                    if (
                        cfg.checkpoints.interval
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and is_optimization_step
                        and self._accelerator.is_main_process
                    ):
                        self._save_checkpoint()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, sampled_videos_paths)

                    self._accelerator.wait_for_everyone()

                    # Update progress
                    if self._accelerator.is_main_process:
                        current_lr = self._optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - train_start_time
                        progress_percentage = self._global_step / cfg.optimization.steps
                        if progress_percentage > 0:
                            total_estimated = elapsed / progress_percentage
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."

                        step_time = time.time() - step_start_time
                        train_progress.update(
                            task,
                            advance=1,
                            loss=loss.item(),
                            lr=current_lr,
                            step_time=step_time,
                            total_time=total_time,
                        )
                        if disable_progress_bars and self._global_step % 20 == 0:
                            logger.info(
                                f"Step {self._global_step}/{cfg.optimization.steps} - "
                                f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, "
                                f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                            )

                    # Sample GPU memory periodically (every 25 steps)
                    if step % 25 == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

                    if self._global_step >= cfg.optimization.steps:
                        break

            # Collect final stats
            train_end_time = time.time()
            end_mem = get_gpu_memory_gb(device)
            peak_mem = max(start_mem, end_mem, peak_mem_during_training)

            # Calculate steps/second excluding compilation time if needed
            if cfg.acceleration.compile_with_inductor:
                training_time = train_end_time - actual_training_start
                steps_per_second = (cfg.optimization.steps - COMPILE_WARMUP_STEPS) / training_time
            else:
                training_time = train_end_time - train_start_time
                steps_per_second = cfg.optimization.steps / training_time

            samples_per_second = steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size

            stats = TrainingStats(
                total_time_seconds=train_end_time - train_start_time,
                training_time=training_time,
                compilation_time_seconds=compilation_time,
                steps_per_second=steps_per_second,
                samples_per_second=samples_per_second,
                peak_gpu_memory_gb=peak_mem,
            )

            train_progress.remove_task(task)
            self._accelerator.end_training()

            if self._accelerator.is_main_process:
                saved_path = self._save_checkpoint()

                # Upload artifacts to hub if enabled
                if cfg.hub.push_to_hub:
                    repo_id = cfg.hub.hub_model_id or Path(cfg.output_dir).name
                    repo_id = create_repo(token=cfg.hub.hub_token, repo_id=repo_id, exist_ok=True)
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=Path(self._config.output_dir),
                    )

                # Log the training statistics
                self._log_training_stats(stats)

            return saved_path, stats

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> Tensor:
        """Perform a single training step."""

        # Get pre-encoded latents
        latent_conditions = batch["latent_conditions"]
        latent_conditions["latents"].squeeze_(1)
        packed_latents = latent_conditions["latents"]
        latent_frames = latent_conditions["num_frames"][0].item()
        latent_height = latent_conditions["height"][0].item()
        latent_width = latent_conditions["width"][0].item()

        # Get pre-encoded text conditions
        text_conditions = batch["text_conditions"]
        prompt_embeds = text_conditions["prompt_embeds"]
        prompt_attention_mask = text_conditions["prompt_attention_mask"]

        sigmas = self._timestep_sampler.sample_for(packed_latents)
        timesteps = torch.round(sigmas * 1000.0).long()

        noise = torch.randn_like(packed_latents, device=self._accelerator.device)
        sigmas = sigmas.view(-1, 1, 1)

        loss_mask = torch.ones_like(packed_latents)
        # If first frame conditioning is enabled, the first latent (first video frame) is left (almost) unchanged.
        if (
            self._config.optimization.first_frame_conditioning_p
            and random.random() < self._config.optimization.first_frame_conditioning_p
        ):
            sigmas = sigmas.repeat(1, packed_latents.shape[1], 1)
            first_frame_end_idx = latent_height * latent_width
            sigmas[:, :first_frame_end_idx] = 1e-5  # Small sigma close to 0 for the first frame.
            loss_mask[:, :first_frame_end_idx] = 0.0  # Mask out the loss for the first frame.
            # TODO: the `timesteps` fed to the transformer should be
            #  adjusted to reflect zero noise level for the first latent.

        noisy_latents = (1 - sigmas) * packed_latents + sigmas * noise
        targets = noise - packed_latents

        # TODO: fetch from data loader
        latent_frame_rate = 24 / 8
        spatial_compression_ratio = 32
        rope_interpolation_scale = [1 / latent_frame_rate, spatial_compression_ratio, spatial_compression_ratio]

        model_pred = self._transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            rope_interpolation_scale=rope_interpolation_scale,
            return_dict=False,
        )[0]

        loss = (model_pred - targets).pow(2)
        loss = loss.mul(loss_mask).div(loss_mask.mean())  # divide by mean to keep the loss scale unchanged.
        return loss.mean()

    def _print_config(self, config: BaseModel) -> None:
        """Print the configuration as a nicely formatted table."""

        from rich.table import Table

        table = Table(title="⚙️ Training Configuration", show_header=True, header_style="bold green")
        table.add_column("Parameter", style="white")
        table.add_column("Value", style="cyan")

        def flatten_config(cfg: BaseModel, prefix: str = "") -> list[tuple[str, str]]:
            rows = []
            for field, value in cfg:
                full_field = f"{prefix}.{field}" if prefix else field
                if isinstance(value, BaseModel):
                    # Recursively flatten nested config
                    rows.extend(flatten_config(value, full_field))
                elif isinstance(value, (list, tuple, set)):
                    # Format list/tuple/set values
                    rows.append((full_field, ", ".join(str(item) for item in value)))
                else:
                    # Add simple values
                    rows.append((full_field, str(value)))
            return rows

        for param, value in flatten_config(config):
            table.add_row(param, value)

        self._console.print(table)

    def _load_models(self) -> None:
        """Load the LTXV model components."""
        prepare = self._accelerator.prepare

        # Load all model components using the new loader
        components = load_ltxv_components(
            model_source=self._config.model.model_source,
            load_text_encoder_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
            transformer_dtype=torch.float32,  # TODO: switch to bfloat16 when training a LoRA?
            vae_dtype=torch.bfloat16,
        )

        # Prepare components with accelerator
        self._scheduler = components.scheduler
        self._tokenizer = components.tokenizer
        self._text_encoder = prepare(components.text_encoder, device_placement=[False])
        self._vae = prepare(components.vae, device_placement=[False])
        transformer = components.transformer

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.warning(f"Quantizing model with precision: {self._config.acceleration.quantization}")
            transformer = quantize_model(
                transformer,
                precision=self._config.acceleration.quantization,
            )

        self._transformer = prepare(transformer)

        # Freeze all models. We later unfreeze the transformer based on training mode.
        self._text_encoder.requires_grad_(False)
        self._vae.requires_grad_(False)
        self._transformer.requires_grad_(False)

        # Enable gradient checkpointing if requested
        if self._config.optimization.enable_gradient_checkpointing:
            self._transformer.enable_gradient_checkpointing()

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def _compile_transformer(self) -> None:
        """Compile the transformer model with Torch Inductor."""

        if not self._config.acceleration.compile_with_inductor:
            return

        torch._dynamo.config.inline_inbuilt_nn_modules = True
        torch._dynamo.config.cache_size_limit = 128

        compile_module = partial(torch.compile, mode=self._config.acceleration.compilation_mode)
        self._transformer.transformer_blocks = nn.ModuleList(
            [compile_module(block) for block in self._transformer.transformer_blocks],
        )

    def _collect_trainable_params(self) -> None:
        """Collect trainable parameters based on training mode."""
        if self._config.model.training_mode == "lora":
            # For LoRA training, first set up LoRA layers
            self._setup_lora()
        elif self._config.model.training_mode == "full":
            # For full training, unfreeze all transformer parameters
            self._transformer.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]
        logger.debug(f"Trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        self._transformer.add_adapter(lora_config)

    def _init_dataloader(self) -> None:
        """Initialize the training data loader."""
        dataset = PrecomputedDataset(self._config.data.preprocessed_data_root)

        # Create dataloader
        data_loader = DataLoader(
            dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
        )

        # noinspection PyTypeChecker
        data_loader = self._accelerator.prepare(data_loader)

        # Create infinite data loader
        self._data_loader = itertools.cycle(data_loader)

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self) -> None:
        """Initialize the optimizer and learning rate scheduler."""
        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            # noinspection PyUnresolvedReferences
            from bitsandbytes.optim import AdamW8bit  # type: ignore

            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        # Add scheduler initialization
        lr_scheduler = self._create_scheduler(optimizer)

        # noinspection PyTypeChecker
        self._optimizer, self._lr_scheduler = self._accelerator.prepare(optimizer, lr_scheduler)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = self._config.optimization.scheduler_params or {}

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.get("start_factor", 1.0),
                end_factor=params.get("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.get("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.get("T_0", steps // 4),  # First restart cycle length
                T_mult=params.get("T_mult", 1),  # Multiplicative factor for cycle lengths
                eta_min=params.get("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.get("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.get("step_size", steps // 2),
                gamma=params.get("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )

    @torch.no_grad()
    @torch.compiler.set_stance("force_eager")
    def _sample_videos(self, progress: Progress) -> list[Path] | None:
        """Run validation by generating images from validation prompts."""

        self._vae.to(self._accelerator.device)
        # Model is already in the correct device if loaded in 8-bit.
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to(self._accelerator.device)

        pipeline = LTXPipeline(
            scheduler=deepcopy(self._scheduler),
            vae=self._vae,
            text_encoder=self._text_encoder,
            tokenizer=self._tokenizer,
            transformer=self._transformer,
        )
        pipeline.set_progress_bar_config(disable=True)

        # Create a task in the sampling progress
        task = progress.add_task(
            "sampling",
            total=len(self._config.validation.prompts),
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        i = 0
        for prompt in self._config.validation.prompts:
            generator = torch.Generator(device=self._accelerator.device).manual_seed(self._config.validation.seed)

            # Generate video
            width, height, frames = self._config.validation.video_dims
            with autocast(self._accelerator.device.type, dtype=torch.bfloat16):
                videos = pipeline(
                    prompt=prompt,
                    negative_prompt=self._config.validation.negative_prompt,
                    width=width,
                    height=height,
                    num_frames=frames,
                    num_inference_steps=self._config.validation.inference_steps,
                    generator=generator,
                ).frames

            for video in videos:
                video_path = output_dir / f"step_{self._global_step:06d}_{i}.mp4"
                export_to_video(video, str(video_path), fps=24)
                video_paths.append(video_path)
                i += 1
            progress.update(task, advance=1)

        progress.remove_task(task)

        # Move unused components back to CPU.
        self._vae.to("cpu")
        if not self._config.acceleration.load_text_encoder_in_8bit:
            self._text_encoder.to("cpu")

        return video_paths

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        logger.info("📊 Training Statistics:")
        logger.info(f"Total time: {stats.total_time_seconds / 60:.1f} minutes")
        logger.info(f"Training time: {stats.training_time / 60:.1f} minutes")
        if stats.compilation_time_seconds is not None:
            logger.info(f"Compilation time: {stats.compilation_time_seconds:.1f} seconds")
        logger.info(f"Training speed: {stats.steps_per_second:.2f} steps/second")
        logger.info(f"Samples/second: {stats.samples_per_second:.2f}")
        logger.info(f"Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB")

    def _save_checkpoint(self) -> Path:
        """Save the model weights."""

        # Create checkpoints directory if it doesn't exist
        save_dir = Path(self._config.output_dir) / "checkpoints"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Create filename with step number
        prefix = "model" if self._config.model.training_mode == "full" else "lora"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename
        rel_saved_weights_path = saved_weights_path.relative_to(self._config.output_dir)

        # Get model state dict
        unwrapped_model = self._accelerator.unwrap_model(self._transformer)

        if self._config.model.training_mode == "full":
            state_dict = unwrapped_model.state_dict()
            save_file(state_dict, saved_weights_path)
            logger.info(f"💾 Model weights for step {self._global_step} saved in {rel_saved_weights_path}")
        elif self._config.model.training_mode == "lora":
            state_dict = get_peft_model_state_dict(unwrapped_model)
            # Adjust layer names to standard formatting.
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            LTXPipeline.save_lora_weights(
                save_directory=save_dir,
                transformer_lora_layers=state_dict,
                weight_name=filename,
            )
            logger.info(f"💾 LoRA weights for step {self._global_step} saved in {rel_saved_weights_path}")
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        # Keep track of checkpoint paths, and cleanup old checkpoints if needed
        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()

        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoints: {old_checkpoint}")
            # Update the list to only contain kept checkpoints
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]
