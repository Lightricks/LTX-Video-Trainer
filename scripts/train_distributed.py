#!/usr/bin/env python

"""
Launch distributed training for LTXV models using Accelerate.

This script sets up distributed training across multiple GPUs on a single host
using Hugging Face Accelerate. It handles launching the training process with the
appropriate configuration.

Basic usage:
    distributed_train.py configs/ltxv_lora_config.yaml
"""

import subprocess
from pathlib import Path

import click
from accelerate.commands.launch import launch_command, launch_command_parser

from ltxv_trainer import logger


@click.command(help="Launch distributed training for LTXV")
@click.argument("config", type=click.Path(exists=True), required=True)
@click.option(
    "--num_processes",
    type=int,
    help="Number of processes/GPUs to use (overrides accelerate config)",
)
@click.option(
    "--disable_progress_bars",
    is_flag=True,
    help="Disable progress bars during training",
)
@click.option(
    "--main_process_port",
    type=int,
    default=None,
    help="Override master port for Accelerate distributed communication",
)
def main(
    config: str,
    num_processes: int | None,
    disable_progress_bars: bool,
    main_process_port: int | None,
) -> None:
    # Get path to the training script
    script_dir = Path(__file__).parent
    training_script = str(script_dir / "train.py")

    if num_processes is None:
        # Get number of available GPUs from nvidia-smi
        try:
            gpu_list = subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8")
            num_processes = len(gpu_list.split("\n")) - 1
            logger.debug(f"Found {num_processes} GPUs:\n{gpu_list}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Failed to get GPU count from nvidia-smi: {e}")
            logger.error("Falling back to 1 GPU")
            num_processes = 1

    # Convert args to form that can be passed to the training script
    training_args = [config]
    if disable_progress_bars:
        training_args.append("--disable_progress_bars")

    # Get the accelerate launch parser
    launch_parser = launch_command_parser()

    # Construct accelerate launch arguments
    launch_args = []
    if num_processes > 1:
        launch_args.append("--multi_gpu")
    launch_args.extend(["--num_processes", str(num_processes)])
    if main_process_port is not None:
        launch_args.extend(["--main_process_port", str(main_process_port)])
    # Add the actual training script and its args
    launch_args.append(training_script)
    launch_args.extend(training_args)

    # Parse the launch arguments
    launch_args = launch_parser.parse_args(launch_args)

    # Launch with accelerate
    launch_command(launch_args)


if __name__ == "__main__":
    main()
