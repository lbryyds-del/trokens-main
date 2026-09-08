#!/usr/bin/env python3

"""Lightweight local launcher for Trokens configs.

This script intentionally keeps dataset/model/few-shot settings in YAML.
Use command-line opts only for local runtime overrides.
"""

import argparse
import os
import secrets
import shlex
import sys
from pathlib import Path


# Change this line when you want `python tools/run_local.py` to run another
# dataset config by default, for example: "configs/trokens/sav.yaml".
DEFAULT_CFG = "configs/trokens/sav.yaml"
DEFAULT_GPUS = "1,2,3,4"


def _default_master_port():
    return int(os.environ.get("MASTER_PORT") or (20000 + secrets.randbelow(10000)))


def _split_gpus(gpus):
    gpu_ids = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id must be provided via --gpus.")
    return gpu_ids


def _append_override(opts, key, value):
    if value is not None and value != "":
        opts.extend([key, str(value)])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Launch tools/run_net.py with a config file. Dataset-specific "
            "settings should live in YAML or be appended after --."
        )
    )
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg",
        default=os.environ.get("TROKENS_CONFIG", DEFAULT_CFG),
        help=(
            "Config file to pass to tools/run_net.py. Defaults to "
            f"TROKENS_CONFIG or {DEFAULT_CFG}."
        ),
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS),
        help="Physical GPU ids to expose, e.g. '0' or '0,1'.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=_default_master_port(),
        help="Distributed master port. Defaults to MASTER_PORT or a random local port.",
    )
    parser.add_argument(
        "--num-workers",
        default=os.environ.get("NUM_WORKERS"),
        help="Optional override for DATA_LOADER.NUM_WORKERS. Omit to use YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR"),
        help="Optional override for OUTPUT_DIR. Omit to use YAML/default config.",
    )
    parser.add_argument(
        "--run-name",
        default=os.environ.get("RUN_NAME"),
        help="Optional override for WANDB.EXP_NAME. Omit to use YAML.",
    )
    parser.add_argument(
        "--wandb-id",
        default=os.environ.get("WANDB_ID"),
        help="Optional override for WANDB.ID. Omit to use YAML.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("offline", "online", "disabled"),
        default=os.environ.get("WANDB_MODE"),
        help="Optional WANDB_MODE environment override.",
    )
    parser.add_argument(
        "--torch-home",
        default=os.environ.get("TORCH_HOME"),
        help="Optional TORCH_HOME environment override. Defaults to repo .torch-cache.",
    )
    parser.add_argument(
        "--train-enable",
        choices=("True", "False"),
        default=None,
        help="Optional override for TRAIN.ENABLE. Omit to use YAML.",
    )
    parser.add_argument(
        "--test-enable",
        choices=("True", "False"),
        default=None,
        help="Optional override for TEST.ENABLE. Omit to use YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Extra config overrides passed to run_net.py. Use after --, e.g. -- DATA.PATH_TO_DATA_DIR /path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {cfg_path}")

    gpu_ids = _split_gpus(args.gpus)

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = repo_root / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    torch_home = Path(args.torch_home) if args.torch_home else repo_root / ".torch-cache"
    if not torch_home.is_absolute():
        torch_home = repo_root / torch_home
    torch_home.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["MASTER_PORT"] = str(args.master_port)
    env["TORCH_HOME"] = str(torch_home)
    if args.wandb_mode:
        env["WANDB_MODE"] = args.wandb_mode

    cfg_overrides = []
    _append_override(cfg_overrides, "MASTER_PORT", args.master_port)
    _append_override(cfg_overrides, "NUM_GPUS", len(gpu_ids))
    _append_override(cfg_overrides, "DATA_LOADER.NUM_WORKERS", args.num_workers)
    _append_override(cfg_overrides, "OUTPUT_DIR", output_dir)
    _append_override(cfg_overrides, "WANDB.EXP_NAME", args.run_name)
    _append_override(cfg_overrides, "WANDB.ID", args.wandb_id)
    _append_override(cfg_overrides, "TRAIN.ENABLE", args.train_enable)
    _append_override(cfg_overrides, "TEST.ENABLE", args.test_enable)

    extra_opts = args.opts
    if extra_opts and extra_opts[0] == "--":
        extra_opts = extra_opts[1:]

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(gpu_ids)}",
        f"--master_port={args.master_port}",
        "tools/run_net.py",
        "--init_method",
        "env://",
        "--new_dist_init",
        "--cfg",
        str(cfg_path),
        *cfg_overrides,
        *extra_opts,
    ]

    print(f"Repo root: {repo_root}")
    print(f"Config: {cfg_path}")
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    if output_dir:
        print(f"OUTPUT_DIR={output_dir}")
    print(f"TORCH_HOME={torch_home}")
    print("Launching:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return
    os.execvpe(sys.executable, command, env)


if __name__ == "__main__":
    main()
