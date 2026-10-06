# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import shutil


CHECKPOINT_SUCCESS_MARKER = "_SUCCESS"
CHECKPOINT_INCOMPLETE_MARKER = "_INCOMPLETE"
CHECKPOINT_EVAL_ONLY_MARKER = "_EVAL_ONLY"

_OPTIMIZER_SHARD_PATTERN = re.compile(r"optimizer\.(\d+)-of-(\d+)\.pt$")
_TRAINING_STATE_FILENAMES = {
    "model.safetensors",
    "scheduler.pt",
    "data_status.pt",
}


def checkpoint_is_complete(checkpoint_path):
    """Return whether a checkpoint is safe to resume from."""
    if os.path.exists(os.path.join(checkpoint_path, CHECKPOINT_INCOMPLETE_MARKER)):
        return False
    if os.path.exists(os.path.join(checkpoint_path, CHECKPOINT_EVAL_ONLY_MARKER)):
        return False
    if os.path.exists(os.path.join(checkpoint_path, CHECKPOINT_SUCCESS_MARKER)):
        return True

    # Checkpoints written before completion markers were introduced remain valid.
    try:
        filenames = os.listdir(checkpoint_path)
    except OSError:
        return False
    optimizer_shards = [
        (int(match.group(1)), int(match.group(2)))
        for name in filenames
        if (match := _OPTIMIZER_SHARD_PATTERN.fullmatch(name)) is not None
    ]
    optimizer_totals = {total for _, total in optimizer_shards}
    has_complete_optimizer = False
    if len(optimizer_totals) == 1:
        total_shards = optimizer_totals.pop()
        shard_indexes = {index for index, _ in optimizer_shards}
        has_complete_optimizer = shard_indexes == set(range(total_shards))
    return (
        "model.safetensors" in filenames
        and "scheduler.pt" in filenames
        and has_complete_optimizer
    )


def checkpoint_is_evaluable(checkpoint_path):
    """Return whether a checkpoint has the EMA weights used by evaluation."""
    return (
        not os.path.exists(os.path.join(checkpoint_path, CHECKPOINT_INCOMPLETE_MARKER))
        and os.path.isfile(os.path.join(checkpoint_path, "ema.safetensors"))
    )


def _mark_checkpoint_eval_only(checkpoint_path, step, logger):
    """Drop resume-only state while preserving the EMA evaluation weights."""
    marker = os.path.join(checkpoint_path, CHECKPOINT_EVAL_ONLY_MARKER)
    marker_tmp = f"{marker}.tmp"
    with open(marker_tmp, "w", encoding="utf-8") as marker_file:
        marker_file.write(f"step={step}\n")
        marker_file.flush()
        os.fsync(marker_file.fileno())
    os.replace(marker_tmp, marker)

    removed_files = []
    for name in os.listdir(checkpoint_path):
        if (
            name in _TRAINING_STATE_FILENAMES
            or _OPTIMIZER_SHARD_PATTERN.fullmatch(name) is not None
        ):
            path = os.path.join(checkpoint_path, name)
            os.remove(path)
            removed_files.append(name)
    logger.info(
        f"Compacted checkpoint {step} for evaluation only; "
        f"removed={sorted(removed_files)}"
    )
    return removed_files


def prune_old_checkpoints(
    checkpoint_dir,
    logger,
    keep=1,
    protected_interval=500,
    newest_step=None,
):
    """Keep one resumable checkpoint and compact older evaluation checkpoints."""
    if keep != 1:
        raise ValueError("checkpoint_keep must be 1")
    if protected_interval < 1:
        raise ValueError("protected_interval must be positive")
    if newest_step is None:
        raise ValueError("newest_step is required before pruning checkpoints")

    newest_path = os.path.join(checkpoint_dir, f"{newest_step:07d}")
    success_marker = os.path.join(newest_path, CHECKPOINT_SUCCESS_MARKER)
    if not os.path.isfile(success_marker):
        logger.warning(
            "Skipping checkpoint pruning because the newest checkpoint has no "
            f"success marker: {newest_path}"
        )
        return []

    checkpoint_steps = sorted(
        int(name)
        for name in os.listdir(checkpoint_dir)
        if name.isdigit() and os.path.isdir(os.path.join(checkpoint_dir, name))
    )
    removed_paths = []
    compacted_steps = []
    for step in (step for step in checkpoint_steps if step < newest_step):
        old_path = os.path.join(checkpoint_dir, f"{step:07d}")
        if step % protected_interval == 0 and checkpoint_is_evaluable(old_path):
            _mark_checkpoint_eval_only(old_path, step, logger)
            compacted_steps.append(step)
            continue

        shutil.rmtree(old_path)
        removed_paths.append(old_path)
        logger.info(f"Deleted old non-evaluation checkpoint: {old_path}")

    logger.info(
        "Checkpoint retention: "
        f"newest_resumable={newest_step}, "
        f"eval_only={compacted_steps}, "
        f"removed={[os.path.basename(path) for path in removed_paths]}"
    )
    return removed_paths


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [
        d
        for d in os.listdir(checkpoint_dir)
        if d.isdigit()
        and os.path.isdir(os.path.join(checkpoint_dir, d))
        and checkpoint_is_complete(os.path.join(checkpoint_dir, d))
    ]
    if not step_dirs:
        return None
    step_dirs.sort(key=int)
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir
