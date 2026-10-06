"""Hooks around BAGEL's trainer: finite-gradient checks, a progress log, and milestone weights only.

By default nothing but the raw model weights of the milestone steps is written (no EMA and no optimizer
state), in the dtype the trainer holds them in. Set VISTA_FULL_STATE=1 to keep the trainer's own
checkpoints (raw weights, EMA weights, optimizer and scheduler) instead.
"""
import gc
import inspect
import json
import math
from contextlib import contextmanager
from pathlib import Path


def fix_loss_logging(trainer):
    """Ranks without image targets contributed an int64 zero to a float32 all_reduce, which corrupted the
    logged MSE (not the loss used for the backward pass). Rewrite that line of trainer.main()."""
    source = inspect.getsource(trainer.main)
    old = 'torch.tensor(value.item(), device=device)'
    assert source.count(old) == 1
    source = source.replace(old, 'torch.tensor(float(value.item()), device=device, dtype=torch.float32)')
    exec(compile(source, trainer.__file__, 'exec'), trainer.__dict__)


@contextmanager
def training_runtime(trainer, output, milestones, model_only=True):
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    output = Path(output)
    milestones = {int(step) for step in milestones}
    checkpoint_class = trainer.FSDPCheckpoint
    save_descriptor = checkpoint_class.__dict__['fsdp_save_ckpt']
    original_save = checkpoint_class.fsdp_save_ckpt
    original_prune = trainer.prune_old_checkpoints
    original_clip, original_step = FSDP.clip_grad_norm_, torch.optim.AdamW.step
    state = {'updates': 0, 'norm': None}

    def rank_zero(function):
        outcome = [None]
        if torch.distributed.get_rank() == 0:
            try:
                function()
            except Exception as exc:
                outcome[0] = f'{type(exc).__name__}: {exc}'
        torch.distributed.broadcast_object_list(outcome, src=0)
        if outcome[0] is not None:
            raise RuntimeError(outcome[0])

    def clip(model, *args, **kwargs):
        state['norm'] = None
        norm = original_clip(model, *args, **kwargs)
        value = float(torch.as_tensor(norm).detach().item())
        if not math.isfinite(value):
            raise FloatingPointError(f'Non-finite global gradient norm: {value}')
        state['norm'] = value
        return norm

    def step(optimizer, *args, **kwargs):
        rates = [float(group['lr']) for group in optimizer.param_groups]
        if state['norm'] is None or not rates or not all(math.isfinite(rate) for rate in rates):
            raise RuntimeError('Optimizer update requires a finite gradient norm and learning rate.')
        result = original_step(optimizer, *args, **kwargs)
        state['updates'] += 1
        if torch.distributed.get_rank() == 0:
            output.mkdir(parents=True, exist_ok=True)
            with (output / 'training_progress.jsonl').open('a') as handle:
                handle.write(json.dumps({'step': state['updates'], 'lr': rates[0],
                                         'grad_norm_before_clip': state['norm']}, allow_nan=False) + '\n')
        state['norm'] = None
        return result

    def save_model_only(**kwargs):
        from safetensors.torch import save_file
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        step_number = kwargs['train_steps']
        if step_number not in milestones:
            return
        folder = output / 'checkpoints' / f'{step_number:07d}'
        # The state dict is a collective call; every rank takes part.
        with FSDP.state_dict_type(kwargs['model'], StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
            weights = kwargs['model'].state_dict()

        def write():
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / 'model.safetensors.tmp'
            save_file(weights, str(temporary))
            temporary.replace(folder / 'model.safetensors')
            (folder / '_SUCCESS').write_text(f'step={step_number}\n')

        try:
            rank_zero(write)
        finally:
            del weights
            gc.collect()
        torch.distributed.barrier()

    FSDP.clip_grad_norm_, torch.optim.AdamW.step = clip, step
    if model_only:
        checkpoint_class.fsdp_save_ckpt = staticmethod(save_model_only)
        trainer.prune_old_checkpoints = lambda *args, **kwargs: []
    try:
        yield
    finally:
        FSDP.clip_grad_norm_, torch.optim.AdamW.step = original_clip, original_step
        checkpoint_class.fsdp_save_ckpt = save_descriptor
        trainer.prune_old_checkpoints = original_prune
