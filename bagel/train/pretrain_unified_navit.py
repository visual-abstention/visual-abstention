# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt, prune_old_checkpoints
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def num_items(value) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return value.shape[0] if value.ndim else 1
    return len(value)


def rank_trace(enabled: bool, stage: str, step=None, phase=None, **details) -> None:
    if not enabled:
        return
    fields = [
        f"time={time():.3f}",
        f"rank={dist.get_rank()}",
        f"stage={stage}",
    ]
    if step is not None:
        fields.append(f"step={step}")
    if phase is not None:
        fields.append(f"phase={phase}")
    fields.extend(f"{key}={value}" for key, value in details.items())
    print("[rank-trace] " + " ".join(fields), flush=True)


def install_fsdp_trace_hooks(model, trace_state, enabled: bool):
    if not enabled:
        return []

    handles = []
    for module_name, module in model.named_modules():
        if not isinstance(module, FSDP):
            continue
        name = module_name or "root"

        def forward_enter(_module, _args, _name=name):
            rank_trace(
                trace_state["enabled"], "fsdp_forward_enter",
                step=trace_state["step"], phase=trace_state["phase"],
                module=_name,
            )

        def forward_exit(_module, _args, _output, _name=name):
            rank_trace(
                trace_state["enabled"], "fsdp_forward_exit",
                step=trace_state["step"], phase=trace_state["phase"],
                module=_name,
            )

        handles.append(module.register_forward_pre_hook(forward_enter))
        handles.append(module.register_forward_hook(forward_exit))
    return handles


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    query_size = num_attention_heads * head_dim
    key_size = num_key_value_heads * head_dim
    value_size = num_key_value_heads * head_dim

    mlp_parameters = hidden_size * intermediate_size * 3
    attention_parameters = hidden_size * (
        query_size + key_size + value_size + num_attention_heads * head_dim
    )
    embedding_and_head_parameters = vocab_size * hidden_size * 2
    dense_parameters = (
        (mlp_parameters + attention_parameters) * num_hidden_layers
        + embedding_and_head_parameters
    )
    dense_token_factor = 6.0 * dense_parameters
    attention_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attention_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Estimate per-device BF16 throughput from the CUDA device name."""
    try:
        device_name = torch.cuda.get_device_name()
    except RuntimeError:
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        return 1336.0
    if any(tag in name for tag in ("H100", "H800", "H200")):
        return 989.0
    if any(tag in name for tag in ("A100", "A800")):
        return 312.0
    if "L40" in name:
        return 181.05
    if "L20" in name:
        return 119.5
    if "H20" in name:
        return 148.0
    if "910B" in name:
        return 354.0
    if "RTX 3070 TI" in name:
        return 21.75
    return default_tflops


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    max_vae_padded_pixels: int = field(
        default=0,
        metadata={"help": "Maximum padded H*W pixels sent to the VAE per packed batch; 0 disables the limit."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    trace_all_ranks: bool = field(
        default=False,
        metadata={"help": "Print fine-grained progress from every distributed rank."}
    )
    trace_steps: int = field(
        default=2,
        metadata={"help": "Number of training steps to trace when trace_all_ranks is enabled."}
    )
    save_final_checkpoint: bool = field(
        default=True,
        metadata={"help": "Save a checkpoint when the training loop exits."}
    )
    save_every: int = field(
        default=250,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    eval_every: int = field(
        default=500,
        metadata={"help": "Keep the checkpoints of every N-th training step when older checkpoints are pruned."}
    )
    checkpoint_keep: int = field(
        default=1,
        metadata={"help": "Must be 1; only the latest checkpoint remains resumable."}
    )
    total_steps: int = field(
        default=20000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-8,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs for MFU; 0 enables device-name detection."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don't fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.trace_steps < 0:
        raise ValueError("trace_steps must be non-negative")
    if training_args.save_every < 1:
        raise ValueError("save_every must be positive")
    if training_args.eval_every < 1:
        raise ValueError("eval_every must be positive")
    if training_args.checkpoint_keep != 1:
        raise ValueError("checkpoint_keep must be 1")
    rank_trace(training_args.trace_all_ranks, "distributed_initialized", device=device)
    if training_args.peak_device_tflops <= 0:
        detected_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if detected_tflops > 0:
            training_args.peak_device_tflops = detected_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120),
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
        if training_args.peak_device_tflops > 0:
            logger.info(
                f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} "
                "TFLOPs per GPU."
            )
        else:
            logger.warning("Peak device TFLOPs unavailable; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    rank_trace(training_args.trace_all_ranks, "startup_barrier_enter")
    dist.barrier()
    rank_trace(training_args.trace_all_ranks, "startup_barrier_exit")
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    auto_resume_checkpoint = None
    if training_args.auto_resume:
        auto_resume_checkpoint = get_latest_ckpt(training_args.checkpoint_dir)
        resume_from = auto_resume_checkpoint
        if auto_resume_checkpoint is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    if auto_resume_checkpoint is not None:
        if dist.get_rank() == 0:
            prune_old_checkpoints(
                training_args.checkpoint_dir,
                logger,
                keep=training_args.checkpoint_keep,
                protected_interval=training_args.eval_every,
                newest_step=int(os.path.basename(auto_resume_checkpoint)),
            )
        dist.barrier()

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    rank_trace(training_args.trace_all_ranks, "model_setup_enter")
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    if training_args.visual_und:  
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )
    rank_trace(training_args.trace_all_ranks, "model_setup_exit")

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(
        f"Model parameter count: {total_param_count / 1e9:.2f}B "
        f"(LM-only: {lm_param_count / 1e9:.2f}B)"
    )

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = deepcopy(model)
    rank_trace(training_args.trace_all_ranks, "checkpoint_load_enter")
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    )
    rank_trace(training_args.trace_all_ranks, "checkpoint_load_exit")
    rank_trace(training_args.trace_all_ranks, "fsdp_setup_enter")
    ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )
    rank_trace(training_args.trace_all_ranks, "fsdp_setup_exit")

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    rank_trace(training_args.trace_all_ranks, "dataloader_setup_enter")
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    print(dataset_meta)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_vae_padded_pixels=data_args.max_vae_padded_pixels,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )
    rank_trace(training_args.trace_all_ranks, "dataloader_setup_exit")

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    token_window = 0.0
    seqlen_square_window = 0.0
    steps_in_window = 0
    dataset_token_window = {}
    skipped_long_samples_window = 0
    skipped_long_samples_total = 0
    dense_token_factor, attention_factor = qwen2_flop_coefficients(
        model.language_model.config
    )
    last_completed_step = train_step
    last_saved_step = train_step if train_step > 0 else None
    trace_state = {"enabled": False, "step": None, "phase": "setup"}
    trace_handles = install_fsdp_trace_hooks(
        fsdp_model, trace_state, training_args.trace_all_ranks
    )
    train_iterator = iter(train_loader)
    for curr_step in range(train_step, training_args.total_steps):
        trace_state.update(
            enabled=(
                training_args.trace_all_ranks
                and curr_step < train_step + training_args.trace_steps
            ),
            step=curr_step,
            phase="data",
        )
        rank_trace(trace_state["enabled"], "dataloader_next_enter", step=curr_step)
        try:
            data = next(train_iterator)
        except StopIteration:
            rank_trace(trace_state["enabled"], "dataloader_exhausted", step=curr_step)
            break
        rank_trace(trace_state["enabled"], "dataloader_next_exit", step=curr_step)
        rank_trace(trace_state["enabled"], "batch_cuda_enter", step=curr_step)
        data = data.cuda(device).to_dict()
        rank_trace(
            trace_state["enabled"], "batch_cuda_exit", step=curr_step,
            sequence_length=data.get("sequence_length", "unknown"),
        )
        data_indexes = data.pop('batch_data_indexes', None)
        skipped_long_samples_window += data.pop('skipped_long_samples', 0)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        padded_images = data.get('padded_images')
        rank_trace(
            trace_state["enabled"], "batch_ready", step=curr_step,
            datasets=",".join(sorted({
                item['dataset_name'] for item in data_indexes or []
            })),
            samples=len(data_indexes or []),
            ce_tokens=num_items(data.get('ce_loss_indexes')),
            mse_tokens=num_items(data.get('mse_loss_indexes')),
            vit_tokens=num_items(data.get('packed_vit_tokens')),
            vae_batch=(padded_images.shape[0] if padded_images is not None else 0),
            vae_hw=(
                f"{padded_images.shape[-2]}x{padded_images.shape[-1]}"
                if padded_images is not None else "none"
            ),
        )
        for item in data_indexes or []:
            dataset_name = item['dataset_name']
            if dataset_name not in dataset_token_window:
                dataset_token_window[dataset_name] = {
                    'token_mix_group': item.get('token_mix_group', dataset_name),
                    'samples': 0,
                    'packed_tokens': 0,
                    'ce_tokens': 0,
                    'mse_tokens': 0,
                }
            dataset_stats = dataset_token_window[dataset_name]
            dataset_stats['samples'] += 1
            dataset_stats['packed_tokens'] += item.get('packed_tokens', 0)
            dataset_stats['ce_tokens'] += item.get('ce_tokens', 0)
            dataset_stats['mse_tokens'] += item.get('mse_tokens', 0)
        rank_trace(trace_state["enabled"], "token_collectives_enter", step=curr_step)
        local_tokens = torch.tensor(float(data['sequence_length']), device=device)
        min_tokens_per_rank = local_tokens.clone()
        max_tokens_per_rank = local_tokens.clone()
        dist.all_reduce(min_tokens_per_rank, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_tokens_per_rank, op=dist.ReduceOp.MAX)
        tokens_tensor = local_tokens.clone()
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        sample_lens_tensor = torch.tensor(
            data['sample_lens'], dtype=torch.float32, device=device
        )
        sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
        dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
        rank_trace(trace_state["enabled"], "token_collectives_exit", step=curr_step)
        seqlen_square_window += sample_square.item()
        steps_in_window += 1
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen and 'padded_images' in data:
                rank_trace(trace_state["enabled"], "vae_encode_enter", step=curr_step)
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
                rank_trace(trace_state["enabled"], "vae_encode_exit", step=curr_step)
            try:
                trace_state["phase"] = "forward"
                rank_trace(trace_state["enabled"], "model_forward_enter", step=curr_step)
                loss_dict = fsdp_model(**data)
                rank_trace(trace_state["enabled"], "model_forward_exit", step=curr_step)
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    dataset_names = sorted(
                        {item['dataset_name'] for item in data_indexes or []}
                    )
                    logger.error(
                        f"CUDA OOM at step {curr_step}, "
                        f"sequence_length={data['sequence_length']}, "
                        f"datasets={dataset_names}: {error}"
                    )
                    torch.cuda.empty_cache()
                raise

        trace_state["phase"] = "loss"
        rank_trace(trace_state["enabled"], "loss_collectives_enter", step=curr_step)
        loss = torch.tensor(0.0, device=device)
        ce = loss_dict["ce"]
        total_ce_tokens = torch.tensor(
            len(data['ce_loss_indexes']) if 'ce_loss_indexes' in data else 0,
            device=device,
        )
        dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
        if training_args.ce_loss_reweighting:
            local_ce_loss_weights = (
                ce_loss_weights.sum()
                if ce_loss_weights is not None
                else torch.tensor(0.0, device=device)
            )
            total_ce_loss_weights = local_ce_loss_weights.clone()
            dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
        if ce is not None:
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)

        mse = loss_dict["mse"]
        total_mse_tokens = torch.tensor(
            len(data['mse_loss_indexes']) if 'mse_loss_indexes' in data else 0,
            device=device,
        )
        dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
        if mse is not None:
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)

        local_nonfinite = (~torch.isfinite(loss.detach())).to(torch.int32)
        dist.all_reduce(local_nonfinite, op=dist.ReduceOp.MAX)
        rank_trace(trace_state["enabled"], "loss_collectives_exit", step=curr_step)
        if local_nonfinite.item():
            logger.warning(
                f"Step {curr_step}: at least one rank has a non-finite loss; "
                "all ranks are skipping this step"
            )
            optimizer.zero_grad()
            continue
        
        rank_trace(trace_state["enabled"], "zero_grad_enter", step=curr_step)
        optimizer.zero_grad()
        rank_trace(trace_state["enabled"], "zero_grad_exit", step=curr_step)
        trace_state["phase"] = "backward"
        rank_trace(trace_state["enabled"], "backward_enter", step=curr_step)
        loss.backward()
        rank_trace(trace_state["enabled"], "backward_exit", step=curr_step)
        rank_trace(trace_state["enabled"], "grad_clip_enter", step=curr_step)
        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        rank_trace(trace_state["enabled"], "grad_clip_exit", step=curr_step)
        rank_trace(trace_state["enabled"], "optimizer_step_enter", step=curr_step)
        optimizer.step()
        scheduler.step()
        rank_trace(trace_state["enabled"], "optimizer_step_exit", step=curr_step)
        rank_trace(trace_state["enabled"], "ema_update_enter", step=curr_step)
        fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)
        rank_trace(trace_state["enabled"], "ema_update_exit", step=curr_step)
        last_completed_step = curr_step + 1

        # Log loss values:
        if curr_step % training_args.log_every == 0:
            rank_trace(trace_state["enabled"], "logging_enter", step=curr_step)
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            skipped_long_samples_tensor = torch.tensor(
                skipped_long_samples_window, dtype=torch.long, device=device
            )
            dist.all_reduce(skipped_long_samples_tensor, op=dist.ReduceOp.SUM)
            global_skipped_long_samples = skipped_long_samples_tensor.item()
            skipped_long_samples_total += global_skipped_long_samples

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = steps_in_window / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / max(steps_in_window, 1)
            estimated_flops = (
                dense_token_factor * token_window
                + attention_factor * seqlen_square_window
            )
            actual_tflops = estimated_flops / elapsed / 1e12
            peak_total_tflops = (
                training_args.peak_device_tflops * dist.get_world_size()
            )
            mfu = (
                actual_tflops / peak_total_tflops
                if peak_total_tflops > 0
                else 0.0
            )
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += (
                f"Train Steps/Sec: {steps_per_sec:.2f}, "
                f"Tokens/Sec: {tokens_per_sec / 1000:.2f}k, "
                f"Tokens/Rank: {min_tokens_per_rank.item():.0f}-"
                f"{max_tokens_per_rank.item():.0f}, "
                f"MFU: {mfu * 100:.1f}%, "
                f"Skipped Long Samples: {global_skipped_long_samples}, "
            )
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['tokens_per_sec'] = tokens_per_sec
            wandb_log['tokens_per_step'] = tokens_per_step
            wandb_log['min_tokens_per_rank'] = min_tokens_per_rank.item()
            wandb_log['max_tokens_per_rank'] = max_tokens_per_rank.item()
            wandb_log['actual_tflops'] = actual_tflops
            wandb_log['mfu'] = mfu
            wandb_log['skipped_long_samples'] = global_skipped_long_samples
            wandb_log['skipped_long_samples_total'] = skipped_long_samples_total

            gathered_dataset_stats = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_dataset_stats, dataset_token_window)
            if dist.get_rank() == 0:
                merged_dataset_stats = {}
                for rank_stats in gathered_dataset_stats:
                    for dataset_name, stats in rank_stats.items():
                        if dataset_name not in merged_dataset_stats:
                            merged_dataset_stats[dataset_name] = {
                                'token_mix_group': stats['token_mix_group'],
                                'samples': 0,
                                'packed_tokens': 0,
                                'ce_tokens': 0,
                                'mse_tokens': 0,
                            }
                        for key in ('samples', 'packed_tokens', 'ce_tokens', 'mse_tokens'):
                            value = stats[key]
                            merged_dataset_stats[dataset_name][key] += value

                total_dataset_ce_tokens = sum(
                    stats['ce_tokens'] for stats in merged_dataset_stats.values()
                )
                total_dataset_mse_tokens = sum(
                    stats['mse_tokens'] for stats in merged_dataset_stats.values()
                )
                mix_parts = []
                for dataset_name in sorted(merged_dataset_stats):
                    stats = merged_dataset_stats[dataset_name]
                    prefix = f'data/{dataset_name}'
                    for key in ('samples', 'packed_tokens', 'ce_tokens', 'mse_tokens'):
                        value = stats[key]
                        wandb_log[f'{prefix}/{key}'] = value
                    ce_fraction = (
                        stats['ce_tokens'] / total_dataset_ce_tokens
                        if total_dataset_ce_tokens > 0
                        else 0.0
                    )
                    wandb_log[f'{prefix}/ce_fraction'] = ce_fraction
                    mse_fraction = (
                        stats['mse_tokens'] / total_dataset_mse_tokens
                        if total_dataset_mse_tokens > 0
                        else 0.0
                    )
                    wandb_log[f'{prefix}/mse_fraction'] = mse_fraction
                    mix_parts.append(
                        f"{dataset_name}: samples={stats['samples']}, "
                        f"tokens={stats['packed_tokens']}, "
                        f"ce={stats['ce_tokens']}, mse={stats['mse_tokens']}, "
                        f"ce_fraction={ce_fraction:.4f}, "
                        f"mse_fraction={mse_fraction:.4f}"
                    )
                logger.info('Dataset mix: ' + '; '.join(mix_parts))

                token_group_stats = {}
                for stats in merged_dataset_stats.values():
                    group = stats['token_mix_group']
                    if group not in token_group_stats:
                        token_group_stats[group] = {
                            'samples': 0,
                            'packed_tokens': 0,
                            'ce_tokens': 0,
                            'mse_tokens': 0,
                        }
                    for key in token_group_stats[group]:
                        token_group_stats[group][key] += stats[key]
                total_group_tokens = sum(
                    stats['packed_tokens'] for stats in token_group_stats.values()
                )
                group_mix_parts = []
                for group in sorted(token_group_stats):
                    stats = token_group_stats[group]
                    packed_fraction = (
                        stats['packed_tokens'] / total_group_tokens
                        if total_group_tokens > 0
                        else 0.0
                    )
                    prefix = f'data/token_group/{group}'
                    for key, value in stats.items():
                        wandb_log[f'{prefix}/{key}'] = value
                    wandb_log[f'{prefix}/packed_fraction'] = packed_fraction
                    group_mix_parts.append(
                        f"{group}: samples={stats['samples']}, "
                        f"tokens={stats['packed_tokens']}, "
                        f"packed_fraction={packed_fraction:.4f}"
                    )
                logger.info('Token-group mix: ' + '; '.join(group_mix_parts))

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0
            steps_in_window = 0
            dataset_token_window = {}
            skipped_long_samples_window = 0
            rank_trace(trace_state["enabled"], "logging_exit", step=curr_step)

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if last_completed_step % training_args.save_every == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            try:
                dist.gather_object(data_status, gather_list, dst=0)
                FSDPCheckpoint.fsdp_save_ckpt(
                    ckpt_dir=training_args.checkpoint_dir,
                    train_steps=last_completed_step,
                    model=fsdp_model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    logger=logger,
                    fsdp_config=fsdp_config,
                    data_status=gather_list,
                )
                last_saved_step = last_completed_step
                if dist.get_rank() == 0:
                    prune_old_checkpoints(
                        training_args.checkpoint_dir,
                        logger,
                        keep=training_args.checkpoint_keep,
                        protected_interval=training_args.eval_every,
                        newest_step=last_completed_step,
                    )
            except RuntimeError as error:
                logger.error(f"Checkpoint failed at step {last_completed_step}: {error}")
                raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    for handle in trace_handles:
        handle.remove()

    if (
        training_args.save_final_checkpoint
        and last_completed_step > 0
        and last_saved_step != last_completed_step
    ):
        logger.info(f"Saving final checkpoint at step {last_completed_step}...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if dist.get_rank() == 0:
            gather_list = [None] * dist.get_world_size()
        else:
            gather_list = None
        try:
            dist.gather_object(data_status, gather_list, dst=0)
            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir,
                train_steps=last_completed_step,
                model=fsdp_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list,
            )
            if dist.get_rank() == 0:
                prune_old_checkpoints(
                    training_args.checkpoint_dir,
                    logger,
                    keep=training_args.checkpoint_keep,
                    protected_interval=training_args.eval_every,
                    newest_step=last_completed_step,
                )
            logger.info(f"Final checkpoint saved at step {last_completed_step}")
        except RuntimeError as error:
            logger.error(f"Final checkpoint failed at step {last_completed_step}: {error}")
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
