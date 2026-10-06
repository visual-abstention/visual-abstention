# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import random
import json

import numpy as np
import torch

from .data_utils import (
    get_flattened_position_ids_interpolate,
    get_flattened_position_ids_extrapolate, 
    len2weight,
    patchify, 
    prepare_attention_mask_per_sample, 
)
from .dataset_info import DATASET_INFO, DATASET_REGISTRY
from .token_mixture import select_token_deficit_index
from .transforms import ImageTransform
from .video_utils import FrameSampler


def get_padded_vae_pixels(sample, packed_vae_images):
    vae_images = list(packed_vae_images)
    image_index = 0
    for item in sample['sequence_plan']:
        if item['type'] not in ('vit_image', 'vae_image'):
            continue
        image_tensor = sample['image_tensor_list'][image_index]
        image_index += 1
        if item['type'] == 'vae_image':
            vae_images.append(image_tensor)

    if not vae_images:
        return 0
    return (
        len(vae_images)
        * max(image.shape[-2] for image in vae_images)
        * max(image.shape[-1] for image in vae_images)
    )


class DataConfig:
    def __init__(
        self, 
        grouped_datasets, 
        text_cond_dropout_prob=0.1,
        vit_cond_dropout_prob=0.4,
        vae_cond_dropout_prob=0.1,
        vae_image_downsample=16,
        max_latent_size=32,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    ):
        self.grouped_datasets = grouped_datasets
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size


class PackedDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, 
        data_config, 
        tokenizer, 
        special_tokens,
        local_rank, 
        world_size, 
        num_workers,
        expected_num_tokens=32768, 
        max_num_tokens_per_sample=16384,
        max_num_tokens=36864,
        max_vae_padded_pixels=0,
        prefer_buffer_before=16384,
        max_buffer_size=50,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    ):
        super().__init__()
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.prefer_buffer_before = prefer_buffer_before
        self.max_num_tokens = max_num_tokens
        self.max_vae_padded_pixels = max_vae_padded_pixels
        self.max_buffer_size = max_buffer_size
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.use_flex = use_flex
        for k, v in special_tokens.items():
            setattr(self, k, v)

        grouped_datasets, is_mandatory, token_weights, token_mix_groups = self.build_datasets(
            data_config.grouped_datasets, data_status
        )
        self.grouped_datasets = grouped_datasets
        self.dataset_iters = [iter(dataset) for dataset in grouped_datasets]
        self.is_mandatory = is_mandatory
        self.token_weights = token_weights
        self.token_mix_groups = token_mix_groups
        self.data_config = data_config
        self.interpolate_pos = interpolate_pos
        if self.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def build_datasets(self, datasets_metainfo, data_status):
        datasets = []
        is_mandatory = []
        token_weights = []
        token_mix_groups = []
        for grouped_dataset_name, dataset_args in datasets_metainfo.items():
            is_mandatory.append(dataset_args.pop('is_mandatory', False))
            if 'weight' in dataset_args:
                raise ValueError(
                    f"{grouped_dataset_name} uses removed sample-based weight; "
                    "configure token_weight instead"
                )
            token_weight = float(dataset_args.pop('token_weight'))
            if token_weight <= 0:
                raise ValueError(
                    f"{grouped_dataset_name} token_weight must be positive"
                )
            token_weights.append(token_weight)
            token_mix_groups.append(
                dataset_args.pop('token_mix_group', grouped_dataset_name)
            )

            if 'frame_sampler_args' in dataset_args.keys():
                frame_sampler = FrameSampler(**dataset_args.pop('frame_sampler_args'))
                dataset_args['frame_sampler'] = frame_sampler
            if 'image_transform_args' in dataset_args.keys():
                transform = ImageTransform(**dataset_args.pop('image_transform_args'))
                dataset_args['transform'] = transform
            if 'vit_image_transform_args' in dataset_args.keys():
                vit_transform = ImageTransform(**dataset_args.pop('vit_image_transform_args'))
                dataset_args['vit_transform'] = vit_transform

            assert 'dataset_names' in dataset_args.keys()
            dataset_names = dataset_args.pop('dataset_names')
            dataset_args['data_dir_list'] = []
            for item in dataset_names:
                if self.local_rank == 0:
                    print(f'Preparing Dataset {grouped_dataset_name}/{item}')
                meta_info = DATASET_INFO[grouped_dataset_name][item]
                if 'data_dir' in meta_info:
                    dataset_args['data_dir_list'].append(meta_info['data_dir'])

                for path_key in (
                    'manifest_path', 'records_path', 'selection_path', 'media_index_path'
                ):
                    if path_key in meta_info:
                        if path_key in dataset_args and dataset_args[path_key] != meta_info[path_key]:
                            raise ValueError(
                                f"{grouped_dataset_name} accepts one {path_key}; "
                                "register each source as a separate flat dataset"
                            )
                        dataset_args[path_key] = meta_info[path_key]

                if "parquet_info_path" in meta_info.keys():
                    if 'parquet_info' not in dataset_args.keys():
                        dataset_args['parquet_info'] = {}
                    with open(meta_info['parquet_info_path'], 'r') as f:
                        parquet_info = json.load(f)
                    dataset_args['parquet_info'].update(parquet_info)

                if 'json_dir' in meta_info.keys():
                    # parquet/tar with json
                    if 'json_dir_list' not in dataset_args.keys():
                        dataset_args['json_dir_list'] = [meta_info['json_dir']]
                    else:
                        dataset_args['json_dir_list'].append(meta_info['json_dir'])

                if 'jsonl_path' in meta_info.keys():
                    # jsonl with jpeg
                    if 'jsonl_path_list' not in dataset_args.keys():
                        dataset_args['jsonl_path_list'] = [meta_info['jsonl_path']]
                    else:
                        dataset_args['jsonl_path_list'].append(meta_info['jsonl_path'])

                if 'image_prefix_dir' in meta_info.keys():
                    dataset_args['image_prefix_dir'] = meta_info['image_prefix_dir']

            resume_data_status = dataset_args.pop('resume_data_status', True)
            if data_status is not None and grouped_dataset_name in data_status.keys() and resume_data_status:
                data_status_per_group = data_status[grouped_dataset_name]
            else:
                data_status_per_group = None
            dataset = DATASET_REGISTRY[grouped_dataset_name](
                dataset_name=grouped_dataset_name,
                tokenizer=self.tokenizer,
                local_rank=self.local_rank,
                world_size=self.world_size,
                num_workers=self.num_workers,
                data_status=data_status_per_group,
                **dataset_args
            )
            datasets.append(dataset)

        return datasets, is_mandatory, token_weights, token_mix_groups

    def set_epoch(self, seed):
        for dataset in self.grouped_datasets:
            dataset.set_epoch(seed)

    def set_sequence_status(self):
        sequence_status = dict(
            curr                        = 0,
            sample_lens                 = list(),
            packed_position_ids         = list(),
            nested_attention_masks      = list(),
            split_lens                  = list(),
            attn_modes                  = list(),
            packed_text_ids             = list(), 
            packed_text_indexes         = list(),
            packed_label_ids            = list(),
            ce_loss_indexes             = list(),
            ce_loss_weights             = list(),
            vae_image_tensors           = list(), 
            packed_latent_position_ids  = list(),
            vae_latent_shapes           = list(), 
            packed_vae_token_indexes    = list(), 
            packed_timesteps            = list(), 
            mse_loss_indexes            = list(),
            packed_vit_tokens           = list(), 
            vit_token_seqlens           = list(),
            packed_vit_position_ids     = list(),
            packed_vit_token_indexes    = list(), 
        )
        return sequence_status

    def to_tensor(self, sequence_status):
        data = dict(
            sequence_length=sum(sequence_status['sample_lens']),
            sample_lens=sequence_status['sample_lens'],
            packed_text_ids=torch.tensor(sequence_status['packed_text_ids']),
            packed_text_indexes=torch.tensor(sequence_status['packed_text_indexes']),
            packed_position_ids=torch.tensor(sequence_status['packed_position_ids']),
        )
        if not self.use_flex:
            data['nested_attention_masks'] = sequence_status['nested_attention_masks']
        else:
            sequence_len = data['sequence_length']
            pad_len = self.max_num_tokens - sequence_len
            data['split_lens'] = sequence_status['split_lens'] + [pad_len]
            data['attn_modes'] = sequence_status['attn_modes'] + ['causal']
            data['sample_lens'] += [pad_len]

        # if the model has a convnet vae (e.g., as visual tokenizer)
        if len(sequence_status['vae_image_tensors']) > 0:
            image_tensors = sequence_status.pop('vae_image_tensors')
            image_sizes = [item.shape for item in image_tensors]
            max_image_size = [max(item) for item in list(zip(*image_sizes))]
            padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
            for i, image_tensor in enumerate(image_tensors):
                padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

            data['padded_images'] = padded_images
            data['patchified_vae_latent_shapes'] = sequence_status['vae_latent_shapes']
            data['packed_latent_position_ids'] = torch.cat(sequence_status['packed_latent_position_ids'], dim=0)
            data['packed_vae_token_indexes'] = torch.tensor(sequence_status['packed_vae_token_indexes'])

        # if the model has a vit (e.g., as visual tokenizer)
        if len(sequence_status['packed_vit_tokens']) > 0:
            data['packed_vit_tokens'] = torch.cat(sequence_status['packed_vit_tokens'], dim=0)
            data['packed_vit_position_ids'] = torch.cat(sequence_status['packed_vit_position_ids'], dim=0)
            data['packed_vit_token_indexes'] = torch.tensor(sequence_status['packed_vit_token_indexes'])
            data['vit_token_seqlens'] = torch.tensor(sequence_status['vit_token_seqlens'])

        # if the model is required to perform visual generation
        if len(sequence_status['packed_timesteps']) > 0:
            data['packed_timesteps'] = torch.tensor(sequence_status['packed_timesteps'])
            data['mse_loss_indexes'] = torch.tensor(sequence_status['mse_loss_indexes'])

        # if the model is required to perform text generation
        if len(sequence_status['packed_label_ids']) > 0:
            data['packed_label_ids'] = torch.tensor(sequence_status['packed_label_ids'])
            data['ce_loss_indexes'] = torch.tensor(sequence_status['ce_loss_indexes'])
            data['ce_loss_weights'] = torch.tensor(sequence_status['ce_loss_weights'])

        # Debug printing for rank 0
        # if self.local_rank == 0:
        #     self.print_debug_info(data, sequence_status)

        return data

    def print_debug_info(self, data, sequence_status):
        """Print detailed debug information in an intuitive table format"""
        print("\n" + "="*120)
        print("DEBUG: Complete Sequence Analysis")
        print("="*120)
        
        # Basic info
        print(f"Sequence Length: {data['sequence_length']}")
        print(f"Sample Lengths: {data['sample_lens']}")
        
        # Get all data
        packed_text_ids = data['packed_text_ids'].tolist()
        packed_text_indexes = data['packed_text_indexes'].tolist()
        
        # Build loss mappings
        ce_loss_indexes = set(data.get('ce_loss_indexes', []).tolist())
        mse_loss_indexes = set(data.get('mse_loss_indexes', []).tolist())
        vit_token_indexes = set(data.get('packed_vit_token_indexes', []).tolist())
        vae_token_indexes = set(data.get('packed_vae_token_indexes', []).tolist())
        
        # Build label mapping
        label_mapping = {}
        if 'ce_loss_indexes' in data:
            ce_indexes = data['ce_loss_indexes'].tolist()
            ce_labels = data['packed_label_ids'].tolist()
            for i, pos in enumerate(ce_indexes):
                label_mapping[pos] = ce_labels[i]
        
        # Print raw token sequence
        print(f"\n1. Raw Token IDs: {packed_text_ids}")
        
        # Print decoded token sequence
        try:
            decoded_text_tokens = []
            for token_id in packed_text_ids:
                decoded = self.tokenizer.decode([token_id])
                decoded_text_tokens.append(decoded)
            print(f"2. Decoded Tokens: {decoded_text_tokens}")
        except Exception as e:
            print(f"2. Error decoding tokens: {e}")
            decoded_text_tokens = ["<ERROR>"] * len(packed_text_ids)
        
        # Create comprehensive sequence table
        print(f"\n3. Complete Sequence Table:")
        print("-" * 120)
        print(f"{'Order':<6} | {'Token Type':<12} | {'Token/Content':<30} | {'Loss Type':<10} | {'Label':<30} | {'Notes':<20}")
        print("-" * 120)
        
        # Track text token index
        text_token_idx = 0
        
        for pos in range(data['sequence_length']):
            # Determine token type and content
            if pos in packed_text_indexes:
                # This is a text token position
                token_id = packed_text_ids[text_token_idx]
                try:
                    decoded_token = self.tokenizer.decode([token_id])
                    token_content = f"ID:{token_id} '{decoded_token}'"
                except:
                    token_content = f"ID:{token_id} '<ERROR>'"
                token_type = "TEXT"
                text_token_idx += 1
                
            elif pos in vit_token_indexes:
                token_type = "VIT_IMAGE"
                token_content = "[VIT Image Patch]"
                
            elif pos in vae_token_indexes:
                token_type = "VAE_IMAGE"  
                token_content = "[VAE Image Latent]"
                
            else:
                token_type = "UNKNOWN"
                token_content = "[Unknown Position]"
            
            # Determine loss type
            if pos in ce_loss_indexes:
                loss_type = "CE"
            elif pos in mse_loss_indexes:
                loss_type = "MSE"
            else:
                loss_type = "None"
            
            # Determine label
            if pos in label_mapping:
                label_id = label_mapping[pos]
                try:
                    decoded_label = self.tokenizer.decode([label_id])
                    label_content = f"ID:{label_id} '{decoded_label}'"
                except:
                    label_content = f"ID:{label_id} '<ERROR>'"
            elif pos in mse_loss_indexes:
                label_content = "[Image Generation Target]"
            else:
                label_content = "N/A"
            
            # Additional notes
            notes = ""
            if pos in mse_loss_indexes and 'packed_timesteps' in data:
                timestep_idx = list(mse_loss_indexes).index(pos) if pos in mse_loss_indexes else -1
                if timestep_idx >= 0 and timestep_idx < len(data['packed_timesteps']):
                    timestep = data['packed_timesteps'][timestep_idx].item()
                    if timestep == float('-inf'):
                        notes = "No noise"
                    else:
                        notes = f"t={timestep:.3f}"
            
            print(f"{pos:<6} | {token_type:<12} | {token_content:<30} | {loss_type:<10} | {label_content:<30} | {notes:<20}")
        
        print("-" * 120)
        
        # Summary statistics
        total_positions = data['sequence_length']
        ce_positions = len(ce_loss_indexes)
        mse_positions = len(mse_loss_indexes)
        vit_positions = len(vit_token_indexes)
        vae_positions = len(vae_token_indexes)
        text_positions = len(packed_text_indexes)
        no_loss_positions = total_positions - ce_positions - mse_positions
        
        print(f"\nSummary Statistics:")
        print(f"  Total positions: {total_positions}")
        print(f"  Text tokens: {text_positions} ({text_positions/total_positions*100:.1f}%)")
        print(f"  VIT image tokens: {vit_positions} ({vit_positions/total_positions*100:.1f}%)")
        print(f"  VAE image tokens: {vae_positions} ({vae_positions/total_positions*100:.1f}%)")
        print(f"  Positions with CE loss: {ce_positions} ({ce_positions/total_positions*100:.1f}%)")
        print(f"  Positions with MSE loss: {mse_positions} ({mse_positions/total_positions*100:.1f}%)")
        print(f"  Positions with no loss: {no_loss_positions} ({no_loss_positions/total_positions*100:.1f}%)")
        
        print("="*120 + "\n")

    def __iter__(self):
        mixed_token_counts = [0] * len(self.token_weights)
        skip_cooldowns = [0] * len(self.token_weights)
        skip_cooldown_steps = max(len(self.token_weights) - 1, 1)
        sequence_status = self.set_sequence_status()
        batch_data_indexes = []

        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        def log_context(group_index=None):
            context = f"rank-{self.local_rank} worker-{worker_id}"
            if group_index is not None:
                dataset_name = self.grouped_datasets[group_index].dataset_name
                context += f" dataset-{dataset_name}"
            return context

        def select_group(candidate_indexes=None):
            selection_scope = candidate_indexes
            if candidate_indexes is None:
                candidate_indexes = list(range(len(skip_cooldowns)))
            candidates = [
                index for index in candidate_indexes
                if skip_cooldowns[index] == 0
            ]
            if not candidates:
                for index in candidate_indexes:
                    skip_cooldowns[index] = 0
                candidates = list(candidate_indexes)
            if selection_scope is None:
                group_index = select_token_deficit_index(
                    mixed_token_counts,
                    self.token_weights,
                    candidate_indexes=candidates,
                )
            else:
                local_candidate_indexes = [
                    candidate_indexes.index(index) for index in candidates
                ]
                local_group_index = select_token_deficit_index(
                    [mixed_token_counts[index] for index in candidate_indexes],
                    [self.token_weights[index] for index in candidate_indexes],
                    candidate_indexes=local_candidate_indexes,
                )
                group_index = candidate_indexes[local_group_index]
            skip_cooldowns[:] = [
                max(cooldown - 1, 0) for cooldown in skip_cooldowns
            ]
            return group_index

        def cool_down_group(group_index):
            skip_cooldowns[group_index] = skip_cooldown_steps

        def sample_limits(sample, status):
            num_tokens = sample['num_tokens'] + 2 * len(sample['sequence_plan'])
            sample_vae_padded_pixels = get_padded_vae_pixels(sample, [])
            packed_vae_padded_pixels = get_padded_vae_pixels(
                sample, status['vae_image_tensors']
            )
            sample_fits = (
                num_tokens <= self.max_num_tokens_per_sample
                and (
                    self.max_vae_padded_pixels <= 0
                    or sample_vae_padded_pixels <= self.max_vae_padded_pixels
                )
            )
            batch_fits = (
                status['curr'] + num_tokens <= self.max_num_tokens
                and (
                    self.max_vae_padded_pixels <= 0
                    or packed_vae_padded_pixels <= self.max_vae_padded_pixels
                )
            )
            return (
                num_tokens,
                sample_vae_padded_pixels,
                packed_vae_padded_pixels,
                sample_fits,
                batch_fits,
            )

        def pack(group_index, sample):
            nonlocal sequence_status
            sequence_status, sample_metadata = self.pack_sample(
                sample, sequence_status
            )
            if sample_metadata['packed_tokens'] == 0:
                return False
            sample_metadata['token_mix_group'] = self.token_mix_groups[group_index]
            mixed_token_counts[group_index] += sample_metadata['packed_tokens']
            batch_data_indexes.append(sample_metadata)
            return True

        def defer_sample(group_index, sample):
            if len(buffer) >= self.max_buffer_size:
                raise RuntimeError(
                    f"{log_context()} cannot reach expected_num_tokens="
                    f"{self.expected_num_tokens} before max_num_tokens="
                    f"{self.max_num_tokens} and max_vae_padded_pixels="
                    f"{self.max_vae_padded_pixels}; current_tokens="
                    f"{sequence_status['curr']}, buffered_samples={len(buffer)}"
                )
            buffer.append((group_index, sample))
            cool_down_group(group_index)

        def pop_fitting_buffer(candidate_indexes=None):
            for buffer_index, (group_index, sample) in enumerate(buffer):
                if (
                    candidate_indexes is not None
                    and group_index not in candidate_indexes
                ):
                    continue
                if sample_limits(sample, sequence_status)[-1]:
                    buffer.pop(buffer_index)
                    return group_index, sample
            return None

        def build_batch():
            data = self.to_tensor(sequence_status)
            data['batch_data_indexes'] = batch_data_indexes
            data['skipped_long_samples'] = skipped_long_samples
            return data

        mandatory_groups = []
        for group_index, mandatory in enumerate(self.is_mandatory):
            token_mix_group = self.token_mix_groups[group_index]
            if mandatory and token_mix_group not in mandatory_groups:
                mandatory_groups.append(token_mix_group)

        buffer = []
        skipped_long_samples = 0
        while True:
            # Ensure one sample from each mandatory token-mix group. Sources
            # within a group are selected by their cumulative token deficit.
            if sequence_status['curr'] == 0:
                for mandatory_group in mandatory_groups:
                    candidate_indexes = [
                        index
                        for index, token_mix_group in enumerate(self.token_mix_groups)
                        if token_mix_group == mandatory_group
                    ]
                    while True:
                        buffered_sample = pop_fitting_buffer(candidate_indexes)
                        if buffered_sample is None:
                            group_index = select_group(candidate_indexes)
                            sample = next(self.dataset_iters[group_index])
                        else:
                            group_index, sample = buffered_sample
                        (
                            num_tokens,
                            sample_vae_padded_pixels,
                            _,
                            sample_fits,
                            batch_fits,
                        ) = sample_limits(sample, sequence_status)
                        if not sample_fits:
                            cool_down_group(group_index)
                            if num_tokens > self.max_num_tokens_per_sample:
                                skipped_long_samples += 1
                                print(
                                    f"{log_context(group_index)} skip sample: "
                                    f"tokens={num_tokens}, "
                                    f"limit={self.max_num_tokens_per_sample}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"{log_context(group_index)} skip sample: "
                                    f"padded_vae_pixels={sample_vae_padded_pixels}, "
                                    f"limit={self.max_vae_padded_pixels}",
                                    flush=True,
                                )
                            continue
                        if not batch_fits:
                            defer_sample(group_index, sample)
                            continue
                        if pack(group_index, sample):
                            break

            if sequence_status['curr'] >= self.expected_num_tokens:
                print(
                    f"{log_context()} yielding batch: "
                    f"tokens={sum(sequence_status['sample_lens'])}, "
                    f"skipped={skipped_long_samples}",
                    flush=True,
                )
                data = build_batch()
                skipped_long_samples = 0
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []
                continue

            buffered_sample = None
            if buffer and sequence_status['curr'] < self.prefer_buffer_before:
                buffered_sample = pop_fitting_buffer()
            if buffered_sample is not None:
                group_index, sample = buffered_sample
                sample_from_buffer = True
            else:
                group_index = select_group()
                buffered_sample = pop_fitting_buffer([group_index])
                if buffered_sample is not None:
                    group_index, sample = buffered_sample
                    sample_from_buffer = True
                else:
                    sample = next(self.dataset_iters[group_index])
                    sample_from_buffer = False

            (
                num_tokens,
                sample_vae_padded_pixels,
                _,
                sample_fits,
                batch_fits,
            ) = sample_limits(sample, sequence_status)
            if not sample_fits:
                cool_down_group(group_index)
                if num_tokens > self.max_num_tokens_per_sample:
                    skipped_long_samples += 1
                    print(
                        f"{log_context(group_index)} skip sample: "
                        f"tokens={num_tokens}, "
                        f"limit={self.max_num_tokens_per_sample}, "
                        f"cooldown={skip_cooldown_steps}",
                        flush=True,
                    )
                else:
                    print(
                        f"{log_context(group_index)} skip sample: "
                        f"padded_vae_pixels={sample_vae_padded_pixels}, "
                        f"limit={self.max_vae_padded_pixels}, "
                        f"cooldown={skip_cooldown_steps}",
                        flush=True,
                    )
                continue

            if not batch_fits:
                if sample_from_buffer:
                    buffer.append((group_index, sample))
                    cool_down_group(group_index)
                else:
                    defer_sample(group_index, sample)
                continue

            pack(group_index, sample)

            if sequence_status['curr'] >= self.expected_num_tokens:
                print(
                    f"{log_context()} yielding batch: "
                    f"tokens={sum(sequence_status['sample_lens'])}, "
                    f"skipped={skipped_long_samples}",
                    flush=True,
                )
                data = build_batch()
                skipped_long_samples = 0
                yield data
                sequence_status = self.set_sequence_status()
                batch_data_indexes = []

    def pack_sample(self, sample, sequence_status):
        before_tokens = sequence_status['curr']
        before_ce_tokens = len(sequence_status['ce_loss_indexes'])
        before_mse_tokens = len(sequence_status['mse_loss_indexes'])

        sample_metadata = dict(sample['data_indexes'])
        sequence_status = self.pack_sequence(sample, sequence_status)
        sample_metadata.update(
            packed_tokens=sequence_status['curr'] - before_tokens,
            ce_tokens=(
                len(sequence_status['ce_loss_indexes']) - before_ce_tokens
            ),
            mse_tokens=(
                len(sequence_status['mse_loss_indexes']) - before_mse_tokens
            ),
        )
        return sequence_status, sample_metadata

    def pack_sequence(self, sample, sequence_status):
        image_tensor_list = sample['image_tensor_list']
        text_ids_list = sample['text_ids_list']
        sequence_plan = sample['sequence_plan']

        split_lens, attn_modes = list(), list()
        curr = sequence_status['curr']
        curr_rope_id = 0
        sample_lens = 0

        for item in sequence_plan:
            split_start = item.get('split_start', True)
            if split_start:
                curr_split_len = 0

            if item['type'] == 'text':
                text_ids = text_ids_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.text_cond_dropout_prob:
                    continue

                shifted_text_ids = [self.bos_token_id] + text_ids
                sequence_status['packed_text_ids'].extend(shifted_text_ids)
                sequence_status['packed_text_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                if item['loss'] == 1:
                    sequence_status['ce_loss_indexes'].extend(range(curr, curr + len(shifted_text_ids)))
                    sequence_status['ce_loss_weights'].extend(
                        [len2weight(len(shifted_text_ids))] * len(shifted_text_ids)
                    )
                    sequence_status['packed_label_ids'].extend(text_ids + [self.eos_token_id])
                curr += len(shifted_text_ids)
                curr_split_len += len(shifted_text_ids)

                # add a <|im_end|> token
                sequence_status['packed_text_ids'].append(self.eos_token_id)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|im_end|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("causal")
                sequence_status['packed_position_ids'].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item['type'] == 'vit_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue

                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                curr += 1
                curr_split_len += 1

                # preprocess image
                vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
                num_img_tokens = vit_tokens.shape[0]
                sequence_status['packed_vit_token_indexes'].extend(range(curr, curr + num_img_tokens))
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                sequence_status['packed_vit_tokens'].append(vit_tokens)
                sequence_status['vit_token_seqlens'].append(num_img_tokens)
                sequence_status['packed_vit_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vit_patch_size, 
                        max_num_patches_per_side=self.data_config.max_num_patch_per_side
                    )
                )

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                if item['special_token_loss'] == 1: # <|endofimage|> may have loss
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item['type'] == 'vae_image':
                image_tensor = image_tensor_list.pop(0)
                if item['enable_cfg'] == 1 and random.random() < self.data_config.vae_cond_dropout_prob:
                    # FIXME fix vae dropout in video2video setting.
                    curr_rope_id += 1
                    continue

                


                # add a <|startofimage|> token
                sequence_status['packed_text_ids'].append(self.start_of_image)
                sequence_status['packed_text_indexes'].append(curr)

                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                        
                curr += 1
                curr_split_len += 1

                # preprocess image
                sequence_status['vae_image_tensors'].append(image_tensor)
                sequence_status['packed_latent_position_ids'].append(
                    self.get_flattened_position_ids(
                        image_tensor.size(1), image_tensor.size(2),
                        self.data_config.vae_image_downsample, 
                        max_num_patches_per_side=self.data_config.max_latent_size
                    )
                )
                H, W = image_tensor.shape[1:]
                h = H // self.data_config.vae_image_downsample
                w = W // self.data_config.vae_image_downsample
                sequence_status['vae_latent_shapes'].append((h, w))

                num_img_tokens = w * h
                sequence_status['packed_vae_token_indexes'].extend(range(curr, curr + num_img_tokens))
                if item['loss'] == 1:
                    sequence_status['mse_loss_indexes'].extend(range(curr, curr + num_img_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float('-inf')

                sequence_status['packed_timesteps'].extend([timestep] * num_img_tokens)
                curr += num_img_tokens
                curr_split_len += num_img_tokens

                # add a <|endofimage|> token
                sequence_status['packed_text_ids'].append(self.end_of_image)
                sequence_status['packed_text_indexes'].append(curr)
                # <|endofimage|> may have loss
                if item['special_token_loss'] == 1:
                    sequence_status['ce_loss_indexes'].append(curr)
                    sequence_status['ce_loss_weights'].append(1.0)
                    sequence_status['packed_label_ids'].append(item['special_token_label'])
                curr += 1
                curr_split_len += 1

                # update sequence status
                if split_start:
                    if item['loss'] == 1 and 'frame_delta' not in item.keys():
                        attn_modes.append("noise")
                    else:
                        attn_modes.append("full")
                sequence_status['packed_position_ids'].extend([curr_rope_id] * (num_img_tokens + 2))
                if 'frame_delta' in item.keys():
                    curr_rope_id += item['frame_delta']
                elif item['loss'] == 0:
                    curr_rope_id += 1

            if item.get('split_end', True):
                split_lens.append(curr_split_len)
                sample_lens += curr_split_len

        # CFG dropout can remove every component of a sample. Do not expose an
        # empty document to packed attention or count it as a batch sample.
        if sample_lens == 0:
            return sequence_status

        sequence_status['curr'] = curr
        sequence_status['sample_lens'].append(sample_lens)
        # prepare attention mask
        if not self.use_flex:
            sequence_status['nested_attention_masks'].append(
                prepare_attention_mask_per_sample(split_lens, attn_modes)
            )
        else:
            sequence_status['split_lens'].extend(split_lens)
            sequence_status['attn_modes'].extend(attn_modes)

        return sequence_status


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data['batch_data_indexes']
        self.skipped_long_samples = data.get('skipped_long_samples', 0)
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, 'padded_images'):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, 'packed_timesteps'):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, 'packed_vit_tokens'):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, 'packed_label_ids'):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length = self.sequence_length,
            sample_lens = self.sample_lens,
            packed_text_ids = self.packed_text_ids,
            packed_text_indexes = self.packed_text_indexes,
            packed_position_ids = self.packed_position_ids,
            batch_data_indexes = self.batch_data_indexes,
            skipped_long_samples = self.skipped_long_samples,
        )

        if not self.use_flex:
            data['nested_attention_masks'] = self.nested_attention_masks
        else:
            data['split_lens'] = self.split_lens
            data['attn_modes'] = self.attn_modes

        if hasattr(self, 'padded_images'):
            data['padded_images'] = self.padded_images
            data['patchified_vae_latent_shapes'] = self.patchified_vae_latent_shapes
            data['packed_latent_position_ids'] = self.packed_latent_position_ids
            data['packed_vae_token_indexes'] = self.packed_vae_token_indexes

        if hasattr(self, 'packed_vit_tokens'):
            data['packed_vit_tokens'] = self.packed_vit_tokens
            data['packed_vit_position_ids'] = self.packed_vit_position_ids
            data['packed_vit_token_indexes'] = self.packed_vit_token_indexes
            data['vit_token_seqlens'] = self.vit_token_seqlens

        if hasattr(self, 'packed_timesteps'):
            data['packed_timesteps'] = self.packed_timesteps
            data['mse_loss_indexes'] = self.mse_loss_indexes

        if hasattr(self, 'packed_label_ids'):
            data['packed_label_ids'] = self.packed_label_ids
            data['ce_loss_indexes'] = self.ce_loss_indexes
            data['ce_loss_weights'] = self.ce_loss_weights

        return data


def collate_wrapper():
    def collate_fn(batch):
        return SimpleCustomBatch(batch)
    return collate_fn
