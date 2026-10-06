# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


# The VisTA training code registers its own dataset under the name 'no_hint_sft' at run time
# (see vista/vista_data.py), so only the datasets shipped with BAGEL are listed here.
DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'think_trace': ThinkTraceJSONLIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i',
            'num_files': 10,
            'num_total_samples': 1000,
        },
    },
    'unified_edit': {
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            'parquet_info_path': 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json',
        },
    },
    'vlm_sft': {
        'llava_ov': {
            'data_dir': 'your_data_path/bagel_example/vlm/images',
            'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
            'num_total_samples': 1000,
        },
    },
    'think_trace': {
        'think_trace_dataset': {
            'data_dir': 'your_data_path/think_trace/images',
            'jsonl_path': 'your_data_path/think_trace/think_trace.jsonl',
            'image_prefix_dir': 'your_data_path/think_trace',
            'num_total_samples': 1000,
        },
    },
}
