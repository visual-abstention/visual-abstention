"""Paired feasible / infeasible training data of VisTA on top of BAGEL's packed-sequence trainer.

A feasible example supervises the reasoning (CoT) and the target image. An infeasible example
supervises only the reasoning, which ends with the fixed marker [ABSTAIN], followed by EOS; it has
no target image. The two examples of a pair are always in the same global step.
"""
import gzip
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image

from data.distributed_iterable_dataset import DistributedIterableDataset
from data.interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset

REPO = Path(__file__).resolve().parents[1]
TRAIN_FOLDER = Path(os.environ.get('VISTA_DATA', REPO / 'data/train_examples'))
SEED = 20260918
WORLD_SIZE = 8

THINK_SYSTEM_PROMPT = (
    "Generation Instructions: You should first think about the planning process in the mind and then generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here")

# The reminder that the evaluation appends in its reminder condition (decline_hint).
DECLINE_HINT = 'If no solution exists under these constraints, state that no solution exists and briefly explain why.'
HINT_SUFFIX = '\n\n' + DECLINE_HINT
FEASIBLE_RANKS, INFEASIBLE_RANKS = (0, 2, 4, 6), (1, 3, 5, 7)


def _read(path):
    """Read a JSONL file; the released data are plain .jsonl, gzip-compressed .jsonl.gz is also accepted."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _file(category, stem):
    """<stem>.jsonl, or <stem>.jsonl.gz if only that exists."""
    path = category / f'{stem}.jsonl'
    return path if path.is_file() else category / f'{stem}.jsonl.gz'


def load(folder=None, check_files=False):
    """Read every category folder; return {case_id: input row} and {pair_id: {'A': feasible, 'B': infeasible}}."""
    folder = Path(folder or TRAIN_FOLDER)
    inputs, labels = {}, []
    for category in sorted(p for p in folder.iterdir() if p.is_dir() and _file(p, 'instruction').is_file()):
        inputs.update({r['case_id']: r for r in _read(_file(category, 'instruction'))})
        labels += _read(_file(category, 'ground_truth_criteria'))
    if not labels or set(inputs) != {a['case_id'] for a in labels}:
        raise ValueError(f'Expected matching instruction and label files under {folder}')
    pairs = {}
    for a in labels:
        row, feasible, cot = inputs[a['case_id']], a['ground_truth']['feasible'], a['training_response']
        assert a['split'] == 'train' and cot.startswith('<think>') and cot.endswith('</think>')
        if feasible:
            assert '[ABSTAIN]' not in cot
        else:
            assert a['training_output_image'] is None and cot.endswith('[ABSTAIN]</think>')
        if check_files:
            assert (folder / row['input_image']).is_file(), row['input_image']
            if feasible:
                assert (folder / a['training_output_image']).is_file(), a['training_output_image']
        a.update(side='A' if feasible else 'B', task=row['task'])
        pair = pairs.setdefault(a['pair_id'], {})
        assert a['side'] not in pair
        pair[a['side']] = a
    assert all(set(p) == {'A', 'B'} for p in pairs.values())
    return inputs, pairs


def order(pairs, rank, world_size, seed, epoch=0):
    """Pair-preserving order. Rank r takes every eighth example, so a pair's feasible side lands on an even
    rank and its infeasible side on the next odd rank, and every step holds four of each."""
    assert world_size == WORLD_SIZE and 0 <= rank < world_size and pairs
    ids = sorted(pairs)
    random.Random(seed + epoch).shuffle(ids)
    # Categories differ in size. Repeat at most three whole pairs so that all ranks take equal steps.
    ids += [ids[i % len(ids)] for i in range((-len(ids)) % 4)]
    sequence = [pairs[p][side] for p in ids for side in ('A', 'B')]
    return sequence[rank::world_size]


def hinted_ranks(seed, epoch, step):
    """MixedHint: the ranks whose example carries the reminder at this step (two feasible + two infeasible)."""
    rng = random.Random(f'dod-mixed-hint:{seed}:{epoch}:{step}')  # str seeds hash stably
    return set(rng.sample(FEASIBLE_RANKS, 2)) | set(rng.sample(INFEASIBLE_RANKS, 2))


def hinted_order(base_order, variant):
    """variant: 'mixed_hint' (half of every step carries the reminder) or 'decline_hint' (all do)."""
    def wrapped(pairs, rank, world_size, seed, epoch=0):
        chosen = []
        for step, annotation in enumerate(base_order(pairs, rank, world_size, seed, epoch)):
            if annotation['ground_truth']['feasible'] != (rank % 2 == 0):
                raise ValueError('Unexpected feasible/rank layout; refuse to assign prompts')
            hint = variant == 'decline_hint' or rank in hinted_ranks(seed, epoch, step)
            chosen.append({**annotation, 'decline_hint_prompt': hint})
        return chosen
    return wrapped


class VistaDataset(InterleavedBaseIterableDataset, DistributedIterableDataset):
    hint = 'mixed_hint'  # 'mixed_hint', 'decline_hint', or None (no reminder in training)

    def __init__(self, dataset_name, tokenizer, transform, vit_transform, local_rank=0,
                 world_size=1, num_workers=1, data_status=None, data_dir_list=None, **kwargs):
        DistributedIterableDataset.__init__(self, dataset_name, local_rank, world_size, num_workers)
        if num_workers != 1:
            raise ValueError('One data worker per rank is required.')
        if data_dir_list and [Path(p).resolve() for p in data_dir_list] != [TRAIN_FOLDER.resolve()]:
            raise ValueError(f'This training reads only {TRAIN_FOLDER}.')
        if world_size != WORLD_SIZE:
            raise ValueError('The paired sampler requires eight ranks.')
        self.tokenizer, self.transform, self.vit_transform = tokenizer, transform, vit_transform
        self.folder = TRAIN_FOLDER
        self.inputs, self.pairs = load(self.folder)
        self.order = order if self.hint is None else hinted_order(order, self.hint)
        self.seed = SEED

    def set_epoch(self, seed=SEED):
        if seed != SEED:
            raise ValueError('--data_seed must equal the training seed.')
        self.seed = seed

    def sample(self, a):
        row = self.inputs[a['case_id']]
        prompt = row['prompt_en'] + (HINT_SUFFIX if a.get('decline_hint_prompt') else '')
        data = self._init_data()
        self._add_text(data, THINK_SYSTEM_PROMPT, need_loss=False, enable_cfg=False)
        with Image.open(self.folder / row['input_image']) as image:
            # The inferencer resizes once for the VAE before preparing both encoders.
            image = self.transform.resize_transform(image.convert('RGB'))
            self._add_image(data, image, need_loss=False, need_vae=True, need_vit=True, enable_cfg=False)
        self._add_text(data, prompt, need_loss=False, enable_cfg=False)
        self._add_text(data, a['training_response'], need_loss=True, enable_cfg=False)
        if a['ground_truth']['feasible']:
            with Image.open(self.folder / a['training_output_image']) as image:
                self._add_image(data, image.convert('RGB'), need_loss=True, need_vae=False, need_vit=False,
                                enable_cfg=False)
        return data

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        assert worker is None or worker.num_workers == 1
        epoch = 0
        while True:
            for annotation in self.order(self.pairs, self.local_rank, self.world_size, self.seed, epoch):
                sample = self.sample(annotation)
                sample['data_indexes'] = {'dataset_name': self.dataset_name, 'worker_id': 0,
                                          'data_indexes': {'epoch': epoch, 'case_id': annotation['case_id']}}
                yield sample
            epoch += 1


def register(hint='mixed_hint'):
    from data.dataset_info import DATASET_INFO, DATASET_REGISTRY
    VistaDataset.hint = None if hint == 'none' else hint
    DATASET_REGISTRY['no_hint_sft'] = VistaDataset
    DATASET_INFO['no_hint_sft'] = {'train': {'data_dir': str(TRAIN_FOLDER)}}
