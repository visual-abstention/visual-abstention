"""CPU checks of the training data and the sampler (no GPU needed).

  MODEL_PATH=<BAGEL-7B-MoT dir> PYTHONPATH=../bagel:. python check_data.py [--files]

  --files   also require every input and target image to exist (after fetching the images)
"""
import os
import sys
from copy import deepcopy

import vista_data as V


def main():
    inputs, pairs = V.load(check_files='--files' in sys.argv)
    print(f'{len(inputs)} examples, {len(pairs)} pairs')

    # Sampler: four feasible and four infeasible examples in every step, each pair in one step, all pairs used.
    orders = [V.order(pairs, r, 8, V.SEED) for r in range(8)]
    assert len({len(o) for o in orders}) == 1
    for i in range(len(orders[0])):
        assert sum(orders[r][i]['ground_truth']['feasible'] for r in range(8)) == 4
        assert orders[0][i]['pair_id'] == orders[1][i]['pair_id']
    assert {a['pair_id'] for o in orders for a in o} == set(pairs)

    # MixedHint: exactly two feasible and two infeasible examples carry the reminder in every step.
    hinted = [V.hinted_order(V.order, 'mixed_hint')(pairs, r, 8, V.SEED) for r in range(8)]
    for i in range(len(hinted[0])):
        assert sum(hinted[r][i]['decline_hint_prompt'] for r in range(0, 8, 2)) == 2
        assert sum(hinted[r][i]['decline_hint_prompt'] for r in range(1, 8, 2)) == 2
    print('sampler and MixedHint assignment: PASS')

    if 'MODEL_PATH' not in os.environ:
        print('MODEL_PATH is not set; skipping the tokenizer and packing checks')
        return
    from data.data_utils import add_special_tokens
    from data.dataset_base import DataConfig, PackedDataset
    from modeling.qwen2 import Qwen2Tokenizer
    tokenizer, tokens, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True))
    # The inference budget is 1000 new tokens including the start token.
    assert all(len(tokenizer.encode(a['training_response'])) <= 998 for p in pairs.values() for a in p.values())
    config = DataConfig({'no_hint_sft': {'dataset_names': ['train'], 'token_weight': 1.,
        'image_transform_args': {'max_image_size': 640, 'min_image_size': 640, 'image_stride': 16},
        'vit_image_transform_args': {'max_image_size': 448, 'min_image_size': 448, 'image_stride': 14}}},
        text_cond_dropout_prob=0, vit_cond_dropout_prob=0, vae_cond_dropout_prob=0, max_latent_size=64)
    V.register('mixed_hint')
    packer = PackedDataset(config, tokenizer, tokens, 0, 8, 1, use_flex=True, expected_num_tokens=1, max_num_tokens=8192)
    dataset, seen = packer.grouped_datasets[0], set()
    for a in (a for p in pairs.values() for a in p.values()):
        key = (a['source']['category'], a['side'])
        if key in seen:
            continue
        seen.add(key)
        for hint in (False, True):
            sample = dataset.sample({**a, 'decline_hint_prompt': hint})
            state = packer.pack_sequence(deepcopy(sample), packer.set_sequence_status())
            # The reasoning is supervised and followed by EOS; an infeasible example has no image loss.
            assert state['packed_label_ids'] == sample['text_ids_list'][-1] + [packer.eos_token_id]
            assert bool(state['mse_loss_indexes']) == a['ground_truth']['feasible']
            assert not set(state['mse_loss_indexes']) & set(state['ce_loss_indexes']) and state['curr'] <= 6000
            expected_prompt = tokenizer.encode(inputs[a['case_id']]['prompt_en'] + (V.HINT_SUFFIX if hint else ''))
            assert sample['text_ids_list'][1] == expected_prompt
    print('tokenizer and packing (all categories, both sides, with and without the reminder): PASS')


if __name__ == '__main__':
    main()
