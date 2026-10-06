"""BAGEL's text decoder with stop metadata, and the upstream gen_text wrapper.

Adapted from the official BAGEL repository (Apache-2.0), modeling/bagel/bagel.py and inferencer.py.
The EOS token is sampled but not returned by the upstream decoder; generate_text records whether it was
observed. Token sampling is unchanged (evaluation uses greedy decoding).
"""
from copy import deepcopy

import torch


@torch.no_grad
def generate_text(self, past_key_values, packed_key_value_indexes, key_values_lens, packed_start_tokens,
                  packed_query_position_ids, max_length, do_sample=False, temperature=1.0, end_token_id=None):
    step = 0
    generated_sequence = []
    curr_tokens = packed_start_tokens
    while step < max_length:
        generated_sequence.append(curr_tokens)
        packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
        query_lens = torch.ones_like(curr_tokens)
        packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
            0, len(key_values_lens), device=key_values_lens.device, dtype=key_values_lens.dtype)

        uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
        for i in range(len(uppacked)):
            uppacked[i] += i
        packed_key_value_indexes = torch.cat(uppacked, dim=0)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding, query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids, packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values, key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes, update_past_key_values=True,
            is_causal=True, **extra_inputs)
        past_key_values = output.past_key_values
        pred_logits = self.language_model.lm_head(output.packed_query_sequence)

        if do_sample:
            probs = torch.nn.functional.softmax(pred_logits / temperature, dim=-1)
            curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            curr_tokens = torch.argmax(pred_logits, dim=-1)

        uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
        for i in range(len(uppacked)):
            uppacked[i] = torch.cat(
                [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0)
        packed_key_value_indexes = torch.cat(uppacked, dim=0)
        key_values_lens = key_values_lens + 1
        packed_query_position_ids = packed_query_position_ids + 1
        step += 1

        if end_token_id is not None and curr_tokens[0] == end_token_id:  # only supports batch size 1
            break

    self.last_text_generation = {
        "version": 1,
        "eos_observed": bool(end_token_id is not None and curr_tokens[0].item() == end_token_id),
        "last_sampled_token_id": int(curr_tokens[0].item()),
        "eos_token_id": end_token_id,
        "steps": step,
        "max_length": max_length,
    }
    output_device = generated_sequence[0].device
    return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)


def gen_text(self, gen_context, max_length=500, do_sample=True, temperature=1.0):
    """InterleaveInferencer.gen_text of the official repository."""
    gen_context = deepcopy(gen_context)
    generation_input = self.model.prepare_start_tokens(gen_context['kv_lens'], gen_context['ropes'], self.new_token_ids)
    unpacked_latent = self.model.generate_text(
        past_key_values=gen_context['past_key_values'], max_length=max_length, do_sample=do_sample,
        temperature=temperature, end_token_id=self.new_token_ids['eos_token_id'], **generation_input)
    output = self.tokenizer.decode(unpacked_latent[:, 0])
    return output.split('<|im_end|>')[0].split('<|im_start|>')[1]
