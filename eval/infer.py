"""Run a BAGEL-family checkpoint on DoD with thinking: reasoning first, then an image unless it abstains.

  PYTHONPATH=../bagel:. python infer.py infer --checkpoint model.safetensors --model-path <BAGEL-7B-MoT dir> \
      --setting no_hint --out runs/vista_no_hint --shards 16 --shard-index 0
  PYTHONPATH=../bagel:. python infer.py merge --out runs/vista_no_hint

The model reads the system prompt, the input image, and the instruction (with the reminder appended in the
decline_hint setting). It writes its reasoning. If the reasoning ends with the marker [ABSTAIN] (and only then),
no image is generated; otherwise the reasoning is followed by an image, as in BAGEL's thinking mode. Every case
has its own seed, so the number of shards does not change any output. Finished cases are skipped on restart.
"""
import argparse
import hashlib
import json
import random
import re
import time
import traceback
from copy import deepcopy
from pathlib import Path
from types import MethodType

DEFAULT_SEED = 20260918
THINK_SYSTEM_PROMPT = (
    "Generation Instructions: You should first think about the planning process in the mind and then generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here")
DECLINE_HINT = 'If no solution exists under these constraints, state that no solution exists and briefly explain why.'
PARAMETERS = {'steps': 50, 'max_think_tokens': 1000, 'do_sample': False, 'text_temperature': 0.3,
              'cfg_text_scale': 4., 'cfg_img_scale': 2., 'cfg_interval': [0., 1.], 'timestep_shift': 3.,
              'cfg_renorm_min': 0., 'cfg_renorm_type': 'text_channel', 'image_max_edge': 640}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data/dod_examples'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temporary.replace(path)


def separate_thinking(text):
    thoughts = re.findall(r'<think>(.*?)</think>', text, flags=re.S)
    final = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if '<think>' in final:
        prefix, unfinished = final.split('<think>', 1)
        thoughts.append(unfinished)
        final = prefix
    return final.strip(), '\n'.join(t.strip() for t in thoughts)


def abstain_flags(raw, generation=None):
    """The reasoning ends with exactly one [ABSTAIN] marker right before </think>."""
    eos = generation['eos_observed'] if generation is not None else (True if '<|im_end|>' in raw else None)
    body = raw.split('<|im_end|>', 1)[0].replace('<|im_start|>', '').strip()
    match = re.fullmatch(r'<think>([\s\S]*?)\s*\[ABSTAIN\]\s*</think>', body)
    marker = bool(match and match.group(1).strip() and body.count('[ABSTAIN]') == 1)
    vision = '<|vision_start|>' in raw
    return {'abstain_marker_valid': marker, 'text_eos_observed': eos, 'marker_stop_applied': marker,
            'native_image_token_observed': vision,
            'natural_text_only_end': None if marker and eos is None else bool(marker and eos and not vision),
            'output_protocol_valid': None if marker and eos is None else bool(marker and eos),
            'stop_reason': 'abstain_stop_unknown' if marker and eos is None else 'abstain_marker_after_eos' if marker and eos
            else 'abstain_marker_without_eos' if marker else 'image_continuation'}


def load_model(checkpoint, model_path):
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer
    import text_generation
    checkpoint, path = Path(checkpoint).expanduser().absolute(), Path(model_path)
    llm = Qwen2Config.from_json_file(str(path / 'llm_config.json'))
    llm.qk_norm, llm.tie_word_embeddings, llm.layer_module = True, False, 'Qwen2MoTDecoderLayer'
    vit = SiglipVisionConfig.from_json_file(str(path / 'vit_config.json'))
    vit.rope = False
    vit.num_hidden_layers -= 1
    vae, vae_config = load_ae(str(path / 'ae.safetensors'))
    vae = vae.eval().to('cuda')
    encode, decode = vae.encode, vae.decode
    vae.encode = lambda x: encode(x.to('cuda'))
    vae.decode = lambda x: decode(x.to('cuda'))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm, vit_config=vit, vae_config=vae_config,
                         vit_max_num_patch_per_side=70, max_latent_size=64)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm), SiglipVisionModel(vit), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit, meta=True)
    tokenizer, tokens, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(str(path)))
    model = load_checkpoint_and_dispatch(model, checkpoint=str(checkpoint), device_map={'': 'cuda:0'},
                                         dtype=torch.bfloat16, force_hooks=True, offload_buffers=True).eval()
    engine = InterleaveInferencer(model, vae, tokenizer, ImageTransform(640, 640, 16), ImageTransform(448, 448, 14), tokens)
    # Upstream EOS stopping, recording whether the end-of-sequence token was reached.
    engine.model.generate_text = MethodType(text_generation.generate_text, engine.model)
    engine.gen_text = MethodType(text_generation.gen_text, engine)
    return engine


def infer(args):
    import numpy as np
    import torch
    from PIL import Image
    from data.transforms import ImageTransform
    data = read_jsonl(Path(args.data) / 'instruction.jsonl')
    out = Path(args.out)
    shard = out / 'shards' / f'{args.shard_index:03d}'
    shard.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(shard / 'results.jsonl') if (shard / 'results.jsonl').exists() else []
    done = {r['case_id'] for r in existing if r['status'] == 'complete'}
    existing = [r for r in existing if r['case_id'] in done]
    engine = load_model(args.checkpoint, args.model_path)
    transform = ImageTransform(640, 640, 16).resize_transform
    for i, row in enumerate(data):
        if i % args.shards != args.shard_index or row['case_id'] in done:
            continue
        cid, started = row['case_id'], time.monotonic()
        prompt = row['prompt_en'] + ('\n\n' + DECLINE_HINT if args.setting == 'decline_hint' else '')
        record = {'case_id': cid, 'prompt': prompt, 'status': 'running'}
        try:
            seed = (args.seed + int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)) % 2 ** 31
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            paths = [Path(args.data) / name for name in row.get('input_images') or [row['input_image']]]
            images = [transform(Image.open(p).convert('RGB')) for p in paths]
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                context = engine.update_context_text(THINK_SYSTEM_PROMPT, engine.init_gen_context())
                without_image = deepcopy(context)
                for image in images:
                    context = engine.update_context_image(image, context, vae=True, vit=True)
                without_text = deepcopy(context)
                context = engine.update_context_text(prompt, context)
                without_image = engine.update_context_text(prompt, without_image)
                raw = engine.gen_text(context, max_length=PARAMETERS['max_think_tokens'],
                                      do_sample=PARAMETERS['do_sample'], temperature=PARAMETERS['text_temperature'])
                final, cot = separate_thinking(raw.replace('<|im_start|>', '').replace('<|im_end|>', '').strip())
                generation = dict(engine.model.last_text_generation)
                flags = abstain_flags(raw, generation)
                record['text_generation'] = generation
                record.update(flags)
                output = None
                if not flags['marker_stop_applied']:
                    context = engine.update_context_text(raw, context)
                    output = engine.gen_image(images[0].size[::-1], context, cfg_text_precontext=without_text,
                        cfg_img_precontext=without_image, num_timesteps=PARAMETERS['steps'],
                        **{k: PARAMETERS[k] for k in ('cfg_text_scale', 'cfg_img_scale', 'cfg_interval', 'timestep_shift',
                                                       'cfg_renorm_min', 'cfg_renorm_type')})
            name = cid + '.png' if output is not None else None
            if output is not None:
                output.save(out / name)
            record.update(status='complete', seed=seed, thinking_text=cot, output_text=final, raw_output=raw,
                          output_image=name, seconds=time.monotonic() - started)
        except Exception as error:
            record.update(status='error', error=str(error), traceback=traceback.format_exc())
        existing = [r for r in existing if r['case_id'] != cid] + [record]
        write_jsonl(shard / 'results.jsonl', existing)
        print(cid, record['status'], flush=True)


def merge(args):
    data = read_jsonl(Path(args.data) / 'instruction.jsonl')
    merged = {}
    for path in Path(args.out).glob('shards/*/results.jsonl'):
        for row in read_jsonl(path):
            if row['case_id'] in merged:
                raise ValueError('Duplicate result: ' + row['case_id'])
            merged[row['case_id']] = row
    missing = [r['case_id'] for r in data if r['case_id'] not in merged]
    if missing:
        raise ValueError(f'{len(missing)} cases have no result yet, e.g. {missing[0]}')
    write_jsonl(Path(args.out) / 'results.jsonl', [merged[r['case_id']] for r in data])
    print(f'{len(data)} results merged; errors: {sum(r["status"] != "complete" for r in merged.values())}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['infer', 'merge'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--setting', choices=['no_hint', 'decline_hint'], default='no_hint')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.command == 'infer':
        if not args.checkpoint or not args.model_path:
            parser.error('infer needs --checkpoint and --model-path')
        infer(args)
    else:
        merge(args)


if __name__ == '__main__':
    main()
