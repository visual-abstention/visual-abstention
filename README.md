# Visual Abstention in Unified Multimodal Models

**Chufan Shi**<sup>1\*</sup>, **Cheng Yang**<sup>2\*</sup>, **Tiannuo Yang**<sup>1</sup>, **Isadora White**<sup>2</sup>,
**Yiwei Chen**<sup>2</sup>, **Taylor Berg-Kirkpatrick**<sup>2</sup>, **Xuezhe Ma**<sup>1</sup>

<sup>1</sup>University of Southern California &nbsp; <sup>2</sup>University of California San Diego &nbsp; <sup>\*</sup>Equal contribution

[Paper](<ARXIV_LINK>) | [DoD benchmark](https://huggingface.co/datasets/visual-abstention/Draw-or-Decline) |
[VisTA training data](https://huggingface.co/datasets/visual-abstention/VisTA-Train) |
[Code](https://github.com/visual-abstention/visual-abstention)

This repository contains the code to evaluate unified multimodal models (UMMs) on the Draw-or-Decline benchmark and
to train VisTA-BAGEL, together with small example subsets of both datasets.

## Overview

* **Visual abstention** (the task). A model receives an input image and an instruction. A request is *feasible* if
  some image satisfies all of its constraints (the requested change, its target, and the content that must stay
  unchanged), and *infeasible* otherwise. For a feasible request the model should draw the edit. For an infeasible
  request it should *visually abstain*: state explicitly that no valid solution exists and end its response without
  generating an image. Returning the input image unchanged does not count as abstention.
* **Draw-or-Decline (DoD)** (the benchmark). 1,050 feasible/infeasible request pairs (2,100 requests), 150 pairs in
  each of 7 categories: material modification, medium interaction, motion state change, pose adjustment, spatial
  arrangement, and temporal evolution (real-world scenes from UniREditBench), and maze solving (programmatically
  generated maps). A real-world pair shares its input image and differs in the instruction; a maze pair shares its
  instruction and differs in one corridor cell that blocks the only path. DoD measures editing success *E* on
  feasible requests, refusal success *D* on infeasible requests, and false refusal *D<sub>f</sub>* on feasible
  requests, with Kimi-K3 as the judge.
* **VisTA** (Visual Transformation and Abstention, the method) and **VisTA-BAGEL** (the model). VisTA trains a model
  on paired feasible and infeasible examples so that it judges feasibility in its reasoning before deciding whether
  to generate. A feasible example supervises the reasoning and the target image; an infeasible example supervises a
  reasoning that explains the conflict and ends with the plain-text marker `[ABSTAIN]`, with no image. VisTA-BAGEL is
  UniREdit-BAGEL fine-tuned with VisTA on 38,224 training pairs.

In the paper, the 8 evaluated UMMs rarely refuse infeasible requests without a reminder (the strongest editor,
UniREdit-BAGEL, edits 68.4% of feasible requests correctly but refuses 0.4% of infeasible ones), and a reminder that
allows refusal raises refusals at the cost of editing accuracy. Without any reminder, VisTA-BAGEL refuses 93.0% of
infeasible requests, falsely refuses 0.8% of feasible ones, and completes 74.3% of feasible edits. See the paper for
all results.

**The weights of VisTA-BAGEL are not released.** This repository provides the training code and data to train it
from the public UniREdit-BAGEL weights.

## Repository layout

```
data/dod_examples/    42 DoD requests (the first 3 pairs of every category) with their images
data/train_examples/  70 training examples (the first 5 pairs of every category) with their images
data/download.py      downloads the full DoD benchmark and training set from Hugging Face (see "Data")
bagel/                the BAGEL code base (trainer, data, model, inferencer) that our runs use; Apache-2.0
vista/                VisTA training: dataset, MixedHint sampler, hooks, entry point, launch script, data checks
eval/                 inference (infer.py) and the Kimi-K3 judge (judge.py)
```

`bagel/` is the code of the official [BAGEL](https://github.com/ByteDance-Seed/Bagel) release (ByteDance Seed,
Apache-2.0) with the modifications that our runs need: training samples without a target image (infeasible requests
have none), token-weighted dataset mixing, and checkpoint options. It also contains the think-trace dataset of
[Bagel-Zebra-CoT](https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT) (Apache-2.0), as used by UniREdit.
Licence headers are unchanged.

## Setup

```
pip install -r requirements.txt          # Python 3.10, one node with 8 GPUs for training, 1 GPU for inference
```

Checkpoints (Hugging Face):

| Use | Repository | Revision |
|---|---|---|
| BAGEL configs, tokenizer, VAE, ViT (`MODEL_PATH`) | `ByteDance-Seed/BAGEL-7B-MoT` | `5019f57d168e5816e8f3f701b17cc816bb7cf24b` |
| Initialization of VisTA-BAGEL (`INIT_WEIGHTS`) | `maplebb/UniREdit-Bagel-bf16` | `2131cfdab2da79172e0ce429cdf28c2150e0b457` |

## Data

The full datasets are on Hugging Face; this repository only holds small example subsets, which have the same format
and run with the code as they are. Download the full data into the folders that the code reads:

```
python data/download.py dod      # visual-abstention/Draw-or-Decline -> data/dod/
python data/download.py train    # visual-abstention/VisTA-Train     -> data/train/ (about 100 GB of disk)
```

| Data | Folder | Used by |
|---|---|---|
| DoD examples (default) | `data/dod_examples/` | `eval/infer.py`, `eval/judge.py` |
| DoD, all 2,100 requests | `data/dod/` (`instruction.jsonl`, `ground_truth_criteria.jsonl`, `image/*.png`) | `--data ../data/dod` |
| Training examples (default) | `data/train_examples/` | `vista/launch.sh`, `vista/check_data.py` |
| Training set, all 76,448 examples | `data/train/` (`<category>/instruction.jsonl`, `<category>/ground_truth_criteria.jsonl`, `image/*.png`) | `VISTA_DATA=../data/train` |

The images of the training set are distributed as tar shards (`images/*.tar` with `images/manifest.json`);
`download.py train` extracts them into `data/train/image/` and deletes each shard after extraction (pass
`--keep-archives` to keep them and `--verify` to check their SHA-256 first). If you download the data another way,
put the files at the paths above: image paths in the annotations are relative to `data/dod/` and `data/train/`.

### DoD

`instruction.jsonl` (what a model sees) and `ground_truth_criteria.jsonl` (labels, hidden from the model):

* `instruction.jsonl`: `case_id`, `input_image(s)`, `instruction_en`, `rules`, `prompt_en` (the text given to the model).
* `ground_truth_criteria.jsonl`: `case_id`, `pair_id`, `paired_case_id`, `ground_truth` (`feasible`, `reason`, and
  `witness` for feasible or `certificate` for infeasible requests), `reference_image` (feasible requests), `source`.

The six real-world categories use the UniREditBench sources (the feasible request keeps the official instruction and
reference image); the maze maps are generated by us, and `source` records the generator seed and grid size.

The images of DoD are PNG files. In `data/dod_examples/`, the photographs are stored as WebP (quality 90) to keep the
repository small, and the maze images as lossless PNG. Lossy compression changes the input pixels, and greedy decoding
is sensitive to that: on a check of four cases, the reasoning text was identical to that of our runs with the original
PNGs in all four, but with the WebP files in only one. Results on the example images are therefore close to, but not
identical with, those on the released PNGs.

### Training data

One folder per category with `instruction.jsonl` and `ground_truth_criteria.jsonl`. A feasible example has a chain of
thought (`training_response`, inside `<think>...</think>`) and a target image (`training_output_image`); an infeasible
example has a reasoning that ends with the marker `[ABSTAIN]` and no target image.

| category | pairs | | category | pairs |
|---|---:|---|---|---:|
| maze | 5,200 | | pose adjustment | 5,695 |
| material modification | 5,958 | | spatial arrangement | 5,837 |
| medium interaction | 5,573 | | temporal evolution | 4,449 |
| motion state change | 5,512 | | **total** | **38,224** |

The 33,024 real-world pairs build on the public UniREdit-Data-100K training corpus: each feasible example keeps its
official instruction, chain of thought, and input and target images, and the paired infeasible example uses the same
input image. The 5,200 maze pairs are generated by us. In `data/train_examples/` the photographs are again stored as
WebP and the mazes as PNG.

## Evaluate

Two settings, both with reasoning enabled: `no_hint` gives the request as it is, and `decline_hint` appends
"If no solution exists under these constraints, state that no solution exists and briefly explain why."
In the paper these are the settings *without* and *with the reminder*.

By default the scripts read `data/dod_examples/`. For the full benchmark, download it (see "Data") and pass
`--data ../data/dod` to `infer.py` and `judge.py`.

```
cd eval
export PYTHONPATH=../bagel:.
# 1. inference (16 shards can run in parallel; every case has its own seed, so the sharding does not change outputs)
python infer.py infer --checkpoint <model.safetensors> --model-path <BAGEL-7B-MoT> --setting no_hint \
       --out runs/vista_no_hint --shards 16 --shard-index 0
python infer.py merge --out runs/vista_no_hint
# 2. judging with Kimi-K3 (any OpenAI-compatible endpoint serving the model name "kimi-k3")
KIMI_BASE_URLS=http://<host>:8000/v1 python judge.py --run runs/vista_no_hint
```

The model reasons first. If the reasoning ends with `[ABSTAIN]` it stops and no image is generated; otherwise it
draws the edit. `judge.py` writes `runs/.../scoring/summary.json` with, over all requests and by category:

* `feasible_edit`: editing success E, the requested edit is made and unrelated content is preserved (judged from the images);
* `infeasible_cot`: refusal success D, the reasoning explicitly concludes that the request has no valid solution;
* `false_refusal`: D<sub>f</sub>, the same conclusion on a feasible request.

Judgments that fail or are malformed stay `pending` and are counted in the denominator, never as failures.
Generation settings (50 image steps, text CFG 4.0, image CFG 2.0, at most 1,000 reasoning tokens, greedy text
decoding) are in `eval/infer.py`; the judge prompts are in `eval/judge.py`. The judge also receives the label of the
request (`ground_truth`); in the released data its explanation fields (`reason`, `witness`, `certificate`) are in
English.

## Train VisTA-BAGEL

VisTA fine-tunes the official UniREdit-BAGEL weights for 16,384 steps on the training pairs. Each step holds 8
examples, one per GPU: four feasible and four infeasible, with both examples of a pair in the same step. In every
step, two of the four feasible and two of the four infeasible instructions carry the decline hint (MixedHint; the paper describes it as including the reminder in 50% of the training examples), so half
of all training instructions do. The learning rate is constant at 2e-6 after 8 warm-up steps, AdamW (betas 0.9/0.95,
no weight decay), gradient clipping at 1.0, seed 20260918; the language model and generation components are
trained, the VAE and ViT are frozen. Text loss covers the reasoning (with `[ABSTAIN]` and the end token for an
infeasible example); the image flow-matching loss covers the target image of feasible examples only.

```
cd vista
# with the 70 example pairs (default); for the full data, download it and set VISTA_DATA=../data/train
MODEL_PATH=<BAGEL-7B-MoT> INIT_WEIGHTS=<UniREdit-BAGEL model.safetensors> OUT=runs/vista bash launch.sh
```

`vista/launch.sh` holds every trainer argument. By default only the raw weights of the last step are saved
(`runs/vista/checkpoints/0016384/model.safetensors`), which is the checkpoint that `eval/infer.py` loads.
`vista/check_data.py` checks the data, the sampler, the MixedHint assignment and the packed loss masks without a GPU:

```
cd vista
MODEL_PATH=<BAGEL-7B-MoT> PYTHONPATH=../bagel:. python check_data.py --files          # VISTA_DATA=../data/train for the full set
```

## Citation

```bibtex
@article{shi2026visual,
  title   = {Visual Abstention in Unified Multimodal Models},
  author  = {Shi, Chufan and Yang, Cheng and Yang, Tiannuo and White, Isadora and Chen, Yiwei and
             Berg-Kirkpatrick, Taylor and Ma, Xuezhe},
  journal = {arXiv preprint arXiv:<ARXIV_ID>},
  year    = {2026}
}
```

## License

The code is released under the Apache License 2.0 (see `LICENSE`); `bagel/` keeps the Apache-2.0 license of BAGEL
(`bagel/LICENSE`). The real-world images and the feasible instructions, reference images, and chains of thought come
from UniREditBench and UniREdit-Data-100K and remain subject to their terms.
