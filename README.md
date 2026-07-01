# FFTquadtrees

REU work on **FFT-guided Quadtree Sibling Attention** — a Vision Transformer
variant that organizes patch tokens into a quadtree and runs sparse "MX-CiF"
Top-K attention between sibling groups, bottom-up, to a single classification
token. The target task is the binary **IllusionAnimals** problem (illusion vs.
real animal).

The code that was scattered across many Colab notebooks and one-off scripts has
been consolidated into **two** clean, runnable files.

## The two files

| File | What it is | FFT? |
|------|------------|------|
| [`quadtree_attention.py`](quadtree_attention.py) | The structural quadtree-attention model. Builds a **fixed 2×2 quadtree pyramid** over the 14×14 ViT patch grid (196 → 49 → 9 → 1 tokens) and runs MX-CiF Top-K sibling attention. Also ships a standalone intensity-based quadtree builder + visualizer. | No |
| [`fourier_quadtree.py`](fourier_quadtree.py) | The Fourier version, consolidating **three** earlier variants into one script with two modes (see below). | Yes |

### `fourier_quadtree.py` modes

* **`--mode adaptive`** (default) — an FFT high-pass magnitude map drives a
  **content-adaptive** quadtree: a region splits into four children only where
  high-frequency detail is large. Sibling groups are precomputed per image and
  cached. *(Consolidates the original `quadtreeAttention.ipynb` /
  `train_quadtree.py` and the fixed-batching `PrecomputeFourier.py`.)*
* **`--mode guided`** — the quadtree is a **fixed** 2×2 pyramid, but the FFT
  high-pass map **guides** attention: it biases both the Top-K sibling
  selection and the final pooling. Runs fully batch-parallel.
  *(Consolidates `FIxedFourier`.)*

> **Bug fixed during consolidation:** the original notebook model concatenated
> the parent features of *every* image in a batch into a single sample
> (`torch.cat(parent_bank, dim=1)`), collapsing the batch and producing one
> output for the whole batch. Both consolidated models now produce **independent
> per-image logits** (adaptive pools each image separately and stacks the roots;
> guided is batch-parallel by construction).

## Quick start

```bash
pip install -r requirements.txt

# 1) Sanity check — no dataset, no pretrained weights, runs on CPU
python quadtree_attention.py --smoke-test
python fourier_quadtree.py   --smoke-test          # tests BOTH modes

# 2) Visualize a quadtree decomposition of an image
python quadtree_attention.py --visualize img.jpg --out qt.png        # intensity quadtree
python fourier_quadtree.py   --visualize img.jpg --out qt_fft.png    # FFT-guided quadtree

# 3) Train on an ImageFolder (<root>/<class>/<img>); add --pretrained to load ViT weights
python quadtree_attention.py --train --data_dir /path/to/IllusionAnimals_train \
    --epochs 10 --batch_size 32 --pretrained

python fourier_quadtree.py --train --mode adaptive \
    --data_dir /path/to/IllusionAnimals_train --cache_dir ./cache \
    --epochs 10 --batch_size 32 --pretrained

python fourier_quadtree.py --train --mode guided \
    --data_dir /path/to/IllusionAnimals_train --cache_dir ./cache \
    --epochs 10 --batch_size 32 --pretrained
```

The FFT/quadtree hyperparameters (`--r_hp`, `--mean_thresh`, `--std_thresh`,
`--max_depth`, `--alpha_topk`, `--beta_pool`, `--k_keep`) are all exposed on the
CLI; defaults are reconciled from the earlier versions.

## Dataset

The IllusionAnimals images are expected in an ImageFolder layout
(`<root>/<class>/<image>`). To fetch the Hugging Face dataset (e.g. in Colab):

```python
!pip install -U datasets huggingface_hub git-lfs
!git lfs install
# clone into Drive / local, then point --data_dir at the folder
!git clone https://huggingface.co/datasets/VQA-Illusion/IllusionAnimals_train
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="IllusionAnimals_train", split="train")
print(ds)
```

The first training run precomputes the FFT cache (`cache/buffers` for `adaptive`,
`cache/hp_maps` for `guided`); subsequent runs reuse it. Pass `--precompute` to
force a rebuild.

## Repository layout

```
quadtree_attention.py   # File 1: structural quadtree attention (no FFT)
fourier_quadtree.py     # File 2: FFT version (adaptive + guided modes)
requirements.txt
legacy/                 # the original un-consolidated scripts, kept for reference
  ├── PrecomputeFourier.py   # adaptive quadtree (batching fixed)  → folded into fourier_quadtree.py (adaptive)
  ├── FIxedFourier           # fixed quadtree + FFT guidance       → folded into fourier_quadtree.py (guided)
  └── train_controlnet.py    # separate ControlNet + MX-CiF Top-K experiment (not one of the two deliverables)
```

## Related repositories

* This repo: <https://github.com/ArnavBali36/FFTquadtrees>
* Author: <https://github.com/ArnavBali36>
* QuadTree Attention reference (Tang et al., ICLR 2022):
  <https://github.com/Tangshitao/QuadTreeAttention>
* QuadRays project: _TODO — add the link_

## Notes

* `timm` provides the ViT patch embedding (`vit_base_patch16_224`). If `timm` is
  unavailable, the models fall back to a plain `Conv2d` patch embed so the smoke
  tests still run.
* The structural model's attention runs in fp32 internally for numerical
  stability of the sharp Top-K masking.
