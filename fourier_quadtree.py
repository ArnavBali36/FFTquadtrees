"""
fourier_quadtree.py — FFT-guided Quadtree Sibling Attention (the "Fourier" version)
===================================================================================

This file consolidates the three Fourier/FFT variants that were scattered across
the project notebooks and earlier repo files into ONE working script with two
selectable modes:

  --mode adaptive   (default)
      An FFT high-pass magnitude map drives a CONTENT-ADAPTIVE quadtree: a region
      is split into 4 children only where the high-frequency detail (std of the
      high-pass map) is large. The resulting sibling groups are precomputed per
      image and fed to MX-CiF sibling attention.
      (Consolidated from `quadtreeAttention.ipynb` / `train_quadtree.py` and the
       fixed-batching `PrecomputeFourier.py`.)

  --mode guided
      The quadtree is a FIXED 2x2 pyramid (data-independent), but the FFT high-pass
      map is used to GUIDE attention: it biases both the Top-K sibling selection
      and the final parent-group pooling. Runs fully batch-parallel.
      (Consolidated from `FIxedFourier`.)

Both modes share one FFT high-pass front end. The original notebook model had a
batching bug (parent features from every image in a batch were concatenated into
a single sample); that is fixed here — `adaptive` pools each image independently
and stacks the per-image roots, and `guided` is batch-parallel by construction.

Usage
-----
Quick check (no dataset, no pretrained weights, CPU; tests BOTH modes):

    python fourier_quadtree.py --smoke-test

Visualize the FFT-guided adaptive quadtree on an image:

    python fourier_quadtree.py --visualize path/to/image.jpg --out qt_fft.png

Train (ImageFolder layout <root>/<class>/<img>); buffers/HP-maps are cached:

    python fourier_quadtree.py --train --mode adaptive \
        --data_dir /path/to/IllusionAnimals_train --cache_dir ./cache \
        --epochs 10 --batch_size 32 --pretrained

    python fourier_quadtree.py --train --mode guided \
        --data_dir /path/to/IllusionAnimals_train --cache_dir ./cache \
        --epochs 10 --batch_size 32 --pretrained
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

PATCH = 16
GRID = 14                     # 224 / 16
N_PATCHES = GRID * GRID       # 196
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ─────────────────────────────────────────────────────────────────────────────
# Config (defaults reconciled across the three earlier versions)
# ─────────────────────────────────────────────────────────────────────────────
# Source configs this reconciles:
#   train_quadtree.py / quadtreeAttention.ipynb : R_HP=8,  MEAN=0.6, STD=0.10, DEPTH=5
#   PrecomputeFourier.py                        : R_HP=50, MEAN=0.2, STD=0.04, DEPTH=3
# The defaults below keep PrecomputeFourier's radius/std/depth but use the original
# MEAN=0.6 cutoff: a high-pass map normalizes to [0,1] with a mean often ~0.2-0.3, so
# MEAN=0.2 would stop-split at the root and produce a degenerate (1-node) tree. 0.6 lets
# the tree actually grow. All knobs are overridable on the CLI.
@dataclass
class Config:
    # FFT high-pass
    r_hp: int = 50            # high-pass radius in pixels
    # adaptive-quadtree split thresholds
    mean_thresh: float = 0.6  # region high-pass mean >= this → stop splitting (leaf)
    std_thresh: float = 0.04  # region high-pass std  >  this → split into 4
    max_depth: int = 3
    min_size: int = 8
    # guided-mode FFT bias strengths
    alpha_topk: float = 0.5   # HP bias added to sibling saliency before Top-K
    beta_pool: float = 0.5    # HP bias added to final pooling logits
    # model
    d_model: int = 768
    heads: int = 12
    k_keep: int = 2
    max_levels: int = 8       # capacity for adaptive trees


# ─────────────────────────────────────────────────────────────────────────────
# FFT high-pass front end (shared by both modes)
# ─────────────────────────────────────────────────────────────────────────────
def highpass_filter(gray_np, r_hp):
    """FFT high-pass magnitude of a 2D array, normalized to [0, 1]."""
    fshift = np.fft.fftshift(np.fft.fft2(gray_np))
    h, w = gray_np.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    mask = ((Y - cy) ** 2 + (X - cx) ** 2) > (r_hp ** 2)
    recon = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
    mag = np.abs(recon)
    mag -= mag.min()
    mag /= (mag.max() + 1e-8)
    return mag


def hp_to_patch_map_14x14(hp224):
    """Average a 224x224 high-pass map over each 16x16 patch → (14,14)."""
    H, W = hp224.shape
    assert H == 224 and W == 224, "HP map must be 224x224"
    return hp224.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))


def _read_gray_224(path):
    """Load an image as a 224x224 grayscale float array in [0,1]."""
    if _HAS_CV2:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        return gray.astype(np.float32) / 255.0
    from PIL import Image
    im = Image.open(path).convert("L").resize((224, 224))
    return np.asarray(im, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive quadtree construction from the FFT high-pass map
# ─────────────────────────────────────────────────────────────────────────────
class QuadNode:
    def __init__(self, x, y, w, h, depth=0):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.depth = depth
        self.leaf = False
        self.children = []


def build_quadtree(node, img_hp, cfg: Config):
    """Split where the high-pass region is detailed (high std)."""
    patch = img_hp[node.y:node.y + node.height, node.x:node.x + node.width]
    m, s = float(patch.mean()), float(patch.std())
    if (m >= cfg.mean_thresh or min(node.width, node.height) < cfg.min_size
            or node.depth >= cfg.max_depth):
        node.leaf = True
    elif s > cfg.std_thresh:
        w2, h2 = node.width // 2, node.height // 2
        for dx, dy in [(0, 0), (w2, 0), (0, h2), (w2, h2)]:
            cw = w2 if dx == 0 else node.width - w2
            ch = h2 if dy == 0 else node.height - h2
            c = QuadNode(node.x + dx, node.y + dy, cw, ch, node.depth + 1)
            node.children.append(c)
            build_quadtree(c, img_hp, cfg)
    else:
        node.leaf = True


def collect_sibling_groups(node, level, groups):
    """Record each internal node's 4 children as patch indices into the 14x14 grid."""
    if level >= len(groups):
        groups.append([])
    if not node.leaf and len(node.children) == 4:
        idxs = []
        for c in node.children:
            py, px = c.y // PATCH, c.x // PATCH       # pixel → patch coordinate
            idxs.append(py * GRID + px)
        groups[level].append(idxs)
        for c in node.children:
            collect_sibling_groups(c, level + 1, groups)


def draw_quadtree(node, canvas):
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for visualization")
    if node.leaf:
        cv2.rectangle(canvas, (node.x, node.y),
                      (node.x + node.width, node.y + node.height), (0, 0, 255), 1)
    else:
        for c in node.children:
            draw_quadtree(c, canvas)


def list_image_paths(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def cache_key(path):
    """Per-image cache filename. Includes the parent folder so that identical
    file stems in different class folders (common in ImageFolder datasets) don't
    collide / overwrite each other's FFT cache."""
    p = Path(path)
    return f"{p.parent.name}__{p.stem}"


# ─────────────────────────────────────────────────────────────────────────────
# Precompute caches
# ─────────────────────────────────────────────────────────────────────────────
def precompute_adaptive_buffers(image_paths, buf_dir, cfg: Config, log_every=200):
    """Pass 1: find (L_max, N_max). Pass 2: save (idxbuf, mskbuf) per image stem."""
    os.makedirs(buf_dir, exist_ok=True)
    print(f"[precompute/adaptive] {len(image_paths)} images | "
          f"R_HP={cfg.r_hp} MEAN>={cfg.mean_thresh} STD>{cfg.std_thresh} DEPTH={cfg.max_depth}")

    L_max = N_max = 0
    for i, p in enumerate(image_paths, 1):
        gray = _read_gray_224(p)
        if gray is None:
            continue
        hp = highpass_filter(gray, cfg.r_hp)
        root = QuadNode(0, 0, 224, 224)
        build_quadtree(root, hp, cfg)
        groups = []
        collect_sibling_groups(root, 0, groups)
        L_max = max(L_max, len(groups))
        if groups:
            N_max = max(N_max, max(len(g) for g in groups))
        if i % log_every == 0:
            print(f"  pass1 {i}/{len(image_paths)} | L_max={L_max} N_max={N_max}")
    L_max, N_max = max(L_max, 1), max(N_max, 1)
    print(f"[precompute/adaptive] buffer dims: L_max={L_max}, N_max={N_max}")

    wrote = 0
    for i, p in enumerate(image_paths, 1):
        gray = _read_gray_224(p)
        if gray is None:
            continue
        hp = highpass_filter(gray, cfg.r_hp)
        root = QuadNode(0, 0, 224, 224)
        build_quadtree(root, hp, cfg)
        groups = []
        collect_sibling_groups(root, 0, groups)

        idxbuf = torch.full((L_max, N_max, 4), -1, dtype=torch.long)
        mskbuf = torch.zeros((L_max, N_max), dtype=torch.bool)
        for lvl, g in enumerate(groups):
            if not g:
                continue
            buf = torch.tensor(g, dtype=torch.long).clamp_(0, N_PATCHES - 1)
            idxbuf[lvl, :buf.size(0), :] = buf
            mskbuf[lvl, :buf.size(0)] = True
        torch.save((idxbuf, mskbuf), os.path.join(buf_dir, f"{cache_key(p)}.pt"))
        wrote += 1
        if i % log_every == 0:
            print(f"  pass2 {i}/{len(image_paths)} saved")
    print(f"[precompute/adaptive] wrote {wrote} buffers → {buf_dir}")


def precompute_hp_maps(image_paths, hp_dir, cfg: Config, log_every=200):
    """Save the flattened 14x14 (=196) high-pass patch map per image stem."""
    os.makedirs(hp_dir, exist_ok=True)
    print(f"[precompute/guided] {len(image_paths)} images | R_HP={cfg.r_hp}")
    wrote = 0
    for i, p in enumerate(image_paths, 1):
        gray = _read_gray_224(p)
        if gray is None:
            continue
        hp224 = highpass_filter(gray, cfg.r_hp)
        hp14 = hp_to_patch_map_14x14(hp224)
        hp196 = torch.from_numpy(hp14.astype(np.float32)).reshape(-1)
        torch.save(hp196, os.path.join(hp_dir, f"{cache_key(p)}.pt"))
        wrote += 1
        if i % log_every == 0:
            print(f"  {i}/{len(image_paths)} saved")
    print(f"[precompute/guided] wrote {wrote} HP maps → {hp_dir}")


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


# ─────────────────────────────────────────────────────────────────────────────
# Patch embedding (timm ViT or Conv2d fallback)
# ─────────────────────────────────────────────────────────────────────────────
class ConvPatchEmbed(nn.Module):
    def __init__(self, in_ch=3, d_model=768, patch=PATCH):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def make_patch_embed(d_model=768, pretrained=False):
    if _HAS_TIMM:
        vit = timm.create_model("vit_base_patch16_224",
                                pretrained=pretrained, num_classes=0)
        return vit.patch_embed
    print("[warn] timm not available — using a plain Conv2d patch embed.")
    return ConvPatchEmbed(3, d_model, PATCH)


# ─────────────────────────────────────────────────────────────────────────────
# Sibling attention (plain, for adaptive mode)
# ─────────────────────────────────────────────────────────────────────────────
class MxCIFSiblingAttention(nn.Module):
    def __init__(self, d_model, n_heads, k_keep=2):
        super().__init__()
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.k = k_keep
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.cif_proj = nn.Linear(d_model, d_model)

    def forward(self, x_grp):                          # [B, N, 4, D]
        B, N, C, D = x_grp.shape
        H, dh = self.n_heads, self.dh
        qkv = self.qkv(x_grp.reshape(-1, D)).view(B, N, C, 3, H, dh)
        q, k, v = qkv.unbind(dim=3)
        sal = q.norm(dim=-1).mean(-1)                  # [B, N, 4]
        k_eff = min(self.k, C)
        _, topk_idx = sal.topk(k_eff, dim=2)           # [B, N, k]

        def gather_kept(t):
            t_flat = t.flatten(-2)                     # [B, N, 4, H*dh]
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, H * dh)
            return torch.gather(t_flat, 2, idx_exp).view(B, N, k_eff, H, dh)

        k_kept, v_kept = gather_kept(k), gather_kept(v)
        q_parent = q.mean(2)                           # [B, N, H, dh]
        scale = 1.0 / math.sqrt(dh)
        att = (q_parent.unsqueeze(2) * k_kept).sum(-1) * scale   # [B, N, k, H]
        w = att.softmax(2)
        o_parent = (w.unsqueeze(-1) * v_kept).sum(2)             # [B, N, H, dh]
        o_parent = o_parent.transpose(2, 3).reshape(B, N, D)
        o_parent = self.proj(o_parent)
        gate = torch.sigmoid(self.cif_proj(x_grp.reshape(-1, D))).view(B, N, C, D)
        x_grp = x_grp + gate * o_parent.unsqueeze(2)
        return o_parent, x_grp


class MxCIFSiblingAttentionHP(nn.Module):
    """Sibling attention whose Top-K saliency is biased by per-child HP scores."""

    def __init__(self, d_model, n_heads, k_keep=2, alpha_topk=0.5):
        super().__init__()
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.k = k_keep
        self.alpha = alpha_topk
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.cif_proj = nn.Linear(d_model, d_model)

    def forward(self, x_grp, hp_child):                # x_grp [B,N,4,D], hp_child [B,N,4]
        B, N, C, D = x_grp.shape
        H, dh = self.n_heads, self.dh
        qkv = self.qkv(x_grp.reshape(-1, D)).view(B, N, C, 3, H, dh)
        q, k, v = qkv.unbind(dim=3)
        sal = q.norm(dim=-1).mean(-1)                  # [B, N, 4]
        if hp_child is not None and self.alpha != 0.0:
            sal = sal + self.alpha * hp_child
        k_eff = min(self.k, C)
        _, topk_idx = sal.topk(k_eff, dim=2)

        def gather_kept(t):
            t_flat = t.flatten(-2)
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, H * dh)
            return torch.gather(t_flat, 2, idx_exp).view(B, N, k_eff, H, dh)

        k_kept, v_kept = gather_kept(k), gather_kept(v)
        q_parent = q.mean(2)
        scale = 1.0 / math.sqrt(dh)
        att = (q_parent.unsqueeze(2) * k_kept).sum(-1) * scale
        w = att.softmax(2)
        o_parent = (w.unsqueeze(-1) * v_kept).sum(2)
        o_parent = o_parent.transpose(2, 3).reshape(B, N, D)
        o_parent = self.proj(o_parent)
        gate = torch.sigmoid(self.cif_proj(x_grp.reshape(-1, D))).view(B, N, C, D)
        x_grp = x_grp + gate * o_parent.unsqueeze(2)
        return o_parent, x_grp


# ─────────────────────────────────────────────────────────────────────────────
# Model A — adaptive quadtree (per-image buffers). Batching fixed: per-image
# roots are pooled independently and stacked to [B, D].
# ─────────────────────────────────────────────────────────────────────────────
class BottomUpAdaptiveQuadtree(nn.Module):
    def __init__(self, cfg: Config, pretrained=False):
        super().__init__()
        d, heads = cfg.d_model, cfg.heads
        self.patch = make_patch_embed(d, pretrained=pretrained)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N_PATCHES, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.level_attn = nn.ModuleList(
            [MxCIFSiblingAttention(d, heads, cfg.k_keep) for _ in range(cfg.max_levels)]
        )
        self.parent_gate = nn.Linear(d, 1, bias=True)
        self.layer_norm = nn.LayerNorm(d)
        self.temperature = 1.0
        self.head = nn.Linear(d, 1)
        self.buf_dir = None                            # set before forward

    def _pool_feats(self, feats):                      # feats: [T, D]
        if feats.size(0) == 0:
            return feats.new_zeros((feats.size(1),))
        logits = self.parent_gate(feats).squeeze(-1)   # [T]
        weights = torch.softmax(logits / self.temperature, dim=0)
        pooled = (weights.unsqueeze(-1) * feats).sum(0)
        return pooled + 0.1 * feats.mean(0)

    def forward(self, imgs, stems):
        assert self.buf_dir is not None, "set model.buf_dir before calling forward"
        B = imgs.size(0)
        dev = imgs.device
        x = self.patch(imgs)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], 1) + self.pos_embed
        x = x[:, 1:]                                   # drop CLS → [B, 196, D]

        roots = []
        for b in range(B):
            x0 = x[b].unsqueeze(0)                      # [1, 196, D] (original patch grid)
            per_feats = []
            buf_path = os.path.join(self.buf_dir, stems[b] + ".pt")
            if os.path.exists(buf_path):
                idxbuf, mskbuf = _torch_load(buf_path, dev)
                # collect_sibling_groups stores indices into the ORIGINAL 196-grid at
                # EVERY level, so each level gathers from x0 (not a reduced grid). This
                # is what makes the deeper levels actually contribute — gathering from a
                # shrinking parent grid silently skipped every level past the first.
                for l in range(min(idxbuf.size(0), len(self.level_attn))):
                    n = int(mskbuf[l].sum())
                    if n == 0:
                        continue
                    idx = idxbuf[l, :n].to(dev)         # [n, 4] indices into 196
                    if idx.numel() == 0 or idx.min() < 0 or idx.max() >= x0.size(1):
                        continue
                    child = x0.index_select(1, idx.reshape(-1)).view(1, n, 4, x0.size(-1))
                    _o_parent, x_grp = self.level_attn[l](child)
                    B2, n2, C2, D2 = x_grp.shape
                    x_grp = self.layer_norm(x_grp.reshape(-1, D2)).view(B2, n2, C2, D2)
                    per_feats.append(x_grp.mean(2).squeeze(0))   # [n, D]
            # Fallback when the FFT tree is degenerate (no sibling groups) or the cache
            # is missing: pool the raw patch tokens, so the logit stays input-dependent
            # and gradient still reaches the backbone (instead of a dead zero vector).
            feats = torch.cat(per_feats, 0) if per_feats else x0.squeeze(0)   # [T, D]
            roots.append(self._pool_feats(feats))
        roots = torch.stack(roots, 0)                  # [B, D]
        return self.head(roots).squeeze(-1)            # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Model B — fixed 2x2 quadtree + FFT-guided attention (batch-parallel)
# ─────────────────────────────────────────────────────────────────────────────
def build_sibling_buffers(H, W):
    """Fixed 2x2 quadtree pyramid (used by 'guided' mode). Ceil division + edge
    clamping so every token feeds the root: a 14x14 grid → levels [49, 16, 4, 1]."""
    levels, h, w = [], H, W
    while h > 1 or w > 1:
        nH, nW = (h + 1) // 2, (w + 1) // 2
        groups = []
        for i in range(nH):
            for j in range(nW):
                r0, r1 = min(2 * i, h - 1), min(2 * i + 1, h - 1)
                c0, c1 = min(2 * j, w - 1), min(2 * j + 1, w - 1)
                groups.append([r0 * w + c0, r0 * w + c1, r1 * w + c0, r1 * w + c1])
        levels.append(torch.tensor(groups, dtype=torch.long))
        h, w = nH, nW
    L = len(levels)
    Nmax = max(buf.size(0) for buf in levels)
    idxbuf = torch.full((L, Nmax, 4), -1, dtype=torch.long)
    mskbuf = torch.zeros((L, Nmax), dtype=torch.bool)
    for i, buf in enumerate(levels):
        idxbuf[i, :buf.size(0)] = buf
        mskbuf[i, :buf.size(0)] = True
    return idxbuf, mskbuf


class BottomUpGuidedQuadtree(nn.Module):
    def __init__(self, cfg: Config, pretrained=False):
        super().__init__()
        d, heads = cfg.d_model, cfg.heads
        self.patch = make_patch_embed(d, pretrained=pretrained)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N_PATCHES, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        idx, msk = build_sibling_buffers(GRID, GRID)
        self.register_buffer("sib_idx", idx)
        self.register_buffer("sib_msk", msk)
        self.level_attn = nn.ModuleList([
            MxCIFSiblingAttentionHP(d, heads, cfg.k_keep, cfg.alpha_topk)
            for _ in range(idx.size(0))
        ])
        self.parent_gate = nn.Linear(d, 1, bias=True)
        self.layer_norm = nn.LayerNorm(d)
        self.temperature = 1.0
        self.beta_pool = cfg.beta_pool
        self.head = nn.Linear(d, 1)
        self.hp_dir = None                             # set before forward

    def forward(self, imgs, stems):
        assert self.hp_dir is not None, "set model.hp_dir before calling forward"
        B = imgs.size(0)
        dev = imgs.device
        x = self.patch(imgs)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], 1) + self.pos_embed
        x = x[:, 1:]

        def _load_hp(s):
            path = os.path.join(self.hp_dir, s + ".pt")
            if os.path.exists(path):
                return _torch_load(path, "cpu").float()
            return torch.zeros(N_PATCHES)              # missing cache → no HP bias
        hp_batch = torch.stack([_load_hp(s) for s in stems], 0).to(dev)   # [B, 196]

        parent_bank, parent_hp = [], []
        for l, attn in enumerate(self.level_attn):
            n = int(self.sib_msk[l].sum().item())
            if n == 0:
                continue
            idx = self.sib_idx[l, :n].to(dev)          # [n, 4]
            flat = idx.reshape(-1)
            child = x.index_select(1, flat).view(B, n, 4, x.size(-1))
            hp_sel = hp_batch.index_select(1, flat).view(B, n, 4)
            o_parent, x_grp = attn(child, hp_sel)
            B2, n2, C2, D2 = x_grp.shape
            x_grp = self.layer_norm(x_grp.reshape(-1, D2)).view(B2, n2, C2, D2)
            parent_bank.append(x_grp.mean(2))          # [B, n, D]
            parent_hp.append(hp_sel.mean(2))           # [B, n]
            x = o_parent                               # [B, n, D]

        feats = torch.cat(parent_bank, 1) if parent_bank else x
        hp_scores = torch.cat(parent_hp, 1) if parent_hp else torch.zeros((B, 0), device=dev)
        logits = self.parent_gate(feats).squeeze(-1)   # [B, T]
        if hp_scores.numel() > 0 and self.beta_pool != 0.0:
            logits = logits + self.beta_pool * hp_scores
        weights = torch.softmax(logits / self.temperature, dim=1)
        gated = (weights.unsqueeze(-1) * feats).sum(1)
        root = gated + 0.1 * feats.mean(1)
        return self.head(root).squeeze(-1)             # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(data_dir, batch_size=32, val_pct=0.2, workers=4, seed=42):
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    class CleanImageFolder(ImageFolder):
        def find_classes(self, directory):
            classes = sorted(e.name for e in os.scandir(directory)
                             if e.is_dir() and not e.name.startswith("."))
            if not classes:
                raise FileNotFoundError(f"No class folders under {directory}")
            return classes, {c: i for i, c in enumerate(classes)}

    class WithStem(Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            path, label = self.base.samples[i]
            img = Image.open(path).convert("RGB")
            img = self.base.transform(img)
            return img, label, cache_key(path)         # matches the precomputed cache

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    full = WithStem(CleanImageFolder(data_dir, transform=tf))
    n_val = int(len(full) * val_pct)
    if len(full) > 1:                       # keep both splits non-empty
        n_val = min(len(full) - 1, max(1, n_val))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(full, [n_train, n_val], generator=g)
    print(f"[data] classes={full.base.classes} | train={len(tr)} | val={len(va)}")
    return (
        DataLoader(tr, batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(va, batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build / cache / train
# ─────────────────────────────────────────────────────────────────────────────
def build_model(mode, cfg, pretrained, cache_dir):
    if mode == "adaptive":
        model = BottomUpAdaptiveQuadtree(cfg, pretrained=pretrained)
        model.buf_dir = os.path.join(cache_dir, "buffers")
    elif mode == "guided":
        model = BottomUpGuidedQuadtree(cfg, pretrained=pretrained)
        model.hp_dir = os.path.join(cache_dir, "hp_maps")
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'adaptive' or 'guided')")
    return model


def ensure_cache(mode, data_dir, cache_dir, cfg, force=False):
    imgs = list_image_paths(data_dir)
    if not imgs:
        raise RuntimeError(f"No images found under {data_dir}")
    sub = "buffers" if mode == "adaptive" else "hp_maps"
    out_dir = os.path.join(cache_dir, sub)
    missing = imgs if force else [p for p in imgs if not (Path(out_dir) / f"{p.stem}.pt").exists()]
    if not missing:
        print(f"[cache] all {len(imgs)} {sub} present in {out_dir}")
        return
    if mode == "adaptive":
        precompute_adaptive_buffers(missing, out_dir, cfg)
    else:
        precompute_hp_maps(missing, out_dir, cfg)


def run_epoch(model, dl, opt, crit, device, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_acc = tot = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, stems in dl:
            x, y = x.to(device), y.to(device).float()
            if train:
                opt.zero_grad(set_to_none=True)
            logits = model(x, list(stems))
            loss = crit(logits, y)
            if train:
                loss.backward()
                opt.step()
            tot_loss += loss.item() * y.size(0)
            tot_acc += ((logits > 0).long() == y.long()).sum().item()
            tot += y.size(0)
    if tot == 0:
        return 0.0, 0.0
    return tot_loss / tot, tot_acc / tot


def train(args, cfg, device):
    ensure_cache(args.mode, args.data_dir, args.cache_dir, cfg, force=args.precompute)
    train_dl, val_dl = build_dataloaders(args.data_dir, args.batch_size,
                                         args.val_pct, args.workers)
    model = build_model(args.mode, cfg, args.pretrained, args.cache_dir).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    print(f"[train] mode={args.mode} | {args.epochs} epochs on {device}")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, opt, crit, device, train=True)
        vl, va = run_epoch(model, val_dl, opt, crit, device, train=False)
        print(f"E{ep:02d} | {time.time()-t0:5.1f}s | "
              f"train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, f"fourier_quadtree_{args.mode}.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[train] saved → {ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualize the FFT-guided adaptive quadtree on an image
# ─────────────────────────────────────────────────────────────────────────────
def visualize(image_path, out_path, cfg: Config, size=224):
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for --visualize")
    from PIL import Image
    im = np.array(Image.open(image_path).convert("RGB").resize((size, size)))
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hp = highpass_filter(gray, cfg.r_hp)
    root = QuadNode(0, 0, size, size)
    build_quadtree(root, hp, cfg)
    overlay = im.copy()
    draw_quadtree(root, overlay)
    Image.fromarray(overlay).save(out_path)
    print(f"[viz] FFT-guided quadtree (R_HP={cfg.r_hp}) → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — synthetic ImageFolder, both modes, forward + backward on CPU
# ─────────────────────────────────────────────────────────────────────────────
def smoke_test(cfg, device):
    import shutil
    import tempfile
    from PIL import Image

    def structured(j):
        """A high-frequency image (checkerboard + shapes) so the adaptive quadtree
        actually splits — pure noise has a uniform high-pass map and never splits."""
        a = np.zeros((224, 224, 3), np.uint8)
        a[:, :, 0] = np.linspace(0, 255, 224).astype(np.uint8)[None, :]
        t = 8 + j * 3
        yy, xx = np.mgrid[0:224, 0:224]
        a[:, :, 1] = (((xx // t + yy // t) % 2) * 255).astype(np.uint8)   # checkerboard
        a[40 + j * 6:130 + j * 6, 50:150] = 255                          # bright square
        return a

    tmp = Path(tempfile.mkdtemp(prefix="fft_qt_smoke_"))
    data_dir = tmp / "data"
    cache_dir = tmp / "cache"
    n_per_class = 4
    for ci, cls in enumerate(("class0", "class1")):
        (data_dir / cls).mkdir(parents=True, exist_ok=True)
        for j in range(n_per_class):
            Image.fromarray(structured(2 * j + ci)).save(data_dir / cls / f"{cls}_{j:03d}.png")
    print(f"[smoke] synthetic dataset at {data_dir} ({2 * n_per_class} images)")

    try:
        for mode in ("adaptive", "guided"):
            print(f"\n[smoke] === mode={mode} ===")
            ensure_cache(mode, str(data_dir), str(cache_dir), cfg, force=True)
            train_dl, _ = build_dataloaders(str(data_dir), batch_size=4,
                                            val_pct=0.25, workers=0)
            model = build_model(mode, cfg, pretrained=False, cache_dir=str(cache_dir)).to(device)
            crit = nn.BCEWithLogitsLoss()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x, y, stems = next(iter(train_dl))
            x, y = x.to(device), y.to(device).float()

            logits = model(x, list(stems))
            assert logits.shape == (x.size(0),), \
                f"[{mode}] expected logits [{x.size(0)}], got {tuple(logits.shape)}"
            assert torch.isfinite(logits).all(), f"[{mode}] non-finite logits"

            # per-image independence: permuting the batch must permute the logits
            model.eval()
            with torch.no_grad():
                perm = torch.tensor([3, 2, 1, 0])[: x.size(0)]
                lp = model(x[perm], [list(stems)[i] for i in perm.tolist()])
                base = model(x, list(stems))
            assert torch.allclose(lp, base[perm], atol=1e-4), \
                f"[{mode}] logits are NOT per-image independent (batch leakage)"
            model.train()

            loss = crit(model(x, list(stems)), y)
            loss.backward()
            patch_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                             for p in model.patch.parameters())
            attn_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for lev in model.level_attn for p in lev.parameters())
            assert patch_grad, f"[{mode}] no gradient reached the patch embed"
            assert attn_grad, f"[{mode}] no gradient reached the sibling attention"
            opt.step()
            print(f"[smoke] {mode}: logits {tuple(logits.shape)} | loss {loss.item():.4f} | "
                  f"per-image independent ✓ | patch+attn grads ✓")
        print("\n[smoke] OK ✅  both modes: forward + backward, independent logits, live gradients.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["adaptive", "guided"], default="adaptive")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--visualize", metavar="IMG")
    ap.add_argument("--out", default="qt_fft.png")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data_dir")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_pct", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--precompute", action="store_true", help="force-rebuild the cache")
    ap.add_argument("--pretrained", action="store_true", help="timm ViT pretrained weights")
    # FFT / quadtree knobs (override Config defaults)
    ap.add_argument("--r_hp", type=int, default=Config.r_hp)
    ap.add_argument("--mean_thresh", type=float, default=Config.mean_thresh)
    ap.add_argument("--std_thresh", type=float, default=Config.std_thresh)
    ap.add_argument("--max_depth", type=int, default=Config.max_depth)
    ap.add_argument("--alpha_topk", type=float, default=Config.alpha_topk)
    ap.add_argument("--beta_pool", type=float, default=Config.beta_pool)
    ap.add_argument("--k_keep", type=int, default=Config.k_keep)
    args = ap.parse_args()

    cfg = Config(r_hp=args.r_hp, mean_thresh=args.mean_thresh, std_thresh=args.std_thresh,
                 max_depth=args.max_depth, alpha_topk=args.alpha_topk,
                 beta_pool=args.beta_pool, k_keep=args.k_keep)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.visualize:
        visualize(args.visualize, args.out, cfg)
    elif args.train:
        if not args.data_dir:
            sys.exit("--train requires --data_dir")
        train(args, cfg, device)
    elif args.smoke_test:
        smoke_test(cfg, device)
    else:
        smoke_test(cfg, device)   # default action when no command is given


if __name__ == "__main__":
    main()
