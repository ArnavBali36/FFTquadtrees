"""
quadtree_attention.py — Quadtree Sibling Attention (structural / no-FFT version)
================================================================================

This is the "just makes quadtrees" version of the model. The quadtree is a
fixed, data-independent 2x2 pyramid built over the ViT patch grid (14x14 = 196
patches for a 224x224 image). At each level, groups of 4 sibling patches attend
to each other (MX-CiF Top-K sibling attention), are pooled into a parent, and
the parent grid is recursed until a single root token remains, which is
classified.

There is NO Fourier / high-pass guidance here — the tree structure is purely
geometric. For the FFT-driven (content-adaptive) variant, see
`fourier_quadtree.py`.

Consolidated from the `Quad2x2FixedKWithHead` model in the project notebooks.

Usage
-----
Quick check (no dataset, no pretrained weights, runs on CPU):

    python quadtree_attention.py --smoke-test

Visualize a pure intensity quadtree decomposition of an image:

    python quadtree_attention.py --visualize path/to/image.jpg --out qt.png

Train on a folder of images (ImageFolder layout: <root>/<class>/<img>):

    python quadtree_attention.py --train --data_dir /path/to/IllusionAnimals_train \
                                 --epochs 10 --batch_size 32
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except Exception:  # pragma: no cover - timm is optional for the smoke test
    _HAS_TIMM = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover - cv2 only needed for --visualize
    _HAS_CV2 = False


# ─────────────────────────────────────────────────────────────────────────────
# 1) Patch embedding (timm ViT if available, else a plain Conv2d fallback)
# ─────────────────────────────────────────────────────────────────────────────
class ConvPatchEmbed(nn.Module):
    """Fallback patch embed (16x16 conv) so the model runs without timm."""

    def __init__(self, in_ch=3, d_model=768, patch=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):                       # x: [B, 3, H, W]
        x = self.proj(x)                        # [B, D, H/16, W/16]
        return x.flatten(2).transpose(1, 2)     # [B, N, D]


def make_patch_embed(d_model=768, pretrained=False):
    """Return a ViT patch_embed from timm, or a Conv2d fallback."""
    if _HAS_TIMM:
        vit = timm.create_model("vit_base_patch16_224",
                                pretrained=pretrained, num_classes=0)
        return vit.patch_embed
    print("[warn] timm not available — using a plain Conv2d patch embed.")
    return ConvPatchEmbed(3, d_model, 16)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Build the fixed quadtree (sibling-group buffers)
#    This is the part that "just makes quadtrees": a uniform 2x2 pyramid.
# ─────────────────────────────────────────────────────────────────────────────
def build_sibling_buffers(H: int, W: int):
    """
    Build a uniform 2x2 quadtree over an H x W token grid.

    Returns
    -------
    idxbuf : LongTensor [L, Nmax, 4]   sibling indices per level (padded with -1)
    mskbuf : BoolTensor [L, Nmax]      valid-group mask per level

    At level 0 the indices address the H*W input tokens; at each subsequent
    level they address the parent tokens produced by the level below.

    Uses ceil division with edge clamping so EVERY token feeds the root (a 14x14
    grid gives levels [49, 16, 4, 1] → 196→49→16→4→1). A naive floor-division
    pyramid would silently drop the right/bottom border at each level and lose
    most of the image; clamping odd-sized borders to the last row/col avoids that.
    """
    levels, h, w = [], H, W
    while h > 1 or w > 1:
        nH, nW = (h + 1) // 2, (w + 1) // 2
        groups = []
        for i in range(nH):
            for j in range(nW):
                r0, r1 = min(2 * i, h - 1), min(2 * i + 1, h - 1)
                c0, c1 = min(2 * j, w - 1), min(2 * j + 1, w - 1)
                groups.append([r0 * w + c0, r0 * w + c1, r1 * w + c0, r1 * w + c1])
        levels.append(torch.tensor(groups, dtype=torch.long))   # [n, 4]
        h, w = nH, nW

    L = len(levels)
    Nmax = max(buf.size(0) for buf in levels)
    idxbuf = torch.full((L, Nmax, 4), -1, dtype=torch.long)
    mskbuf = torch.zeros((L, Nmax), dtype=torch.bool)
    for i, buf in enumerate(levels):
        idxbuf[i, :buf.size(0)] = buf
        mskbuf[i, :buf.size(0)] = True
    return idxbuf, mskbuf


# ─────────────────────────────────────────────────────────────────────────────
# 3) MX-CiF Top-K sibling attention (pairwise within a group of 4)
# ─────────────────────────────────────────────────────────────────────────────
class MxCIFSiblingAttention(nn.Module):
    """
    Within each group of 4 sibling tokens, every child attends to its `k_keep`
    most relevant siblings (Top-K sparsity), then a CiF (conditional-input
    fusion) gate updates the children. The group is pooled (mean) into a parent.
    """

    def __init__(self, d_model: int, n_heads: int, k_keep: int = 3):
        super().__init__()
        assert d_model % n_heads == 0
        self.nh, self.dh = n_heads, d_model // n_heads
        self.k_keep = max(1, min(4, k_keep))
        self.pre_ln = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.cif_proj = nn.Linear(d_model, d_model)
        self.out_scale = nn.Parameter(torch.tensor([0.35]))
        self.local_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(4 * d_model, d_model), nn.Dropout(0.10),
        )

    def forward(self, x_grp):                      # x_grp: [B, N, 4, D]
        B, N, C, D = x_grp.shape
        H, dh = self.nh, self.dh
        x_in = self.pre_ln(x_grp)
        qkv = self.qkv(x_in.reshape(-1, D)).view(B, N, C, 3, H, dh)
        q, k, v = (qkv[..., i, :, :] for i in (0, 1, 2))
        qh, kh, vh = (t.permute(0, 1, 3, 2, 4).contiguous() for t in (q, k, v))
        scale = dh ** -0.5

        # Attention in fp32 for numerical stability (the Top-K masking is sharp).
        with torch.amp.autocast("cuda", enabled=False):
            q32, k32, v32 = qh.float(), kh.float(), vh.float()
            scores = torch.einsum("bnhcd,bnhkd->bnhck", q32, k32) * scale
            if self.k_keep < 4:
                _topv, topi = torch.topk(scores, k=self.k_keep, dim=-1)
                with torch.no_grad():
                    mu = scores.detach().mean(dim=-1, keepdim=True)
                    std = scores.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)
                    floor = mu - 6.0 * std
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask.scatter_(-1, topi, True)
                scores = torch.where(mask, scores, floor)
            scores = scores - scores.amax(dim=-1, keepdim=True)
            w = torch.softmax(scores, dim=-1)
            mix = torch.einsum("bnhck,bnhkd->bnhcd", w, v32)
            child_upd = mix.permute(0, 1, 3, 2, 4).contiguous().view(B, N, 4, D).to(x_grp.dtype)

        gate = torch.sigmoid(self.cif_proj(x_grp.reshape(-1, D))).view(B, N, 4, D)
        x_grp_out = x_grp + torch.clamp(self.out_scale, 0.05, 1.20) * gate * (child_upd - x_grp)
        x_ffn = self.local_ffn(x_grp_out.reshape(-1, D)).view(B, N, 4, D)
        x_grp_out = x_grp_out + 0.5 * x_ffn
        return x_grp_out.mean(dim=2)               # parent tokens: [B, N, D]


# ─────────────────────────────────────────────────────────────────────────────
# 4) Full model: ViT patch embed → fixed quadtree sibling attention → head
# ─────────────────────────────────────────────────────────────────────────────
class Quad2x2SiblingAttention(nn.Module):
    """
    Bottom-up quadtree classifier with a fixed 2x2 sibling structure.
    Output is a single logit per image (binary classification by default).
    """

    def __init__(self, d_model=768, heads=12, patch_hw=14, k_keep=3,
                 num_classes=1, pretrained=False):
        super().__init__()
        self.patch = make_patch_embed(d_model, pretrained=pretrained)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + patch_hw * patch_hw, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        idx, msk = build_sibling_buffers(patch_hw, patch_hw)
        self.register_buffer("sib_idx", idx)       # [L, Nmax, 4]
        self.register_buffer("sib_msk", msk)       # [L, Nmax]

        self.levels = nn.ModuleList([
            MxCIFSiblingAttention(d_model, heads, k_keep=k_keep)
            for _ in range(self.sib_idx.size(0))
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(4 * d_model, d_model), nn.Dropout(0.10),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, imgs):                       # imgs: [B, 3, 224, 224]
        B = imgs.size(0)
        x = self.patch(imgs)                       # [B, 196, D]
        x = torch.cat([self.cls_token.expand(B, 1, -1), x], 1) + self.pos_embed
        x = x[:, 1:]                               # drop CLS → [B, 196, D]

        for lvl, attn in enumerate(self.levels):
            n = int(self.sib_msk[lvl].sum())
            idx = self.sib_idx[lvl, :n].to(x.device)   # [n, 4]
            child = x[:, idx]                          # [B, n, 4, D]
            x = attn(child)                            # [B, n, D]

        x = self.norm(x).mean(dim=1)               # [B, D]
        return self.head(x).squeeze(-1)            # [B] (num_classes=1)


def build_quadtree_model(img_size=224, num_classes=1, k_keep=3, pretrained=False):
    return Quad2x2SiblingAttention(
        d_model=768, heads=12, patch_hw=img_size // 16,
        k_keep=k_keep, num_classes=num_classes, pretrained=pretrained,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5) Pure quadtree construction + visualization (intensity-based, no model)
#    This is the literal "make a quadtree from an image" utility.
# ─────────────────────────────────────────────────────────────────────────────
class QuadNode:
    def __init__(self, x, y, w, h, depth=0):
        self.x, self.y, self.width, self.height = x, y, w, h
        self.depth = depth
        self.leaf = False
        self.children = []


def build_intensity_quadtree(node, img, mean_thresh=0.6, std_thresh=0.1,
                             max_depth=5, min_size=8):
    """Recursively split a node when its region is high-variance (detailed)."""
    patch = img[node.y:node.y + node.height, node.x:node.x + node.width]
    m, s = float(patch.mean()), float(patch.std())
    if m >= mean_thresh or min(node.width, node.height) < min_size or node.depth >= max_depth:
        node.leaf = True
    elif s > std_thresh:
        w2, h2 = node.width // 2, node.height // 2
        for dx, dy in [(0, 0), (w2, 0), (0, h2), (w2, h2)]:
            cw = w2 if dx == 0 else node.width - w2
            ch = h2 if dy == 0 else node.height - h2
            c = QuadNode(node.x + dx, node.y + dy, cw, ch, node.depth + 1)
            node.children.append(c)
            build_intensity_quadtree(c, img, mean_thresh, std_thresh, max_depth, min_size)
    else:
        node.leaf = True


def draw_quadtree(node, canvas):
    """Draw leaf-node rectangles onto an RGB uint8 canvas (needs OpenCV)."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for --visualize; pip install opencv-python")
    if node.leaf:
        cv2.rectangle(canvas, (node.x, node.y),
                      (node.x + node.width, node.y + node.height), (0, 0, 255), 1)
    else:
        for c in node.children:
            draw_quadtree(c, canvas)


def visualize_quadtree(image_path, out_path="quadtree.png", size=224,
                       mean_thresh=0.6, std_thresh=0.1, max_depth=5):
    """Build an intensity quadtree on an image and save the overlay."""
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for --visualize; pip install opencv-python")
    from PIL import Image

    im = np.array(Image.open(image_path).convert("RGB").resize((size, size)))
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    root = QuadNode(0, 0, size, size)
    build_intensity_quadtree(root, gray, mean_thresh, std_thresh, max_depth)
    overlay = im.copy()
    draw_quadtree(root, overlay)
    Image.fromarray(overlay).save(out_path)
    print(f"[viz] wrote quadtree overlay → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6) Optional training on an ImageFolder (binary: BCEWithLogits)
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(data_dir, batch_size=32, val_pct=0.2, workers=4, seed=42):
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    class CleanImageFolder(ImageFolder):
        """ImageFolder that skips hidden dirs (.git, .ipynb_checkpoints, …)."""
        def find_classes(self, directory):
            classes = sorted(e.name for e in os.scandir(directory)
                             if e.is_dir() and not e.name.startswith("."))
            if not classes:
                raise FileNotFoundError(f"No class folders under {directory}")
            return classes, {c: i for i, c in enumerate(classes)}

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    full = CleanImageFolder(data_dir, transform=tf)
    n_val = int(len(full) * val_pct)
    if len(full) > 1:                       # keep both splits non-empty
        n_val = min(len(full) - 1, max(1, n_val))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(full, [n_train, n_val], generator=g)
    print(f"[data] classes={full.classes} | train={len(tr)} | val={len(va)}")
    return (
        DataLoader(tr, batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(va, batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    )


def run_epoch(model, dl, opt, crit, device, train=True):
    model.train() if train else model.eval()
    tot_loss = tot_acc = tot = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in dl:
            x, y = x.to(device), y.to(device).float()
            if train:
                opt.zero_grad(set_to_none=True)
            logits = model(x)
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


def train(args, device):
    train_dl, val_dl = build_dataloaders(args.data_dir, args.batch_size,
                                         args.val_pct, args.workers)
    model = build_quadtree_model(pretrained=args.pretrained, k_keep=args.k_keep).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    print(f"[train] {args.epochs} epochs on {device}")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, opt, crit, device, train=True)
        vl, va = run_epoch(model, val_dl, opt, crit, device, train=False)
        print(f"E{ep:02d} | {time.time()-t0:5.1f}s | "
              f"train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
    ckpt = os.path.join(args.out_dir, "quadtree_attention.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[train] saved → {ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# 7) Smoke test — verifies forward + backward without data / pretrained weights
# ─────────────────────────────────────────────────────────────────────────────
def smoke_test(device):
    print("[smoke] building model (pretrained=False)…")
    model = build_quadtree_model(pretrained=False, k_keep=3).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.tensor([0.0, 1.0], device=device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    logits = model(x)
    assert logits.shape == (2,), f"expected logits [2], got {tuple(logits.shape)}"
    loss = crit(logits, y)
    loss.backward()
    opt.step()

    # the patch embed and the sibling attention must both receive gradient
    patch_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in model.patch.parameters())
    attn_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.levels[0].parameters())
    assert patch_grad, "no gradient reached the patch embed"
    assert attn_grad, "no gradient reached the sibling attention"

    idx, msk = build_sibling_buffers(14, 14)
    levels = [int(msk[l].sum()) for l in range(idx.size(0))]
    covered = len(set(int(v) for v in idx[0][msk[0]].reshape(-1).tolist()))
    print(f"[smoke] quadtree levels (groups per level): {levels}  (196 → … → 1)")
    print(f"[smoke] level-0 token coverage: {covered}/196 patches")
    assert levels[0] == 49 and covered == 196, "quadtree does not cover all 196 patches"
    print(f"[smoke] logits={logits.detach().cpu().numpy().round(4)} | loss={loss.item():.4f}")
    print("[smoke] OK ✅  forward + backward pass succeeded, gradients flow.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke-test", action="store_true", help="run a tiny CPU sanity check")
    ap.add_argument("--visualize", metavar="IMG", help="draw an intensity quadtree on IMG")
    ap.add_argument("--out", default="quadtree.png", help="output path for --visualize")
    ap.add_argument("--train", action="store_true", help="train on --data_dir")
    ap.add_argument("--data_dir", help="ImageFolder root (<root>/<class>/<img>)")
    ap.add_argument("--out_dir", default="runs", help="checkpoint dir for --train")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_pct", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k_keep", type=int, default=3, help="Top-K siblings to keep (1..4)")
    ap.add_argument("--pretrained", action="store_true", help="load timm ViT pretrained weights")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.visualize:
        visualize_quadtree(args.visualize, args.out)
    elif args.train:
        if not args.data_dir:
            sys.exit("--train requires --data_dir")
        train(args, device)
    elif args.smoke_test:
        smoke_test(device)
    else:
        smoke_test(device)        # default action when no command is given


if __name__ == "__main__":
    main()
