# === ONE-CELL PIPELINE: Precompute (your HP quadtree) + Train (MX-CiF vs ViT) with verbose logs ===
import os, math, time, gc, sys, hashlib
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ------------------------------ CONFIG ------------------------------
# Point this to your IllusionAnimals dataset:
# If your data is a directory with class subfolders (0,1, or animal classes):
DATA_SRC = "/content/drive/MyDrive/IllusionAnimals_train"   # <-- CHANGE if needed

# Where to store your per-image quadtree buffers (*.pt)
BUF_DIR  = "/content/IllusionAnimals_buffers"
os.makedirs(BUF_DIR, exist_ok=True)

# Your quadtree parameters (as requested)
R_HP        = 50
MEAN_THRESH = 0.2
STD_THRESH  = 0.04
MAX_DEPTH   = 3

# Training params (kept same as friend’s code)
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 3e-4
VAL_PCT    = 0.2
NUM_WORKERS = 4

# Logging knobs
LOG_PRECOMPUTE_EVERY = 25   # print progress every N images
LOG_TRAIN_EVERY      = 10   # print batch metrics every N batches

# ------------------------- UTIL: device + mem ------------------------
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gpu_mem():
    if torch.cuda.is_available():
        return f"GPU mem: {torch.cuda.memory_allocated()/1e6:.1f}MB / {torch.cuda.memory_reserved()/1e6:.1f}MB"
    return "CPU"

# --------------------- PRECOMPUTE: HP + Quadtree ---------------------
def highpass_filter(img_np, r_hp):
    # FFT high-pass magnitude in [0,1]
    fshift = np.fft.fftshift(np.fft.fft2(img_np))
    h,w = img_np.shape
    cy,cx = h//2, w//2
    Y,X = np.ogrid[:h,:w]
    mask = ((Y-cy)**2+(X-cx)**2) > (r_hp**2)
    recon = np.fft.ifft2(np.fft.ifftshift(fshift * mask))
    mag = np.abs(recon)
    mag -= mag.min()
    mag /= (mag.max() + 1e-8)
    return mag

class QuadNode:
    def __init__(self,x,y,w,h,depth=0):
        self.x,self.y,self.width,self.height = x,y,w,h
        self.depth  = depth
        self.leaf   = False
        self.children = []

def build_quadtree(node, img_hp):
    patch = img_hp[node.y:node.y+node.height, node.x:node.x+node.width]
    m, s  = patch.mean(), patch.std()
    if m >= MEAN_THRESH or min(node.width, node.height) < 8 or node.depth >= MAX_DEPTH:
        node.leaf = True
    elif s > STD_THRESH:
        w2, h2 = node.width // 2, node.height // 2
        for dx, dy in [(0,0),(w2,0),(0,h2),(w2,h2)]:
            cw = w2 if dx == 0 else node.width - w2
            ch = h2 if dy == 0 else node.height - h2
            c  = QuadNode(node.x + dx, node.y + dy, cw, ch, node.depth + 1)
            node.children.append(c)
            build_quadtree(c, img_hp)
    else:
        node.leaf = True

def collect_sibling_groups(node, level, groups):
    # groups[level] : list of [i00, i01, i10, i11] indices into 14x14 patches
    if level >= len(groups): groups.append([])
    if not node.leaf and len(node.children) == 4:
        idxs = []
        for c in node.children:
            py, px = c.y // 16, c.x // 16   # 224/16 = 14; map pixel to patch index
            idxs.append(py * 14 + px)
        groups[level].append(idxs)
        for c in node.children:
            collect_sibling_groups(c, level+1, groups)

def list_image_paths(root):
    root = Path(root)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def precompute_buffers(image_paths):
    print(f"\n[Precompute] Start | images: {len(image_paths)} | params: R_HP={R_HP}, MEAN>={MEAN_THRESH}, STD>{STD_THRESH}, MAX_DEPTH={MAX_DEPTH}")
    t0 = time.time()
    # pass 1: find maxima
    L_max = N_max = 0
    for i, p in enumerate(image_paths, 1):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[Precompute][WARN] Could not read {p}; skipping.")
            continue
        gray = cv2.resize(gray, (224,224))
        hp   = highpass_filter(gray, R_HP)
        root = QuadNode(0,0,224,224)
        build_quadtree(root, hp)
        groups = []
        collect_sibling_groups(root, 0, groups)
        L_max = max(L_max, len(groups))
        if groups:
            N_max = max(N_max, max(len(g) for g in groups))
        if i % LOG_PRECOMPUTE_EVERY == 0:
            print(f"[Precompute] pass1 {i}/{len(image_paths)} | L_max={L_max} N_max={N_max}")
    L_max = max(L_max, 1)
    N_max = max(N_max, 1)
    print(f"[Precompute] pass1 done | L_max={L_max}, N_max={N_max} | {time.time()-t0:.1f}s")

    # pass 2: save buffers
    wrote = 0
    for i, p in enumerate(image_paths, 1):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        gray = cv2.resize(gray, (224,224))
        hp   = highpass_filter(gray, R_HP)
        root = QuadNode(0,0,224,224)
        build_quadtree(root, hp)
        groups = []
        collect_sibling_groups(root, 0, groups)

        idxbuf = torch.full((L_max, N_max, 4), -1, dtype=torch.long)
        mskbuf = torch.zeros((L_max, N_max), dtype=torch.bool)
        for lvl, g in enumerate(groups):
            if not g: continue
            buf = torch.tensor(g, dtype=torch.long)
            # Clamp to valid 14x14 patch range just in case
            buf = buf.clamp_(min=0, max=14*14 - 1)
            idxbuf[lvl, :buf.size(0), :] = buf
            mskbuf[lvl, :buf.size(0)]    = 1

        stem = p.stem
        torch.save((idxbuf, mskbuf), os.path.join(BUF_DIR, f"{stem}.pt"))
        wrote += 1
        if i % LOG_PRECOMPUTE_EVERY == 0:
            print(f"[Precompute] pass2 {i}/{len(image_paths)} | saved {stem}.pt | {gpu_mem()}")
    print(f"[Precompute] done | saved {wrote} buffers in {time.time()-t0:.1f}s | {BUF_DIR}")

# ---------------------- DATA: ImageFolder loader ---------------------
# This mirrors your friend’s build_dataloaders, but robust to local folder datasets.
class HFDatasetWithName(Dataset):
    def __init__(self, recs, tf):
        self.recs = recs
        self.tf = tf
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        img = self.tf(r['image'].convert('RGB'))
        label = r['label']
        # Try to get stem from HF sample; fallback to hash if missing
        stem = Path(getattr(r['image'], 'filename', f"sample_{i}")).stem
        return img, label, stem

def load_hf_records_from_imagefolder(data_dir):
    # Build HF-like list of dicts: {'image': PIL, 'label': int}
    items = []
    classes = sorted([p for p in Path(data_dir).iterdir() if p.is_dir()])
    label_map = {cls.name: idx for idx, cls in enumerate(classes)}
    for cls in classes:
        for p in list_image_paths(cls):
            try:
                img = Image.open(p).convert('RGB')
            except Exception:
                continue
            items.append({'image': img, 'label': label_map[cls.name]})
            # attach filename attribute so we can recover stem
            setattr(items[-1]['image'], 'filename', str(p))
    return items

def build_dataloaders_with_names(path, bs=BATCH_SIZE, val_pct=VAL_PCT):
    # Try HF datasets.load_dataset(path) first; if that fails, fall back to local imagefolder
    recs = None
    try:
        from datasets import load_dataset
        print(f"[Loader] Trying datasets.load_dataset('{path}', split='train')...")
        recs = list(load_dataset(path, split='train'))
        print(f"[Loader] Loaded {len(recs)} samples via HF hub/script id='{path}'.")
    except Exception as e:
        print(f"[Loader] Fallback to local imagefolder at: {path} ({e})")
        recs = load_hf_records_from_imagefolder(path)
        print(f"[Loader] Local imagefolder loaded: {len(recs)} samples.")

    # stratify if possible
    y = [r['label'] for r in recs]
    tr, val = train_test_split(recs, test_size=val_pct,
                               stratify=y if len(set(y))>1 else None,
                               random_state=42)
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3,(0.5,)*3),
    ])
    train_dl = DataLoader(HFDatasetWithName(tr,tf), bs, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(HFDatasetWithName(val,tf), bs, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_dl, val_dl

# ------------------------ MODEL: Sibling attention -------------------
class MxCIFSiblingAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, k_keep:int=2):
        super().__init__()
        self.n_heads = n_heads
        self.dh      = d_model // n_heads
        self.k       = k_keep
        self.qkv      = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj     = nn.Linear(d_model, d_model, bias=False)
        self.cif_proj = nn.Linear(d_model, d_model)

    def forward(self, x_grp: torch.Tensor):
        B, N, C, D = x_grp.shape
        H, dh = self.n_heads, self.dh
        qkv = self.qkv(x_grp.reshape(-1, D)).view(B, N, C, 3, H, dh)
        q, k, v = qkv.unbind(dim=3)
        sal = q.norm(dim=-1).mean(-1)          # [B,N,4]
        _, topk_idx = sal.topk(self.k, dim=2)  # [B,N,k]
        def gather_kept(t):
            t_flat = t.flatten(-2)             # [B,N,4,H*dh]
            idx_exp = topk_idx.unsqueeze(-1).expand(-1,-1,-1,H*dh)
            kept = torch.gather(t_flat, 2, idx_exp)
            return kept.view(B, N, self.k, H, dh)
        k_kept, v_kept = gather_kept(k), gather_kept(v)
        q_parent = q.mean(2)                   # [B,N,H,dh]
        scale = 1.0 / math.sqrt(dh)
        att = (q_parent.unsqueeze(2) * k_kept).sum(-1) * scale  # [B,N,k,H]
        w = att.softmax(2)
        o_parent = (w.unsqueeze(-1) * v_kept).sum(2)            # [B,N,H,dh]
        o_parent = o_parent.transpose(2,3).reshape(B, N, D)
        o_parent = self.proj(o_parent)

        # CiF fusion (residual gate on children)
        gate = torch.sigmoid(self.cif_proj(x_grp.reshape(-1, D))).view(B, N, C, D)
        x_grp = x_grp + gate * o_parent.unsqueeze(2)

        return o_parent, x_grp

# --------------------- MODEL: Quadtree + MX-CiF ---------------------
class BottomUpQuadtreeMxCiF(nn.Module):
    """
    Same hierarchy attention as your friend's code, but sibling groups come from
    YOUR precomputed quadtree buffers saved per image stem in BUF_DIR.
    """
    def __init__(self, d_model=768, heads=12, k_keep=2, max_levels=8):
        super().__init__()
        vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.patch     = vit.patch_embed
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1,1+14*14,d_model))  # 224/16=14
        self.level_attn  = nn.ModuleList([
            MxCIFSiblingAttention(d_model, heads, k_keep)
            for _ in range(max_levels)
        ])
        self.parent_gate = nn.Linear(d_model,1,bias=True)
        self.layer_norm  = nn.LayerNorm(d_model)
        self.temperature = 1.0
        self.head        = nn.Linear(d_model,1)

    def _pool_feats(self, feats):
        # feats: [n_groups, D] per-image
        if feats.size(0) == 0:
            return feats.new_zeros((feats.size(1),))  # [D]
        logits = self.parent_gate(feats).squeeze(-1)           # [T]
        weights = torch.softmax(logits/self.temperature, dim=0)
        pooled  = (weights.unsqueeze(-1)*feats).sum(0)         # [D]
        root    = pooled + 0.1*feats.mean(0)
        return root

    def forward(self, imgs, buf_names):
        B = imgs.size(0)
        x = self.patch(imgs)  # [B, 196, D]
        x = torch.cat([self.cls_token.expand(B,-1,-1), x],1) + self.pos_embed
        x = x[:,1:]  # drop CLS
        roots = []
        dev = imgs.device

        for b in range(B):
            stem = buf_names[b]
            idxbuf, mskbuf = torch.load(os.path.join(BUF_DIR, stem + ".pt"),
                                        map_location=dev)
            xb = x[b].unsqueeze(0)  # [1, 196, D]
            per_feats = []

            L_use = min(idxbuf.size(0), len(self.level_attn))
            for l in range(L_use):
                n = int(mskbuf[l].sum())
                if n == 0:
                    continue
                idx = idxbuf[l, :n].to(dev)                 # [n,4]
                # sanity check
                if idx.numel()==0 or idx.min()<0 or idx.max()>=xb.size(1):
                    continue
                flat_idx = idx.reshape(-1)                  # [n*4]
                sel = xb.index_select(dim=1, index=flat_idx)  # [1, n*4, D]
                child = sel.view(1, n, 4, xb.size(-1))        # [1, n, 4, D]
                o_parent, x_grp = self.level_attn[l](child)

                # LayerNorm over D
                B2,n2,C2,D2 = x_grp.shape
                x_grp = self.layer_norm(x_grp.reshape(-1, D2)).view(B2, n2, C2, D2)

                per_feats.append(x_grp.mean(2).squeeze(0))  # [n, D]
                xb = o_parent                                # [1, n, D]

            feats = torch.cat(per_feats, dim=0) if per_feats else xb.new_zeros((0, xb.size(-1)))
            roots.append(self._pool_feats(feats))

        roots = torch.stack(roots, dim=0)  # [B, D]
        return self.head(roots).squeeze(-1)

# --------------------- Data + Precompute orchestration ----------------
def ensure_buffers_for_dataset(data_root):
    # Build a list of image paths and precompute any missing buffers
    imgs = list_image_paths(data_root)
    if not imgs:
        raise RuntimeError(f"No images found under {data_root}")
    missing = []
    for p in imgs:
        if not (Path(BUF_DIR)/f"{p.stem}.pt").exists():
            missing.append(p)
    if missing:
        print(f"[Precompute] Missing buffers: {len(missing)}/{len(imgs)}. Generating now…")
        precompute_buffers(missing)
    else:
        print(f"[Precompute] All buffers present for {len(imgs)} images.")

# --------------------------- Training utils --------------------------
def run_epoch_with_names(model, dl, opt, crit, train=True, device='cuda', name="model"):
    model.train() if train else model.eval()
    tot_loss = tot_acc = tot = 0
    t0 = time.time()
    for b_idx, (x,y,stems) in enumerate(dl, 1):
        x = x.to(device); y = y.to(device).float()
        if train:
            opt.zero_grad(set_to_none=True)
            logits = model(x, list(stems)).squeeze(-1)
            loss   = crit(logits, y)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                logits = model(x, list(stems)).squeeze(-1)
                loss   = crit(logits, y)
        tot_loss  += loss.item() * y.size(0)
        tot_acc   += ((logits>0).long()==y.long()).sum().item()
        tot       += y.size(0)

        if b_idx % LOG_TRAIN_EVERY == 0 or b_idx == len(dl):
            print(f"[{name}][{'train' if train else 'val'}] "
                  f"batch {b_idx}/{len(dl)} | "
                  f"loss {tot_loss/tot:.4f} | acc {tot_acc/tot:.4f} | {gpu_mem()}")
            sys.stdout.flush()
    return tot_loss/tot, tot_acc/tot, time.time()-t0

# Wrap ViT so it accepts (x, stems) and ignores stems (to reuse the same runner)
class VitAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, _stems=None):
        return self.model(x)

# --------------------------------- GO --------------------------------
print(f"[Setup] Device: {device} | {gpu_mem()}")
print(f"[Paths] DATA_SRC={DATA_SRC} | BUF_DIR={BUF_DIR}")

# Make sure we have buffers (if this is a local imagefolder)
try:
    # Only try precompute when DATA_SRC is a directory
    if Path(DATA_SRC).exists() and Path(DATA_SRC).is_dir():
        ensure_buffers_for_dataset(DATA_SRC)
    else:
        print("[Precompute] Skipped (DATA_SRC not a local directory).")
except Exception as e:
    print(f"[Precompute][WARN] {e}")

# Build dataloaders (returns (img, label, stem))
train_dl, val_dl = build_dataloaders_with_names(DATA_SRC, bs=BATCH_SIZE, val_pct=VAL_PCT)
print(f"[Data] Train batches: {len(train_dl)} | Val batches: {len(val_dl)} | batch_size={BATCH_SIZE}")

# Models
model_qt  = BottomUpQuadtreeMxCiF().to(device)
model_vit = VitAdapter(timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)).to(device)

opt_qt  = torch.optim.AdamW(model_qt.parameters(),  lr=LR)
opt_vit = torch.optim.AdamW(model_vit.parameters(), lr=LR)
crit = nn.BCEWithLogitsLoss()

# Warm-up
print("\n[Warmup] Running one forward pass on each model…")
xb,yb,stems = next(iter(train_dl))
_ = model_qt(xb.to(device), list(stems))
_ = model_vit(xb.to(device), list(stems))
print(f"[Warmup] OK. {gpu_mem()}")

# Train both models for 10 epochs (exactly as friend’s schedule)
total_start = time.time()
for model, opt, name in [(model_qt, opt_qt, 'MX-CiF-QT'),
                         (model_vit, opt_vit, 'Baseline-ViT')]:
    print(f"\n=== {name} Training ===")
    start = time.time()
    for ep in range(1, EPOCHS+1):
        ep_start = time.time()
        tl, ta, tt = run_epoch_with_names(model, train_dl, opt, crit, True,  device, name=name)
        vl, va, vt = run_epoch_with_names(model, val_dl,   opt, crit, False, device, name=name)
        ep_time = time.time() - ep_start
        print(f"{name} E{ep:02d} | {ep_time:5.1f}s "
              f"| train loss {tl:.4f} acc {ta:.4f} (t={tt:.1f}s) "
              f"| val loss {vl:.4f} acc {va:.4f} (t={vt:.1f}s)")
        sys.stdout.flush()
        # Optional: free cache between epochs for stability on small GPUs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    elapsed = time.time() - start
    print(f"{name} total time: {elapsed:.1f}s")
total_elapsed = time.time() - total_start
print(f"\nAll models total time: {total_elapsed:.1f}s | {gpu_mem()}")

