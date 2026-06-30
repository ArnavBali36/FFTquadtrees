"""
KITTI ControlNet training:
 - Auto-downloads KITTI (nateraw/kitti) via `datasets`
 - Trains Baseline (stock ControlNet) then Ours (MX-CiF Top-K attention inside ControlNet)
 - Saves models, samples, logs, and a summary comparison in one run

Run (one command):
    python kitti_controlnet_allinone.py

Defaults are HPC-friendly but conservative. You can tweak the constants near the top.
"""

import os, sys, json, time, math, random, argparse, datetime
from pathlib import Path

# ---------- lightweight auto-install for non-torch deps ----------
def _pip_install(pkg):
    import subprocess, sys as _sys
    try:
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q", pkg])
    except Exception as e:
        print(f"[WARN] pip install failed for {pkg}: {e}")

try:
    import torch, torch.nn as nn, torch.nn.functional as F
except Exception:
    raise SystemExit("[ERROR] Please install PyTorch matching your CUDA/MPS/CPU first: https://pytorch.org/get-started")

for m in ["diffusers>=0.35", "transformers>=4.50", "datasets>=2.19", "safetensors", "Pillow", "opencv-python", "numpy"]:
    try:
        __import__(m.split(">=")[0].split("==")[0].replace("-", "_"))
    except Exception:
        print(f"[Setup] Installing {m} ..."); _pip_install(m)

import numpy as np, cv2
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available

# --------------------- Defaults you can tune ----------------------
DATASET_NAME     = "nateraw/kitti"      # HF dataset id (auto-download)
DATASET_SPLIT    = "train"              # use train split
DATA_LIMIT       = 20000                # how many images to sample for training (None = all)
RESOLUTION       = 512                  # training/inference resolution
PROMPT           = "a realistic street scene photo taken from a dash camera"
VAL_SAMPLES      = 8                    # fixed validation controls (edges) reused across both models

# Training schedule (per model)
TRAIN_STEPS      = 3000                 # steps for Baseline, then same for Ours
BATCH_SIZE       = 2
LR               = 1e-5
MIXED_PRECISION  = "fp16"               # "fp16", "bf16", or "no"
SAMPLE_EVERY     = 500                  # save inference images every N steps

# Models (auto-download from HF Hub; may require an HF token for SD 1.5)
BASE_MODEL       = "runwayml/stable-diffusion-v1-5"
CONTROLNET_REPO  = "lllyasviel/sd-controlnet-canny"

# MX-CiF attention sparsity (Top-K keep ratio)
MXCIF_KEEP_RATIO = 0.5                  # 0.5 keeps top 50% per query/head


# ------------------ MX-CiF Top-K attention (robust) ------------------
class MXCiFTopKAttnProcessor(nn.Module):
    """
    Drop-in attention processor that follows diffusers' helpers for shaping.
    It applies Top-K sparsity over keys per query/head (keep_ratio in (0,1]).
    """
    def __init__(self, keep_ratio: float = 0.5):
        super().__init__()
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        # Projection to Q, K, V
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # [B,N,C] -> [B*H,N,D]
        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Scaled scores
        scale = attn.scale if getattr(attn, "scale", None) is not None else (query.shape[-1] ** -0.5)
        scores = torch.bmm(query, key.transpose(1, 2)) * scale

        # Optional mask
        if attention_mask is not None:
            try:
                attention_mask = attn.prepare_attention_mask(attention_mask, scores.shape[0], scores.shape[1], dtype=scores.dtype)
            except Exception:
                pass
            if attention_mask is not None:
                scores = scores + attention_mask

        # Top-K sparsification
        if self.keep_ratio < 1.0:
            k_keep = max(1, int(scores.shape[-1] * self.keep_ratio))
            topv, topi = torch.topk(scores, k=k_keep, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            scores = mask.scatter(-1, topi, topv)

        probs = torch.softmax(scores, dim=-1)
        out = torch.bmm(probs, value)  # [B*H,N,D]

        # Back to [B,N,C]
        out = attn.batch_to_head_dim(out)

        # Output projection
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1:
            out = attn.to_out[1](out)
        return out


# ------------------------- Helpers -------------------------
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def dtype_for(device):
    if device.type == "cuda" and MIXED_PRECISION == "fp16": return torch.float16
    if MIXED_PRECISION == "bf16": return torch.bfloat16
    return torch.float32

def canny_edges(pil_img: Image.Image, size=RESOLUTION) -> Image.Image:
    gray = pil_img.convert("L").resize((size, size), Image.BICUBIC)
    arr = np.array(gray)
    e = cv2.Canny(arr, 100, 200)
    e = cv2.cvtColor(e, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(e)

def to_tensor_01(img: Image.Image, size=RESOLUTION):
    x = np.array(img.resize((size, size), Image.BICUBIC)).astype(np.float32) / 255.0
    if x.ndim == 2: x = np.stack([x,x,x], -1)
    return torch.from_numpy(x).permute(2,0,1)  # [3,H,W] in [0,1]

def to_tensor_m11(img: Image.Image, size=RESOLUTION):
    return to_tensor_01(img, size) * 2.0 - 1.0

def tensor_to_pil01(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0,1)
    x = (x.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(x)

# ------------------------- Data -------------------------
class HFKittiDataset(Dataset):
    """
    Minimal dataset: returns dict(image=[-1,1], cond=[0,1], prompt=str).
    """
    def __init__(self, name=DATASET_NAME, split=DATASET_SPLIT, limit=DATA_LIMIT, res=RESOLUTION, prompt=PROMPT):
        ds = load_dataset(name, split=split, streaming=False)
        cols = [c for c in ds.column_names if c.lower() in {"image","img","file","filepath","file_path"}]
        if not cols:
            raise RuntimeError(f"No image-like column in {name}/{split}. Columns: {ds.column_names}")
        col = cols[0]
        n = len(ds) if limit is None else min(limit, len(ds))
        self.imgs = []
        for i in range(n):
            r = ds[i][col]
            pil = r if isinstance(r, Image.Image) else Image.open(r).convert("RGB")
            self.imgs.append(pil)
        self.res = res
        self.prompt = prompt
        print(f"[Data] Loaded {len(self.imgs)} images from {name}/{split} (limit={n})")

    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = self.imgs[idx].convert("RGB")
        cond = canny_edges(img, size=self.res)
        return {
            "image": to_tensor_m11(img, self.res),
            "cond":  to_tensor_01(cond, self.res),
            "prompt": self.prompt
        }

# --------------------- Build model / pipeline ---------------------
def build_pipeline(device, dtype, use_mxcif=False, keep_ratio=MXCIF_KEEP_RATIO):
    hf_token = os.environ.get("HF_TOKEN", None)
    token_kw = {"use_auth_token": hf_token} if hf_token else {}
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_REPO, torch_dtype=dtype, **token_kw)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, torch_dtype=dtype, safety_checker=None, **token_kw
    )
    # DDPM scheduler for training objective (eps-pred)
    pipe.scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    # Freeze base, train only ControlNet
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.controlnet.requires_grad_(True)

    # Optional: attention swap
    if use_mxcif:
        pipe.controlnet.set_attn_processor(MXCiFTopKAttnProcessor(keep_ratio=keep_ratio))

    if is_xformers_available() and device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[Models] xFormers enabled")
        except Exception as e:
            print(f"[Models][WARN] xFormers failed: {e}")

    return pipe.to(device)

# -------------------------- Training core --------------------------
def train_one(pipe, dl, out_dir, steps=TRAIN_STEPS, sample_every=SAMPLE_EVERY, out_samples=VAL_SAMPLES, device=None, dtype=torch.float16):
    out_dir = Path(out_dir); (out_dir/"samples").mkdir(parents=True, exist_ok=True)
    # components
    vae, unet, cn, tok, txt, sched = pipe.vae, pipe.unet, pipe.controlnet, pipe.tokenizer, pipe.text_encoder, pipe.scheduler
    latent_scale = 0.18215
    opt = torch.optim.AdamW([p for p in cn.parameters() if p.requires_grad], lr=LR)
    use_amp = (device.type == "cuda" and MIXED_PRECISION in {"fp16","bf16"})
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and MIXED_PRECISION=="fp16"))

    # Build a fixed validation control set (PIL) from the first few batches
    controls_val = []
    for batch in dl:
        for c in batch["cond"][:max(VAL_SAMPLES - len(controls_val), 0)]:
            controls_val.append(tensor_to_pil01(c))
        if len(controls_val) >= VAL_SAMPLES: break

    # Train loop
    step = 0
    t0 = time.time()
    logs_path = out_dir / "logs.jsonl"
    if logs_path.exists(): logs_path.unlink()
    while step < steps:
        for batch in dl:
            if step >= steps: break
            images = batch["image"].to(device)
            conds  = batch["cond"].to(device)
            prompts = batch["prompt"]
            if isinstance(prompts, str): prompts = [prompts]*images.size(0)

            # text encode + VAE latent encode (frozen)
            tok_out = tok(prompts, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
            ids = tok_out.input_ids.to(device)
            with torch.no_grad():
                enc_h = txt(ids)[0]
                lat   = vae.encode(images).latent_dist.sample() * latent_scale

            # diffusion noise
            noise = torch.randn_like(lat)
            t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=device, dtype=torch.long)
            noisy = sched.add_noise(lat, noise, t)

            control = conds * 2.0 - 1.0  # [-1,1] as ControlNet expects
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
                down_res, mid_res = cn(noisy, t, encoder_hidden_states=enc_h, controlnet_cond=control, return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=enc_h,
                            down_block_additional_residuals=down_res,
                            mid_block_additional_residual=mid_res).sample
                loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

            step += 1
            if step % 20 == 0 or step == 1:
                print(f"[{out_dir.name}] step {step:06d} | loss {loss.item():.4f}")
                with open(logs_path, "a") as f:
                    f.write(json.dumps({"step": int(step), "loss": float(loss.item())}) + "\n")

            if sample_every and (step % sample_every == 0 or step == steps):
                for i, c in enumerate(controls_val):
                    img = pipe(PROMPT, image=c, num_inference_steps=30, guidance_scale=7.5).images[0]
                    img.save(out_dir / "samples" / f"step{step:06d}_{i:02d}.png")

    wall = time.time() - t0
    # Save trained ControlNet
    pipe.controlnet.save_pretrained(str(out_dir / "controlnet"))
    return {"steps": step, "wall_s": wall, "logs": str(logs_path), "samples_dir": str(out_dir / "samples"), "controlnet_dir": str(out_dir / "controlnet")}

# -------------------------- Comparison grid --------------------------
def make_compare_grid(baseline_dir: Path, ours_dir: Path, out_path: Path, take=VAL_SAMPLES):
    """
    Make a simple 2-row grid: top = baseline samples (latest), bottom = ours samples (latest).
    Align by index (0..take-1).
    """
    from PIL import ImageDraw, ImageFont
    def latest_samples(d: Path):
        files = sorted((d/"samples").glob("*.png"))
        if not files: return []
        # group by step (parse stepNNNNNN)
        by_step = {}
        for p in files:
            stem = p.stem  # step000500_00
            step = int(stem.split("_")[0].replace("step",""))
            by_step.setdefault(step, []).append(p)
        last_step = sorted(by_step.keys())[-1]
        imgs = sorted(by_step[last_step])[:take]
        return [Image.open(p).convert("RGB") for p in imgs]

    a = latest_samples(baseline_dir)
    b = latest_samples(ours_dir)
    if not a or not b:
        print("[Grid] Not enough samples to build a grid."); return
    n = min(len(a), len(b), take)
    if n == 0: return

    W, H = a[0].size
    pad = 10
    grid = Image.new("RGB", (n*W + (n+1)*pad, 2*H + 3*pad), (240,240,240))
    draw = ImageDraw.Draw(grid)
    # Titles
    title = "Baseline (top) vs Ours (bottom)"
    draw.text((pad, pad//2), title, fill=(0,0,0))
    # Paste
    y1 = pad*2
    y2 = y1 + H + pad
    for i in range(n):
        grid.paste(a[i], (pad + i*(W+pad), y1))
        grid.paste(b[i], (pad + i*(W+pad), y2))
    grid.save(out_path)
    print(f"[Grid] Wrote comparison grid: {out_path}")

# ----------------------------- MAIN -----------------------------
def main():
    seed_all(42)
    device = pick_device()
    dtype  = dtype_for(device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(f"runs/kitti_allinone_{timestamp}")
    (run_root/"baseline").mkdir(parents=True, exist_ok=True)
    (run_root/"ours").mkdir(parents=True, exist_ok=True)

    # Reports
    try:
        import diffusers, transformers, datasets as _ds
        env_report = {
            "python": sys.version,
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "datasets": _ds.__version__,
            "device": str(device),
            "dtype": str(dtype),
            "mixed_precision": MIXED_PRECISION,
            "base_model": BASE_MODEL,
            "controlnet": CONTROLNET_REPO,
            "dataset": f"{DATASET_NAME}/{DATASET_SPLIT}",
            "data_limit": DATA_LIMIT,
        }
        (run_root/"env_report.json").write_text(json.dumps(env_report, indent=2))
        print("[ENV]", env_report)
    except Exception:
        pass

    # Data
    ds = HFKittiDataset(name=DATASET_NAME, split=DATASET_SPLIT, limit=DATA_LIMIT, res=RESOLUTION, prompt=PROMPT)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=(device.type=="cuda"))

    # Baseline
    print("\n=== Training Baseline (stock ControlNet + Canny) ===")
    pipe_base = build_pipeline(device, dtype, use_mxcif=False)
    res_base = train_one(pipe_base, dl, run_root/"baseline", steps=TRAIN_STEPS, device=device, dtype=dtype)

    # Ours (MX-CiF Top-K inside ControlNet)
    print("\n=== Training Ours (ControlNet + MX-CiF Top-K) ===")
    pipe_ours = build_pipeline(device, dtype, use_mxcif=True, keep_ratio=MXCIF_KEEP_RATIO)
    res_ours = train_one(pipe_ours, dl, run_root/"ours", steps=TRAIN_STEPS, device=device, dtype=dtype)

    # Grid & summary
    make_compare_grid(run_root/"baseline", run_root/"ours", run_root/"compare_grid.png", take=VAL_SAMPLES)
    summary = {
        "baseline": res_base,
        "ours": res_ours,
        "run_root": str(run_root),
        "notes": "Both models trained on the same data stream with identical hyperparameters. Only ControlNet attention differs (ours uses Top-K)."
    }
    (run_root/"summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== DONE ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts:\n - {run_root/'baseline'}\n - {run_root/'ours'}\n - {run_root/'compare_grid.png'}\n - {run_root/'summary.json'}")

if __name__ == "__main__":
    main()