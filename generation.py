# infer_sr_single.py
# -*- coding: utf-8 -*-
"""
Single-patch SR inference + save LR/HR/SR (and optional triplet collage).

It will:
- Read config.yml -> data_loader.out_img_dir, data_loader.down_scale
- Locate:
    LR: <out_img_dir>/lr_png/<slide_id>/patch_XXXXXX.png
    HR: <out_img_dir>/hr_png/<slide_id>/patch_XXXXXX.png
- Load trained checkpoint
- Run model(lr) -> sr
- Save 3 pngs into:
    <save_dir>/<slide_id>/patch_XXXXXX_lr.png
    <save_dir>/<slide_id>/patch_XXXXXX_sr.png
    <save_dir>/<slide_id>/patch_XXXXXX_hr.png
- Optional: save a collage (LR_up | SR | HR) as:
    <save_dir>/<slide_id>/patch_XXXXXX_triplet.png

Usage:
python infer_sr_single.py \
  --config config.yml \
  --ckpt /path/to/best_model.pt \
  --slide_id TCGA-05-4244-01Z-00-DX1 \
  --n 1 \
  --save_dir /mnt/liangjm/SpRR_data/sr_png_single \
  --save_triplet

Notes:
- If HR does not exist for that patch, it will error (by design).
"""

import os
import yaml
import argparse
from typing import Dict, Any, Tuple, Optional

import torch
from PIL import Image

from src.model import SRModel, SRModelConfig


# ----------------------
# helpers: config access
# ----------------------
def _cfg_get(cfg: Dict[str, Any], path: str, default=None):
    cur: Any = cfg
    for k in path.split("."):
        if isinstance(cur, dict) and (k in cur):
            cur = cur[k]
        else:
            return default
    return cur


def _build_model_from_config(cfg: Dict[str, Any], device: torch.device) -> SRModel:
    """
    Build SRModel with only supported SRModelConfig fields,
    mirroring train.py behavior (filter unexpected keys).
    """
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(SRModelConfig)}

    model_cfg = _cfg_get(cfg, "model", {}) or {}
    if not isinstance(model_cfg, dict):
        model_cfg = dict(model_cfg)

    # scale from data_loader.down_scale
    model_cfg["scale"] = int(_cfg_get(cfg, "data_loader.down_scale", 4))

    filtered = {k: v for k, v in model_cfg.items() if k in valid_fields}
    mcfg = SRModelConfig(**filtered)

    model = SRModel(cfg=mcfg).to(device)
    model.eval()
    return model


def _load_checkpoint_to_model(model: SRModel, ckpt_path: str, device: torch.device) -> Tuple[int, Optional[float]]:
    """
    Load weights from:
      - dict with key "model"
      - dict with key "state_dict"
      - raw state_dict
    Returns: (epoch, best_metric)
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    epoch = -1
    best_metric = None

    state_dict = None
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        else:
            # might already be a state_dict-like mapping
            maybe_keys = list(ckpt.keys())
            if maybe_keys and isinstance(ckpt[maybe_keys[0]], torch.Tensor):
                state_dict = ckpt

        epoch = int(ckpt.get("epoch", -1)) if "epoch" in ckpt else -1
        best_metric = ckpt.get("best_metric", None)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    if state_dict is None:
        raise ValueError(f"Failed to find model weights in checkpoint: {ckpt_path}")

    # Handle potential "module." prefix (DDP)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        new_sd = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=True)

    return epoch, (float(best_metric) if best_metric is not None else None)


def _format_patch_name(n: int) -> str:
    if n < 0:
        raise ValueError("n must be >= 0")
    return f"patch_{n:06d}.png"


def _read_rgb_png(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pil_to_tensor01(img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    PIL RGB -> tensor [1,3,H,W], float in [0,1]
    """
    import numpy as np

    arr = np.array(img, dtype=np.uint8)  # HWC
    x = torch.from_numpy(arr).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return x.to(device)


def _tensor01_to_pil(x01: torch.Tensor) -> Image.Image:
    """
    tensor [1,3,H,W] or [3,H,W] in [0,1] -> PIL RGB
    """
    import numpy as np

    if x01.dim() == 4:
        x01 = x01[0]
    x01 = x01.detach().cpu().clamp(0, 1)
    arr = (x01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _bicubic_up(lr_img: Image.Image, scale: int) -> Image.Image:
    w, h = lr_img.size
    return lr_img.resize((w * scale, h * scale), resample=Image.BICUBIC)


def _make_triplet(lr_up: Image.Image, sr: Image.Image, hr: Image.Image) -> Image.Image:
    """
    Concatenate (LR_up | SR | HR) horizontally.
    Assumes same size; if not, will center-crop to min common size.
    """
    # Make same size by cropping to min (w,h)
    ws = [lr_up.size[0], sr.size[0], hr.size[0]]
    hs = [lr_up.size[1], sr.size[1], hr.size[1]]
    W, H = min(ws), min(hs)

    def center_crop(im: Image.Image) -> Image.Image:
        w, h = im.size
        left = (w - W) // 2
        top = (h - H) // 2
        return im.crop((left, top, left + W, top + H))

    lr_up_c = center_crop(lr_up)
    sr_c = center_crop(sr)
    hr_c = center_crop(hr)

    out = Image.new("RGB", (W * 3, H))
    out.paste(lr_up_c, (0, 0))
    out.paste(sr_c, (W, 0))
    out.paste(hr_c, (W * 2, 0))
    return out


@torch.no_grad()
def run_single_sr(
    config_path: str,
    ckpt_path: str,
    slide_id: str,
    n: int,
    save_dir: Optional[str] = None,
    device_str: str = "cuda",
    use_amp: bool = False,
    save_triplet: bool = False,
) -> Dict[str, str]:
    """
    Returns dict with saved paths: {"lr":..., "sr":..., "hr":..., "triplet":...optional}
    """
    cfg = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise ValueError("config.yml is not a dict-like YAML")

    out_img_dir = str(_cfg_get(cfg, "data_loader.out_img_dir", None))
    if not out_img_dir:
        raise ValueError("Missing data_loader.out_img_dir in config.yml")
    out_img_dir = os.path.abspath(out_img_dir)

    scale = int(_cfg_get(cfg, "data_loader.down_scale", 4))

    lr_root = os.path.join(out_img_dir, "lr_png")
    hr_root = os.path.join(out_img_dir, "hr_png")

    patch_name = _format_patch_name(int(n))
    lr_path = os.path.join(lr_root, slide_id, patch_name)
    hr_path = os.path.join(hr_root, slide_id, patch_name)

    if not os.path.exists(lr_path):
        raise FileNotFoundError(f"LR patch not found: {lr_path}")
    if not os.path.exists(hr_path):
        raise FileNotFoundError(f"HR patch not found: {hr_path}")

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    # build model + load ckpt
    model = _build_model_from_config(cfg, device=device)
    epoch, best_metric = _load_checkpoint_to_model(model, ckpt_path, device=device)

    # read images
    lr_img = _read_rgb_png(lr_path)
    hr_img = _read_rgb_png(hr_path)

    lr = _pil_to_tensor01(lr_img, device=device)

    # infer SR
    if use_amp and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            sr_t = model(lr).clamp(0, 1)
    else:
        sr_t = model(lr).clamp(0, 1)

    sr_img = _tensor01_to_pil(sr_t)

    # determine save dir
    if save_dir is None:
        save_dir = os.path.join(out_img_dir, "sr_png_single")
    save_dir = os.path.abspath(save_dir)

    save_slide_dir = os.path.join(save_dir, slide_id)
    os.makedirs(save_slide_dir, exist_ok=True)

    stem = patch_name.replace(".png", "")
    out_lr = os.path.join(save_slide_dir, f"{stem}_lr.png")
    out_sr = os.path.join(save_slide_dir, f"{stem}_sr.png")
    out_hr = os.path.join(save_slide_dir, f"{stem}_hr.png")

    # save 3 pngs
    lr_img.save(out_lr)
    sr_img.save(out_sr)
    hr_img.save(out_hr)

    out = {"lr": out_lr, "sr": out_sr, "hr": out_hr}

    # optional triplet
    if save_triplet:
        lr_up = _bicubic_up(lr_img, scale=scale)
        trip = _make_triplet(lr_up=lr_up, sr=sr_img, hr=hr_img)
        out_trip = os.path.join(save_slide_dir, f"{stem}_triplet.png")
        trip.save(out_trip)
        out["triplet"] = out_trip

    print(f"[OK] slide_id={slide_id} n={n} | scale={scale}")
    print(f"[OK] lr={lr_path}")
    print(f"[OK] hr={hr_path}")
    print(f"[OK] ckpt={ckpt_path} | loaded_epoch={epoch} best_metric={best_metric}")
    print(f"[OK] saved_lr={out_lr}")
    print(f"[OK] saved_sr={out_sr}")
    print(f"[OK] saved_hr={out_hr}")
    if "triplet" in out:
        print(f"[OK] saved_triplet={out['triplet']}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml", help="Path to config.yml")
    ap.add_argument("--ckpt", type=str, default="/workspace/SPR_new/logs/INR_SR_Scheme2_V23_20260130_024205/best_model.pt", help="Path to trained weights (best_model.pt or checkpoint_latest.pt)")
    ap.add_argument("--slide_id", type=str,default="TCGA-49-4505-01Z-00-DX4")
    ap.add_argument("--n", type=int, default=240, help="Patch index, e.g. 1 -> patch_000001.png")
    ap.add_argument("--save_dir", type=str, default="/workspace/SPR_new/sr_png_single", help="Output root dir. Default: <out_img_dir>/sr_png_single")
    ap.add_argument("--device", type=str, default="cuda", help="cuda | cuda:0 | cpu")
    ap.add_argument("--amp", action="store_true", help="Use autocast fp16 on CUDA")
    ap.add_argument("--save_triplet", action="store_true", help="Also save (LR_up | SR | HR) collage")

    args = ap.parse_args()

    run_single_sr(
        config_path=args.config,
        ckpt_path=args.ckpt,
        slide_id=args.slide_id,
        n=int(args.n),
        save_dir=args.save_dir,
        device_str=args.device,
        use_amp=bool(args.amp),
        save_triplet=bool(args.save_triplet),
    )


if __name__ == "__main__":
    main()
