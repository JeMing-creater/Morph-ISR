# eval_all.py
from ast import Pass
import os
import glob
import json
import random
from dataclasses import asdict
from typing import Optional, Dict, Tuple, Callable, Any, List, Union

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F




# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)



# -----------------------------
# Loader builder (hyperparam-based)
# -----------------------------
def build_loaders_by_hparams(
    out_img_dir: str,
    batch_size: int = 4,
    patch_num: int = 200,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    split_seed: int = 2025,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    require_done: bool = True,
    shuffle_train: bool = True,
    disc_root: Optional[str] = None,
):
    """
    Wrapper around src.loader.build_case_split_dataloaders using the correct signature.

    IMPORTANT:
      - Use out_img_dir (not data_root).
      - out_img_dir should contain: hr_png/, lr_png/, clinical.tsv (and optionally disc_maps/).
    """
    from src import loader as loader_mod

    if not hasattr(loader_mod, "build_case_split_dataloaders"):
        raise RuntimeError("src.loader.build_case_split_dataloaders not found.")

    return loader_mod.build_case_split_dataloaders(
        out_img_dir=out_img_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        patch_num=patch_num,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        require_done=require_done,
        shuffle_train=shuffle_train
    )




# -----------------------------
# Validation (match trainer.validate)
# -----------------------------


def make_infer_fn_morph_isr(
    morph_pack,
    do_color_cc: bool = True,
    force_out_size: int | None = 512,
):
    """
    Build infer_fn for Morph-ISR.

    IMPORTANT: infer_fn is made *signature-flexible* to match eval.py:
      - sometimes called as infer_fn(lr, hr)
      - sometimes called as infer_fn(lr, hr, meta, device)
    See eval.py branching behavior :contentReference[oaicite:4]{index=4} and folder infer :contentReference[oaicite:5]{index=5}
    """
    import torch
    import torch.nn.functional as F

    model = morph_pack["model"]
    default_device = morph_pack.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _color_match_batch(sr: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Simple per-image, per-channel affine color matching:
            sr' = (sr - mu_sr) * (std_ref/std_sr) + mu_ref
        ref is LR_up (bicubic), consistent with your eval.py color policy idea :contentReference[oaicite:6]{index=6}
        """
        # [B,3,H,W]
        mu_s = sr.mean(dim=(-2, -1), keepdim=True)
        mu_r = ref.mean(dim=(-2, -1), keepdim=True)
        std_s = sr.var(dim=(-2, -1), keepdim=True, unbiased=False).sqrt().clamp_min(eps)
        std_r = ref.var(dim=(-2, -1), keepdim=True, unbiased=False).sqrt().clamp_min(eps)
        out = (sr - mu_s) * (std_r / std_s) + mu_r
        return out

    @torch.no_grad()
    def infer_fn(*args):
        """
        Accept both:
          (lr, hr)
          (lr, hr, meta, device)
        """
        if len(args) < 2:
            raise TypeError("infer_fn expects at least (lr, hr)")
        lr = args[0]
        hr = args[1]
        meta = args[2] if len(args) >= 3 else {}
        device = args[3] if len(args) >= 4 else default_device

        lr = lr.to(device, non_blocking=True)
        if hr is not None:
            hr = hr.to(device, non_blocking=True)

        # target size policy (same as eval.py's make_infer_fn_ours) :contentReference[oaicite:7]{index=7}
        if hr is not None:
            target_hw = hr.shape[-2:]
        elif force_out_size is not None:
            target_hw = (int(force_out_size), int(force_out_size))
        else:
            target_hw = None

        # ---- Morph-ISR forward ----
        # If your SRModel already outputs HR size, great; if not, we resize to target_hw.
        sr = model(lr).clamp(0, 1)

        if target_hw is not None and sr.shape[-2:] != target_hw:
            sr = F.interpolate(sr, size=target_hw, mode="bicubic", align_corners=False)

        # ---- optional color correction (reference from LR_up) ----
        if do_color_cc and target_hw is not None:
            lr_up = F.interpolate(lr, size=target_hw, mode="bicubic", align_corners=False)
            sr = _color_match_batch(sr, lr_up).clamp(0, 1)

        return sr

    return infer_fn



@torch.no_grad()
def eval_with_validate_by_project(
    loader,
    infer_fn,
    device: torch.device,
    max_batches: int = 10,              # 解释为：每个project至少纳入max_batches个样本
    split_name: str = "val",
    print_every: int = 10,
    meta_defaults: Optional[Dict[str, Any]] = None,
    method: str = "S3_Diff",
) -> Dict[str, Any]:
    """
    Project-wise validation with per-project minimum sample budget.

    Returns:
      {
        "overall": {...},
        "by_project": { "<project>": {...}, ... },
        "target_samples_per_project": int
      }
    """
    if meta_defaults is None:
        meta_defaults = {}

    # 兼容导入
    try:
        from src.sr_metrics import SRMetrics
    except Exception:
        from sr_metrics import SRMetrics

    metrics = SRMetrics(device=device)

    # -------------------------
    # 1) Discover projects
    # -------------------------
    projects = set()
    ds = loader.dataset

    # case A) loader.dataset is a Subset -> ds.dataset is base dataset
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset

        # 最快：从 base.items + subset.indices 直接拿到当前 split 的 project
        if hasattr(base, "items") and isinstance(base.items, list) and len(base.items) > 0:
            for base_i in ds.indices:
                it = base.items[base_i]
                pj = it.get("project", None)
                if pj is None and hasattr(base, "project_map"):
                    cid = it.get("case_id", "")
                    pj = base.project_map.get(cid, None)
                if pj is not None:
                    projects.add(str(pj))

        # 备选：直接用 project_map 但仅统计 subset 覆盖到的 case
        elif hasattr(base, "project_map") and isinstance(base.project_map, dict):
            # 若 base 没有 items，就退回：统计 project_map 的全部值（可能包含别的 split）
            projects.update([str(v) for v in base.project_map.values() if v is not None])

    # case B) loader.dataset is directly the base dataset
    else:
        if hasattr(ds, "project_map") and isinstance(ds.project_map, dict):
            projects.update([str(v) for v in ds.project_map.values() if v is not None])

    projects = sorted([p for p in projects if p and p.lower() != "none"])
    if len(projects) == 0:
        projects = ["UNKNOWN"]

    # -------------------------
    # 2) Accumulators
    # -------------------------
    def _new_bucket():
        return {"psnr": [], "ssim": [], "lpips": [], "stlpips": [], "gradl1": [], "n": 0}

    acc = {p: _new_bucket() for p in projects}
    overall = _new_bucket()

    target_n = int(max_batches)
    print(
        f"\n[eval_by_project] split={split_name} target_samples_per_project={target_n} projects={projects}",
        flush=True
    )

    # -------------------------
    # 3) Iterate & accumulate
    # -------------------------
    for bi, batch in enumerate(loader):
        print(f"[eval_by_project] processing batch {bi} ...", flush=True)
        # early stop if all projects have enough samples
        
        if all(acc[p]["n"] >= target_n for p in projects):
            print(f"[eval_by_project] early stop at batch {bi} (all projects reached target).", flush=True)
            break

        if not isinstance(batch, dict):
            raise TypeError("Expected dict batch with keys: lr/hr/project/...")

        proj_list = batch.get("project", None)
        if proj_list is not None:
            if not isinstance(proj_list, (list, tuple)):
                proj_list = [str(proj_list)]
            else:
                proj_list = [str(p) if p else "UNKNOWN" for p in proj_list]

            # 若 batch 中所有样本的 project 都已达标，则直接跳过
            all_done = True
            for pj in proj_list:
                if pj not in acc or acc[pj]["n"] < target_n:
                    all_done = False
                    break

            if all_done:
                # 这个 batch 不会贡献任何新样本，直接跳过
                continue
        
        print(f"[acc] current counts: ", acc, flush=True)
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)

        proj_list = batch.get("project", ["UNKNOWN"] * lr.shape[0])
        if not isinstance(proj_list, (list, tuple)):
            proj_list = [str(proj_list)] * lr.shape[0]
        proj_list = [str(x) if str(x) else "UNKNOWN" for x in proj_list]

        # meta passthrough (if your infer_fn uses it)
        meta = {k: v for k, v in batch.items() if k not in ["lr", "hr"]}
        meta = dict(meta_defaults, **meta)

        # infer SR
        if method == "S3_Diff":
            sr = infer_fn(lr, hr, meta, device)
        else:
            sr = infer_fn(lr, hr)

        sr = sr.clamp(0, 1)
        hr = hr.clamp(0, 1)

        if sr.shape != hr.shape:
            raise ValueError(f"sr/hr shape mismatch: sr={tuple(sr.shape)} hr={tuple(hr.shape)}")

        B = sr.shape[0]

        # 逐样本分配到 project bucket（保证每个 project 不少于 target_n）
        for i in range(B):
            pj = proj_list[i] if i < len(proj_list) else "UNKNOWN"

            if pj not in acc:
                acc[pj] = _new_bucket()
                projects.append(pj)

            # 达标后不再为该 project 累计，避免某个 project 支配 overall
            if acc[pj]["n"] >= target_n:
                continue

            s1 = sr[i:i+1]
            h1 = hr[i:i+1]

            psnr_v = float(metrics.psnr(s1, h1))
            ssim_v = float(metrics.ssim(s1, h1))
            lpips_v = float(metrics.lpips(s1, h1))

            st = metrics.shift_tolerant_lpips(s1, h1)
            if isinstance(st, torch.Tensor):
                st_v = float(st.view(-1)[0].item()) if st.numel() else 0.0
            else:
                st_v = float(st)

            gl = metrics.gradient_l1(s1, h1)
            if isinstance(gl, torch.Tensor):
                gl_v = float(gl.view(-1)[0].item()) if gl.numel() else 0.0
            else:
                gl_v = float(gl)

            acc[pj]["psnr"].append(psnr_v)
            acc[pj]["ssim"].append(ssim_v)
            acc[pj]["lpips"].append(lpips_v)
            acc[pj]["stlpips"].append(st_v)
            acc[pj]["gradl1"].append(gl_v)
            acc[pj]["n"] += 1

            overall["psnr"].append(psnr_v)
            overall["ssim"].append(ssim_v)
            overall["lpips"].append(lpips_v)
            overall["stlpips"].append(st_v)
            overall["gradl1"].append(gl_v)
            overall["n"] += 1

        if print_every and (bi + 1) % int(print_every) == 0:
            prog = {p: acc[p]["n"] for p in projects}
            print(f"[eval_by_project] batch {bi+1}: per-project counts {prog}", flush=True)

    # -------------------------
    # 4) Reduce
    # -------------------------
    def _reduce(d):
        return {
            "PSNR": float(np.mean(d["psnr"])) if d["psnr"] else 0.0,
            "SSIM": float(np.mean(d["ssim"])) if d["ssim"] else 0.0,
            "LPIPS": float(np.mean(d["lpips"])) if d["lpips"] else 0.0,
            "ST-LPIPS": float(np.mean(d["stlpips"])) if d["stlpips"] else 0.0,
            "Grad-L1": float(np.mean(d["gradl1"])) if d["gradl1"] else 0.0,
            "samples": int(d["n"]),
        }

    out = {
        "overall": _reduce(overall),
        "by_project": {p: _reduce(acc[p]) for p in sorted(acc.keys())},
        "target_samples_per_project": target_n,
    }

    print(f"[eval_by_project] overall: {out['overall']}", flush=True)
    for p in sorted(out["by_project"].keys()):
        print(f"[eval_by_project] {p}: {out['by_project'][p]}", flush=True)

    return out
# -----------------------------
# Inference on a LR folder
# -----------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _list_imgs(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    return paths


def _img_to_tensor01(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return t


def _tensor01_to_u8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    return (x.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def _infer_pair_hr_path(lr_path: str, lr_root: str, hr_root: str) -> str:
    """
    Keep relative path: hr_root/<relpath of lr under lr_root>.
    This is robust when lr_root contains nested folders.
    """
    rel = os.path.relpath(lr_path, lr_root)
    return os.path.join(hr_root, rel)



def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _list_imgs(folder: str) -> List[str]:
    exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff")
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(folder, e)))
    return sorted(out)

def _img_to_tensor01(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

def _tensor01_to_u8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().clamp(0,1)
    if x.ndim == 4: x = x[0]
    if x.shape[0] == 3: x = x.permute(1,2,0)
    return (x.cpu().numpy()*255.0 + 0.5).astype(np.uint8)

def _infer_pair_hr_path(lr_path: str, lr_root: str, hr_root: str) -> str:
    rel = os.path.relpath(lr_path, lr_root)
    return os.path.join(hr_root, rel)

@torch.no_grad()
def infer_folder_lr_to_sr_generic(
    lr_dir: str,
    out_root: str,
    name: str,
    infer_fn: Callable[[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], torch.device], torch.Tensor],
    device: torch.device,
    hr_dir: Optional[str] = None,
    lr_up_size: int = 512,
    limit: Optional[int] = None,
    meta_defaults: Optional[Dict[str, Any]] = None,
):
    """
    Save only: LR_up, SR, (optional HR) using the SAME infer_fn as validation.
    """
    if meta_defaults is None:
        meta_defaults = {}

    lr_dir = os.path.abspath(lr_dir)
    case_id = os.path.basename(os.path.normpath(lr_dir))

    base_out = os.path.join(out_root, "eval_image", name, case_id)
    out_lr_up = _ensure_dir(os.path.join(base_out, "LR_up"))
    out_sr = _ensure_dir(os.path.join(base_out, "SR"))
    out_hr = _ensure_dir(os.path.join(base_out, "HR")) if hr_dir else None

    lr_paths = _list_imgs(lr_dir)
    if limit is not None and limit > 0:
        lr_paths = lr_paths[: int(limit)]
    if len(lr_paths) == 0:
        raise RuntimeError(f"No images found under: {lr_dir}")

    print(f"[infer] lr_dir={lr_dir} num={len(lr_paths)} out={base_out}", flush=True)

    for idx, lr_path in enumerate(lr_paths):
        print(f"Processing {lr_path} ...", flush=True)
        rel = os.path.relpath(lr_path, lr_dir)
        stem = os.path.splitext(os.path.basename(lr_path))[0]
        subdir = os.path.dirname(rel)

        # make subdirs
        if subdir:
            os.makedirs(os.path.join(out_lr_up, subdir), exist_ok=True)
            os.makedirs(os.path.join(out_sr, subdir), exist_ok=True)
            if out_hr: os.makedirs(os.path.join(out_hr, subdir), exist_ok=True)

        # load LR
        lr = _img_to_tensor01(lr_path, device=device)

        # optional HR load (for saving only, not mandatory)
        hr = None
        if hr_dir:
            hr_path = _infer_pair_hr_path(lr_path, lr_dir, hr_dir)
            if os.path.isfile(hr_path):
                hr = _img_to_tensor01(hr_path, device=device)

        meta = dict(meta_defaults)
        meta["lr_path"] = lr_path
        if hr_dir:
            meta["hr_path"] = _infer_pair_hr_path(lr_path, lr_dir, hr_dir)

        # infer SR (same function as validation)
        if name == "S3_Diff":
            sr = infer_fn(lr, hr, meta, device).clamp(0,1)
        else:
            sr = infer_fn(lr, hr).clamp(0,1)

        # save LR_up (always size=lr_up_size for visualization)
        lr_up = F.interpolate(lr, size=(lr_up_size, lr_up_size), mode="bicubic", align_corners=False)
        Image.fromarray(_tensor01_to_u8(lr_up)).save(os.path.join(out_lr_up, subdir, f"{stem}.png"))

        # save SR (also resize to lr_up_size for consistent viewing)
        if sr.shape[-2:] != (lr_up_size, lr_up_size):
            sr_save = F.interpolate(sr, size=(lr_up_size, lr_up_size), mode="bicubic", align_corners=False)
        else:
            sr_save = sr
        Image.fromarray(_tensor01_to_u8(sr_save)).save(os.path.join(out_sr, subdir, f"{stem}.png"))

        # save HR if available
        if out_hr is not None and hr is not None:
            hr_save = hr
            if hr_save.shape[-2:] != (lr_up_size, lr_up_size):
                hr_save = F.interpolate(hr_save, size=(lr_up_size, lr_up_size), mode="bicubic", align_corners=False)
            Image.fromarray(_tensor01_to_u8(hr_save)).save(os.path.join(out_hr, subdir, f"{stem}.png"))

        if (idx + 1) % 20 == 0:
            print(f"[infer] {idx+1}/{len(lr_paths)}", flush=True)

    print(f"[infer] done. saved to {base_out}", flush=True)
    return base_out




# ----------models------------
# load (model here)
# ----------models------------
def Morph_ISR(
    OUT_ROOT,
    METHOD_NAME,
    OURS_OUTPUT_DIR,
    CKPT_PREFER,
    DEVICE,
    SAMPLE_STEPS,
    MAX_BATCHES,
    CONFIG_PATH: str | None = None,
):
    """
    Morph-ISR loader (drop-in replacement for S3_Diff in eval.py):
      1) build model using *train.py build_model_and_optim* style:
         - load config.yml (eval.py is same level as train.py)
         - take config.model
         - force model_cfg["scale"] = config.data_loader.down_scale
         - filter keys by SRModelConfig dataclass fields
      2) load weights from best_model.pt / checkpoint_latest.pt (expects dict["model"])
    """
    import os
    import glob
    import yaml
    import torch
    import dataclasses
    from easydict import EasyDict

    # -------------------------
    # helper: cfg_get (same logic as train.py)
    # -------------------------
    def _cfg_get(cfg: EasyDict, path: str, default=None):
        cur = cfg
        for k in path.split("."):
            if not hasattr(cur, k):
                return default
            cur = getattr(cur, k)
        return cur

    # -------------------------
    # helper: resolve ckpt path
    # -------------------------
    def _resolve_ckpt_path(ckpt_dir: str, prefer: str) -> str:
        # 1) direct path
        if prefer=="best":
            return ckpt_dir + "/" + "best_model.pt"
        else:
            return ckpt_dir + "/" + "checkpoint_latest.pt"
        
    # -------------------------
    # 0) locate config.yml (eval.py same level as train.py)
    # -------------------------
    if CONFIG_PATH is not None:
        cfg_path = CONFIG_PATH
    else:
        # eval.py file location (preferred)
        try:
            cfg_path = os.path.join(os.path.dirname(__file__), "config.yml")
        except Exception:
            cfg_path = "config.yml"  # fallback

        if not os.path.isfile(cfg_path):
            # fallback to cwd
            cfg_path = os.path.join(os.getcwd(), "config.yml")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"[Morph_ISR] config.yml not found. Tried: {cfg_path}. "
            f"Pass CONFIG_PATH explicitly."
        )

    config = EasyDict(yaml.load(open(cfg_path, "r", encoding="utf-8"), Loader=yaml.FullLoader))

    # -------------------------
    # 1) build model (train.py build_model_and_optim style)
    # -------------------------
    try:
        from src.model import SRModel, SRModelConfig
    except Exception:
        from model import SRModel, SRModelConfig

    valid_fields = {f.name for f in dataclasses.fields(SRModelConfig)}  # :contentReference[oaicite:3]{index=3}

    model_cfg = {}
    if hasattr(config, "model") and isinstance(config.model, (dict, EasyDict)):
        model_cfg = dict(config.model)  # :contentReference[oaicite:4]{index=4}

    # force scale from data_loader.down_scale (train.py behavior) :contentReference[oaicite:5]{index=5}
    model_cfg["scale"] = int(_cfg_get(config, "data_loader.down_scale", 4))

    filtered = {k: v for k, v in model_cfg.items() if k in valid_fields}  # :contentReference[oaicite:6]{index=6}

    mcfg = SRModelConfig(**filtered)  # :contentReference[oaicite:7]{index=7}
    model = SRModel(cfg=mcfg)

    # -------------------------
    # 2) load weights
    # train.py best_model.pt: {"model": state_dict} :contentReference[oaicite:8]{index=8}
    # train.py checkpoint_latest.pt: {"model": state_dict, ...} :contentReference[oaicite:9]{index=9}
    # -------------------------
    ckpt_path = _resolve_ckpt_path(OURS_OUTPUT_DIR, CKPT_PREFER)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if not (isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict)):
        raise RuntimeError(f"[Morph_ISR] Unexpected ckpt format (missing dict['model']): {ckpt_path}")

    model.load_state_dict(ckpt["model"], strict=True)

    device = torch.device(DEVICE)
    model = model.to(device)
    model.eval()

    print(f"[Morph_ISR] cfg={cfg_path}", flush=True)
    print(f"[Morph_ISR] ckpt={ckpt_path}", flush=True)

    return {
        "model": model,
        "device": device,
        "method_name": METHOD_NAME,
        "sample_steps": int(SAMPLE_STEPS),
        "max_batches": int(MAX_BATCHES),
        "cfg_path": cfg_path,
        "ckpt_path": ckpt_path,
    }


@torch.no_grad()
def infer_out_img_dir_lr_png_to_sr_png(
    out_img_dir: str,
    infer_fn: Callable[[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], torch.device], torch.Tensor],
    device: torch.device,
    name: str = "S3_Diff",
    force_out_size: int = 512,
    overwrite: bool = False,
    limit: Optional[int] = None,
    log_every: int = 50,
    meta_defaults: Optional[Dict[str, Any]] = None,
):
    """
    Recursively infer ALL pngs under:
        <out_img_dir>/lr_png/**.png
    and save to:
        <out_img_dir>/sr_png/**.png
    keeping the SAME relative path.

    This uses the SAME infer_fn as your validation (so颜色矫正也会一致地生效，如果infer_fn内部做了cc)。
    """
    if meta_defaults is None:
        meta_defaults = {}

    out_img_dir = os.path.abspath(out_img_dir)
    lr_root = os.path.join(out_img_dir, "lr_png")
    sr_root = os.path.join(out_img_dir, "sr_png")
    os.makedirs(sr_root, exist_ok=True)

    if not os.path.isdir(lr_root):
        raise FileNotFoundError(f"lr_png not found: {lr_root}")

    # recursive collect
    lr_paths: List[str] = []
    for r, _, files in os.walk(lr_root):
        for fn in files:
            if fn.lower().endswith(".png"):
                lr_paths.append(os.path.join(r, fn))
    lr_paths = sorted(lr_paths)

    if limit is not None and limit > 0:
        lr_paths = lr_paths[: int(limit)]

    if len(lr_paths) == 0:
        raise RuntimeError(f"No png found under: {lr_root}")

    print(f"[infer_all] lr_root={lr_root}")
    print(f"[infer_all] sr_root={sr_root}")
    print(f"[infer_all] num_png={len(lr_paths)} overwrite={overwrite} force_out_size={force_out_size}", flush=True)

    for i, lr_path in enumerate(lr_paths):
        print(f"[infer_all] processing {lr_path} ...", flush=True)
        rel = os.path.relpath(lr_path, lr_root)  # keep folder structure
        out_path = os.path.join(sr_root, rel)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        if (not overwrite) and os.path.isfile(out_path):
            continue

        # load LR [1,3,h,w] in [0,1]
        lr = _img_to_tensor01(lr_path, device=device)

        meta = dict(meta_defaults)
        meta["lr_path"] = lr_path

        # run infer (same branch behavior as your infer_folder_lr_to_sr_generic)
        # 你的infer_fn_ours通常签名是 infer_fn(lr, hr=None, meta={}, device=...)
        if name == "S3_Diff":
            sr = infer_fn(lr, None, meta, device).clamp(0, 1)
        else:
            # if you later plug other baselines, keep a compatible fallback
            sr = infer_fn(lr, None, meta, device).clamp(0, 1)

        # force save size (optional but recommended for统一512可视化/评测)
        if force_out_size is not None:
            if sr.shape[-2:] != (force_out_size, force_out_size):
                sr = F.interpolate(sr, size=(force_out_size, force_out_size), mode="bicubic", align_corners=False)

        Image.fromarray(_tensor01_to_u8(sr)).save(out_path)

        if (i + 1) % int(log_every) == 0:
            print(f"[infer_all] {i+1}/{len(lr_paths)} saved: {out_path}", flush=True)

    print(f"[infer_all] done. sr saved under: {sr_root}", flush=True)
    return sr_root

def _is_slide_of_case(slide_id: str, case_id: str) -> bool:
    slide_id = str(slide_id)
    case_id = str(case_id)
    if slide_id.startswith(case_id + "-"):
        return True
    s_parts = slide_id.split("-")
    c_parts = case_id.split("-")
    if len(s_parts) >= 3 and len(c_parts) >= 3:
        return "-".join(s_parts[:3]) == "-".join(c_parts[:3])
    return False


@torch.no_grad()
def infer_out_img_dir_lr_png_case_to_sr_png(
    out_img_dir: str,
    case_id: str,
    infer_fn: Callable[[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], torch.device], torch.Tensor],
    device: torch.device,
    name: str = "S3_Diff",
    force_out_size: int = 512,
    overwrite: bool = False,
    limit: Optional[int] = None,
    limit_per_slide: Optional[int] = None,
    log_every: int = 50,
    meta_defaults: Optional[Dict[str, Any]] = None,
    # ✅ NEW:
    choose_id: Optional[Union[int, str]] = None,   # e.g. 292 or "000292"
) -> str:
    """
    - choose_id is None: old behavior (scan and run all patches for the case)
    - choose_id is not None: FAST path
        -> only process patch_{choose_id:06d}.png for each slide of this case
        -> no full os.walk of lr_png
    """
    if meta_defaults is None:
        meta_defaults = {}

    out_img_dir = os.path.abspath(out_img_dir)
    lr_root = os.path.join(out_img_dir, "lr_png")
    sr_root = os.path.join(out_img_dir, "sr_png_single")
    os.makedirs(sr_root, exist_ok=True)

    if not os.path.isdir(lr_root):
        raise FileNotFoundError(f"lr_png not found: {lr_root}")

    # -------------------------
    # 1) FAST mode: choose_id provided
    # -------------------------
    if choose_id is not None:
        # normalize choose_id -> 6-digit
        if isinstance(choose_id, str):
            cid = int(choose_id)   # accept "292" or "000292"
        else:
            cid = int(choose_id)
        fn = f"patch_{cid:06d}.png"

        lr_paths: List[str] = []

        # ✅ 关键：case_id 只是 slide 目录名前缀，所以只 list lr_root 一级目录并做前缀匹配
        # 例：case_id="TCGA-35-3615"
        # slide_dir="TCGA-35-3615-01Z-00-DX1" -> match
        prefix = case_id + "-"

        for sd in sorted(os.listdir(lr_root)):
            slide_dir = os.path.join(lr_root, sd)
            if not os.path.isdir(slide_dir):
                continue

            # 前缀匹配（比 _is_slide_of_case 更直接更快）
            if not sd.startswith(prefix):
                continue

            p = os.path.join(slide_dir, fn)
            if os.path.isfile(p):
                lr_paths.append(p)

        if len(lr_paths) == 0:
            raise RuntimeError(
                f"[choose_id] file not found for case_id={case_id}, choose_id={cid} ({fn}). "
                f"Checked: {lr_root}/<slide startswith '{prefix}'>/{fn}"
            )

        print(f"[infer_case FAST] case_id={case_id} prefix='{prefix}' choose_id={cid} matched_files={len(lr_paths)}", flush=True)
    # -------------------------
    # 2) FULL mode: old behavior (walk all png)
    # -------------------------
    else:
        lr_paths = []
        for r, _, files in os.walk(lr_root):
            for f in files:
                if not f.lower().endswith(".png"):
                    continue
                full = os.path.join(r, f)
                rel = os.path.relpath(full, lr_root)
                parts = rel.split(os.sep)

                slide_guess = None
                if len(parts) >= 1 and _is_slide_of_case(parts[0], case_id):
                    slide_guess = parts[0]
                elif len(parts) >= 2 and (parts[0] == case_id) and _is_slide_of_case(parts[1], case_id):
                    slide_guess = parts[1]
                else:
                    for p in parts[:-1]:
                        if _is_slide_of_case(p, case_id):
                            slide_guess = p
                            break
                if slide_guess is None:
                    continue
                lr_paths.append(full)

        lr_paths = sorted(lr_paths)
        if len(lr_paths) == 0:
            raise RuntimeError(f"No LR png found for case_id={case_id} under: {lr_root}")

        # optional per-slide limit / global limit（保持你原来的逻辑）
        if limit_per_slide is not None and int(limit_per_slide) > 0:
            per_slide = {}
            kept = []
            for p in lr_paths:
                rel = os.path.relpath(p, lr_root)
                parts = rel.split(os.sep)
                slide = parts[0] if _is_slide_of_case(parts[0], case_id) else (parts[1] if len(parts) > 1 else "UNK")
                per_slide.setdefault(slide, 0)
                if per_slide[slide] < int(limit_per_slide):
                    kept.append(p)
                    per_slide[slide] += 1
            lr_paths = kept

        if limit is not None and int(limit) > 0:
            lr_paths = lr_paths[: int(limit)]

        print(f"[infer_case FULL] case_id={case_id} num_png={len(lr_paths)}", flush=True)

    # -------------------------
    # 3) inference loop (common)
    # -------------------------
    saved = 0
    for i, lr_path in enumerate(lr_paths):
        rel = os.path.relpath(lr_path, lr_root)
        out_path = os.path.join(sr_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if (not overwrite) and os.path.isfile(out_path):
            continue

        lr = _img_to_tensor01(lr_path, device=device)  # expect your existing helper
        meta = dict(meta_defaults)
        meta.update({"lr_path": lr_path, "case_id": case_id})

        sr = infer_fn(lr, None, meta, device).clamp(0, 1)

        if force_out_size is not None and sr.shape[-2:] != (int(force_out_size), int(force_out_size)):
            sr = F.interpolate(sr, size=(int(force_out_size), int(force_out_size)),
                               mode="bicubic", align_corners=False)

        Image.fromarray(_tensor01_to_u8(sr)).save(out_path)  # expect your existing helper
        saved += 1

        if log_every and (saved % int(log_every) == 0):
            print(f"[infer_case] saved={saved} last={out_path}", flush=True)

    print(f"[infer_case] done. saved={saved} -> {sr_root}", flush=True)
    return sr_root

# -----------------------------
# MAIN (hyperparams live here)
# -----------------------------
if __name__ == "__main__":
    # -------------------------
    # 0) hyperparams (edit here)
    # -------------------------
    seed = 2025
    set_seed(seed)

    # data/loader
    # 说明：一些数据加载超参数，不用改
    OUT_IMG_DIR = "/mnt/liangjm/SpRR_data"  # <-- change to yours
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    PATCH_NUM = 300 # 只选择一个样本随机的前300个patch进行训练和验证

    # ours model
    # 说明：选择你训练好的OURS模型进行验证和推理
    OURS_OUTPUT_DIR = "/workspace/SPR_new/logs/INR_SR_Scheme2_V23_20260130_024205/"  # this is training output_dir that contains checkpoints/
    CKPT_PREFER = "best"  # "best" or "latest"
    DEVICE = "cuda"

    # validate sampling
    MAX_BATCHES = 20
    SAMPLE_STEPS = 20  # override sampling steps for inference (optional)

    # inference single folder
    # 说明：选择一个LR和对应HR文件夹进行推理，生成SR结果，方法名可以写在METHOD_NAME
    LR_FOLDER_TO_INFER = "/mnt/liangjm/SpRR_data/lr_png/TCGA-ZP-A9D4-01Z-00-DX1/"
    HR_FOLDER_TO_INFER = "/mnt/liangjm/SpRR_data/hr_png/TCGA-ZP-A9D4-01Z-00-DX1/"
    OUT_ROOT = "./output"  # will save to ./output/eval_image/{name}/{case_id}/
    METHOD_NAME = "Morph-ISR"

    # # -------------------------
    # # 1) loaders
    # # -------------------------
    print("[main] building dataloaders ...")
    train_loader, val_loader, test_loader = build_loaders_by_hparams(
        out_img_dir=OUT_IMG_DIR,
        batch_size=BATCH_SIZE,
        patch_num=PATCH_NUM,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        split_seed=2025,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        require_done=True,
        disc_root=None,  # 或者 "/mnt/liangjm/SpRR_data/disc_maps"
    )
    print("[main] dataloaders ready.")

    # -------------------------
    # 2) load OURS trainer like main.py flow (init -> load weights)
    # -------------------------
    # 加载模型
    # S3_Diff = S3_Diff(OUT_ROOT, METHOD_NAME, OURS_OUTPUT_DIR, CKPT_PREFER, DEVICE, SAMPLE_STEPS, MAX_BATCHES)
    # # 构建输入输出函数。
    # infer_fn_ours = make_infer_fn_ours(S3_Diff, sample_steps=SAMPLE_STEPS, do_color_cc=True, force_out_size=512)
    
    MorphPack = Morph_ISR(OUT_ROOT, METHOD_NAME, OURS_OUTPUT_DIR, CKPT_PREFER, DEVICE, SAMPLE_STEPS, MAX_BATCHES, CONFIG_PATH="config.yml")
    infer_fn_ours = make_infer_fn_morph_isr(MorphPack, do_color_cc=True, force_out_size=512)


    # # -------------------------
    # # 3) validate on val/test (same logic as training validate)
    # # -------------------------
    m_val = eval_with_validate_by_project(
        val_loader, infer_fn_ours, device=DEVICE, 
        max_batches=10, split_name="val", method="S3_Diff")

    m_test = eval_with_validate_by_project(
        test_loader, infer_fn_ours, device=DEVICE, 
        max_batches=10, split_name="test", method="S3_Diff")
    print(f"[main] {METHOD_NAME} val metrics: {json.dumps(m_val, indent=2)}")
    print(f"[main] {METHOD_NAME} test metrics: {json.dumps(m_test, indent=2)}")
    

    # save metrics
    output_dir = os.path.join(OUT_ROOT, "eval_runs", METHOD_NAME)
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "metrics_val.json"), "w") as f:
        json.dump(m_val, f, indent=2)
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(m_test, f, indent=2)
    print(f"[main] saved metrics to: {output_dir}")

    
    