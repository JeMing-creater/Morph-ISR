import os
import sys
import json
import yaml
import random
from datetime import datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from accelerate import Accelerator

# ======================
# CLAM / slicing imports
# ======================
CLAM_ROOT = "CLAM"
sys.path.append(CLAM_ROOT)
import src.preprocess_tcga_luad_with_clam as slic_tool

from src.loader import build_case_split_dataloaders
from src.utils import Logger, same_seeds
from src.sr_metrics import SRMetrics
from src.model import SRModel, SRModelConfig


# ----------------------
# helpers: config access
# ----------------------
def cfg_get(cfg: EasyDict, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if not hasattr(cur, k):
            return default
        cur = getattr(cur, k)
    return cur


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ----------------------
# model / optim / sched
# ----------------------
def build_model_and_optim(config: EasyDict, device: torch.device):
    """
    Build SRModel + optimizer.
    关键：只把 config.model 中 SRModelConfig 真正支持的字段传进去，
    自动过滤旧参数/多余参数，避免“unexpected keyword argument”。
    """
    import dataclasses
    from torch.optim import AdamW

    # 1) 收集 SRModelConfig 的合法字段
    valid_fields = {f.name for f in dataclasses.fields(SRModelConfig)}

    # 2) 从 config.model 读取参数（没有就给空 dict）
    model_cfg = {}
    if hasattr(config, "model") and isinstance(config.model, (dict, EasyDict)):
        model_cfg = dict(config.model)

    # 3) 强制写入 scale（通常来自 down_scale）
    model_cfg["scale"] = int(cfg_get(config, "data_loader.down_scale", 4))

    # 4) 过滤掉 SRModelConfig 不认识的字段
    filtered = {k: v for k, v in model_cfg.items() if k in valid_fields}

    # 5) 构建模型
    mcfg = SRModelConfig(**filtered)
    model = SRModel(cfg=mcfg).to(device)

    # 6) Optimizer
    lr = float(cfg_get(config, "trainer.lr", 1e-4))
    wd = float(cfg_get(config, "trainer.weight_decay", 1e-4))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    return model, optimizer, scheduler

# ----------------------
# checkpoint
# ----------------------
def save_checkpoint_latest(save_dir: Path, accelerator: Accelerator, model, optimizer, scheduler, epoch: int, best_metric):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "checkpoint_latest.pt"

    unwrap = accelerator.unwrap_model(model)
    obj = {
        "epoch": int(epoch),
        "best_metric": float(best_metric) if best_metric is not None else None,
        "model": unwrap.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    accelerator.save(obj, str(ckpt_path))


def try_resume_from_latest(save_dir: Path, accelerator: Accelerator, model, optimizer, scheduler):
    ckpt_path = save_dir / "checkpoint_latest.pt"
    if not ckpt_path.exists():
        return 0, None

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    unwrap = accelerator.unwrap_model(model)
    unwrap.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = ckpt.get("best_metric", None)
    return start_epoch, best_metric


def save_best_model(save_dir: Path, accelerator: Accelerator, model, epoch: int, best_metric: float):
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pt"
    meta_path = save_dir / "best_meta.json"

    unwrap = accelerator.unwrap_model(model)
    accelerator.save({"model": unwrap.state_dict()}, str(best_path))

    if accelerator.is_local_main_process:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"epoch": int(epoch), "best_metric": float(best_metric)}, f, indent=2)


# ----------------------
# visuals
# ----------------------
@torch.no_grad()
def save_triplet_visual(save_path: Path, lr, hr, sr):
    """Save HR | LR_up | SR triplet as a horizontal concatenated PNG."""
    lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)

    def to_uint8(img):
        img = img[0].detach().cpu().permute(1, 2, 0).numpy()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    a = to_uint8(hr)
    b = to_uint8(lr_up)
    c = to_uint8(sr)
    cat = np.concatenate([a, b, c], axis=1)

    from PIL import Image
    Image.fromarray(cat).save(str(save_path))


@torch.no_grad()
def save_random_train_visual_to_val_vis(
    accelerator: Accelerator,
    model,
    train_loader,
    epoch: int,
    val_vis_dir: Path,
):
    """
    Save ONE random training sample visualization into the SAME val_vis folder,
    distinguished by filename: train_epoch_XXXX.png
    """
    if not accelerator.is_local_main_process:
        return
    if len(train_loader) <= 0:
        return

    vis_idx = random.randint(0, max(len(train_loader) - 1, 0))

    picked = None
    for i, batch in enumerate(train_loader):
        if i == vis_idx:
            picked = batch
            break
    if picked is None:
        return

    was_training = model.training
    model.eval()

    lr = picked["lr"].to(accelerator.device, non_blocking=True)
    hr = picked["hr"].to(accelerator.device, non_blocking=True)
    sr = model(lr).clamp(0, 1)

    save_triplet_visual(val_vis_dir / f"train_epoch_{epoch:04d}.png", lr, hr, sr)

    if was_training:
        model.train()


# ----------------------
# train / val
# ----------------------
def train_one_epoch(accelerator: Accelerator, model, optimizer, train_loader, epoch: int, config: EasyDict):
    model.train()

    if hasattr(model, "set_epoch") and callable(getattr(model, "set_epoch")):
        try:
            model.set_epoch(epoch)
        except Exception:
            pass

    log_every = int(cfg_get(config, "trainer.log_every", 50))
    grad_clip = cfg_get(config, "trainer.grad_clip", None)

    run_total = 0.0
    run_l1 = 0.0
    run_mask_l1 = 0.0
    run_hp_bnd = 0.0
    run_hp_msk = 0.0
    run_fft = 0.0
    step = 0

    for batch in train_loader:
        lr = batch["lr"].to(accelerator.device, non_blocking=True)
        hr = batch["hr"].to(accelerator.device, non_blocking=True)

        hover_bnd = batch.get("hover_bnd", None)
        hover_mask = batch.get("hover_mask", None)
        if hover_bnd is not None:
            hover_bnd = hover_bnd.to(accelerator.device, non_blocking=True)
        if hover_mask is not None:
            hover_mask = hover_mask.to(accelerator.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss, dbg = model.compute_loss(
            lr,
            hr,
            hover_bnd=hover_bnd,
            hover_mask=hover_mask,
            return_debug=True,
        )

        accelerator.backward(loss)

        if grad_clip is not None:
            accelerator.clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()

        total = float(loss.detach().item())
        l1 = float(dbg.get("loss_crop_l1", 0.0))
        mask_l1 = float(dbg.get("loss_mask_l1", 0.0))
        hp_bnd = float(dbg.get("loss_hp_bnd", 0.0))
        hp_msk = float(dbg.get("loss_hp_mask", 0.0))
        fft = float(dbg.get("loss_fft", 0.0))

        run_total += total
        run_l1 += l1
        run_mask_l1 += mask_l1
        run_hp_bnd += hp_bnd
        run_hp_msk += hp_msk
        run_fft += fft
        step += 1

        if accelerator.is_local_main_process and (step % log_every == 0):
            avg_total = run_total / step
            denom = max(avg_total, 1e-8)

            def _ratio(x): return (x / step) / denom * 100.0

            def _fmt(x, p=4):
                if x is None:
                    return "NA"
                try:
                    return f"{float(x):.{p}f}"
                except Exception:
                    return "NA"

            tau = dbg.get("tau", None)
            gateH = dbg.get("gate_entropy", None)
            gateMax = dbg.get("gate_max", None)

            accelerator.print(
                f"[Epoch {epoch}][{step}] "
                f"total={total:.4f} (avg={avg_total:.4f}) | "
                f"l1={l1:.4f} ({_ratio(run_l1):.1f}%) | "
                f"maskL1={mask_l1:.4f} ({_ratio(run_mask_l1):.1f}%) | "
                f"hpB={hp_bnd:.4f} ({_ratio(run_hp_bnd):.1f}%) | "
                f"hpM={hp_msk:.4f} ({_ratio(run_hp_msk):.1f}%) | "
                f"fft={fft:.4f} ({_ratio(run_fft):.1f}%)"
                f" || tau={_fmt(tau,3)}"
                f" | gateH={_fmt(gateH,3)} gateMax={_fmt(gateMax,3)}"
            )

    return run_total / max(step, 1)


@torch.no_grad()
def validate_one_epoch(accelerator: Accelerator, model, val_loader, epoch: int, val_vis_dir: Path):
    """
    Validation: compute PSNR/SSIM/LPIPS and save ONE random validation visualization
    into val_vis folder: val_epoch_XXXX.png
    """
    model.eval()
    metrics = SRMetrics(device=str(accelerator.device))

    psnr_list, ssim_list, lpips_list = [], [], []

    # choose a batch index for visualization (sync across processes)
    if accelerator.is_local_main_process:
        vis_idx = random.randint(0, max(len(val_loader) - 1, 0))
        t = torch.tensor([vis_idx], device=accelerator.device, dtype=torch.long)
    else:
        t = torch.tensor([0], device=accelerator.device, dtype=torch.long)

    if accelerator.num_processes > 1:
        accelerator.broadcast(t, from_process=0)
    vis_idx = int(t.item())

    for i, batch in enumerate(val_loader):
        lr = batch["lr"].to(accelerator.device, non_blocking=True)
        hr = batch["hr"].to(accelerator.device, non_blocking=True)

        sr = model(lr).clamp(0, 1)

        psnr_list.append(metrics.psnr(sr, hr))
        ssim_list.append(metrics.ssim(sr, hr))
        lpips_list.append(metrics.lpips(sr, hr))

        if accelerator.is_local_main_process and i == vis_idx:
            save_triplet_visual(val_vis_dir / f"val_epoch_{epoch:04d}.png", lr, hr, sr)

    val_metrics = {
        "psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "lpips": float(np.mean(lpips_list)) if lpips_list else 0.0,
    }
    return val_metrics


def is_better(val_metrics: dict, best_metric, config: EasyDict):
    key = str(cfg_get(config, "validator.metric_for_best", "psnr"))
    mode = str(cfg_get(config, "validator.best_mode", "max")).lower()
    val = float(val_metrics.get(key, 0.0))

    if best_metric is None:
        return True
    if mode == "max":
        return val > float(best_metric)
    return val < float(best_metric)


# ----------------------
# main
# ----------------------
if __name__ == "__main__":
    config = EasyDict(yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader))

    # seed
    same_seeds(int(cfg_get(config, "data_loader.seed", 2025)))

    # logging dir
    ckpt_name = str(cfg_get(config, "checkpoint", "exp"))
    logging_dir = Path(os.getcwd()) / "logs" / f"{ckpt_name}_{now_str()}"
    logging_dir.mkdir(parents=True, exist_ok=True)

    # accelerator
    mp = "fp16" if bool(cfg_get(config, "trainer.use_amp", False)) else "no"
    accelerator = Accelerator(
        cpu=False,
        log_with=["tensorboard"],
        project_dir=str(logging_dir),
        mixed_precision=mp,
    )

    Logger(str(logging_dir) if accelerator.is_local_main_process else None)

    # ---------------------------------
    # slicing (optional)
    # ---------------------------------
    image_dict = {
        "TCGA-LUAD": [config.TCGA_LUAD.root, config.TCGA_LUAD.choose_WSI],
        "TCGA-KIRC": [config.TCGA_KIRC.root, config.TCGA_KIRC.choose_WSI],
        "TCGA-LIHC": [config.TCGA_LIHC.root, config.TCGA_LIHC.choose_WSI],
    }

    skip_slicing = bool(cfg_get(config, "data_loader.skip_slicing", False))
    if not skip_slicing:
        slic_tool.run_clam_and_export(
            data_cfg=image_dict,
            out_img_dir=config.data_loader.out_img_dir,
            patch_size=config.data_loader.patch_size,
            step_size=config.data_loader.step_size,
            patch_level=config.data_loader.patch_level,
            down_scale=config.data_loader.down_scale,
            min_tissue_ratio=config.data_loader.min_tissue_ratio,
            seed=config.data_loader.seed,
            limit_samples=config.data_loader.patch_num,
        )
        accelerator.print("[CHECK] slicing done.")
    else:
        accelerator.print("[CHECK] skip slicing (data_loader.skip_slicing=true).")

    # ---------------------------------
    # dataloaders
    # ---------------------------------
    train_loader, val_loader, test_loader = build_case_split_dataloaders(
        out_img_dir=config.data_loader.out_img_dir,
        batch_size=int(cfg_get(config, "trainer.batch_size", 1)),
        patch_num=int(cfg_get(config, "data_loader.patch_num", 200)),
        train_ratio=float(cfg_get(config, "data_loader.train_ratio", 0.8)),
        val_ratio=float(cfg_get(config, "data_loader.val_ratio", 0.1)),
        test_ratio=float(cfg_get(config, "data_loader.test_ratio", 0.1)),
        split_seed=int(cfg_get(config, "data_loader.seed", 2025)),
        num_workers=int(cfg_get(config, "data_loader.num_workers", 4)),
        pin_memory=bool(cfg_get(config, "data_loader.pin_memory", True)),
    )

    # ---------------------------------
    # model / optim
    # ---------------------------------
    model, optimizer, scheduler = build_model_and_optim(config, accelerator.device)

    # prepare
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    save_dir = logging_dir
    start_epoch, best_metric = try_resume_from_latest(save_dir, accelerator, model, optimizer, scheduler)
    num_epochs = int(cfg_get(config, "trainer.epochs", 10))

    # val_vis dir (save BOTH val and train visuals here)
    vis_subdir = str(cfg_get(config, "validator.vis_subdir", "val_vis"))
    val_vis_dir = save_dir / vis_subdir
    val_vis_dir.mkdir(parents=True, exist_ok=True)

    accelerator.print(f"[CHECK] start_epoch={start_epoch} best_metric={best_metric}")

    for epoch in range(start_epoch, num_epochs):
        # enable tau annealing
        unwrap = accelerator.unwrap_model(model)
        if hasattr(unwrap, "set_epoch"):
            unwrap.set_epoch(epoch)

        train_loss = train_one_epoch(accelerator, model, optimizer, train_loader, epoch, config)

        # validate and save val_epoch_XXXX.png
        val_metrics = validate_one_epoch(accelerator, model, val_loader, epoch, val_vis_dir)

        # additionally save train_epoch_XXXX.png in the SAME folder
        save_random_train_visual_to_val_vis(accelerator, model, train_loader, epoch, val_vis_dir)

        metric_key = str(cfg_get(config, "validator.metric_for_best", "psnr"))
        if is_better(val_metrics, best_metric, config):
            best_metric = float(val_metrics.get(metric_key, 0.0))
            save_best_model(save_dir, accelerator, model, epoch, best_metric)

        save_checkpoint_latest(save_dir, accelerator, model, optimizer, scheduler, epoch, best_metric)

        if accelerator.is_local_main_process:
            accelerator.print(
                f"[Epoch {epoch}] train_loss={train_loss:.4f} "
                f"val={val_metrics} best({metric_key})={best_metric}"
            )
