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


def test_weight(model, x, warmup=3, repeat=1, use_cuda_sync=True):
    """
    返回: flops, params, fps
    - warmup: 预热次数
    - repeat: 正式计时重复次数（取平均）
    - use_cuda_sync: GPU 上计时是否加入 synchronize
    """
    model.eval()
    import time
    # warmup
    with torch.no_grad():
        for _ in range(int(warmup)):
            _ = model(x)

    if x.is_cuda and use_cuda_sync:
        torch.cuda.synchronize()

    # timed
    with torch.no_grad():
        start_time = time.time()
        out = None
        for _ in range(int(repeat)):
            out = model(x)
        if x.is_cuda and use_cuda_sync:
            torch.cuda.synchronize()
        end_time = time.time()

    need_time = (end_time - start_time) / max(int(repeat), 1)

    # thop
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x,), verbose=False)
    except Exception as e:
        print(f"[WARN] thop profile failed: {e}")
        flops, params = -1, -1

    fps = round(x.shape[0] / max(need_time, 1e-12), 3)
    return flops, params, fps, out


def Unitconversion(flops, params, fps):
    if params >= 0:
        print("params : {} M".format(round(params / (1000 ** 2), 2)))
    else:
        print("params : NA")

    if flops >= 0:
        print("flop  : {} G".format(round(flops / (1000 ** 3), 2)))
    else:
        print("flop  : NA")

    print("throughout: {} FPS".format(fps))




if __name__ == "__main__":
    config = EasyDict(yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader))

    # seed
    same_seeds(int(cfg_get(config, "data_loader.seed", 2025)))


    ckpt_name = str(cfg_get(config, "checkpoint", "exp"))
    logging_dir = Path(os.getcwd()) / "logs" / f"TV"
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


    
    # ---------------------------------
    # model / optim
    # ---------------------------------
    model, _, _ = build_model_and_optim(config, accelerator.device)

    # prepare
    model= accelerator.prepare(model)
    x = torch.randn(1, 3, 128, 128, device=accelerator.device)
    flops, params, fps, _ = test_weight(
            model=model,
            x=x,
            warmup=2,
            repeat=10,
            use_cuda_sync=True,
        )

    Unitconversion(flops, params, fps)