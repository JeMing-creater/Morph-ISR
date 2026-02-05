# model.py
# -*- coding: utf-8 -*-
"""
Pathology Super-Resolution (SR) model (Strategy 1 + 3).

Goals:
- Inference uses ONLY LR: model(lr) -> sr (no HoVer dependency).
- Training can use hover_bnd / hover_mask as LOSS WEIGHTS ONLY.
- Dense HR-crop supervision + boundary/mask high-frequency guidance + small FFT HF loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# small utils
# -------------------------
def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _make_hr_int_grid(H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(dtype=dtype)


def _hr_int_to_hr_norm(hr_xy_int: torch.Tensor, H: int, W: int) -> torch.Tensor:
    x = hr_xy_int[:, 0].to(torch.float32)
    y = hr_xy_int[:, 1].to(torch.float32)
    x_norm = (x + 0.5) / float(W) * 2.0 - 1.0
    y_norm = (y + 0.5) / float(H) * 2.0 - 1.0
    return torch.stack([x_norm, y_norm], dim=1).to(dtype=torch.float32, device=hr_xy_int.device)


def _fuse_keep_color(lr_up: torch.Tensor, sr_pred: torch.Tensor) -> torch.Tensor:
    """
    Texture-friendly stain stabilization.

    Old behavior: enforce LR_up chroma by only injecting luminance residual -> tends to suppress cross-channel texture.
    New behavior: match SR's global per-channel mean/std to LR_up (stain distribution),
                  while keeping SR local texture/edges; then blend slightly for stability.

    Notes:
    - No extra deps.
    - Inference-only postprocess; does NOT affect training gradients if used after clamp.
    """
    eps = 1e-6

    # ensure float
    lr = lr_up.to(dtype=torch.float32)
    sr = sr_pred.to(dtype=torch.float32)

    # per-sample, per-channel global stats (B,C,1,1)
    lr_mean = lr.mean(dim=(2, 3), keepdim=True)
    lr_std = lr.std(dim=(2, 3), keepdim=True).clamp_min(eps)

    sr_mean = sr.mean(dim=(2, 3), keepdim=True)
    sr_std = sr.std(dim=(2, 3), keepdim=True).clamp_min(eps)

    # match SR distribution to LR distribution (global stain correction only)
    sr_match = (sr - sr_mean) / sr_std * lr_std + lr_mean

    # blend: keep SR texture dominant, only lightly correct stain distribution
    alpha = 0.85  # 0.8~0.9 usually good; higher => more SR texture, lower => more stain stabilization
    out = alpha * sr + (1.0 - alpha) * sr_match

    return out.clamp(0.0, 1.0)



def _blur01(m: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return m
    k = int(round(sigma * 4 + 1))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    mp = F.pad(m, (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(mp, kernel_size=k, stride=1)


def _gray_highpass(x_rgb: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    g = _rgb_to_gray(x_rgb)
    if sigma <= 0:
        return g
    k = int(round(sigma * 4 + 1))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    g_pad = F.pad(g, (pad, pad, pad, pad), mode="replicate")
    blur = F.avg_pool2d(g_pad, kernel_size=k, stride=1)
    return g - blur


def _safe01(x: torch.Tensor) -> torch.Tensor:
    # assume input in [0,1] but keep safe
    return x.clamp(0.0, 1.0)


def _fft_hf_mag(gray: torch.Tensor, hf_ratio: float = 0.35) -> torch.Tensor:
    """
    gray: [B,1,H,W]
    return: magnitude of high-frequency ring (masked)
    """
    B, C, H, W = gray.shape
    assert C == 1
    # FFT
    f = torch.fft.fft2(gray, norm="ortho")
    mag = torch.abs(f)

    # radial high-frequency mask (keep outside radius)
    yy = torch.linspace(-1.0, 1.0, steps=H, device=gray.device).view(H, 1).expand(H, W)
    xx = torch.linspace(-1.0, 1.0, steps=W, device=gray.device).view(1, W).expand(H, W)
    rr = torch.sqrt(xx * xx + yy * yy)  # 0~sqrt(2)
    # keep high freq region
    thr = float(hf_ratio) * math.sqrt(2.0)
    mask = (rr >= thr).to(gray.dtype).view(1, 1, H, W)
    return mag * mask


# -------------------------
# modules
# -------------------------
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_bands: int = 10):
        super().__init__()
        self.num_bands = int(num_bands)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: [N,2] in [-1,1]
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        outs = []
        for k in range(self.num_bands):
            freq = (2.0 ** k) * math.pi
            outs += [torch.sin(freq * x), torch.cos(freq * x), torch.sin(freq * y), torch.cos(freq * y)]
        return torch.cat(outs, dim=1)


class LocalEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, feat_ch: int = 64):
        super().__init__()
        c = int(feat_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.GELU(),
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.net(lr)


class KernelMLP(nn.Module):
    """
    input  = [local_feat(center), PE(hr_xy_norm), lr_frac(dx,dy)]
    output = kernel_logits [*, heads, K*K] + gate_logits [*, heads]
    """
    def __init__(
        self,
        feat_dim: int,
        pe_dim: int,
        kernel_size: int = 7,
        hidden: int = 256,
        depth: int = 5,
        num_heads: int = 4,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.num_heads = int(num_heads)
        K2 = self.kernel_size * self.kernel_size
        in_dim = int(feat_dim + pe_dim + 2)

        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.backbone = nn.Sequential(*layers)

        self.head_kernel = nn.Linear(hidden, self.num_heads * K2)
        self.head_gate = nn.Linear(hidden, self.num_heads)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.head_kernel(h), self.head_gate(h)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class SRRefiner(nn.Module):
    """
    stronger refiner than old groups=3 head, but still small.
    input: concat([lr_up, sr0]) -> residual rgb
    """
    def __init__(self, in_ch: int = 6, base: int = 48, depth: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, base, 3, 1, 1), nn.GELU())
        self.blocks = nn.Sequential(*[ResBlock(base) for _ in range(int(depth))])
        self.head = nn.Conv2d(base, 3, 3, 1, 1)

    def forward(self, lr_up: torch.Tensor, sr0: torch.Tensor) -> torch.Tensor:
        x = torch.cat([lr_up, sr0], dim=1)
        h = self.blocks(self.stem(x))
        return self.head(h)


# -------------------------
# config
# -------------------------
@dataclass
class SRModelConfig:
    scale: int = 4

    feat_ch: int = 64
    pe_bands: int = 10
    mlp_hidden: int = 256
    mlp_depth: int = 5
    kernel_heads: int = 4

    kernel_size: int = 7
    kernel_allow_negative: bool = False
    kernel_logit_norm: bool = True

    # refiner
    use_refiner: bool = True
    refiner_base: int = 48
    refiner_depth: int = 4

    # inference chunk
    infer_chunk: int = 8192

    # tau schedules
    tau_start: float = 1.0
    tau_end: float = 0.5
    tau_warm_epochs: int = 2
    tau_anneal_epochs: int = 8

    kernel_gate_tau_start: float = 8.0
    kernel_gate_tau_end: float = 1.0
    kernel_gate_tau_warm_steps: int = 500
    kernel_gate_tau_anneal_steps: int = 3000

    # training: dense crop
    train_crop: int = 224
    train_crops_per_img: int = 2
    train_crop_chunk: int = 8192

    lambda_crop_l1: float = 1.0

    # hover as loss weights only
    use_hover: bool = True
    hover_bnd_sigma: float = 2.0
    hover_mask_sigma: float = 2.0

    # high-pass guidance
    lambda_hover_hp_bnd: float = 0.25
    lambda_hover_hp_mask: float = 0.15
    hover_hp_sigma: float = 1.2

    # nucleus interior extra L1 (helps push away from bicubic in nuclei)
    lambda_mask_l1: float = 0.25

    # small FFT HF loss
    lambda_fft: float = 0.02
    fft_hf_ratio: float = 0.35


# -------------------------
# model
# -------------------------
class SRModel(nn.Module):
    def __init__(self, cfg: Optional[SRModelConfig] = None):
        super().__init__()
        self.cfg = cfg or SRModelConfig()

        if int(self.cfg.kernel_size) % 2 != 1:
            raise ValueError("kernel_size must be odd.")

        self.encoder = LocalEncoder(in_ch=3, feat_ch=self.cfg.feat_ch)
        self.pe = FourierPositionalEncoding(num_bands=self.cfg.pe_bands)
        pe_dim = 4 * int(self.cfg.pe_bands)

        self.kernel_heads = max(1, int(self.cfg.kernel_heads))
        self.kernel_mlp = KernelMLP(
            feat_dim=int(self.cfg.feat_ch),
            pe_dim=pe_dim,
            kernel_size=int(self.cfg.kernel_size),
            hidden=int(self.cfg.mlp_hidden),
            depth=int(self.cfg.mlp_depth),
            num_heads=self.kernel_heads,
        )

        self.refiner = None
        if bool(getattr(self.cfg, "use_refiner", True)):
            self.refiner = SRRefiner(
                in_ch=6,
                base=int(getattr(self.cfg, "refiner_base", 48)),
                depth=int(getattr(self.cfg, "refiner_depth", 4)),
            )

        # state
        self._epoch = 0
        self._tau = float(self.cfg.tau_start)
        self._global_step = 0
        self._gate_tau = float(self.cfg.kernel_gate_tau_start)

        # debug gate
        self._last_gate_prob = None
        self._last_gate_used = False

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _compute_tau(self, epoch: int) -> float:
        s = float(self.cfg.tau_start)
        e = float(self.cfg.tau_end)
        warm = int(self.cfg.tau_warm_epochs)
        anneal = int(self.cfg.tau_anneal_epochs)
        if anneal <= 0:
            return s
        if epoch <= warm:
            return s
        t = min(1.0, max(0.0, (epoch - warm) / float(anneal)))
        return (1.0 - t) * s + t * e

    def _update_gate_tau(self):
        tau_s = float(getattr(self.cfg, "kernel_gate_tau_start", 1.0))
        tau_e = float(getattr(self.cfg, "kernel_gate_tau_end", tau_s))
        warm = int(getattr(self.cfg, "kernel_gate_tau_warm_steps", 0))
        anneal = int(getattr(self.cfg, "kernel_gate_tau_anneal_steps", 0))
        if warm > 0 and self._global_step < warm:
            self._gate_tau = tau_s
        elif anneal > 0:
            t = (self._global_step - warm) / float(max(1, anneal))
            t = max(0.0, min(1.0, t))
            self._gate_tau = tau_s + (tau_e - tau_s) * t
        else:
            self._gate_tau = tau_e

    def _lr_continuous_from_hr_int(
        self,
        hr_xy_int: torch.Tensor,
        lr_hw: Tuple[int, int],
        hr_hw: Tuple[int, int],
    ):
        scale = float(self.cfg.scale)
        H_lr, W_lr = lr_hw
        H_hr, W_hr = hr_hw

        x_hr = hr_xy_int[:, 0].to(torch.float32)
        y_hr = hr_xy_int[:, 1].to(torch.float32)

        x_lr = (x_hr + 0.5) / scale - 0.5
        y_lr = (y_hr + 0.5) / scale - 0.5
        lr_xy_cont = torch.stack([x_lr, y_lr], dim=1).to(device=hr_xy_int.device, dtype=torch.float32)

        x0 = torch.floor(x_lr)
        y0 = torch.floor(y_lr)
        dx = (x_lr - x0).clamp(0.0, 1.0)
        dy = (y_lr - y0).clamp(0.0, 1.0)
        lr_frac = torch.stack([dx, dy], dim=1).to(device=hr_xy_int.device, dtype=torch.float32)

        x_norm = (x_lr + 0.5) / float(W_lr) * 2.0 - 1.0
        y_norm = (y_lr + 0.5) / float(H_lr) * 2.0 - 1.0
        lr_xy_norm = torch.stack([x_norm, y_norm], dim=1).to(device=hr_xy_int.device, dtype=torch.float32)
        return lr_xy_cont, lr_frac, lr_xy_norm

    def _kernel_weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self.cfg, "kernel_logit_norm", True)):
            m = logits.mean(dim=-1, keepdim=True)
            s = logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = (logits - m) / s

        logits = logits / max(float(self._tau), 1e-6)

        if not bool(self.cfg.kernel_allow_negative):
            return torch.softmax(logits, dim=-1)

        # optional negative weights (rarely needed)
        w = torch.tanh(logits)
        K2 = w.shape[-1]
        center = K2 // 2
        oh = F.one_hot(torch.tensor(center, device=w.device), num_classes=K2).to(w.dtype)
        w = w + oh.view(*([1] * (w.ndim - 1)), K2)
        return w / (w.sum(dim=-1, keepdim=True) + 1e-6)

    def _predict_kernel_weights(
        self,
        feat_lr: torch.Tensor,
        hr_xy_norm: torch.Tensor,
        lr_xy_norm_center: torch.Tensor,
        lr_frac: torch.Tensor,
    ) -> torch.Tensor:
        B = feat_lr.shape[0]
        N = hr_xy_norm.shape[0]
        K2 = int(self.cfg.kernel_size) ** 2
        Hh = self.kernel_heads

        grid = lr_xy_norm_center.view(1, 1, -1, 2).repeat(B, 1, 1, 1)
        cen = F.grid_sample(feat_lr, grid, mode="bilinear", align_corners=False)  # [B,C,1,N]
        cen = cen.squeeze(2).transpose(1, 2).contiguous()  # [B,N,C]

        pe = self.pe(hr_xy_norm).unsqueeze(0).expand(B, -1, -1)
        frac = lr_frac.unsqueeze(0).expand(B, -1, -1)

        x = torch.cat([cen, pe, frac], dim=-1).contiguous().view(B * N, -1)

        klogits, glogits = self.kernel_mlp(x)
        klogits = klogits.view(B, N, Hh, K2)
        glogits = glogits.view(B, N, Hh)

        gate = torch.softmax(glogits / max(float(self._gate_tau), 1e-6), dim=-1)  # [B,N,H]
        self._last_gate_prob = gate.detach()
        self._last_gate_used = True

        w_head = self._kernel_weights_from_logits(klogits.view(B, N * Hh, K2)).view(B, N, Hh, K2)
        w = (gate.unsqueeze(-1) * w_head).sum(dim=2)  # [B,N,K2]
        return w

    def _gather_lr_patch(self, lr: torch.Tensor, lr_xy_cont: torch.Tensor) -> torch.Tensor:
        B, C, H, W = lr.shape
        if C != 3:
            raise ValueError("SRModel assumes RGB input (C=3).")

        K = int(self.cfg.kernel_size)
        r = K // 2
        device = lr.device

        x0 = torch.floor(lr_xy_cont[:, 0]).long()
        y0 = torch.floor(lr_xy_cont[:, 1]).long()

        offs = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        ox = ox.reshape(-1)
        oy = oy.reshape(-1)

        xx = (x0[:, None] + ox[None, :]).clamp(0, W - 1)
        yy = (y0[:, None] + oy[None, :]).clamp(0, H - 1)

        idx = (yy * W + xx).view(-1)
        lr_flat = lr.view(B, C, H * W)
        gathered = lr_flat[:, :, idx]  # [B,3,N*K2]
        return gathered.view(B, C, -1, ox.numel()).permute(0, 2, 3, 1).contiguous()  # [B,N,K2,3]

    def predict_points_base(
        self,
        lr: torch.Tensor,
        hr_xy_int: torch.Tensor,
        hr_hw: Tuple[int, int],
        feat_lr: torch.Tensor,
        lr_up: torch.Tensor,
    ) -> torch.Tensor:
        B, _, H_lr, W_lr = lr.shape
        H_hr, W_hr = hr_hw

        hr_xy_norm = _hr_int_to_hr_norm(hr_xy_int, H_hr, W_hr)
        lr_xy_cont, lr_frac, lr_xy_norm_center = self._lr_continuous_from_hr_int(
            hr_xy_int=hr_xy_int,
            lr_hw=(H_lr, W_lr),
            hr_hw=(H_hr, W_hr),
        )

        xh = hr_xy_int[:, 0].long().clamp(0, W_hr - 1)
        yh = hr_xy_int[:, 1].long().clamp(0, H_hr - 1)
        lr_up_pts = lr_up[:, :, yh, xh].permute(0, 2, 1).contiguous()  # [B,N,3]

        w = self._predict_kernel_weights(
            feat_lr=feat_lr,
            hr_xy_norm=hr_xy_norm,
            lr_xy_norm_center=lr_xy_norm_center,
            lr_frac=lr_frac,
        )  # [B,N,K2]

        patch = self._gather_lr_patch(lr, lr_xy_cont)  # [B,N,K2,3]
        sr_like = (patch * w.unsqueeze(-1)).sum(dim=2)  # [B,N,3]
        return sr_like - lr_up_pts  # residual

    def _predict_sr_crop(self, lr: torch.Tensor, feat_lr: torch.Tensor, lr_up: torch.Tensor, top: int, left: int, crop: int):
        device = lr.device
        B = lr.shape[0]
        H_hr, W_hr = lr_up.shape[-2], lr_up.shape[-1]

        ys = torch.arange(top, top + crop, device=device)
        xs = torch.arange(left, left + crop, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        hr_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(torch.float32)
        N = hr_xy.shape[0]

        chunk = int(getattr(self.cfg, "train_crop_chunk", 8192))
        outs = []
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            res_pts = self.predict_points_base(lr, hr_xy[st:ed], (H_hr, W_hr), feat_lr, lr_up)  # [B,n,3]
            outs.append(res_pts)
        res_all = torch.cat(outs, dim=1)  # [B,N,3]
        res_map = res_all.view(B, crop, crop, 3).permute(0, 3, 1, 2).contiguous()

        lr_up_crop = lr_up[:, :, top:top + crop, left:left + crop]
        sr0 = lr_up_crop + res_map  # do NOT clamp here (keep gradients)
        if self.refiner is not None:
            sr = sr0 + self.refiner(lr_up_crop, _safe01(sr0))  # refiner sees bounded sr0
        else:
            sr = sr0
        return sr, lr_up_crop, sr0

    @torch.no_grad()
    def super_resolve(self, lr: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        B, _, H_lr, W_lr = lr.shape
        if out_hw is None:
            out_hw = (H_lr * int(self.cfg.scale), W_lr * int(self.cfg.scale))
        H_hr, W_hr = out_hw

        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        feat_lr = self.encoder(lr)

        hr_xy_int_full = _make_hr_int_grid(H_hr, W_hr, device=lr.device, dtype=torch.float32)
        N = hr_xy_int_full.shape[0]
        chunk = int(self.cfg.infer_chunk)

        outs = []
        for st in range(0, N, chunk):
            ed = min(N, st + chunk)
            res_pts = self.predict_points_base(lr, hr_xy_int_full[st:ed], (H_hr, W_hr), feat_lr, lr_up)
            outs.append(res_pts)

        res_all = torch.cat(outs, dim=1)
        res_map = res_all.view(B, H_hr, W_hr, 3).permute(0, 3, 1, 2).contiguous()

        sr0 = lr_up + res_map
        if self.refiner is not None:
            sr = sr0 + self.refiner(lr_up, _safe01(sr0))
        else:
            sr = sr0
        sr = sr.clamp(0, 1)
        return _fuse_keep_color(lr_up, sr)

    def forward(self, lr: torch.Tensor, hover_bnd: Optional[torch.Tensor] = None, hover_mask: Optional[torch.Tensor] = None):
        # hover_* accepted only for compatibility, ignored in inference
        return self.super_resolve(lr, out_hw=None)

    def compute_loss(
        self,
        lr: torch.Tensor,
        hr: torch.Tensor,
        hover_bnd: Optional[torch.Tensor] = None,
        hover_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        B, _, H_lr, W_lr = lr.shape
        Bh, _, H_hr, W_hr = hr.shape
        if B != Bh:
            raise ValueError("lr/hr batch size mismatch")

        device = lr.device

        if self.training:
            self._global_step += 1
            self._update_gate_tau()

        self._tau = float(self._compute_tau(self._epoch))

        lr_up = F.interpolate(lr, size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)
        feat_lr = self.encoder(lr)

        # ---- build weight maps (TRAIN ONLY) ----
        W_bnd = torch.zeros((B, 1, H_hr, W_hr), device=device, dtype=torch.float32)
        W_msk = torch.zeros((B, 1, H_hr, W_hr), device=device, dtype=torch.float32)

        if bool(getattr(self.cfg, "use_hover", True)) and (hover_bnd is not None):
            hb = hover_bnd
            if hb.ndim == 3:
                hb = hb.unsqueeze(1)
            hb = hb[:, :1].to(device=device, dtype=torch.float32)
            if hb.max() > 1.5:
                hb = hb / 255.0
            hb = hb.clamp(0.0, 1.0)
            if hb.shape[-2:] != (H_hr, W_hr):
                hb = F.interpolate(hb, size=(H_hr, W_hr), mode="nearest")
            W_bnd = _blur01(hb, sigma=float(getattr(self.cfg, "hover_bnd_sigma", 2.0))).clamp(0.0, 1.0)

        if bool(getattr(self.cfg, "use_hover", True)) and (hover_mask is not None):
            hm = hover_mask
            if hm.ndim == 3:
                hm = hm.unsqueeze(1)
            hm = hm[:, :1].to(device=device, dtype=torch.float32)
            if hm.max() > 1.5:
                hm = hm / 255.0
            hm = hm.clamp(0.0, 1.0)
            if hm.shape[-2:] != (H_hr, W_hr):
                hm = F.interpolate(hm, size=(H_hr, W_hr), mode="nearest")
            W_msk = _blur01(hm, sigma=float(getattr(self.cfg, "hover_mask_sigma", 2.0))).clamp(0.0, 1.0)

        crop = int(getattr(self.cfg, "train_crop", 224))
        crop = max(96, min(crop, H_hr, W_hr))
        crops_per_img = int(getattr(self.cfg, "train_crops_per_img", 2))

        lam_l1 = float(getattr(self.cfg, "lambda_crop_l1", 1.0))
        lam_hp_bnd = float(getattr(self.cfg, "lambda_hover_hp_bnd", 0.0))
        lam_hp_msk = float(getattr(self.cfg, "lambda_hover_hp_mask", 0.0))
        lam_msk_l1 = float(getattr(self.cfg, "lambda_mask_l1", 0.0))
        hp_sigma = float(getattr(self.cfg, "hover_hp_sigma", 1.2))

        lam_fft = float(getattr(self.cfg, "lambda_fft", 0.0))
        fft_ratio = float(getattr(self.cfg, "fft_hf_ratio", 0.35))

        loss_l1 = torch.tensor(0.0, device=device)
        loss_hp_bnd = torch.tensor(0.0, device=device)
        loss_hp_msk = torch.tensor(0.0, device=device)
        loss_msk_l1 = torch.tensor(0.0, device=device)
        loss_fft = torch.tensor(0.0, device=device)

        for b in range(B):
            for _ in range(crops_per_img):
                # bias crop selection to bnd or mask if available
                mix = (W_bnd[b, 0] + 0.5 * W_msk[b, 0]).reshape(-1)
                if float(mix.sum().detach().cpu()) > 1e-6:
                    w = mix + 1e-6
                    prob = w / w.sum()
                    idx0 = torch.multinomial(prob, num_samples=1, replacement=True)[0].item()
                    cy = idx0 // W_hr
                    cx = idx0 - cy * W_hr
                    top = int(max(0, min(H_hr - crop, cy - crop // 2)))
                    left = int(max(0, min(W_hr - crop, cx - crop // 2)))
                else:
                    top = torch.randint(0, H_hr - crop + 1, (1,), device=device).item()
                    left = torch.randint(0, W_hr - crop + 1, (1,), device=device).item()

                sr_crop, lr_up_crop, _sr0 = self._predict_sr_crop(
                    lr=lr[b:b + 1],
                    feat_lr=feat_lr[b:b + 1],
                    lr_up=lr_up[b:b + 1],
                    top=top,
                    left=left,
                    crop=crop,
                )
                hr_crop = hr[b:b + 1, :, top:top + crop, left:left + crop]

                # base L1 (dense)
                loss_l1 = loss_l1 + (sr_crop - hr_crop).abs().mean()

                # mask-weighted extra L1 (push nucleus details)
                if lam_msk_l1 > 0:
                    w_m = W_msk[b:b + 1, :, top:top + crop, left:left + crop]
                    denom = w_m.mean().clamp_min(1e-6)
                    loss_msk_l1 = loss_msk_l1 + (w_m * (sr_crop - hr_crop).abs()).mean() / denom

                # high-pass on boundary and mask
                if (lam_hp_bnd > 0) or (lam_hp_msk > 0):
                    hp_sr = _gray_highpass(_safe01(sr_crop), sigma=hp_sigma)
                    hp_hr = _gray_highpass(hr_crop, sigma=hp_sigma)

                    if lam_hp_bnd > 0:
                        w_b = W_bnd[b:b + 1, :, top:top + crop, left:left + crop]
                        denom = w_b.mean().clamp_min(1e-6)
                        loss_hp_bnd = loss_hp_bnd + (w_b * (hp_sr - hp_hr).abs()).mean() / denom

                    if lam_hp_msk > 0:
                        w_m = W_msk[b:b + 1, :, top:top + crop, left:left + crop]
                        denom = w_m.mean().clamp_min(1e-6)
                        loss_hp_msk = loss_hp_msk + (w_m * (hp_sr - hp_hr).abs()).mean() / denom

                # small FFT HF magnitude loss (no hover needed)
                if lam_fft > 0:
                    g_sr = _rgb_to_gray(_safe01(sr_crop))
                    g_hr = _rgb_to_gray(hr_crop)
                    mag_sr = _fft_hf_mag(g_sr, hf_ratio=fft_ratio)
                    mag_hr = _fft_hf_mag(g_hr, hf_ratio=fft_ratio)
                    loss_fft = loss_fft + (mag_sr - mag_hr).abs().mean()

        norm = float(B * crops_per_img)
        loss_l1 = loss_l1 / norm
        loss_msk_l1 = loss_msk_l1 / norm
        loss_hp_bnd = loss_hp_bnd / norm
        loss_hp_msk = loss_hp_msk / norm
        loss_fft = loss_fft / norm

        total = (
            lam_l1 * loss_l1
            + lam_msk_l1 * loss_msk_l1
            + lam_hp_bnd * loss_hp_bnd
            + lam_hp_msk * loss_hp_msk
            + lam_fft * loss_fft
        )

        # gate diagnostics (optional, for logging only)
        gateH = None
        gateMax = None
        if self._last_gate_used and (self._last_gate_prob is not None):
            g = self._last_gate_prob  # [B,N,H]
            gpos = g.clamp_min(1e-12)
            ent = (-(gpos * gpos.log()).sum(dim=-1)).mean()
            gmax = g.max(dim=-1).values.mean()
            gateH = float(ent.detach().cpu())
            gateMax = float(gmax.detach().cpu())

        if return_debug:
            dbg = {
                "loss_crop_l1": float(loss_l1.detach().cpu()),
                "loss_mask_l1": float(loss_msk_l1.detach().cpu()),
                "loss_hp_bnd": float(loss_hp_bnd.detach().cpu()),
                "loss_hp_mask": float(loss_hp_msk.detach().cpu()),
                "loss_fft": float(loss_fft.detach().cpu()),
                "tau": float(getattr(self, "_tau", 1.0)),
                "gate_tau": float(getattr(self, "_gate_tau", 1.0)),
                "global_step": int(getattr(self, "_global_step", 0)),
                "gate_entropy": gateH,
                "gate_max": gateMax,
            }
            return total, dbg

        return total


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SRModelConfig()
    model = SRModel(cfg=cfg).to(device)

    lr = torch.rand(2, 3, 128, 128, device=device)
    hr = torch.rand(2, 3, 512, 512, device=device)
    hb = torch.rand(2, 1, 512, 512, device=device)
    hm = torch.rand(2, 1, 512, 512, device=device)

    sr = model(lr)
    print("sr:", sr.shape)

    loss, dbg = model.compute_loss(lr, hr, hover_bnd=hb, hover_mask=hm, return_debug=True)
    print("loss:", float(loss), dbg)
