# src/sr_metrics.py
import torch
import torch.nn.functional as F
import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class SRMetrics:
    def __init__(self, device="cuda"):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def psnr(self, sr, hr):
        # sr, hr: [B,3,H,W] in [0,1]
        sr = sr.clamp(0, 1).cpu().numpy()
        hr = hr.clamp(0, 1).cpu().numpy()

        vals = []
        for i in range(sr.shape[0]):
            vals.append(
                peak_signal_noise_ratio(
                    hr[i].transpose(1,2,0),
                    sr[i].transpose(1,2,0),
                    data_range=1.0
                )
            )
        return float(np.mean(vals))

    @torch.no_grad()
    def ssim(self, sr, hr):
        sr = sr.clamp(0, 1).cpu().numpy()
        hr = hr.clamp(0, 1).cpu().numpy()

        vals = []
        for i in range(sr.shape[0]):
            vals.append(
                structural_similarity(
                    hr[i].transpose(1,2,0),
                    sr[i].transpose(1,2,0),
                    channel_axis=2,
                    data_range=1.0
                )
            )
        return float(np.mean(vals))

    @torch.no_grad()
    def lpips(self, sr, hr):
        # LPIPS expects [-1,1]
        sr = sr * 2 - 1
        hr = hr * 2 - 1
        d = self.lpips_fn(sr, hr)
        return float(d.mean().item())

    @torch.no_grad()
    def shift_tolerant_lpips(self, sr, hr, max_shift=4):
        """
        Shift-Tolerant LPIPS:
        min LPIPS over small spatial shifts (±max_shift pixels)
        using overlap cropping (NO wrap-around).
        """
        # LPIPS expects [-1, 1]
        sr = sr * 2 - 1
        hr = hr * 2 - 1

        B, C, H, W = sr.shape
        best = torch.full((B,), float("inf"), device=sr.device)

        for dx in range(-max_shift, max_shift + 1):
            for dy in range(-max_shift, max_shift + 1):
                # overlap crop indices
                x1_s = max(0, dx)
                x2_s = H + min(0, dx)
                y1_s = max(0, dy)
                y2_s = W + min(0, dy)

                x1_h = max(0, -dx)
                x2_h = H + min(0, -dx)
                y1_h = max(0, -dy)
                y2_h = W + min(0, -dy)

                # crop to overlap
                sr_c = sr[:, :, x1_s:x2_s, y1_s:y2_s]
                hr_c = hr[:, :, x1_h:x2_h, y1_h:y2_h]

                # safety (should always hold)
                if sr_c.numel() == 0 or hr_c.numel() == 0:
                    continue

                d = self.lpips_fn(sr_c, hr_c).view(B)
                best = torch.minimum(best, d)

        return float(best.mean().item())

    @torch.no_grad()
    def gradient_l1(self, sr, hr):
        # sr, hr: [B,3,H,W] in [0,1]
        def grad(x):
            gx = x[..., :, 1:] - x[..., :, :-1]
            gy = x[..., 1:, :] - x[..., :-1, :]
            return gx.abs().mean() + gy.abs().mean()

        return float((grad(sr) - grad(hr)).abs().mean().item())