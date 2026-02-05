import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, PngImagePlugin

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# --- Pillow robustness (your PNG issue) ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


# Optional fallback reader
import cv2




def slide_to_case_id(slide_id: str) -> str:
    parts = slide_id.split("-")
    return "-".join(parts[:3])


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


@dataclass
class SurvivalInfo:
    time: float
    event: int  # 1 if days_to_death used, else 0


class ClinicalSurvivalIndex:
    def __init__(
        self,
        clin_path: str,
        id_col: str = "case_submitter_id",
        death_col: str = "days_to_death",
        follow_col: str = "days_to_last_follow_up",
    ):
        if not os.path.exists(clin_path):
            raise FileNotFoundError(f"clinical.tsv not found: {clin_path}")

        self.id_col = id_col
        self.death_col = death_col
        self.follow_col = follow_col

        df = pd.read_csv(clin_path, sep="\t")
        if id_col not in df.columns:
            raise ValueError(f"clinical.tsv missing column: {id_col}")
        if death_col not in df.columns and follow_col not in df.columns:
            raise ValueError(f"clinical.tsv missing both: {death_col}, {follow_col}")

        df[id_col] = df[id_col].astype(str)
        self.df = df.set_index(id_col, drop=False)

    def get(self, case_id: str) -> Optional[SurvivalInfo]:
        if case_id not in self.df.index:
            return None

        row = self.df.loc[case_id]

        # Prefer vital_status when available
        vital = None
        if "vital_status" in self.df.columns:
            vital = str(row.get("vital_status", "")).strip().lower()

        if vital in ("dead", "deceased"):
            t = _safe_float(row.get(self.death_col, None))
            if t is not None:
                return SurvivalInfo(time=t, event=1)
            return None

        if vital in ("alive", "living"):
            t = _safe_float(row.get(self.follow_col, None))
            if t is not None:
                return SurvivalInfo(time=t, event=0)
            return None

        # Fallback: old behavior
        t = _safe_float(row.get(self.death_col, None)) if self.death_col in self.df.columns else None
        if t is not None:
            return SurvivalInfo(time=t, event=1)

        t = _safe_float(row.get(self.follow_col, None)) if self.follow_col in self.df.columns else None
        if t is not None:
            return SurvivalInfo(time=t, event=0)

        return None



def build_survival_map(
    df: pd.DataFrame,
    id_col: str = "case_submitter_id",
    death_col: str = "days_to_death",
    follow_col: str = "days_to_last_follow_up",
) -> Dict[str, SurvivalInfo]:
    """
    Build {case_id -> SurvivalInfo(time, event)} from clinical.tsv DataFrame.

    - If vital_status exists:
        dead/deceased -> use days_to_death, event=1
        alive/living  -> use days_to_last_follow_up, event=0
    - Else fallback:
        if days_to_death valid -> event=1
        elif days_to_last_follow_up valid -> event=0
    """
    if id_col not in df.columns:
        raise ValueError(f"clinical.tsv missing column: {id_col}")
    if (death_col not in df.columns) and (follow_col not in df.columns):
        raise ValueError(f"clinical.tsv missing both: {death_col}, {follow_col}")

    out: Dict[str, SurvivalInfo] = {}

    has_vital = "vital_status" in df.columns
    for _, row in df.iterrows():
        case_id = str(row[id_col])

        vital = None
        if has_vital:
            vital = str(row.get("vital_status", "")).strip().lower()

        # preferred branch if vital_status available
        if vital in ("dead", "deceased"):
            t = _safe_float(row.get(death_col, None))
            if t is not None:
                out[case_id] = SurvivalInfo(time=t, event=1)
            continue

        if vital in ("alive", "living"):
            t = _safe_float(row.get(follow_col, None))
            if t is not None:
                out[case_id] = SurvivalInfo(time=t, event=0)
            continue

        # fallback
        t = _safe_float(row.get(death_col, None)) if death_col in df.columns else None
        if t is not None:
            out[case_id] = SurvivalInfo(time=t, event=1)
            continue

        t = _safe_float(row.get(follow_col, None)) if follow_col in df.columns else None
        if t is not None:
            out[case_id] = SurvivalInfo(time=t, event=0)
            continue

    return out



def _list_pngs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.png")))


def _has_done_flag(hr_slide_dir: str) -> bool:
    return os.path.exists(os.path.join(hr_slide_dir, ".DONE"))


class PathologySRSurvivalDataset(Dataset):
    """
    Patch-level dataset for SR + survival:
      - Reads paired (lr, hr) PNG patches
      - (NEW) Reads HoVer-Net boundary png as condition map
      - Filters cases without valid survival time
      - Ensures lr/hr/(hover) are aligned by patch filename
      - patch_num: per-slide cap (take first N patches sorted by name)

    Return dict:
      {
        "lr": Tensor[3,128,128],
        "hr": Tensor[3,512,512],
        "hover": Tensor[1,512,512] (0/1 float)   # NEW
        "time": Tensor[],
        "event": Tensor[],
        "meta": {...}
      }
    """
    def __init__(
        self,
        out_img_dir: str,
        id_col: str = "case_submitter_id",
        death_col: str = "days_to_death",
        follow_col: str = "days_to_last_follow_up",
        require_done: bool = True,
        patch_num: int = 200,
        transform_lr=None,
        transform_hr=None,
        # --- NEW: two hover dirs ---
        hover_bnd_subdir: str = "hovernet_boundary_png",
        hover_mask_subdir: str = "hovernet_nucmask_png",
        require_hover: bool = False,
        hover_is_binary: bool = True,
        hover_threshold: int = 8,  # 0~255
    ):
        self.out_img_dir = out_img_dir
        self.hr_root = os.path.join(out_img_dir, "hr_png")
        self.lr_root = os.path.join(out_img_dir, "lr_png")
        self.clin_path = os.path.join(out_img_dir, "clinical.tsv")

        # --- NEW ---
        self.hover_bnd_root = os.path.join(out_img_dir, hover_bnd_subdir)
        self.hover_mask_root = os.path.join(out_img_dir, hover_mask_subdir)
        self.require_hover = bool(require_hover)
        self.hover_is_binary = bool(hover_is_binary)
        self.hover_threshold = int(hover_threshold)

        if not os.path.isdir(self.hr_root):
            raise FileNotFoundError(f"hr_png not found: {self.hr_root}")
        if not os.path.isdir(self.lr_root):
            raise FileNotFoundError(f"lr_png not found: {self.lr_root}")
        if not os.path.exists(self.clin_path):
            raise FileNotFoundError(f"clinical.tsv not found: {self.clin_path}")

        # hover roots can be missing if not generated yet
        if self.require_hover:
            if not os.path.isdir(self.hover_bnd_root):
                raise FileNotFoundError(f"require_hover=True but not found: {self.hover_bnd_root}")
            if not os.path.isdir(self.hover_mask_root):
                raise FileNotFoundError(f"require_hover=True but not found: {self.hover_mask_root}")
        else:
            if not os.path.isdir(self.hover_bnd_root):
                print(f"[dataset][WARN] hover boundary dir not found: {self.hover_bnd_root}")
                self.hover_bnd_root = None
            if not os.path.isdir(self.hover_mask_root):
                print(f"[dataset][WARN] hover mask dir not found: {self.hover_mask_root}")
                self.hover_mask_root = None

        self.require_done = bool(require_done)
        self.patch_num = int(patch_num)

        if transform_lr is None:
            self.transform_lr = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform_lr = transform_lr

        if transform_hr is None:
            self.transform_hr = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform_hr = transform_hr

        # --- survival table ---
        df = pd.read_csv(self.clin_path, sep="\t")
        surv_map = build_survival_map(df, id_col=id_col, death_col=death_col, follow_col=follow_col)
        df[id_col] = df[id_col].astype(str)

        if "project_id" in df.columns:
            self.project_map = dict(zip(df[id_col].tolist(), df["project_id"].astype(str).tolist()))
        elif "dataset_name" in df.columns:
            self.project_map = dict(zip(df[id_col].tolist(), df["dataset_name"].astype(str).tolist()))
        else:
            self.project_map = {}

        # placeholders
        self._time_min = 0.0
        self._time_max = 0.0
        self._time_edges = (0.0, 0.0, 0.0)

        # --- detect layout ---
        first_level = sorted([d for d in os.listdir(self.hr_root) if os.path.isdir(os.path.join(self.hr_root, d))])

        def _dir_has_pngs(d: str) -> bool:
            try:
                for f in os.listdir(d):
                    if f.lower().endswith(".png"):
                        return True
            except Exception:
                return False
            return False

        layout_is_slide_level = False
        for name in first_level[: min(20, len(first_level))]:
            if _dir_has_pngs(os.path.join(self.hr_root, name)):
                layout_is_slide_level = True
                break

        # --- helper for hover paths ---
        def _get_hover_path(root: Optional[str], slide_id: str, patch_name: str, case_id: Optional[str] = None) -> Optional[str]:
            if root is None:
                return None

            # preferred: <root>/<slide>/<patch>
            p1 = os.path.join(root, slide_id, patch_name)
            if os.path.exists(p1):
                return p1

            # fallback: <root>/<case>/<slide>/<patch>
            if case_id is not None:
                p2 = os.path.join(root, case_id, slide_id, patch_name)
                if os.path.exists(p2):
                    return p2

            return p1  # return expected location for debug

        # --- build items ---
        self.items = []
        skipped_missing_hover = 0
        skipped_missing_lr = 0
        skipped_no_surv = 0
        skipped_no_lr_slide = 0
        skipped_no_hr_slide = 0
        skipped_no_done = 0

        if layout_is_slide_level:
            hr_slides = first_level
            for slide_id in hr_slides:
                hr_slide_dir = os.path.join(self.hr_root, slide_id)
                lr_slide_dir = os.path.join(self.lr_root, slide_id)
                if not os.path.isdir(lr_slide_dir):
                    skipped_no_lr_slide += 1
                    continue

                case_id = slide_to_case_id(slide_id)
                surv = surv_map.get(case_id, None)
                if surv is None or surv.time is None:
                    skipped_no_surv += 1
                    continue

                hr_pngs = sorted([p for p in os.listdir(hr_slide_dir) if p.lower().endswith(".png")])
                if not hr_pngs:
                    skipped_no_hr_slide += 1
                    continue
                if self.patch_num > 0:
                    hr_pngs = hr_pngs[: self.patch_num]

                for patch_name in hr_pngs:
                    hr_path = os.path.join(hr_slide_dir, patch_name)
                    lr_path = os.path.join(lr_slide_dir, patch_name)
                    if not os.path.exists(lr_path):
                        skipped_missing_lr += 1
                        continue

                    bnd_path = _get_hover_path(self.hover_bnd_root, slide_id, patch_name, case_id)
                    msk_path = _get_hover_path(self.hover_mask_root, slide_id, patch_name, case_id)

                    if self.require_hover:
                        if (bnd_path is None) or (not os.path.exists(bnd_path)) or (msk_path is None) or (not os.path.exists(msk_path)):
                            skipped_missing_hover += 1
                            continue

                    self.items.append(
                        dict(
                            case_id=case_id,
                            slide_id=slide_id,
                            hr_path=hr_path,
                            lr_path=lr_path,
                            hover_bnd_path=bnd_path,
                            hover_mask_path=msk_path,
                            time=float(surv.time),
                            event=int(surv.event),
                            project=self.project_map.get(case_id, ""),
                        )
                    )
        else:
            hr_cases = first_level
            for case_id in hr_cases:
                case_dir = os.path.join(self.hr_root, case_id)
                if not os.path.isdir(case_dir):
                    continue

                surv = surv_map.get(case_id, None)
                if surv is None or surv.time is None:
                    skipped_no_surv += 1
                    continue

                slides = sorted([d for d in os.listdir(case_dir) if os.path.isdir(os.path.join(case_dir, d))])
                if not slides:
                    skipped_no_hr_slide += 1
                    continue

                for slide_id in slides:
                    hr_slide_dir = os.path.join(self.hr_root, case_id, slide_id)
                    lr_slide_dir = os.path.join(self.lr_root, case_id, slide_id)
                    if not os.path.isdir(lr_slide_dir):
                        skipped_no_lr_slide += 1
                        continue

                    if self.require_done and (not _has_done_flag(hr_slide_dir)):
                        skipped_no_done += 1
                        continue

                    hr_pngs = sorted([p for p in os.listdir(hr_slide_dir) if p.lower().endswith(".png")])
                    if self.patch_num > 0:
                        hr_pngs = hr_pngs[: self.patch_num]

                    for patch_name in hr_pngs:
                        hr_path = os.path.join(hr_slide_dir, patch_name)
                        lr_path = os.path.join(lr_slide_dir, patch_name)
                        if not os.path.exists(lr_path):
                            skipped_missing_lr += 1
                            continue

                        bnd_path = _get_hover_path(self.hover_bnd_root, slide_id, patch_name, case_id)
                        msk_path = _get_hover_path(self.hover_mask_root, slide_id, patch_name, case_id)

                        if self.require_hover:
                            if (bnd_path is None) or (not os.path.exists(bnd_path)) or (msk_path is None) or (not os.path.exists(msk_path)):
                                skipped_missing_hover += 1
                                continue

                        self.items.append(
                            dict(
                                case_id=case_id,
                                slide_id=slide_id,
                                hr_path=hr_path,
                                lr_path=lr_path,
                                hover_bnd_path=bnd_path,
                                hover_mask_path=msk_path,
                                time=float(surv.time),
                                event=int(surv.event),
                                project=self.project_map.get(case_id, ""),
                            )
                        )

        print(
            f"[dataset] layout={'slide_level' if layout_is_slide_level else 'case_level'} | "
            f"items={len(self.items)} | skipped_missing_hover={skipped_missing_hover} | "
            f"skipped_missing_lr={skipped_missing_lr} | skipped_no_surv={skipped_no_surv} | "
            f"hover_bnd_root={self.hover_bnd_root} hover_mask_root={self.hover_mask_root}"
        )

        # --- time edges ---
        case_to_time = {}
        for it in self.items:
            cid = it["case_id"]
            if cid not in case_to_time:
                case_to_time[cid] = float(it["time"])

        if len(case_to_time) == 0:
            self._time_min = 0.0
            self._time_max = 0.0
            self._time_edges = (0.0, 0.0, 0.0)
        else:
            times = np.array(list(case_to_time.values()), dtype=np.float64)
            self._time_min = float(times.min())
            self._time_max = float(times.max())
            if self._time_max <= self._time_min:
                self._time_edges = (self._time_min, self._time_min, self._time_min)
            else:
                step = (self._time_max - self._time_min) / 4.0
                self._time_edges = (float(self._time_min + step), float(self._time_min + 2 * step), float(self._time_min + 3 * step))



    # -------- robust readers --------
    @staticmethod
    def _read_rgb_pil_or_cv2(path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read RGB image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)

    @staticmethod
    def _read_gray_u8_pil_or_cv2(path: str) -> np.ndarray:
        """Read grayscale png as uint8 HxW (0~255)."""
        try:
            img = Image.open(path).convert("L")
            return np.array(img, dtype=np.uint8)
        except Exception:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read grayscale image: {path}")
            return img.astype(np.uint8, copy=False)

    def _time_to_timeY_equal_range(self, t: float) -> int:
        e1, e2, e3 = self._time_edges
        if self._time_max <= self._time_min:
            return 0
        if t < e1:
            return 0
        elif t < e2:
            return 1
        elif t < e3:
            return 2
        else:
            return 3

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]

        lr_img = self._read_rgb_pil_or_cv2(it["lr_path"])
        hr_img = self._read_rgb_pil_or_cv2(it["hr_path"])

        lr = self.transform_lr(lr_img)  # [3,128,128]
        hr = self.transform_hr(hr_img)  # [3,512,512]

        H, W = int(hr.shape[1]), int(hr.shape[2])

        def _read_hover(path: Optional[str]) -> torch.Tensor:
            if (path is None) or (not os.path.exists(path)):
                return torch.zeros((1, H, W), dtype=torch.float32)
            try:
                img = Image.open(path).convert("L")
                t = transforms.ToTensor()(img).float()  # [1,h,w] in [0,1]
            except Exception:
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    return torch.zeros((1, H, W), dtype=torch.float32)
                t = torch.from_numpy(m.astype(np.float32) / 255.0).unsqueeze(0)

            if self.hover_is_binary:
                thr = float(self.hover_threshold) / 255.0
                t = (t > thr).float()

            if (t.shape[1] != H) or (t.shape[2] != W):
                t = torch.nn.functional.interpolate(t.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)
            return t

        hover_bnd = _read_hover(it.get("hover_bnd_path", None))
        hover_mask = _read_hover(it.get("hover_mask_path", None))

        time_val = float(it["time"])
        time_y = self._time_to_timeY_equal_range(time_val)

        out = {
            "lr": lr,
            "hr": hr,
            "hover_bnd": hover_bnd,     # NEW
            "hover_mask": hover_mask,   # NEW
            "time": torch.tensor(time_val, dtype=torch.float32),
            "event": torch.tensor(int(it["event"]), dtype=torch.long),
            "time_Y": torch.tensor(time_y, dtype=torch.long),
            "project": it.get("project", ""),
            "meta": {
                "case_id": it["case_id"],
                "slide_id": it["slide_id"],
                "lr_path": it["lr_path"],
                "hr_path": it["hr_path"],
                "hover_bnd_path": it.get("hover_bnd_path", None),
                "hover_mask_path": it.get("hover_mask_path", None),
                "project": it.get("project", ""),
            },
        }
        return out

        
def _split_cases(
    cases: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("ratios must be non-negative")
    s = float(ratios.sum())
    if s <= 0:
        raise ValueError("sum of ratios must be > 0")
    ratios = ratios / s

    rng = np.random.default_rng(seed)
    cases = list(cases)
    rng.shuffle(cases)

    n = len(cases)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    # ensure total = n
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]

    return train_cases, val_cases, test_cases


def build_case_split_dataloaders(
    out_img_dir: str,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    split_seed: int = 2025,
    patch_num: int = 200,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
    require_done: bool = True,
    id_col: str = "case_submitter_id",
    death_col: str = "days_to_death",
    follow_col: str = "days_to_last_follow_up",
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns: train_loader, val_loader, test_loader
    Split is by case_id to avoid leakage.
    """
    ds = PathologySRSurvivalDataset(
        out_img_dir=out_img_dir,
        id_col=id_col,
        death_col=death_col,
        follow_col=follow_col,
        require_done=require_done,
        patch_num=patch_num,
    )

    cases = sorted({it["case_id"] for it in ds.items})
    if len(cases) == 0:
        raise RuntimeError("No valid cases found after filtering clinical + png pairs.")
    

    train_cases, val_cases, test_cases = _split_cases(
        cases=cases,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    train_set = set(train_cases)
    val_set = set(val_cases)
    test_set = set(test_cases)

    train_idx = [i for i, it in enumerate(ds.items) if it["case_id"] in train_set]
    val_idx   = [i for i, it in enumerate(ds.items) if it["case_id"] in val_set]
    test_idx  = [i for i, it in enumerate(ds.items) if it["case_id"] in test_set]

    print(f"[split] cases: total={len(cases)} | train={len(train_cases)} | val={len(val_cases)} | test={len(test_cases)}")
    print(f"[split] patches: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader




if __name__ == "__main__":
    import yaml
    from easydict import EasyDict
    
    config = EasyDict(
        yaml.load(open("/workspace/SPR_new/config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    
    train_loader, val_loader, test_loader = build_case_split_dataloaders(
        out_img_dir=config.data_loader.out_img_dir,
        batch_size=config.trainer.batch_size,
        patch_num=getattr(config.data_loader, "patch_num", 200),
        train_ratio=config.data_loader.train_ratio,
        val_ratio=config.data_loader.val_ratio,
        test_ratio=config.data_loader.test_ratio,
        split_seed=getattr(config.data_loader, "seed", 2025),
        num_workers=config.data_loader.num_workers,
        pin_memory=config.data_loader.pin_memory,
    )  
    for batch in train_loader:
        print(batch["lr"])
        print(batch["hr"])
        print(batch["hover_bnd"])
        print(batch["hover_mask"])
        print(batch["time"])
    
    for batch in val_loader:
        print(batch["lr"])
        print(batch["hr"])
        print(batch["hover_bnd"])
        print(batch["hover_mask"])
        print(batch["time"])
    
    for batch in test_loader:
        print(batch["lr"])
        print(batch["hr"])
        print(batch["hover_bnd"])
        print(batch["hover_mask"])
        print(batch["time"])
    