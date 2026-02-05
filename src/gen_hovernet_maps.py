# gen_hovernet_maps.py
# ------------------------------------------------------------
# TIAToolbox HoVer-Net predict -> tmp/*.dat -> decode -> (boundary png + filled mask png)
#
# Outputs:
#   out_img_dir/hovernet_boundary_png/<slide>/<patch>.png     (outline)
#   out_img_dir/hovernet_nucmask_png/<slide>/<patch>.png      (filled nuclei mask)
#
# Designed for your environment:
# - Many TIAToolbox versions store contours in .dat (no inst_map)
# - This script:
#     1) tries inst_map (if exists) -> boundary+mask
#     2) else uses contours -> draw polylines + fillPoly -> boundary+mask
#
# Robustness:
# - Unique tmp save_dir (must NOT exist)
# - Skips non-index artifacts (file_map.dat etc.)
# - Chunk report: __debug_artifact_report_<tmp>.txt
# - Optional debug previews: __debug_boundary_preview.png + __debug_mask_preview.png
# - Resume-friendly: skip existing outputs; slide DONE marker
# ------------------------------------------------------------

import os
import sys
import time
import uuid
import glob
import shutil
import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import cv2
import joblib
from tqdm import tqdm

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.setrecursionlimit(20000)

from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.architecture.utils import compile_model


# -------------------------
# FS helpers
# -------------------------
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_touch(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def list_slide_dirs(hr_root: Path) -> List[Path]:
    slide_dirs = sorted([p for p in hr_root.iterdir() if p.is_dir()])
    return slide_dirs if slide_dirs else [hr_root]


def make_unique_tmp_dir(parent: Path, slide_id: str) -> Path:
    """Return a path that DOES NOT exist (do not mkdir)."""
    pid = os.getpid()
    ts = int(time.time())
    for _ in range(500):
        u = uuid.uuid4().hex[:10]
        p = parent / f"__tmp_{slide_id}_{pid}_{ts}_{u}"
        if not p.exists():
            return p
    raise RuntimeError("Failed to allocate a non-existing tmp dir name.")


def find_prediction_artifacts(tmp_save: Path) -> List[Path]:
    files = sorted(list(tmp_save.rglob("*.dat")))
    files += sorted(list(tmp_save.rglob("*.joblib")))
    # de-dup
    seen = set()
    uniq = []
    for f in files:
        s = f.as_posix()
        if s not in seen:
            uniq.append(f)
            seen.add(s)
    return uniq


# -------------------------
# Morphology helpers
# -------------------------
def dilate_binary(img: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return img
    kk = int(k)
    if kk % 2 == 0:
        kk += 1
    kernel = np.ones((kk, kk), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


# -------------------------
# Artifact mapping
# -------------------------
def artifact_index(artifact_path: Path) -> Optional[int]:
    stem = artifact_path.stem
    return int(stem) if stem.isdigit() else None


def artifact_to_patch_stem_and_src(artifact_path: Path, sub_pngs: List[str]) -> Tuple[Optional[str], Optional[str]]:
    idx = artifact_index(artifact_path)
    if idx is not None and 0 <= idx < len(sub_pngs):
        src = sub_pngs[idx]
        return Path(src).stem, src
    # fallback: try exact stem match
    a = artifact_path.stem
    for p in sub_pngs:
        if Path(p).stem == a:
            return a, p
    return None, None


# -------------------------
# Recursive extractors
# -------------------------
def _iter_items(obj: Any, prefix: str = "", max_depth: int = 8, depth: int = 0):
    """Yield (path, value) recursively."""
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            yield kp, v
            yield from _iter_items(v, kp, max_depth=max_depth, depth=depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            kp = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield kp, v
            yield from _iter_items(v, kp, max_depth=max_depth, depth=depth + 1)
    else:
        return


def try_extract_inst_map(obj: Any) -> Optional[np.ndarray]:
    """
    Conservative: only accept a real 2D integer map with background zeros.
    """
    if isinstance(obj, dict):
        for k in ("inst_map", "instance_map"):
            if k in obj and isinstance(obj[k], np.ndarray) and obj[k].ndim == 2:
                a = obj[k]
                if np.issubdtype(a.dtype, np.integer):
                    if (a == 0).mean() > 0.01 and int(a.max()) > 0:
                        return a.astype(np.int32, copy=False)

        for k in ("inst_dict", "pred", "result", "output", "pred_inst"):
            if k in obj and isinstance(obj[k], dict):
                for kk in ("inst_map", "instance_map"):
                    if kk in obj[k] and isinstance(obj[k][kk], np.ndarray) and obj[k][kk].ndim == 2:
                        a = obj[k][kk]
                        if np.issubdtype(a.dtype, np.integer):
                            if (a == 0).mean() > 0.01 and int(a.max()) > 0:
                                return a.astype(np.int32, copy=False)

    for p, v in _iter_items(obj):
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.integer):
            if v.size >= 256 and (v == 0).mean() > 0.01 and int(v.max()) > 0 and int(v.max()) < 50000:
                return v.astype(np.int32, copy=False)

    return None


def extract_contours(obj: Any) -> List[np.ndarray]:
    """
    Extract contour coordinate arrays.
    Accept arrays that look like Nx2 coordinates and whose path contains 'contour'.
    """
    contours: List[np.ndarray] = []
    for p, v in _iter_items(obj):
        if "contour" not in p.lower():
            continue

        if isinstance(v, np.ndarray):
            a = v
            if a.ndim == 2 and (a.shape[1] == 2 or a.shape[0] == 2) and a.size >= 6:
                contours.append(a)
            elif a.ndim == 3 and a.shape[-1] == 2 and a.size >= 6:
                contours.append(a)

        if isinstance(v, (list, tuple)):
            for vv in v:
                if isinstance(vv, np.ndarray):
                    a = vv
                    if a.ndim == 2 and (a.shape[1] == 2 or a.shape[0] == 2) and a.size >= 6:
                        contours.append(a)
                    elif a.ndim == 3 and a.shape[-1] == 2 and a.size >= 6:
                        contours.append(a)

    return contours


def infer_canvas_hw_from_contours(contours: List[np.ndarray], fallback_hw: Tuple[int, int]) -> Tuple[int, int]:
    """
    Infer contour canvas size from max coords; fallback to source patch size.
    """
    if not contours:
        return fallback_hw
    max_x = 0
    max_y = 0
    for c in contours[: min(len(contours), 2000)]:
        a = np.asarray(c).reshape(-1, 2)
        if a.size == 0:
            continue
        max_x = max(max_x, int(np.nanmax(a[:, 0])))
        max_y = max(max_y, int(np.nanmax(a[:, 1])))
    H = max_y + 1
    W = max_x + 1
    if H >= 64 and W >= 64 and H <= 6000 and W <= 6000:
        return (H, W)
    return fallback_hw


def _normalize_contour_pts(c: np.ndarray, H: int, W: int) -> Optional[np.ndarray]:
    """
    Convert various contour shapes to Nx2 int32 (x,y) and clip.
    """
    if c is None:
        return None
    a = np.asarray(c)
    if a.size < 6:
        return None

    # normalize to Nx2
    if a.ndim == 3 and a.shape[-1] == 2:
        pts = a.reshape(-1, 2)
    elif a.ndim == 2 and a.shape[1] == 2:
        pts = a
    elif a.ndim == 2 and a.shape[0] == 2:
        pts = a.T
    else:
        return None

    pts = np.round(pts).astype(np.int32)

    # clip
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    return pts


# -------------------------
# Draw from inst_map / contours
# -------------------------
def inst_map_to_boundary_and_mask(inst_map: np.ndarray, dilate: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    inst_map: [H,W] int
    returns:
      boundary_u8 [H,W] 0/255
      mask_u8     [H,W] 0/255  (filled nuclei regions)
    """
    m = inst_map.astype(np.int32, copy=False)
    H, W = m.shape
    boundary = np.zeros((H, W), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    ids = np.unique(m)
    ids = ids[ids != 0]

    for _id in ids:
        binm = (m == _id).astype(np.uint8)
        if binm.max() == 0:
            continue
        mask[binm > 0] = 255
        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cv2.drawContours(boundary, cnts, -1, 255, thickness=1)

    boundary = dilate_binary(boundary, dilate)
    return boundary, mask


def contours_to_boundary_and_mask(
    contours: List[np.ndarray],
    canvas_hw: Tuple[int, int],
    dilate: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    contours: list of coordinate arrays
    canvas_hw: (H,W) for drawing in contour coordinate space
    returns boundary+mask in that canvas space
    """
    H, W = int(canvas_hw[0]), int(canvas_hw[1])
    boundary = np.zeros((H, W), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)

    for c in contours:
        pts = _normalize_contour_pts(c, H, W)
        if pts is None or pts.shape[0] < 3:
            continue

        pts_cv = pts.reshape(-1, 1, 2)
        # outline
        cv2.polylines(boundary, [pts_cv], isClosed=True, color=255, thickness=1)
        # filled region
        cv2.fillPoly(mask, [pts_cv], color=255)

    boundary = dilate_binary(boundary, dilate)
    return boundary, mask


# -------------------------
# Decode artifact -> boundary+mask
# -------------------------
def decode_artifact_to_maps(
    artifact_path: Path,
    src_patch_path: Optional[str],
    dilate: int,
    report_lines: List[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Returns (boundary_u8, mask_u8, mode_used)
      - boundary_u8/mask_u8 are in FINAL src patch size if src_patch_path exists, else in canvas size.
      - mode_used: 'inst_map' | 'contour' | 'none'
    """
    obj = joblib.load(artifact_path)

    # final target size from src patch
    src_hw = None
    if src_patch_path is not None:
        img = cv2.imread(src_patch_path, cv2.IMREAD_COLOR)
        if img is not None:
            src_hw = (img.shape[0], img.shape[1])
    if src_hw is None:
        src_hw = (512, 512)

    # 1) inst_map path
    inst_map = try_extract_inst_map(obj)
    if inst_map is not None:
        bnd, msk = inst_map_to_boundary_and_mask(inst_map, dilate=dilate)

        # resize to src
        if bnd.shape[:2] != src_hw:
            bnd = cv2.resize(bnd, (src_hw[1], src_hw[0]), interpolation=cv2.INTER_NEAREST)
        if msk.shape[:2] != src_hw:
            msk = cv2.resize(msk, (src_hw[1], src_hw[0]), interpolation=cv2.INTER_NEAREST)

        uniq = np.unique(inst_map)
        report_lines.append(
            f"  [USE inst_map] inst_shape={inst_map.shape} min={int(inst_map.min())} max={int(inst_map.max())} "
            f"uniq={len(uniq)} bg_ratio={(inst_map==0).mean():.3f} final_hw={src_hw}"
        )
        return bnd, msk, "inst_map"

    # 2) contour path
    contours = extract_contours(obj)
    report_lines.append(f"  contour_count={len(contours)}")
    if contours:
        canvas_hw = infer_canvas_hw_from_contours(contours, fallback_hw=src_hw)
        bnd, msk = contours_to_boundary_and_mask(contours, canvas_hw=canvas_hw, dilate=dilate)

        # resize to src patch size
        if bnd.shape[:2] != src_hw:
            bnd = cv2.resize(bnd, (src_hw[1], src_hw[0]), interpolation=cv2.INTER_NEAREST)
        if msk.shape[:2] != src_hw:
            msk = cv2.resize(msk, (src_hw[1], src_hw[0]), interpolation=cv2.INTER_NEAREST)

        bden = float((bnd > 0).mean())
        mden = float((msk > 0).mean())
        report_lines.append(
            f"  [USE contour] canvas_hw={canvas_hw} final_hw={src_hw} boundary_density={bden:.4f} mask_density={mden:.4f}"
        )
        return bnd, msk, "contour"

    report_lines.append("  [USE none] no inst_map and no contour found")
    return None, None, "none"


# -------------------------
# Build segmentor
# -------------------------
def build_segmentor(pretrained_model: str, batch_size: int, num_loader_workers: int, num_postproc_workers: int):
    model, ioconfig = get_pretrained_model(pretrained_model=pretrained_model)
    model = compile_model(model, mode="disable")
    seg = NucleusInstanceSegmentor(
        model=model,
        batch_size=batch_size,
        num_loader_workers=num_loader_workers,
        num_postproc_workers=num_postproc_workers,
    )
    return seg, ioconfig


# -------------------------
# Process one chunk
# -------------------------
def process_one_chunk(
    inst_segmentor: NucleusInstanceSegmentor,
    ioconfig,
    sub_pngs: List[str],
    tmp_save: Path,
    out_bnd_dir: Path,
    out_msk_dir: Path,
    device: str,
    dilate: int,
    overwrite: bool,
    patch_level_skip: bool,
    debug_preview: bool,
) -> Tuple[int, int, int, int, int, int, Path]:
    """
    Returns:
      artifacts_count, mapped_count, instmap_used, contour_used, written, failed, report_path
    """
    _ = inst_segmentor.predict(
        sub_pngs,
        save_dir=str(tmp_save),
        mode="tile",
        device=device,
        ioconfig=ioconfig,
        crash_on_exception=True,
    )

    artifacts = find_prediction_artifacts(tmp_save)
    artifacts_count = len(artifacts)

    mapped_count = 0
    instmap_used = 0
    contour_used = 0
    written = 0
    failed = 0

    report_path = tmp_save.parent / f"__debug_artifact_report_{tmp_save.name}.txt"
    report_lines: List[str] = []
    report_lines.append(f"[TMP] {tmp_save}")
    report_lines.append(f"artifacts_count={artifacts_count}")

    did_preview = False

    for art in artifacts:
        patch_stem, src_path = artifact_to_patch_stem_and_src(art, sub_pngs)
        if patch_stem is None:
            report_lines.append(f"[SKIP] {art.name} -> cannot map to patch stem")
            continue
        mapped_count += 1

        out_bnd = out_bnd_dir / f"{patch_stem}.png"
        out_msk = out_msk_dir / f"{patch_stem}.png"

        if patch_level_skip and (out_bnd.exists() and out_msk.exists()) and (not overwrite):
            continue

        report_lines.append(f"\n[ARTIFACT] {art}")

        bnd, msk, mode_used = decode_artifact_to_maps(
            artifact_path=art,
            src_patch_path=src_path,
            dilate=dilate,
            report_lines=report_lines,
        )

        if bnd is None or msk is None:
            failed += 1
            continue

        if mode_used == "inst_map":
            instmap_used += 1
        elif mode_used == "contour":
            contour_used += 1

        ok1 = cv2.imwrite(str(out_bnd), bnd)
        ok2 = cv2.imwrite(str(out_msk), msk)
        if ok1 and ok2:
            written += 1
        else:
            failed += 1

        if debug_preview and (not did_preview):
            try:
                cv2.imwrite(str(out_bnd_dir / "__debug_boundary_preview.png"), bnd)
                cv2.imwrite(str(out_msk_dir / "__debug_mask_preview.png"), msk)
            except Exception:
                pass
            did_preview = True

    report_path.write_text("\n".join(report_lines))
    return artifacts_count, mapped_count, instmap_used, contour_used, written, failed, report_path


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_img_dir", type=str, default="/mnt/liangjm/SpRR_data/")
    ap.add_argument("--pretrained_model", type=str, default="hovernet_fast-pannuke")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--predict_chunk", type=int, default=64)
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--num_loader_workers", type=int, default=0)
    ap.add_argument("--num_postproc_workers", type=int, default=0)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no_skip_done", action="store_true")
    ap.add_argument("--no_patch_skip", action="store_true")

    ap.add_argument("--keep_tmp", action="store_true")
    ap.add_argument("--keep_tmp_on_failure", action="store_true")
    ap.add_argument("--debug_preview", action="store_true")

    args = ap.parse_args()

    out_img_dir = Path(args.out_img_dir)
    hr_root = out_img_dir / "hr_png"

    out_bnd_root = out_img_dir / "hovernet_boundary_png"
    out_msk_root = out_img_dir / "hovernet_nucmask_png"

    safe_mkdir(out_bnd_root)
    safe_mkdir(out_msk_root)

    if not hr_root.exists():
        raise FileNotFoundError(f"hr_png not found: {hr_root}")

    slide_dirs = list_slide_dirs(hr_root)

    seg, ioconfig = build_segmentor(
        pretrained_model=args.pretrained_model,
        batch_size=int(args.batch_size),
        num_loader_workers=int(args.num_loader_workers),
        num_postproc_workers=int(args.num_postproc_workers),
    )

    skip_if_done = not bool(args.no_skip_done)
    patch_level_skip = not bool(args.no_patch_skip)

    print(f"[INFO] slides={len(slide_dirs)} model={args.pretrained_model} device={args.device} bs={args.batch_size}")
    print(f"[INFO] out_boundary_root={out_bnd_root}")
    print(f"[INFO] out_nucmask_root={out_msk_root}")
    print(f"[INFO] predict_chunk={args.predict_chunk} dilate={args.dilate}")
    print(f"[INFO] keep_tmp={args.keep_tmp} keep_tmp_on_failure={args.keep_tmp_on_failure} overwrite={args.overwrite}")

    for sdir in slide_dirs:
        slide_id = sdir.name if sdir != hr_root else "hr_png_flat"
        pngs = sorted(glob.glob(str(sdir / "*.png")))
        if not pngs:
            continue

        out_bnd_dir = out_bnd_root / slide_id
        out_msk_dir = out_msk_root / slide_id
        safe_mkdir(out_bnd_dir)
        safe_mkdir(out_msk_dir)

        done_flag = out_bnd_dir / ".HOVERNET_DONE"  # use boundary dir for DONE

        if skip_if_done and done_flag.exists() and (not args.overwrite):
            print(f"[SKIP] {slide_id} (DONE exists)")
            continue

        if args.overwrite:
            for f in out_bnd_dir.glob("*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass
            for f in out_msk_dir.glob("*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass
            if done_flag.exists():
                try:
                    done_flag.unlink()
                except Exception:
                    pass

        print(f"[RUN] {slide_id} patches={len(pngs)}")

        totals = {
            "artifacts": 0,
            "mapped": 0,
            "instmap_used": 0,
            "contour_used": 0,
            "written": 0,
            "failed": 0,
        }
        kept_tmps: List[Path] = []

        for start in tqdm(range(0, len(pngs), max(1, int(args.predict_chunk))), desc=slide_id):
            sub_pngs = pngs[start:start + max(1, int(args.predict_chunk))]

            # fast skip if all outputs exist
            if patch_level_skip and (not args.overwrite):
                all_exist = True
                for p in sub_pngs:
                    stem = Path(p).stem
                    if not (out_bnd_dir / f"{stem}.png").exists():
                        all_exist = False
                        break
                    if not (out_msk_dir / f"{stem}.png").exists():
                        all_exist = False
                        break
                if all_exist:
                    continue

            tmp_save = make_unique_tmp_dir(out_bnd_root, slide_id)  # tmp in boundary root is fine

            a = m = im = co = w = f = 0
            report = None

            try:
                a, m, im, co, w, f, report = process_one_chunk(
                    inst_segmentor=seg,
                    ioconfig=ioconfig,
                    sub_pngs=sub_pngs,
                    tmp_save=tmp_save,
                    out_bnd_dir=out_bnd_dir,
                    out_msk_dir=out_msk_dir,
                    device=args.device,
                    dilate=int(args.dilate),
                    overwrite=bool(args.overwrite),
                    patch_level_skip=patch_level_skip,
                    debug_preview=bool(args.debug_preview),
                )
                totals["artifacts"] += a
                totals["mapped"] += m
                totals["instmap_used"] += im
                totals["contour_used"] += co
                totals["written"] += w
                totals["failed"] += f

                print(
                    f"[CHUNK] {slide_id} start={start} artifacts={a} mapped={m} "
                    f"instmap_used={im} contour_used={co} written={w} failed={f} report={report}"
                )
            except Exception as e:
                totals["failed"] += 1
                print(f"[ERROR] {slide_id} start={start}: {repr(e)}")

            # keep tmp?
            keep_this = False
            if args.keep_tmp:
                keep_this = True
            if args.keep_tmp_on_failure and (w == 0):
                keep_this = True

            if keep_this:
                kept_tmps.append(tmp_save)
                print(f"[KEEP] tmp kept: {tmp_save}")
            else:
                if tmp_save.exists():
                    shutil.rmtree(tmp_save, ignore_errors=True)

        existing_bnd = len(list(out_bnd_dir.glob("*.png")))
        existing_msk = len(list(out_msk_dir.glob("*.png")))

        atomic_touch(
            done_flag,
            text=(
                f"done\n"
                f"existing_boundary={existing_bnd}\n"
                f"existing_nucmask={existing_msk}\n"
                f"artifacts_total={totals['artifacts']}\n"
                f"mapped_total={totals['mapped']}\n"
                f"instmap_used_total={totals['instmap_used']}\n"
                f"contour_used_total={totals['contour_used']}\n"
                f"written_total={totals['written']}\n"
                f"failed_total={totals['failed']}\n"
                f"kept_tmps={len(kept_tmps)}\n"
            ),
        )

        print(
            f"[DONE] {slide_id} boundary={existing_bnd} nucmask={existing_msk} "
            f"instmap_used={totals['instmap_used']} contour_used={totals['contour_used']} "
            f"written={totals['written']} failed={totals['failed']}"
        )

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
