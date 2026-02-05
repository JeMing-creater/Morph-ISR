import os
import re
import sys
import h5py
import tqdm
from PIL import Image
import numpy as np
import openslide
import random
import pandas as pd
import json
import glob
from typing import Optional, Dict, Any

from create_patches_fp import seg_and_patch


def slide_to_case_id(slide_name: str) -> str:
    """
    TCGA slide -> case id
    default: 前三段（12 chars with hyphen） e.g. TCGA-S2-AA1A
    """
    base = os.path.basename(slide_name)
    stem = base.replace(".svs", "")
    parts = stem.split("-")
    case_id = "-".join(parts[:3])
    return case_id


def slide_id_from_path(svs_path: str) -> str:
    return os.path.basename(svs_path).replace(".svs", "")


def compute_tissue_ratio(
    patch: Image.Image,
    white_thr: float = 0.9,
) -> float:
    """
    粗略估计组织占比：认为 “灰度 < white_thr” 的像素是组织（或非纯白背景）。
    返回 [0,1] 之间的比例。
    """
    arr = np.asarray(patch).astype(np.float32) / 255.0  # [H,W,3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0
    gray = arr.mean(axis=2)  # [H,W]
    tissue_mask = gray < white_thr  # 非白背景
    ratio = tissue_mask.mean().item()
    return float(ratio)


def export_h5_to_pngs(
    h5_path: str,
    svs_path: str,
    hr_dir: str,
    lr_dir: str,
    scale: int = 4,
    min_tissue_ratio: float = 0.7,  # 至少 70% 组织
    white_thr: float = 0.9,         # 灰度 > 0.9 视作近白背景
    limit_samples: Optional[int] = None,  # ✅ NEW: 每个 slide 最多保存多少张（达到后中止并视为 DONE）
):
    """
    用 h5 里的 coords + 对应 svs，从原始 WSI 裁 patch，并保存 HR/LR png。
    特性：
      - 只保存组织占比 >= min_tissue_ratio 的 patch；
      - 使用 .PROGRESS 记录当前处理到的 coord 索引，实现断点续切；
      - 处理完所有 coords 后写 .DONE，自检时可直接跳过该样本；
      - ✅ 若 limit_samples 设定：保存到上限即中止，并写 .DONE（下次直接跳过）
    """
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    done_flag = os.path.join(hr_dir, ".DONE")
    progress_file = os.path.join(hr_dir, ".PROGRESS")

    # 0) 如果已经有 .DONE，说明整个 sample 已经处理完（或已达到 limit），直接跳过
    if os.path.exists(done_flag):
        print(f"[SKIP] {os.path.basename(svs_path)} already DONE.")
        return

    # 0.5) 如果已有 PNG 达到 limit_samples：直接标记 DONE 并跳过
    existing_hr = [
        fn for fn in os.listdir(hr_dir)
        if fn.startswith("patch_") and fn.endswith(".png")
    ]
    kept_count = len(existing_hr)

    if (limit_samples is not None) and (int(limit_samples) > 0) and (kept_count >= int(limit_samples)):
        print(
            f"[SKIP] {os.path.basename(svs_path)} reached limit_samples={int(limit_samples)} "
            f"(existing_png={kept_count}). Mark DONE."
        )
        with open(done_flag, "w") as f:
            f.write("done")
        if os.path.exists(progress_file):
            os.remove(progress_file)
        return

    # 1) 读取 coords & patch 属性
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(
                f"{h5_path} 中不存在 'coords' dataset，"
                "请确认 create_patches_fp.py 是否成功运行。"
            )
        dset = f["coords"]
        coords = dset[:]  # [N, 2] (x, y)
        patch_level = int(dset.attrs["patch_level"])
        patch_size = int(dset.attrs["patch_size"])

    N = coords.shape[0]

    # 2) 读取上一次处理进度（最后处理到的 coord 索引）
    last_idx = -1
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                line = f.read().strip()
                if line:
                    last_idx = int(line)
        except Exception:
            last_idx = -1  # 损坏就从头来

    # 如果已经处理到最后一个索引，直接认定 DONE
    if last_idx >= N - 1:
        print(f"[INFO] {os.path.basename(svs_path)} already fully processed, marking DONE.")
        with open(done_flag, "w") as f:
            f.write("done")
        if os.path.exists(progress_file):
            os.remove(progress_file)
        return

    start_idx = last_idx + 1

    # 3) 打开对应的 WSI
    if not os.path.exists(svs_path):
        raise FileNotFoundError(f"找不到对应的 svs 文件：{svs_path}")

    wsi = openslide.OpenSlide(svs_path)
    slide_name = os.path.basename(svs_path)

    print(
        f"[RESUME] slide={slide_name}, total_coords={N}, "
        f"last_idx={last_idx}, start_idx={start_idx}, "
        f"existing_png={kept_count}, limit_samples={limit_samples}"
    )

    # 4) 从 start_idx 开始处理，tqdm 显示进度
    stopped_by_limit = False
    for idx in tqdm.tqdm(
        range(start_idx, N),
        desc=f"{slide_name} patches",
        unit="patch",
    ):
        # ✅ 达到上限：直接停止该 slide 的导出
        if (limit_samples is not None) and (int(limit_samples) > 0) and (kept_count >= int(limit_samples)):
            stopped_by_limit = True
            print(f"[STOP] {slide_name}: reached limit_samples={int(limit_samples)}, stop exporting.")
            break

        xy = coords[idx]
        x, y = int(xy[0]), int(xy[1])

        patch = wsi.read_region(
            (x, y),
            patch_level,
            (patch_size, patch_size),
        ).convert("RGB")

        # 计算组织占比
        ratio = compute_tissue_ratio(patch, white_thr=white_thr)

        # 不满足阈值：认为该 coord 已处理，但不保存 png
        if ratio < min_tissue_ratio:
            # 仍然更新 progress，表示这个 coord 已经访问过
            with open(progress_file, "w") as f:
                f.write(str(idx))
            continue

        # 命名：对“真正保存的 patch”单独编号，避免因过滤造成大量空号
        hr_name = f"patch_{kept_count:06d}.png"
        hr_path = os.path.join(hr_dir, hr_name)
        patch.save(hr_path)

        w, h = patch.size
        lr_img = patch.resize(
            (w // scale, h // scale),
            resample=Image.BICUBIC,
        )
        lr_path = os.path.join(lr_dir, hr_name)
        lr_img.save(lr_path)

        kept_count += 1

        # 每处理一个 coord 都更新 progress（你也可以改成每隔 K 次更新一次）
        with open(progress_file, "w") as f:
            f.write(str(idx))

        # ✅ 刚好达到上限：立刻结束
        if (limit_samples is not None) and (int(limit_samples) > 0) and (kept_count >= int(limit_samples)):
            stopped_by_limit = True
            print(f"[STOP] {slide_name}: reached limit_samples={int(limit_samples)} after save, stop exporting.")
            break

    # 5) 无论是否提前 stop，都标记 DONE（达到上限也视作完成，便于下次直接跳过）
    wsi.close()
    with open(done_flag, "w") as f:
        f.write("done")
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(
        f"[DONE] {slide_name}: "
        f"{kept_count} patches saved "
        f"(coords_total={N}, min_tissue_ratio={min_tissue_ratio}, limit_samples={limit_samples}, stopped_by_limit={stopped_by_limit})"
    )


def _load_json(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _case_has_done_slide(out_hr_root: str, case_id: str) -> bool:
    # slide_id 目录名一般形如：TCGA-S2-AA1A-01Z-00-DX1
    pattern = os.path.join(out_hr_root, f"{case_id}-*")
    for slide_dir in glob.glob(pattern):
        if os.path.isdir(slide_dir) and os.path.exists(os.path.join(slide_dir, ".DONE")):
            return True
    return False


def _get_cases_in_out_clin(out_clin_path: str, data_name: str, id_col: str) -> set:
    if not os.path.exists(out_clin_path):
        return set()
    df = pd.read_csv(out_clin_path, sep="\t")
    if id_col not in df.columns:
        return set()
    if "project_id" in df.columns:
        sub = df[df["project_id"].astype(str) == str(data_name)]
    elif "dataset_name" in df.columns:
        sub = df[df["dataset_name"].astype(str) == str(data_name)]
    else:
        return set()
    return set(sub[id_col].astype(str).tolist())


def _append_to_out_clin(out_clin_path: str, rows, id_col: str):
    if not rows:
        return
    add_df = pd.DataFrame(rows)

    if os.path.exists(out_clin_path):
        base = pd.read_csv(out_clin_path, sep="\t")
        merged = pd.concat([base, add_df], axis=0, ignore_index=True)
    else:
        merged = add_df

    if "project_id" in merged.columns:
        merged = merged.drop_duplicates(subset=[id_col, "project_id"], keep="first")
    else:
        merged = merged.drop_duplicates(subset=[id_col], keep="first")

    merged.to_csv(out_clin_path, sep="\t", index=False)


def run_clam_and_export(
    data_cfg: Dict[str, list],
    out_img_dir: str,
    patch_size: int = 512,
    step_size: int = 512,
    patch_level: int = 0,
    down_scale: int = 4,
    min_tissue_ratio: float = 0.7,
    id_col: str = "case_submitter_id",
    seed: int = 0,
    limit_samples: Optional[int] = None,  # ✅ NEW: 每个 slide 的最多导出 PNG 数
):
    """
    支持 choose_WSI 递增：
      - 每次以 data_cfg 当前 choose_WSI 为目标；
      - 若 out_img_dir 已有部分结果，则在原基础上补齐；
      - 优先补完 manifest 中“已选未完成”的病例，再新增未选病例。
    """
    random.seed(seed)

    os.makedirs(out_img_dir, exist_ok=True)
    out_hr_root = os.path.join(out_img_dir, "hr_png")
    out_lr_root = os.path.join(out_img_dir, "lr_png")
    os.makedirs(out_hr_root, exist_ok=True)
    os.makedirs(out_lr_root, exist_ok=True)

    out_clin_path = os.path.join(out_img_dir, "clinical.tsv")
    manifest_path = os.path.join(out_img_dir, "selection_manifest.json")
    manifest = _load_json(manifest_path)
    # manifest 结构：{ data_name: {"cases": [...]} }

    for data_name, cfg in data_cfg.items():
        if not isinstance(cfg, (list, tuple)) or len(cfg) != 2:
            print(f"[WARN] data_cfg[{data_name}] 不是形如 [image_root, choose_WSI]，跳过。")
            continue

        image_root, choose_WSI = cfg
        choose_WSI = int(choose_WSI)

        print(f"\n========== DATASET: {data_name} ==========")
        print(f"image_root={image_root}, target choose_WSI={choose_WSI}")
        print(f"limit_samples(per-slide)={limit_samples}")

        svs_dir = os.path.join(image_root, "img")
        clam_dir = os.path.join(image_root, "clam_results")
        patch_save_dir = os.path.join(clam_dir, "patches")
        mask_save_dir = os.path.join(clam_dir, "masks")
        stitch_save_dir = os.path.join(clam_dir, "stitches")
        os.makedirs(patch_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)
        os.makedirs(stitch_save_dir, exist_ok=True)

        clin_path = os.path.join(image_root, "clinical.tsv")
        if not os.path.exists(clin_path):
            print(f"[WARN] {data_name}: clinical.tsv 不存在，跳过。")
            continue
        clin_df = pd.read_csv(clin_path, sep="\t")
        if id_col not in clin_df.columns:
            print(f"[WARN] {data_name}: clinical.tsv 缺少列 {id_col}，跳过。")
            continue
        clin_df[id_col] = clin_df[id_col].astype(str)
        id_set = set(clin_df[id_col].tolist())

        if not os.path.exists(svs_dir):
            print(f"[WARN] {data_name}: svs_dir 不存在，跳过。")
            continue

        # case -> slides
        all_svs_files = sorted([f for f in os.listdir(svs_dir) if f.lower().endswith(".svs")])
        case_to_slides = {}
        for fname in all_svs_files:
            case_id = slide_to_case_id(fname)
            if case_id in id_set:
                case_to_slides.setdefault(case_id, []).append(fname)
        all_case_ids = sorted(case_to_slides.keys())
        if not all_case_ids:
            print(f"[WARN] {data_name}: 找不到匹配病例，跳过。")
            continue

        # --- 关键：自检“已完成病例数” ---
        out_cases_in_clin = _get_cases_in_out_clin(out_clin_path, data_name, id_col)
        done_cases = sorted([cid for cid in out_cases_in_clin if _case_has_done_slide(out_hr_root, cid)])
        done_set = set(done_cases)

        # manifest 初始化
        if data_name not in manifest:
            manifest[data_name] = {"cases": []}
        chosen_set = set(map(str, manifest[data_name].get("cases", [])))

        # 把 out_clin 里已有的病例也并入 chosen_set（兼容旧结果无 manifest）
        chosen_set |= set(map(str, out_cases_in_clin))
        manifest[data_name]["cases"] = sorted(chosen_set)

        # 目标缺口
        need = max(0, choose_WSI - len(done_set))
        print(f"[CHECK] {data_name}: done={len(done_set)}, target={choose_WSI}, need_more={need}")

        if need == 0:
            _save_json(manifest_path, manifest)
            print(f"[OK] {data_name}: 已满足目标，不需要补齐。")
            continue

        # --- 新增：优先补完“已选但未完成”的病例 ---
        selected_not_done = sorted([cid for cid in chosen_set if cid in all_case_ids and cid not in done_set])
        fix_cases = selected_not_done[:need]

        remain = need - len(fix_cases)
        add_cases = []
        if remain > 0:
            candidates = [cid for cid in all_case_ids if cid not in chosen_set]
            if candidates:
                add_k = min(remain, len(candidates))
                add_cases = sorted(random.sample(candidates, add_k))
                chosen_set |= set(add_cases)
                manifest[data_name]["cases"] = sorted(chosen_set)

        target_cases = sorted(set(fix_cases + add_cases))
        if not target_cases:
            print(f"[WARN] {data_name}: 无法补齐（没有可用病例）。")
            _save_json(manifest_path, manifest)
            continue

        print(f"[PLAN] {data_name}: fix_cases={fix_cases}, add_cases={add_cases}")

        # 这些病例对应的 slides
        target_slides = []
        for cid in target_cases:
            target_slides.extend(case_to_slides.get(cid, []))
        target_slides = sorted(set(target_slides))
        if not target_slides:
            print(f"[WARN] {data_name}: 目标病例无 slide，跳过。")
            _save_json(manifest_path, manifest)
            continue

        # 生成 process_list（增量）
        process_list_path = os.path.join(clam_dir, f"process_list_{data_name}_increment.csv")
        pd.DataFrame({"slide_id": target_slides, "process": [1] * len(target_slides)}).to_csv(process_list_path, index=False)

        # seg_and_patch 参数（保持你原样）
        seg_params = {
            "seg_level": -1,
            "sthresh": 8,
            "mthresh": 7,
            "close": 4,
            "use_otsu": False,
            "keep_ids": "none",
            "exclude_ids": "none",
        }
        filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
        vis_params = {"vis_level": -1, "line_thickness": 250}
        patch_params = {"use_padding": True, "contour_fn": "four_pt"}

        print(f">>> [CLAM] {data_name}: seg_and_patch increment slides={len(target_slides)}")
        seg_and_patch(
            source=svs_dir,
            save_dir=clam_dir,
            patch_save_dir=patch_save_dir,
            mask_save_dir=mask_save_dir,
            stitch_save_dir=stitch_save_dir,
            patch_size=patch_size,
            step_size=step_size,
            seg_params=seg_params,
            filter_params=filter_params,
            vis_params=vis_params,
            patch_params=patch_params,
            patch_level=patch_level,
            use_default_params=False,
            seg=True,
            save_mask=True,
            stitch=False,
            patch=True,
            auto_skip=True,
            process_list=process_list_path,
        )

        # 导出到统一 out_img_dir
        for slide_file in target_slides:
            slide_id = slide_id_from_path(slide_file)
            h5_path = os.path.join(patch_save_dir, slide_id + ".h5")
            if not os.path.exists(h5_path):
                print(f"[WARN] {data_name}: h5 缺失 {h5_path}，跳过。")
                continue

            hr_dir = os.path.join(out_hr_root, slide_id)
            lr_dir = os.path.join(out_lr_root, slide_id)

            export_h5_to_pngs(
                h5_path=h5_path,
                svs_path=os.path.join(svs_dir, slide_file),
                hr_dir=hr_dir,
                lr_dir=lr_dir,
                scale=down_scale,
                min_tissue_ratio=min_tissue_ratio,
                limit_samples=limit_samples,  # ✅ NEW
            )

        # 补 clinical：对 fix_cases/add_cases 都确保写入
        new_rows = []
        for cid in target_cases:
            sub = clin_df[clin_df[id_col] == cid]
            if sub.empty:
                continue
            row = sub.iloc[0].copy()
            row["project_id"] = data_name
            new_rows.append(row)
        _append_to_out_clin(out_clin_path, new_rows, id_col)

        _save_json(manifest_path, manifest)

    print(f"\n[INFO] manifest saved: {manifest_path}")
    print(f"[INFO] out clinical: {out_clin_path}")
