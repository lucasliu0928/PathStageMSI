#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate ensemble attention maps and their plotting-data CSVs.

Per-fold attention is averaged in memory. Each slide produces only
<SAMPLE_ID>_ensemble_attention.png and
<SAMPLE_ID>_ensemble_attention_plot_data.csv under ensemble_attention/.
With --plot_attention false, the plotting-data CSV is saved for later rendering
with --plots_only or the standalone R helper. No per-fold or intermediate
ensemble CSVs, configuration files or manifests are written.
"""

import argparse
import importlib.util
import ast
import re
import sys
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


THIS_DIR = Path(__file__).resolve().parent
PUBLIC_INFER_PATH = THIS_DIR / "inference.py"
locked = None
torch = None


def load_runtime():
    """Reuse the public inference loader, including its legacy import fallbacks."""
    global locked, torch
    import torch as torch_module
    torch = torch_module
    spec = importlib.util.spec_from_file_location("pathstage_prediction_helpers", PUBLIC_INFER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load inference helpers from {PUBLIC_INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    locked = module.locked


def str2bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "1", "yes", "y"}:
        return True
    if str(value).lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def build_parser():
    parser = argparse.ArgumentParser("Generate ensemble attention PNG maps and plotting-data CSVs")
    parser.add_argument("--test_data", nargs="*", default=None, help="One or more model-ready .h5/.hdf5/.pth files.")
    parser.add_argument("--feature_dir", default=None, help="Folder of per-slide embedding HDF5 files from 4_get_feature.py.")
    parser.add_argument("--feature_pattern", default="**/*.h5", help="Glob pattern used inside --feature_dir.")
    parser.add_argument("--feature_key", default="feature", help="HDF5 key for embedding matrix in per-slide feature files.")
    parser.add_argument("--tile_info_key", default="tile_info", help="HDF5 key for tile metadata in per-slide feature files.")
    parser.add_argument("--label_csv", default=None, help="Optional labels CSV; dummy zeros are used if omitted.")
    parser.add_argument("--sample_id_col", default="SAMPLE_ID")
    parser.add_argument("--patient_id_col", default="PATIENT_ID")
    parser.add_argument("--tumor_fraction_col", default="TUMOR_PIXEL_PERC")
    parser.add_argument("--site_local_col", default="SITE_LOCAL")
    parser.add_argument("--location_col", default="pred_map_location", help="Tile coordinate/location column to keep in *_att.csv.")

    parser.add_argument("--project_dir", default=None, help="Project root containing intermediate_data/.")
    parser.add_argument("--model_dir", default=None, help="Direct path to locked_models/. Overrides constructed path.")
    parser.add_argument("--out_folder", default="pred_out_100125_union2", help="Folder under intermediate_data/.")
    parser.add_argument("--model_folder", default=None, help="Folder containing locked_models; inferred if omitted.")
    parser.add_argument("--output_dir", default=None, help="Root output directory; final files are placed under ensemble_attention/.")
    parser.add_argument("--dataset_name", default=None, help="Name used in default output path.")

    parser.add_argument("--mutation", default="MSI", choices=["AR", "HR1", "HR2", "PTEN", "RB1", "TP53", "TMB", "MSI"])
    parser.add_argument("--train_tumor_frac", default=0.9, type=float)
    parser.add_argument("--infer_tumor_frac", default=0.0, type=float)
    parser.add_argument("--fe_method", default="uni2")
    parser.add_argument("--model_name", default="Transfer_MIL", choices=["Transfer_MIL", "ABMIL", "TransMIL"])
    parser.add_argument("--folds", default="all", help="'all' or a comma-separated list such as 0,1,2.")

    parser.add_argument("--split", default="all", choices=["all", "TRAIN", "VALID", "TEST"])
    parser.add_argument("--split_fold", default=0, type=int, help="Fold column used if --split is not all.")
    parser.add_argument("--concat_tf", default=False, type=str2bool)
    parser.add_argument("--cuda_device", default="auto", help="auto, cpu, cuda:0, cuda:1, ...")
    parser.set_defaults(include_sample_id=True)
    parser.add_argument("--plot_attention", default=True, type=str2bool,
                        help="Render PNG maps; false saves only the final plotting-data CSVs.")
    parser.add_argument("--plots_only", action="store_true",
                        help="Render saved plotting-data CSVs under --output_dir without loading models.")
    parser.add_argument("--plot_dpi", default=300, type=int)
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable with ggplot2.")
    return parser


def load_records(args):
    records = []
    if args.test_data:
        records.extend(locked.load_case_records(args.test_data))
    if args.feature_dir is not None:
        records.extend(locked.load_feature_records(args.feature_dir, args))
    if len(records) == 0:
        raise ValueError("Provide either --test_data model-ready files or --feature_dir per-slide embedding files.")
    return records


def tile_info_from_record(record):
    if isinstance(record, dict):
        tile_info = record.get("tile_info")
    elif isinstance(record, (tuple, list)) and len(record) > 3 and isinstance(record[3], pd.DataFrame):
        tile_info = record[3]
    else:
        tile_info = None

    if not isinstance(tile_info, pd.DataFrame):
        raise ValueError(
            "Attention CSV export requires tile_info for every sample. "
            "Use feature-folder input or model-ready HDF5 data that stores tile_info."
        )
    return tile_info.reset_index(drop=True).copy()


def prepare_attention_inputs(records, args):
    label_idx = locked.ALL_LABELS.index(args.mutation)
    eval_data = []
    sample_ids = []
    tile_infos = []
    full_tumor_fractions = []

    for i, record in enumerate(records):
        if args.split != "all":
            value = locked.split_value(record, args.split_fold)
            if value != args.split:
                continue

        item, sample_id = locked.normalize_case(
            record,
            label_idx=label_idx,
            concat_tf=args.concat_tf,
            fallback_id=f"case_{i:05d}",
        )
        eval_data.append(item)
        sample_ids.append(sample_id)
        tile_infos.append(tile_info_from_record(record))
        full_tumor_fractions.append(item[2].detach().cpu().clone())

    if len(eval_data) == 0:
        raise ValueError(f"No samples remained after applying split={args.split}.")

    filtered_data, masks = locked.filter_by_tumor_fraction(eval_data, threshold=args.infer_tumor_frac)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate SAMPLE_ID inputs would overwrite attention outputs; supply each slide once.")
    if any(item[0].shape[0] == 0 for item in filtered_data):
        raise ValueError("At least one slide has no tiles after tumor-fraction filtering.")
    if any(args.location_col not in ti.columns for ti in tile_infos):
        raise ValueError(f"Tile metadata must contain {args.location_col} for ensemble alignment.")
    return filtered_data, sample_ids, masks, tile_infos, full_tumor_fractions


def attention_vector(attention, expected_n):
    att = torch.as_tensor(attention).detach().cpu().float().squeeze()

    if att.ndim == 0:
        att = att.reshape(1)

    if att.numel() == expected_n:
        return att.reshape(-1).numpy()

    if att.ndim == 2:
        if att.shape[-1] == expected_n:
            return att.mean(dim=0).reshape(-1).numpy()
        if att.shape[0] == expected_n:
            return att.mean(dim=1).reshape(-1).numpy()

    raise ValueError(f"Attention shape {tuple(att.shape)} cannot be aligned to {expected_n} selected tiles.")


def forward_attention(model, model_name, x):
    if model_name == "TransMIL":
        raise NotImplementedError("TransMIL attention export is not implemented in this script.")

    output = model(x, return_attention=True, return_slide_feats=True)
    if not isinstance(output, tuple) or len(output) < 2:
        raise ValueError(f"{model_name} did not return an attention log_dict.")

    log_dict = output[1]
    if not isinstance(log_dict, dict) or "attention" not in log_dict:
        raise ValueError(f"{model_name} log_dict does not contain an 'attention' entry.")
    return log_dict["attention"]


def build_attention_dataframe(tile_info, tumor_fraction, mask, attention_values, sample_id, args):
    mask_np = mask.detach().cpu().numpy().astype(bool)
    if len(tile_info) != len(mask_np):
        raise ValueError(
            f"{sample_id}: tile_info has {len(tile_info)} rows, but tumor-fraction mask has {len(mask_np)} rows."
        )
    if int(mask_np.sum()) != len(attention_values):
        raise ValueError(
            f"{sample_id}: mask keeps {int(mask_np.sum())} tiles, but attention has {len(attention_values)} values."
        )

    if isinstance(tumor_fraction, pd.Series):
        tumor_fraction_np = tumor_fraction.to_numpy()
    else:
        tumor_fraction_np = torch.as_tensor(tumor_fraction).detach().cpu().numpy()
    if len(tumor_fraction_np) != len(tile_info):
        raise ValueError(
            f"{sample_id}: tumor_fraction has {len(tumor_fraction_np)} rows, "
            f"but tile_info has {len(tile_info)} rows."
        )
    tile_info = tile_info.copy()
    tile_info["att"] = pd.NA
    tile_info.loc[mask_np, "att"] = attention_values
    tile_info["tumor_fraction"] = tumor_fraction_np

    out_cols = []
    if args.include_sample_id:
        tile_info["SAMPLE_ID"] = sample_id
        out_cols.append("SAMPLE_ID")
    if args.location_col in tile_info.columns:
        out_cols.append(args.location_col)
    else:
        print(f"{sample_id}: {args.location_col} missing; writing only att and tumor_fraction.")

    out_cols.extend(["att", "tumor_fraction"])
    return tile_info[out_cols]


def load_model_for_fold(model_dir, fold, args, device, n_feature):
    checkpoint_path = locked.find_checkpoint(model_dir, fold)
    model = locked.load_model(checkpoint_path, args.model_name, device, n_feature=n_feature)
    return model, checkpoint_path


def ensemble_attention_frames(frames, folds, location_col="pred_map_location"):
    """Mean raw attention by slide, coordinate and tumor fraction, across all folds.

    Row order is irrelevant. Repeated tile keys follow the original project's
    grouping convention; n_att_values records how many scores were averaged.
    Excluded tiles retain missing mean_att and zero scored-fold counts.
    """
    if not frames:
        raise ValueError("No fold attention frames were generated.")
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("Select a nonempty list of distinct folds.")
    combined = pd.concat(frames, ignore_index=True)
    keys = ["SAMPLE_ID", location_col, "tumor_fraction"]
    required = keys + ["FOLD", "att"]
    missing = set(required) - set(combined.columns)
    if missing:
        raise ValueError(f"Missing ensemble columns: {sorted(missing)}")
    if combined[keys].isna().any().any():
        raise ValueError("Missing sample IDs, tile coordinates or tumor fractions.")
    combined["att"] = pd.to_numeric(combined["att"], errors="raise")
    scored = combined["att"].notna()
    if not np.isfinite(combined.loc[scored, "att"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite attention scores cannot be ensembled.")
    for sample_id, group in combined.groupby("SAMPLE_ID", sort=False):
        if set(group["FOLD"]) != set(folds):
            raise ValueError(f"{sample_id}: missing or unexpected fold outputs.")
    # Count only folds that contribute an actual score, not excluded-tile rows.
    combined["scored_fold"] = combined["FOLD"].where(scored)
    result = combined.groupby(keys, sort=False, dropna=False).agg(
        mean_att=("att", "mean"),
        n_folds=("scored_fold", "nunique"),
        n_att_values=("att", "count"),
    ).reset_index()
    partial = (result["n_folds"] > 0) & (result["n_folds"] != len(folds))
    if partial.any():
        raise ValueError("Tile coordinates or filtering differ between folds; refusing a partial ensemble.")
    return result


def write_plot_data(ensemble, output_dir, location_col="pred_map_location"):
    """Save only each slide's final raw scores, coordinates and manuscript scaling."""
    plot_dir = output_dir / "ensemble_attention"
    paths = []
    used_stems = set()
    for sample_id, group in ensemble.groupby("SAMPLE_ID", sort=False):
        plot_data = group.loc[group["mean_att"].notna()].copy()
        if plot_data.empty:
            raise ValueError(f"{sample_id}: no attention scores to plot.")
        if not np.isfinite(plot_data["mean_att"].to_numpy(dtype=float)).all():
            raise ValueError(f"{sample_id}: non-finite attention scores.")
        coords = []
        for location in plot_data[location_col]:
            try:
                values = ast.literal_eval(str(location))
                if not isinstance(values, (tuple, list)) or len(values) != 4:
                    raise ValueError("expected four coordinates")
                coords.append([float(v) for v in values])
            except (ValueError, SyntaxError, TypeError) as exc:
                raise ValueError(f"Invalid tile coordinates: {location}") from exc
        coordinates = np.asarray(coords)
        if (not np.isfinite(coordinates).all()
                or np.any(coordinates[:, 1] <= coordinates[:, 0])
                or np.any(coordinates[:, 3] <= coordinates[:, 2])):
            raise ValueError(f"{sample_id}: tile rectangle bounds must be finite and increase.")
        plot_data[["x1", "x2", "y1", "y2"]] = coordinates
        low, high = plot_data["mean_att"].min(), plot_data["mean_att"].max()
        plot_data["attention_heatmap_value"] = (
            (plot_data["mean_att"] - low) / (high - low) if high > low else 0.5
        )
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id))
        stem = re.sub(r"_+", "_", stem).strip("_")[:180]
        if not stem or stem in used_stems:
            raise ValueError("Sample IDs do not produce distinct valid output filenames.")
        used_stems.add(stem)
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / f"{stem}_ensemble_attention_plot_data.csv"
        plot_data.to_csv(path, index=False)
        paths.append(path)
    return paths


def plot_ensemble_attention(output_dir, args):
    """Render final plotting data with the manuscript's original R functions."""
    plot_dir = output_dir / "ensemble_attention"
    inputs = sorted(plot_dir.glob("*_ensemble_attention_plot_data.csv"))
    # Read older runs for compatibility; all new outputs use ensemble_attention/.
    if not inputs:
        inputs = sorted((output_dir / "ensemble" / "plots").glob("*_ensemble_attention_plot_data.csv"))
    legacy_csv = output_dir / "ensemble" / "ensemble_attention_df.csv"
    if not inputs and legacy_csv.exists():
        inputs = [legacy_csv]
    if not inputs:
        raise FileNotFoundError(f"No saved attention plotting data found under {plot_dir}")
    if args.plot_dpi <= 0:
        raise ValueError("--plot_dpi must be positive.")
    rscript = shutil.which(args.rscript)
    if rscript is None:
        raise FileNotFoundError(
            "Rscript is required for manuscript-style maps (R package ggplot2). "
            "Final plotting-data CSVs are retained. Use --plots_only after enabling R."
        )
    for input_csv in inputs:
        subprocess.run([
            rscript, str(THIS_DIR / "Utils" / "plot_attention.R"), "--input", str(input_csv),
            "--output_dir", str(plot_dir), "--dpi", str(args.plot_dpi),
        ], check=True)


def main():
    args = build_parser().parse_args()
    if args.plots_only:
        if not args.output_dir:
            raise ValueError("--plots_only requires --output_dir pointing to existing attention output.")
        output_dir = Path(args.output_dir).expanduser().resolve()
        plot_ensemble_attention(output_dir, args)
        return
    load_runtime()
    folds = locked.parse_folds(args.folds)
    if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
        raise ValueError("--folds must select distinct folds from 0,1,2,3,4.")
    if args.model_name == "TransMIL":
        raise NotImplementedError("Attention export supports Transfer_MIL and ABMIL only.")
    project_dir = locked.resolve_project_dir(args.project_dir)
    model_dir = locked.resolve_model_dir(args, project_dir)
    output_dir = locked.resolve_output_dir(args, project_dir)
    device = locked.resolve_device(args.cuda_device)

    if not model_dir.exists():
        raise FileNotFoundError(f"Locked model directory does not exist: {model_dir}")

    records = load_records(args)
    test_data, sample_ids, masks, tile_infos, full_tumor_fractions = prepare_attention_inputs(records, args)
    n_feature = test_data[0][0].shape[1]

    fold_frames = []
    for fold in tqdm(folds, desc="Running folds", unit="fold"):
        model, checkpoint_path = load_model_for_fold(model_dir, fold, args, device, n_feature)

        model.eval()
        with torch.no_grad():
            iterator = zip(test_data, sample_ids, masks, tile_infos, full_tumor_fractions)
            for item, sample_id, mask, tile_info, full_tumor_fraction in tqdm(
                iterator,
                total=len(sample_ids),
                desc=f"FOLD{fold} attention",
                unit="slide",
            ):
                x, _y, _tumor_fraction, _site_loc = item
                attention = forward_attention(model, args.model_name, x.unsqueeze(0).to(device))
                att_values = attention_vector(attention, expected_n=x.shape[0])
                if not np.isfinite(att_values).all():
                    raise ValueError(f"{sample_id}: model returned non-finite attention scores.")
                att_df = build_attention_dataframe(
                    tile_info=tile_info,
                    tumor_fraction=full_tumor_fraction,
                    mask=mask,
                    attention_values=att_values,
                    sample_id=sample_id,
                    args=args,
                )

                ensemble_input = att_df.copy()
                ensemble_input["SAMPLE_ID"] = sample_id
                ensemble_input["FOLD"] = fold
                fold_frames.append(ensemble_input)
    ensemble = ensemble_attention_frames(fold_frames, folds, args.location_col)
    write_plot_data(ensemble, output_dir, args.location_col)
    print(f"Final attention plotting-data CSVs: {output_dir}/ensemble_attention")
    if args.plot_attention:
        plot_ensemble_attention(output_dir, args)


if __name__ == "__main__":
    main()
