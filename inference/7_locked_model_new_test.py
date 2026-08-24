#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run locked TransferMIL/ABMIL checkpoints on a new test dataset.

Input can be either model-ready HDF5/PTH data or a folder of per-slide
embedding HDF5 files written by 4_get_feature.py. All supplied samples are
treated as test samples unless --split is set.
"""

import argparse
import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DEFAULT = THIS_DIR.parents[1]


def find_utils_dir():
    for base_dir in [THIS_DIR, *THIS_DIR.parents]:
        candidate = base_dir / "Utils"
        if (candidate / "data_loader.py").exists():
            return candidate
    raise FileNotFoundError("Could not find code_s/Utils/data_loader.py from this script location.")


sys.path.insert(0, str(find_utils_dir()))

from data_loader import H5Cases, filter_by_tumor_fraction, h5_to_list  # noqa: E402
from Eval import compute_performance  # noqa: E402
from Loss import compute_logit_adjustment  # noqa: E402
from misc_utils import str2bool  # noqa: E402
from TransferMIL_utils import build_model  # noqa: E402


ALL_LABELS = ["AR", "HR1", "HR2", "PTEN", "RB1", "TP53", "TMB", "MSI"]
LABEL_ALIASES = {
    "TMB": ["TMB", "TMB_HIGHorINTERMEDITATE"],
    "MSI": ["MSI", "MSI_POS"],
}


def parse_folds(value):
    if value.lower() == "all":
        return [0, 1, 2, 3, 4]
    return [int(v.strip()) for v in value.split(",") if v.strip() != ""]


def parse_class_priors(value):
    priors = np.array([float(v.strip()) for v in value.split(",")], dtype=float)
    if priors.shape[0] != 2:
        raise ValueError("--class_priors must contain two comma-separated values.")
    if np.any(priors <= 0):
        raise ValueError("--class_priors values must be positive.")
    return priors / priors.sum()


def fmt_float(value):
    return str(float(value))


def output_subdir_from_args(args):
    value = str(args.output_subdir).strip()
    if value.lower() == "auto":
        return f"inference_tf{fmt_float(args.infer_tumor_frac)}"
    if value.lower() in {"", "none", "false", "no"}:
        return None
    return value


def model_folder_from_args(args):
    if args.model_folder:
        return args.model_folder
    return (
        f"{args.mutation}_traintf{fmt_float(args.train_tumor_frac)}_"
        f"{args.model_name}_{args.fe_method}"
    )


def resolve_project_dir(value):
    if value is not None:
        return Path(value).expanduser().resolve()
    return PROJECT_DEFAULT


def resolve_model_dir(args, project_dir):
    if args.model_dir:
        return Path(args.model_dir).expanduser().resolve()
    return (
        project_dir
        / "intermediate_data"
        / args.out_folder
        / model_folder_from_args(args)
        / "locked_models"
    )


def resolve_output_dir(args, project_dir):
    subdir = output_subdir_from_args(args)
    if args.output_dir:
        base_dir = Path(args.output_dir).expanduser().resolve()
        return base_dir / subdir if subdir else base_dir
    dataset_name = args.dataset_name
    if dataset_name is None:
        stems = [Path(p).stem for p in (args.test_data or [])]
        if args.feature_dir:
            stems.append(Path(args.feature_dir).expanduser().resolve().name)
        dataset_name = "_".join(stems)[:120]
    base_dir = (
        project_dir
        / "intermediate_data"
        / "new_test_predictions"
        / model_folder_from_args(args)
        / dataset_name
    )
    return base_dir / subdir if subdir else base_dir


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is not available; using CPU instead of {device_arg}.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_case_records(paths):
    records = []
    for data_path in paths:
        path = Path(data_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input data file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            records.extend(h5_to_list(H5Cases(path)))
        elif suffix in {".pth", ".pt"}:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(loaded, list):
                records.extend(loaded)
            elif hasattr(loaded, "__len__") and hasattr(loaded, "__getitem__"):
                records.extend([loaded[i] for i in range(len(loaded))])
            else:
                raise TypeError(f"Unsupported torch dataset object in {path}: {type(loaded)}")
        else:
            raise ValueError(f"Unsupported input file type: {path}")

    if len(records) == 0:
        raise ValueError("No samples were loaded from --test_data.")
    return records


def load_label_table(label_csv, sample_id_col):
    if label_csv is None:
        return None

    label_path = Path(label_csv).expanduser().resolve()
    if not label_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_path}")

    label_df = pd.read_csv(label_path)
    if sample_id_col not in label_df.columns:
        raise ValueError(f"{sample_id_col} is missing from label CSV: {label_path}")
    return label_df.drop_duplicates(subset=[sample_id_col]).set_index(sample_id_col)


def first_existing_value(row, columns, default=0):
    if row is None:
        return default
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def label_vector(label_row):
    values = []
    for label in ALL_LABELS:
        candidate_cols = LABEL_ALIASES.get(label, [label])
        values.append(float(first_existing_value(label_row, candidate_cols, default=0)))
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1)


def infer_sample_id(feature_file, tile_info):
    if "SAMPLE_ID" in tile_info.columns:
        values = tile_info["SAMPLE_ID"].dropna().unique()
        if len(values) > 0:
            return str(values[0])

    if feature_file.name.startswith("features_alltiles_"):
        return feature_file.parent.parent.name
    return feature_file.stem


def infer_patient_id(sample_id, tile_info, label_row, patient_id_col):
    if label_row is not None and patient_id_col in label_row.index and pd.notna(label_row[patient_id_col]):
        return str(label_row[patient_id_col])
    if "PATIENT_ID" in tile_info.columns:
        values = tile_info["PATIENT_ID"].dropna().unique()
        if len(values) > 0:
            return str(values[0])
    return sample_id


def load_feature_records(feature_dir, args):
    folder = Path(feature_dir).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Feature directory not found: {folder}")

    feature_files = sorted(folder.glob(args.feature_pattern))
    if len(feature_files) == 0:
        raise FileNotFoundError(f"No files matching {args.feature_pattern} found under {folder}")

    label_table = load_label_table(args.label_csv, args.sample_id_col)
    records = []
    skipped = []

    for feature_file in tqdm(feature_files, desc="Loading feature files", unit="slide"):
        try:
            feature_df = pd.read_hdf(feature_file, key=args.feature_key)
            tile_info = pd.read_hdf(feature_file, key=args.tile_info_key)
        except (KeyError, OSError, ValueError) as exc:
            skipped.append((feature_file, str(exc)))
            continue

        feature_df.columns = feature_df.columns.astype(str)
        feature_df = feature_df.reset_index(drop=True)
        tile_info = tile_info.reset_index(drop=True)
        if feature_df.shape[0] != tile_info.shape[0]:
            raise ValueError(f"Feature/tile row mismatch in {feature_file}")

        sample_id = infer_sample_id(feature_file, tile_info)
        label_row = None
        if label_table is not None and sample_id in label_table.index:
            label_row = label_table.loc[sample_id]
        patient_id = infer_patient_id(sample_id, tile_info, label_row, args.patient_id_col)

        if args.tumor_fraction_col in tile_info.columns:
            tumor_fraction = torch.tensor(tile_info[args.tumor_fraction_col].to_numpy(), dtype=torch.float32)
        else:
            warnings.warn(
                f"{args.tumor_fraction_col} missing in {feature_file}; using tumor_fraction=1 for all tiles.",
                RuntimeWarning,
            )
            tumor_fraction = torch.ones(feature_df.shape[0], dtype=torch.float32)

        if args.site_local_col in tile_info.columns:
            site_location = torch.tensor(tile_info[args.site_local_col].to_numpy(), dtype=torch.float32)
        else:
            site_location = torch.zeros(feature_df.shape[0], dtype=torch.float32)

        records.append(
            {
                "x": torch.tensor(feature_df.to_numpy(), dtype=torch.float32),
                "y": label_vector(label_row),
                "tumor_fraction": tumor_fraction,
                "site_location": site_location,
                "tile_info": tile_info,
                "sample_id": sample_id,
                "patient_id": patient_id,
                "fold0": "TEST",
                "fold1": "TEST",
                "fold2": "TEST",
                "fold3": "TEST",
                "fold4": "TEST",
            }
        )

    if skipped:
        print(f"Skipped {len(skipped)} non-feature HDF5 file(s) under {folder}.")
    if len(records) == 0:
        raise ValueError(f"No usable per-slide feature files found under {folder}.")
    return records


def selected_label_tensor(y, label_idx):
    y = torch.as_tensor(y)
    if y.ndim == 0:
        return y.reshape(1, 1).float()
    if y.ndim == 1:
        if y.numel() == 1:
            return y.reshape(1, 1).float()
        return y[label_idx].reshape(1, 1).float()
    if y.shape[-1] == 1:
        return y.reshape(1, 1).float()
    return y[:, [label_idx]].float()


def sample_id_from_tile_info(tile_info, fallback):
    if isinstance(tile_info, pd.DataFrame) and "SAMPLE_ID" in tile_info.columns:
        values = tile_info["SAMPLE_ID"].dropna().unique()
        if len(values) > 0:
            return str(values[0])
    return fallback


def normalize_case(record, label_idx, concat_tf, fallback_id):
    if isinstance(record, dict):
        x = torch.as_tensor(record["x"]).float()
        y = selected_label_tensor(record["y"], label_idx)
        tf = torch.as_tensor(record["tumor_fraction"]).float()
        site_loc = torch.as_tensor(record.get("site_location", torch.zeros_like(tf))).float()
        sample_id = str(record.get("sample_id", fallback_id))
    elif isinstance(record, (tuple, list)):
        x = torch.as_tensor(record[0]).float()
        y = selected_label_tensor(record[1], label_idx)
        tf = torch.as_tensor(record[2]).float()

        if len(record) > 3 and isinstance(record[3], pd.DataFrame):
            site_loc = torch.zeros_like(tf)
            sample_id = sample_id_from_tile_info(record[3], fallback_id)
        else:
            site_loc = torch.as_tensor(record[3] if len(record) > 3 else torch.zeros_like(tf)).float()
            sample_id = str(record[4]) if len(record) > 4 and isinstance(record[4], str) else fallback_id
    else:
        raise TypeError(f"Unsupported case record type: {type(record)}")

    if concat_tf:
        x = torch.cat([x, tf.unsqueeze(1)], dim=1)

    return (x, y, tf, site_loc), sample_id


def split_value(record, fold):
    if isinstance(record, dict):
        value = record.get(f"fold{fold}")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value
    return None


def prepare_eval_data(records, args, split="all"):
    label_idx = ALL_LABELS.index(args.mutation)
    eval_data = []
    sample_ids = []

    for i, record in enumerate(records):
        if split != "all":
            value = split_value(record, args.split_fold)
            if value != split:
                continue

        item, sample_id = normalize_case(
            record,
            label_idx=label_idx,
            concat_tf=args.concat_tf,
            fallback_id=f"case_{i:05d}",
        )
        eval_data.append(item)
        sample_ids.append(sample_id)

    if len(eval_data) == 0:
        raise ValueError(f"No samples remained after applying split={split}.")

    filtered_data, _ = filter_by_tumor_fraction(eval_data, threshold=args.infer_tumor_frac)
    return filtered_data, sample_ids


def label_priors(eval_data):
    labels = [int(item[1].reshape(-1)[0].item()) for item in eval_data]
    counts = np.bincount(labels, minlength=2).astype(float)
    if counts.sum() == 0:
        raise ValueError("Cannot compute class priors from an empty validation set.")
    return counts / counts.sum()


def resolve_saved_threshold_root(args, model_dir):
    if args.saved_threshold_root:
        return Path(args.saved_threshold_root).expanduser().resolve()
    return model_dir.parent


def load_saved_threshold(args, model_dir, fold):
    threshold_root = resolve_saved_threshold_root(args, model_dir)
    perf_dir = threshold_root / f"FOLD{fold}" / f"perf_{fmt_float(args.infer_tumor_frac)}"
    perf_path = perf_dir / args.saved_threshold_file
    if not perf_path.exists():
        raise FileNotFoundError(f"Saved threshold file not found for FOLD{fold}: {perf_path}")

    perf_df = pd.read_csv(perf_path, index_col=0)
    if args.saved_threshold_row not in perf_df.index:
        raise ValueError(
            f"{args.saved_threshold_row} is missing from saved threshold file: {perf_path}"
        )
    if args.saved_threshold_col not in perf_df.columns:
        raise ValueError(
            f"{args.saved_threshold_col} is missing from saved threshold file: {perf_path}"
        )

    value = perf_df.loc[args.saved_threshold_row, args.saved_threshold_col]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(
            f"Empty {args.saved_threshold_col} for {args.saved_threshold_row} in {perf_path}"
        )
    return float(value), perf_path


def find_checkpoint(model_dir, fold):
    candidates = sorted(model_dir.glob(f"*FOLD{fold}*.pth"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No checkpoint matching FOLD{fold} found in {model_dir}")
    if len(candidates) > 1:
        print(f"Multiple FOLD{fold} checkpoints found; using {candidates[0].name}")
    return candidates[0]


def forward_logits(model, model_name, x):
    if model_name == "TransMIL":
        results = model(data=x)
    else:
        results = model(x)
        if isinstance(results, tuple):
            results = results[0]

    if isinstance(results, dict):
        return results["logits"]
    return results


def predict_dataframe(
    eval_data,
    sample_ids,
    model,
    model_name,
    device,
    logit_adjustments,
    threshold,
    use_logit_adjustment,
    desc="Predicting",
):
    model.eval()
    rows = []

    with torch.no_grad():
        iterator = zip(eval_data, sample_ids)
        for item, sample_id in tqdm(iterator, total=len(sample_ids), desc=desc, unit="slide"):
            x, y, *_ = item
            logits = forward_logits(model, model_name, x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

            row = {
                "SAMPLE_ID": sample_id,
                "True_y": int(y.reshape(-1)[0].item()),
                "logit_0": float(logits.squeeze(0).detach().cpu()[0]),
                "logit_1": float(logits.squeeze(0).detach().cpu()[1]),
                "prob_0": float(probs[0]),
                "prob_1": float(probs[1]),
                "Pred_Class": int(probs[1] > threshold),
            }

            if use_logit_adjustment:
                adj_logits = logits - logit_adjustments.to(device)
                adj_probs = torch.softmax(adj_logits, dim=1).squeeze(0).detach().cpu()
                row.update(
                    {
                        "adj_logit_0": float(adj_logits.squeeze(0).detach().cpu()[0]),
                        "adj_logit_1": float(adj_logits.squeeze(0).detach().cpu()[1]),
                        "adj_prob_0": float(adj_probs[0]),
                        "adj_prob_1": float(adj_probs[1]),
                        "Pred_Class_adj": int(adj_probs[1] > threshold),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def safe_performance(pred_df, cohort_name, use_logit_adjustment):
    if pred_df["True_y"].nunique(dropna=True) < 2:
        print(f"Skipping performance for {cohort_name}: only one label class is present.")
        return None

    if use_logit_adjustment:
        return compute_performance(
            pred_df["True_y"],
            pred_df["adj_prob_1"],
            pred_df["Pred_Class_adj"],
            cohort_name,
        )
    return compute_performance(
        pred_df["True_y"],
        pred_df["prob_1"],
        pred_df["Pred_Class"],
        cohort_name,
    )


def majority_vote(values):
    values = pd.Series(values).dropna().astype(int)
    n_pos = int((values == 1).sum())
    n_neg = int((values == 0).sum())
    return 1 if n_pos > n_neg else 0


def build_majority_vote_ensemble(all_pred, use_logit_adjustment, threshold):
    prob_col = "adj_prob_1" if use_logit_adjustment else "prob_1"
    class_col = "Pred_Class_adj" if use_logit_adjustment else "Pred_Class"

    rows = []
    for sample_id, group in all_pred.groupby("SAMPLE_ID", sort=False):
        majority_class = majority_vote(group[class_col])
        voted = group.loc[group[class_col].astype(int) == majority_class]
        folds_voted = ",".join(f"FOLD{int(fold)}" for fold in voted["FOLD"].tolist())
        mean_prob_allfolds = float(group[prob_col].mean())
        mean_prob_votedclass = float(voted[prob_col].mean())

        row = {
            "SAMPLE_ID": sample_id,
            "True_y": int(group["True_y"].iloc[0]),
            "majority_class": majority_class,
            "folds_voted": folds_voted,
            "ensemble_threshold": threshold,
            "majority_class_confidence": (
                mean_prob_votedclass if majority_class == 1 else 1.0 - mean_prob_votedclass
            ),
        }

        if use_logit_adjustment:
            row.update(
                {
                    "mean_adj_prob": mean_prob_allfolds,
                    "mean_adj_prob_votedclass": mean_prob_votedclass,
                    "adj_prob_1": mean_prob_votedclass,
                    "adj_prob_0": 1.0 - mean_prob_votedclass,
                    "Pred_Class_adj": majority_class,
                    "mean_prob": float(group["prob_1"].mean()),
                    "prob_1": float(group["prob_1"].mean()),
                    "prob_0": float(group["prob_0"].mean()),
                    "Pred_Class": int(float(group["prob_1"].mean()) > threshold),
                }
            )
        else:
            row.update(
                {
                    "mean_prob": mean_prob_allfolds,
                    "mean_prob_votedclass": mean_prob_votedclass,
                    "prob_1": mean_prob_votedclass,
                    "prob_0": 1.0 - mean_prob_votedclass,
                    "Pred_Class": majority_class,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def load_model(checkpoint_path, model_name, device, n_feature):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(model_name=model_name, device=device, num_classes=2, n_feature=n_feature)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def build_parser():
    parser = argparse.ArgumentParser("Locked-model inference for a new test dataset")
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
    parser.add_argument("--val_data", nargs="*", default=None, help="Optional validation files for threshold/priors.")
    parser.add_argument("--project_dir", default=None, help="Project root containing intermediate_data/.")
    parser.add_argument("--model_dir", default=None, help="Direct path to locked_models/. Overrides constructed path.")
    parser.add_argument("--out_folder", default="pred_out_100125_union2", help="Folder under intermediate_data/.")
    parser.add_argument("--model_folder", default=None, help="Folder containing locked_models; inferred if omitted.")
    parser.add_argument("--output_dir", default=None, help="Where prediction CSVs should be written.")
    parser.add_argument(
        "--output_subdir",
        default="auto",
        help="'auto' writes to inference_tf<infer_tumor_frac>; use 'none' to write directly to --output_dir.",
    )
    parser.add_argument("--dataset_name", default=None, help="Name used in default output path.")

    parser.add_argument("--mutation", default="MSI", choices=ALL_LABELS)
    parser.add_argument("--train_tumor_frac", default=0.9, type=float)
    parser.add_argument("--infer_tumor_frac", default=0.0, type=float)
    parser.add_argument("--fe_method", default="uni2")
    parser.add_argument("--model_name", default="Transfer_MIL", choices=["Transfer_MIL", "ABMIL", "TransMIL"])
    parser.add_argument("--folds", default="all", help="'all' or a comma-separated list such as 0,1,2.")

    parser.add_argument("--split", default="all", choices=["all", "TRAIN", "VALID", "TEST"])
    parser.add_argument("--split_fold", default=0, type=int, help="Fold column used if --split is not all.")
    parser.add_argument("--concat_tf", default=False, type=str2bool)

    parser.add_argument("--cuda_device", default="auto", help="auto, cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--logit_adj_train", default=False, type=str2bool)
    parser.add_argument("--logit_adj_infer", default=True, type=str2bool)
    parser.add_argument("--logit_adj_tau", default=0.1, type=float)
    parser.add_argument("--class_priors", default="0.98,0.02", help="Fallback class priors for logit adjustment.")
    parser.add_argument("--pred_threshold", default=0.5, type=float)
    parser.add_argument("--use_val_threshold", default=True, type=str2bool)
    parser.add_argument(
        "--use_saved_threshold",
        default=False,
        type=str2bool,
        help="Load each fold threshold from FOLD*/perf_<infer_tumor_frac>/after_finetune_performance.csv.",
    )
    parser.add_argument(
        "--saved_threshold_root",
        default=None,
        help="Folder containing FOLD*/perf_*/ threshold CSVs. Defaults to parent of --model_dir.",
    )
    parser.add_argument("--saved_threshold_file", default="after_finetune_performance.csv")
    parser.add_argument("--saved_threshold_row", default="OPX_TCGA_valid")
    parser.add_argument("--saved_threshold_col", default="best_thresh")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.test_data and args.feature_dir is None:
        raise ValueError("Provide either --test_data model-ready files or --feature_dir per-slide embedding files.")

    folds = parse_folds(args.folds)
    project_dir = resolve_project_dir(args.project_dir)
    model_dir = resolve_model_dir(args, project_dir)
    output_dir = resolve_output_dir(args, project_dir)
    device = resolve_device(args.cuda_device)

    if not model_dir.exists():
        raise FileNotFoundError(f"Locked model directory does not exist: {model_dir}")

    test_records = []
    if args.test_data:
        test_records.extend(load_case_records(args.test_data))
    if args.feature_dir is not None:
        test_records.extend(load_feature_records(args.feature_dir, args))
    test_data, test_sample_ids = prepare_eval_data(test_records, args, split=args.split)

    val_data = None
    val_sample_ids = None
    if args.val_data:
        val_records = load_case_records(args.val_data)
        val_data, val_sample_ids = prepare_eval_data(val_records, args, split="all")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "project_dir": str(project_dir),
            "model_dir": str(model_dir),
            "output_dir": str(output_dir),
            "folds": folds,
            "device": str(device),
        }
    )
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.logit_adj_infer:
        priors = parse_class_priors(args.class_priors)
        if val_data is not None:
            val_priors = label_priors(val_data)
            if np.count_nonzero(val_priors) == 2:
                priors = val_priors
            else:
                print("Validation data has one class; using --class_priors for logit adjustment.")
        logit_adjustments = compute_logit_adjustment(priors, tau=args.logit_adj_tau).float()
        print(f"Using logit-adjustment priors: {priors.tolist()}")
    else:
        logit_adjustments = torch.zeros(2)

    n_feature = test_data[0][0].shape[1]
    all_fold_predictions = []
    fold_thresholds = []

    for fold in tqdm(folds, desc="Running folds", unit="fold"):
        checkpoint_path = find_checkpoint(model_dir, fold)
        print(f"FOLD{fold}: loading {checkpoint_path}")
        model = load_model(checkpoint_path, args.model_name, device, n_feature=n_feature)

        threshold = args.pred_threshold
        threshold_source = "pred_threshold"
        threshold_path = None
        fold_dir = output_dir / f"FOLD{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if args.use_saved_threshold:
            threshold, threshold_path = load_saved_threshold(args, model_dir, fold)
            threshold_source = "saved_threshold"
            print(f"FOLD{fold}: using saved threshold {threshold} from {threshold_path}")
        elif val_data is not None and args.use_val_threshold:
            val_pred = predict_dataframe(
                val_data,
                val_sample_ids,
                model,
                args.model_name,
                device,
                logit_adjustments,
                threshold=args.pred_threshold,
                use_logit_adjustment=args.logit_adj_infer,
                desc=f"FOLD{fold} validation",
            )
            val_perf = safe_performance(val_pred, f"FOLD{fold}_validation", args.logit_adj_infer)
            val_pred.to_csv(fold_dir / "validation_prediction.csv", index=False)
            if val_perf is not None:
                val_perf.to_csv(fold_dir / "validation_performance.csv")
                threshold = float(val_perf["best_thresh"].iloc[0])
                threshold_source = "validation_best_thresh"

        fold_thresholds.append(threshold)
        pred_df = predict_dataframe(
            test_data,
            test_sample_ids,
            model,
            args.model_name,
            device,
            logit_adjustments,
            threshold=threshold,
            use_logit_adjustment=args.logit_adj_infer,
            desc=f"FOLD{fold} test",
        )
        pred_df.insert(0, "FOLD", fold)
        pred_df["threshold"] = threshold
        pred_df["threshold_source"] = threshold_source
        pred_df["threshold_file"] = str(threshold_path) if threshold_path is not None else pd.NA
        pred_df["checkpoint"] = checkpoint_path.name
        pred_df.to_csv(fold_dir / "test_prediction.csv", index=False)

        perf = safe_performance(pred_df, f"FOLD{fold}_test", args.logit_adj_infer)
        if perf is not None:
            perf.to_csv(fold_dir / "test_performance.csv")

        all_fold_predictions.append(pred_df)

    all_pred = pd.concat(all_fold_predictions, ignore_index=True)
    all_pred.to_csv(output_dir / "all_folds_prediction.csv", index=False)

    threshold_ensemble = (
        float(np.mean(fold_thresholds))
        if args.use_saved_threshold or args.use_val_threshold
        else args.pred_threshold
    )
    ensemble = build_majority_vote_ensemble(
        all_pred,
        use_logit_adjustment=args.logit_adj_infer,
        threshold=threshold_ensemble,
    )

    ensemble.to_csv(output_dir / "ensemble_prediction.csv", index=False)
    ensemble_perf = safe_performance(ensemble, "ensemble_test", args.logit_adj_infer)
    if ensemble_perf is not None:
        ensemble_perf.to_csv(output_dir / "ensemble_performance.csv")

    print(f"Done. Predictions written to: {output_dir}")


if __name__ == "__main__":
    main()
