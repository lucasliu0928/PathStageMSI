#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public-facing locked-model inference entry point.

This script writes ensemble_prediction.csv and ensemble_performance.csv. Slide-level point performance is printed
after inference. Optional bootstrapped confidence intervals can be computed
separately with bootstrap_performance.py.

Run from the PathStageMSI root, for example:

    python inference/inference.py \
        --test_data examples/test.pth \
        --model_dir locked_models \
        --output_dir output/predictions_tcga \
        --threshold_csv configs/thresholds_summary.csv \
        --logit_priors_csv configs/logit_priors_summary.csv
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bootstrap_performance import compute_metrics

THIS_DIR = Path(__file__).resolve().parent
LOCKED_INFER_PATH = THIS_DIR / "Utils" / "7_locked_model_new_test.py"


def unavailable_training_helper(*_args, **_kwargs):
    raise RuntimeError(
        "This training-only helper is not bundled with the public inference package."
    )


def register_training_import_fallbacks():
    # These modules are imported by legacy training utilities but are not used for
    # locked Transfer_MIL inference. Stubbing them keeps the public package small.
    try:
        import RandomSplit  # noqa: F401
    except ModuleNotFoundError:
        random_split = types.ModuleType("RandomSplit")
        random_split.MakeBalancedCrossValidation = unavailable_training_helper
        sys.modules["RandomSplit"] = random_split

    try:
        import ACMIL  # noqa: F401
    except ModuleNotFoundError:
        acmil = types.ModuleType("ACMIL")
        acmil.compute_loss_singletask = unavailable_training_helper
        sys.modules["ACMIL"] = acmil

    try:
        import TransMIL  # noqa: F401
    except ModuleNotFoundError:
        transmil = types.ModuleType("TransMIL")

        class TransMILUnavailable:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError(
                    "TransMIL is not bundled with this public inference package. "
                    "Use --model_name Transfer_MIL with the provided locked checkpoints."
                )

        transmil.TransMIL = TransMILUnavailable
        sys.modules["TransMIL"] = transmil


register_training_import_fallbacks()


def load_locked_infer_module():
    spec = importlib.util.spec_from_file_location("locked_model_new_test", LOCKED_INFER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {LOCKED_INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


locked = load_locked_infer_module()
_original_build_parser = locked.build_parser
_original_load_saved_threshold = locked.load_saved_threshold


def build_parser():
    parser = _original_build_parser()
    parser.prog = "inference.py"
    parser.description = "Run PathStageMSI locked-model inference and write ensemble predictions and point performance."
    parser.add_argument(
        "--threshold_csv",
        default=None,
        help=(
            "Compact CSV with at least columns fold and threshold. Optional columns "
            "infer_tumor_frac, saved_threshold_row, and saved_threshold_col are used "
            "to disambiguate rows when present."
        ),
    )
    parser.add_argument(
        "--logit_priors_csv",
        default=None,
        help=(
            "Optional CSV with fold-specific inference logit-adjustment priors. "
            "Expected columns: fold, prior_0, prior_1; optional: infer_tumor_frac, "
            "logit_adj_tau. Use this to reproduce the paper's TCGA run."
        ),
    )
    return parser


def threshold_from_summary(threshold_csv, fold, infer_tumor_frac, row_name, col_name):
    path = Path(threshold_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Threshold CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"fold", "threshold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing column(s) in threshold CSV {path}: {sorted(missing)}")

    mask = df["fold"].astype(int) == int(fold)
    if "infer_tumor_frac" in df.columns:
        mask &= np.isclose(df["infer_tumor_frac"].astype(float), float(infer_tumor_frac))
    if "saved_threshold_row" in df.columns:
        mask &= df["saved_threshold_row"].astype(str) == str(row_name)
    if "saved_threshold_col" in df.columns:
        mask &= df["saved_threshold_col"].astype(str) == str(col_name)

    matches = df.loc[mask]
    if matches.empty:
        raise ValueError(
            f"No threshold found in {path} for FOLD{fold}, "
            f"infer_tumor_frac={infer_tumor_frac}, row={row_name}, col={col_name}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple thresholds found in {path} for FOLD{fold}, "
            f"infer_tumor_frac={infer_tumor_frac}, row={row_name}, col={col_name}."
        )

    value = matches.iloc[0]["threshold"]
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"Empty threshold in {path} for FOLD{fold}.")
    return float(value), path


def load_saved_threshold(args, model_dir, fold):
    if getattr(args, "threshold_csv", None):
        threshold, threshold_path = threshold_from_summary(
            args.threshold_csv,
            fold=fold,
            infer_tumor_frac=args.infer_tumor_frac,
            row_name=args.saved_threshold_row,
            col_name=args.saved_threshold_col,
        )
        return threshold, threshold_path
    return _original_load_saved_threshold(args, model_dir, fold)


def prior_columns(df):
    candidates = [
        ("prior_0", "prior_1"),
        ("prior_MSS", "prior_MSI"),
        ("prior_negative", "prior_positive"),
    ]
    for col0, col1 in candidates:
        if col0 in df.columns and col1 in df.columns:
            return col0, col1
    raise ValueError(
        "Logit-priors CSV must contain either prior_0/prior_1, "
        "prior_MSS/prior_MSI, or prior_negative/prior_positive."
    )


def priors_from_summary(logit_priors_csv, fold, infer_tumor_frac):
    path = Path(logit_priors_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Logit-priors CSV not found: {path}")

    df = pd.read_csv(path)
    if "fold" not in df.columns:
        raise ValueError(f"Missing column 'fold' in logit-priors CSV: {path}")

    col0, col1 = prior_columns(df)
    mask = df["fold"].astype(int) == int(fold)
    if "infer_tumor_frac" in df.columns:
        mask &= np.isclose(df["infer_tumor_frac"].astype(float), float(infer_tumor_frac))

    matches = df.loc[mask]
    if matches.empty:
        raise ValueError(
            f"No priors found in {path} for FOLD{fold}, infer_tumor_frac={infer_tumor_frac}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple priors found in {path} for FOLD{fold}, infer_tumor_frac={infer_tumor_frac}."
        )

    row = matches.iloc[0]
    priors = np.array([float(row[col0]), float(row[col1])], dtype=float)
    if np.any(priors <= 0):
        raise ValueError(f"Priors must be positive in {path} for FOLD{fold}.")
    priors = priors / priors.sum()
    tau = float(row["logit_adj_tau"]) if "logit_adj_tau" in matches.columns else None
    return priors, tau, path


def fold_logit_adjustment(args, val_data, fold):
    if not args.logit_adj_infer:
        return torch.zeros(2), None, None

    tau = args.logit_adj_tau
    priors_path = None
    if getattr(args, "logit_priors_csv", None):
        priors, file_tau, priors_path = priors_from_summary(
            args.logit_priors_csv,
            fold=fold,
            infer_tumor_frac=args.infer_tumor_frac,
        )
        if file_tau is not None:
            tau = file_tau
    elif val_data is not None:
        priors = locked.label_priors(val_data)
        if np.count_nonzero(priors) != 2:
            print("Validation data has one class; using --class_priors for logit adjustment.")
            priors = locked.parse_class_priors(args.class_priors)
    else:
        priors = locked.parse_class_priors(args.class_priors)

    logit_adjustments = locked.compute_logit_adjustment(priors, tau=tau).float()
    return logit_adjustments, priors, tau if priors_path is not None else tau


def collect_patient_ids(records):
    """Preserve patient identifiers for patient-cluster performance bootstraps."""
    patients = {}
    for index, record in enumerate(records):
        fallback = f"case_{index:05d}"
        tile_info = None
        patient_id = None
        if isinstance(record, dict):
            sample_id = str(record.get("sample_id", fallback))
            patient_id = record.get("patient_id", record.get("PATIENT_ID"))
            tile_info = record.get("tile_info")
        elif isinstance(record, (tuple, list)):
            if len(record) > 3 and isinstance(record[3], pd.DataFrame):
                tile_info = record[3]
                values = tile_info.get("SAMPLE_ID", pd.Series(dtype=str)).dropna().unique()
                sample_id = str(values[0]) if len(values) else fallback
            else:
                sample_id = str(record[4]) if len(record) > 4 and isinstance(record[4], str) else fallback
        else:
            continue
        if patient_id is None or pd.isna(patient_id) or str(patient_id).strip() == "":
            if isinstance(tile_info, pd.DataFrame) and "PATIENT_ID" in tile_info:
                values = tile_info["PATIENT_ID"].dropna().unique()
                if len(values) > 1:
                    raise ValueError(f"Multiple patient IDs for slide {sample_id}")
                patient_id = values[0] if len(values) else None
        if patient_id is None or pd.isna(patient_id) or str(patient_id).strip() == "":
            # TCGA slide barcodes contain the participant ID in the first three fields.
            patient_id = "-".join(sample_id.split("-")[:3]) if sample_id.startswith("TCGA-") else pd.NA
        if isinstance(patient_id, bytes):
            patient_id = patient_id.decode("utf-8")
        patients[sample_id] = patient_id
    return patients


def minimal_prediction_output(ensemble, patient_ids, use_logit_adjustment):
    """Keep all-fold scores and majority-vote labels used for final performance."""
    score_column = "mean_adj_prob" if use_logit_adjustment else "mean_prob"
    result = ensemble[["SAMPLE_ID", "True_y", score_column, "majority_class"]].copy()
    result.insert(1, "PATIENT_ID", result["SAMPLE_ID"].map(patient_ids))
    return result


def load_test_data(args):
    test_records = []
    if args.test_data:
        test_records.extend(locked.load_case_records(args.test_data))
    if args.feature_dir is not None:
        test_records.extend(locked.load_feature_records(args.feature_dir, args))
    eval_data, sample_ids = locked.prepare_eval_data(test_records, args, split=args.split)
    return eval_data, sample_ids, collect_patient_ids(test_records)


def print_point_performance(ensemble, use_logit_adjustment):
    """Evaluate the ensemble once, with the manuscript metric definitions."""
    score_column = "mean_adj_prob" if use_logit_adjustment else "mean_prob"
    metrics = compute_metrics(
        ensemble["True_y"], ensemble[score_column], ensemble["majority_class"],
        pr_auc_method="trapezoid",
    )
    positive = int((ensemble["True_y"] == 1).sum())
    negative = int((ensemble["True_y"] == 0).sum())
    print("\nPoint performance (slide level; no bootstrapping)")
    print(f"Slides: {len(ensemble)} | Positive: {positive} | Negative: {negative}")
    if "PATIENT_ID" in ensemble and ensemble["PATIENT_ID"].notna().all():
        print(f"Patients: {ensemble['PATIENT_ID'].nunique()}")
    labels = {
        "ROC_AUC": "AUROC", "PR_AUC": "PR-AUC (trapezoidal)",
        "Recall": "Sensitivity", "Precision": "PPV", "ACC": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
    }
    for name, value in metrics.items():
        if name in {"TP", "TN", "FP", "FN"}:
            formatted = str(int(value))
        else:
            formatted = f"{value:.4f}" if np.isfinite(value) else "N/A"
        print(f"{labels.get(name, name):<25} {formatted}")
    return metrics


def main():
    locked.build_parser = build_parser
    locked.load_saved_threshold = load_saved_threshold
    args = build_parser().parse_args()

    if not args.test_data and args.feature_dir is None:
        raise ValueError("Provide either --test_data model-ready files or --feature_dir per-slide embedding files.")
    if args.threshold_csv and not args.use_saved_threshold:
        args.use_saved_threshold = True

    folds = locked.parse_folds(args.folds)
    project_dir = locked.resolve_project_dir(args.project_dir)
    model_dir = locked.resolve_model_dir(args, project_dir)
    output_dir = locked.resolve_output_dir(args, project_dir)
    device = locked.resolve_device(args.cuda_device)

    if not model_dir.exists():
        raise FileNotFoundError(f"Locked model directory does not exist: {model_dir}")

    test_data, test_sample_ids, patient_ids = load_test_data(args)

    val_data = None
    if args.val_data:
        val_records = locked.load_case_records(args.val_data)
        val_data, _ = locked.prepare_eval_data(val_records, args, split="all")

    output_dir.mkdir(parents=True, exist_ok=True)
    n_feature = test_data[0][0].shape[1]
    all_fold_predictions = []
    fold_thresholds = []

    for fold in locked.tqdm(folds, desc="Running folds", unit="fold"):
        checkpoint_path = locked.find_checkpoint(model_dir, fold)
        print(f"FOLD{fold}: loading {checkpoint_path}")
        model = locked.load_model(checkpoint_path, args.model_name, device, n_feature=n_feature)

        logit_adjustments, priors, tau = fold_logit_adjustment(args, val_data, fold)
        if args.logit_adj_infer:
            prior_msg = priors.tolist() if priors is not None else None
            print(f"FOLD{fold}: using logit-adjustment priors {prior_msg} with tau={tau}")

        threshold = args.pred_threshold
        threshold_path = None

        if args.use_saved_threshold:
            threshold, threshold_path = load_saved_threshold(args, model_dir, fold)
            print(f"FOLD{fold}: using threshold {threshold} from {threshold_path}")
        fold_thresholds.append(threshold)
        pred_df = locked.predict_dataframe(
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
        pred_df["checkpoint"] = checkpoint_path.name
        all_fold_predictions.append(pred_df)

    all_pred = pd.concat(all_fold_predictions, ignore_index=True)

    threshold_ensemble = (
        float(np.mean(fold_thresholds))
        if args.use_saved_threshold
        else args.pred_threshold
    )
    ensemble = locked.build_majority_vote_ensemble(
        all_pred,
        use_logit_adjustment=args.logit_adj_infer,
        threshold=threshold_ensemble,
    )
    ensemble = minimal_prediction_output(ensemble, patient_ids, args.logit_adj_infer)
    ensemble.to_csv(output_dir / "ensemble_prediction.csv", index=False)

    print(f"Done. Ensemble predictions written to: {output_dir / 'ensemble_prediction.csv'}")
    metrics = print_point_performance(ensemble, args.logit_adj_infer)
    point_metrics = {
        "N_slides": len(ensemble),
        "N_patients": ensemble["PATIENT_ID"].nunique() if ensemble["PATIENT_ID"].notna().all() else pd.NA,
        "N_positive_slides": int((ensemble["True_y"] == 1).sum()),
        "N_negative_slides": int((ensemble["True_y"] == 0).sum()),
        **metrics,
    }
    pd.DataFrame([point_metrics]).to_csv(output_dir / "ensemble_performance.csv", index=False)
    print(f"Point performance written to: {output_dir / 'ensemble_performance.csv'}")


if __name__ == "__main__":
    main()
