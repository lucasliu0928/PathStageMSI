#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap performance summaries from PathStageMSI prediction CSVs.

This script is intentionally separate from inference. It reads prediction rows,
optionally merges metadata for patient IDs/cohort labels, and reports metrics as
mean [95% CI] across bootstrap samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "ROC_AUC",
    "PR_AUC",
    "Recall",
    "Precision",
    "NPV",
    "Specificity",
    "False_Positive_Rate",
    "False_Negative_Rate",
    "balanced_accuracy",
    "ACC",
    "F1",
    "F2",
    "F3",
    "MCC",
    "TP",
    "TN",
    "FP",
    "FN",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def safe_divide(num, denom):
    return np.nan if denom == 0 else num / denom


def fbeta(precision, recall, beta):
    if np.isnan(precision) or np.isnan(recall):
        return np.nan
    beta2 = beta * beta
    denom = beta2 * precision + recall
    return np.nan if denom == 0 else (1 + beta2) * precision * recall / denom


def roc_auc_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y_true == 1].sum()
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def precision_recall_points(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None, None

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    distinct = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    return precision, recall


def average_precision_binary(y_true, y_score):
    precision, recall = precision_recall_points(y_true, y_score)
    if precision is None:
        return np.nan
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def trapezoid_pr_auc_binary(y_true, y_score):
    precision, recall = precision_recall_points(y_true, y_score)
    if precision is None:
        return np.nan
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


def pr_auc_binary(y_true, y_score, method):
    if method == "average_precision":
        return average_precision_binary(y_true, y_score)
    if method == "trapezoid":
        return trapezoid_pr_auc_binary(y_true, y_score)
    raise ValueError(f"Unknown PR-AUC method: {method}")


def compute_metrics(y_true, y_prob, y_pred, pr_auc_method="average_precision"):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    recall = safe_divide(tp, tp + fn)
    precision = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    specificity = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, fp + tn)
    fnr = safe_divide(fn, fn + tp)
    acc = safe_divide(tp + tn, len(y_true))
    balanced_accuracy = np.nanmean([recall, specificity])
    f1 = fbeta(precision, recall, beta=1)
    f2 = fbeta(precision, recall, beta=2)
    f3 = fbeta(precision, recall, beta=3)

    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.nan if mcc_denom == 0 else ((tp * tn) - (fp * fn)) / mcc_denom

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_binary(y_true, y_prob)
        pr_auc = pr_auc_binary(y_true, y_prob, pr_auc_method)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Recall": recall,
        "Precision": precision,
        "NPV": npv,
        "Specificity": specificity,
        "False_Positive_Rate": fpr,
        "False_Negative_Rate": fnr,
        "balanced_accuracy": balanced_accuracy,
        "ACC": acc,
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "MCC": mcc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def choose_column(df, requested, candidates, label):
    if requested and requested.lower() != "auto":
        if requested not in df.columns:
            raise ValueError(f"{label} column not found: {requested}")
        return requested
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer {label} column. Tried: {candidates}")


def load_predictions(args):
    pred_path = Path(args.prediction_csv).expanduser().resolve()
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_path}")
    df = pd.read_csv(pred_path)

    metadata_path = None
    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv).expanduser().resolve()
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        if args.sample_id_col not in df.columns:
            raise ValueError(f"{args.sample_id_col} is missing from prediction CSV: {pred_path}")
        if args.metadata_sample_id_col not in metadata.columns:
            raise ValueError(
                f"{args.metadata_sample_id_col} is missing from metadata CSV: {metadata_path}"
            )
        metadata = metadata.drop_duplicates(subset=[args.metadata_sample_id_col])
        df = df.merge(
            metadata,
            left_on=args.sample_id_col,
            right_on=args.metadata_sample_id_col,
            how="left",
            suffixes=("", "_metadata"),
        )

    return df, pred_path, metadata_path


def apply_optional_filter(df, filter_col, filter_value):
    if not filter_col:
        return df
    if filter_col not in df.columns:
        raise ValueError(f"Filter column not found: {filter_col}")
    if filter_value is None:
        raise ValueError("--filter_value is required when --filter_col is provided.")

    series = df[filter_col]
    try:
        target = float(filter_value)
        mask = np.isclose(series.astype(float), target)
    except (TypeError, ValueError):
        mask = series.astype(str) == str(filter_value)

    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError(f"No rows remain after filtering {filter_col}={filter_value!r}.")
    return filtered


def validate_columns(df, y_true_col, prob_col, pred_col):
    for col in [y_true_col, prob_col, pred_col]:
        if col not in df.columns:
            raise ValueError(f"Required column is missing: {col}")
    if df[[y_true_col, prob_col, pred_col]].isna().any().any():
        raise ValueError("Prediction, probability, or label column contains missing values.")


def patient_unit_column(df, requested_unit, patient_id_col):
    if requested_unit == "slide":
        return None, "slide"
    if patient_id_col in df.columns and df[patient_id_col].notna().all():
        return patient_id_col, "patient_cluster"
    if requested_unit == "patient":
        raise ValueError(f"--bootstrap_unit patient requires non-missing column: {patient_id_col}")
    return None, "slide"


def patient_label_groups(df, unit_col, y_true_col):
    unit_labels = df.groupby(unit_col)[y_true_col].nunique(dropna=False)
    inconsistent = unit_labels[unit_labels > 1]
    if not inconsistent.empty:
        examples = ", ".join(map(str, inconsistent.index[:5]))
        raise ValueError(f"Patient clusters with inconsistent labels: {examples}")

    first_labels = df.groupby(unit_col)[y_true_col].first().astype(int)
    groups = {
        int(label): first_labels.index[first_labels == label].to_numpy()
        for label in sorted(first_labels.unique())
    }
    return groups


def resample_dataframe(df, rng, y_true_col, unit_col, sampling):
    if sampling == "vanilla":
        if unit_col is None:
            draw = rng.choice(df.index.to_numpy(), size=len(df), replace=True)
            return df.loc[draw].reset_index(drop=True)

        unit_ids = df[unit_col].drop_duplicates().to_numpy()
        draw_units = rng.choice(unit_ids, size=len(unit_ids), replace=True)
        pieces = [df.loc[df[unit_col] == unit_id] for unit_id in draw_units]
        return pd.concat(pieces, ignore_index=True)

    if sampling == "stratified":
        if unit_col is None:
            pieces = []
            for _label, group in df.groupby(y_true_col, sort=False):
                draw = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
                pieces.append(df.loc[draw])
            return pd.concat(pieces, ignore_index=True)

        groups = patient_label_groups(df, unit_col, y_true_col)
        pieces = []
        for unit_ids in groups.values():
            draw_units = rng.choice(unit_ids, size=len(unit_ids), replace=True)
            for unit_id in draw_units:
                pieces.append(df.loc[df[unit_col] == unit_id])
        return pd.concat(pieces, ignore_index=True)

    raise ValueError(f"Unknown bootstrap sampling mode: {sampling}")


def summarize_bootstrap(perf_df, ci):
    alpha = (100 - ci) / 2.0
    lower_q = alpha / 100.0
    upper_q = 1.0 - lower_q

    summary = {}
    for col in METRIC_COLUMNS:
        values = perf_df[col].dropna()
        if values.empty:
            summary[col] = "NA"
            continue
        mean = values.mean()
        low = values.quantile(lower_q)
        high = values.quantile(upper_q)
        summary[col] = f"{mean:.2f} [{low:.2f} - {high:.2f}]"
    return summary


def summarize_cohort(df, cohort_name, args, prob_col, pred_col):
    validate_columns(df, args.y_true_col, prob_col, pred_col)
    unit_col, bootstrap_unit = patient_unit_column(df, args.bootstrap_unit, args.patient_id_col)

    labels = df[args.y_true_col].astype(int)
    if labels.nunique() < 2:
        raise ValueError(f"Cohort {cohort_name!r} has fewer than two label classes.")

    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.n_bootstrap):
        sample = resample_dataframe(df, rng, args.y_true_col, unit_col, args.bootstrap_sampling)
        rows.append(compute_metrics(sample[args.y_true_col], sample[prob_col], sample[pred_col], args.pr_auc_method))
    perf_df = pd.DataFrame(rows)
    summary = summarize_bootstrap(perf_df, args.ci)

    n_patients = df[unit_col].nunique() if unit_col is not None else pd.NA
    result = {
        "Cohort": cohort_name,
        "N_slides": len(df),
        "N_patients": n_patients,
        "N_positive_slides": int((labels == 1).sum()),
        "N_negative_slides": int((labels == 0).sum()),
        "n_bootstrap": args.n_bootstrap,
        "ci": args.ci,
        "probability_column": prob_col,
        "prediction_column": pred_col,
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_sampling": args.bootstrap_sampling,
        "valid_auc_bootstrap": int(perf_df["ROC_AUC"].notna().sum()),
        "pr_auc_method": args.pr_auc_method,
    }
    result.update(summary)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute bootstrapped PathStageMSI performance from prediction CSVs."
    )
    parser.add_argument("--prediction_csv", required=True, help="Usually output/.../ensemble_prediction.csv")
    parser.add_argument("--output_csv", default=None, help="Output CSV. Defaults beside --prediction_csv.")
    parser.add_argument("--metadata_csv", default=None, help="Optional sample metadata/label CSV for PATIENT_ID and cohort columns.")
    parser.add_argument("--sample_id_col", default="SAMPLE_ID")
    parser.add_argument("--metadata_sample_id_col", default="SAMPLE_ID")
    parser.add_argument("--patient_id_col", default="PATIENT_ID")
    parser.add_argument("--cohort_col", default=None, help="Optional column used to summarize each cohort separately.")
    parser.add_argument("--cohort_name", default="test", help="Cohort name used when --cohort_col is omitted.")
    parser.add_argument("--filter_col", default=None, help="Optional column to filter before bootstrapping.")
    parser.add_argument("--filter_value", default=None, help="Value used with --filter_col, for example 0.0.")
    parser.add_argument("--y_true_col", default="True_y")
    parser.add_argument(
        "--prob_col",
        default="auto",
        help="Probability column. Auto prefers mean_adj_prob, adj_prob_1, mean_prob, then prob_1.",
    )
    parser.add_argument(
        "--pred_col",
        default="auto",
        help="Predicted-class column. Auto prefers majority_class, Pred_Class_adj, then Pred_Class.",
    )
    parser.add_argument("--n_bootstrap", default=1000, type=int)
    parser.add_argument(
        "--pr_auc_method",
        default="trapezoid",
        choices=["trapezoid", "average_precision"],
        help="Definition used for PR_AUC. The default matches the paper-publication workflow.",
    )
    parser.add_argument("--ci", default=95, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--bootstrap_sampling",
        default="vanilla",
        choices=["vanilla", "stratified"],
        help="vanilla matches the paper-publication workflow; stratified preserves label counts.",
    )
    parser.add_argument("--bootstrap_unit", default="auto", choices=["auto", "patient", "slide"])
    return parser


def main():
    args = build_parser().parse_args()
    df, pred_path, metadata_path = load_predictions(args)
    df = apply_optional_filter(df, args.filter_col, args.filter_value)

    prob_col = choose_column(
        df,
        args.prob_col,
        candidates=["mean_adj_prob", "ensemble_mean_adj_prob", "adj_prob_1", "mean_prob", "prob_1"],
        label="probability",
    )
    pred_col = choose_column(
        df,
        args.pred_col,
        candidates=["majority_class", "ensemble_majority_class", "Pred_Class_adj", "Pred_Class"],
        label="prediction",
    )

    if args.cohort_col:
        if args.cohort_col not in df.columns:
            raise ValueError(f"Cohort column not found: {args.cohort_col}")
        results = [
            summarize_cohort(group.copy(), str(cohort), args, prob_col, pred_col)
            for cohort, group in df.groupby(args.cohort_col, sort=False)
        ]
    else:
        results = [summarize_cohort(df.copy(), args.cohort_name, args, prob_col, pred_col)]

    out_df = pd.DataFrame(results)
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else pred_path.with_name("bootstrap_performance.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"Prediction CSV: {pred_path}")
    if metadata_path is not None:
        print(f"Metadata CSV: {metadata_path}")
    print(f"Probability column: {prob_col}")
    print(f"Prediction column: {pred_col}")
    print(f"Bootstrap performance written to: {output_csv}")


if __name__ == "__main__":
    main()
