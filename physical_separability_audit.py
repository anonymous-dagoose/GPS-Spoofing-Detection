"""Reproducible audit of GPS spoofing feature separability.

This script measures separability in the supplied feature map. It does not
establish a receiver-independent physical limit: the attacks are simulated,
recording-window IDs are inferred from RX gaps, and no flight/session metadata
are present in the spreadsheet.

Example:
    python physical_separability_audit.py --data GPS_Data_Simplified_2D_Feature_Map.xlsx \
        --out audit_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier


PHYSICAL = ["DO", "CP", "EC", "LC", "PC", "PIP", "PQP", "TCD", "CN0"]
CONTEXT = ["PD", "RX", "TOW", "PRN"]
ALL = ["PRN", "DO", "PD", "RX", "TOW", "CP", "EC", "LC", "PC", "PIP", "PQP", "TCD", "CN0"]
FEATURE_SETS = {"all_13": ALL, "physical_9": PHYSICAL, "context_4": CONTEXT}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Original 14-column Excel feature map")
    parser.add_argument("--out", type=Path, required=True, help="Directory for CSV/JSON results")
    parser.add_argument("--gap-seconds", type=float, default=60.0,
                        help="RX gap used to infer recording windows; these are not verified sessions")
    parser.add_argument("--target-val-fpr", type=float, default=0.05,
                        help="Threshold chosen from legitimate validation scores only")
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--bootstrap-block-seconds", type=float, default=10.0)
    parser.add_argument("--max-trees", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--core-only", action="store_true",
                        help="Run data audit and matched XGBoost ablations; omit additional models")
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    required = set(ALL + ["Output"])
    if set(df.columns) != required or len(df.columns) != 14:
        raise ValueError(f"Expected exactly 13 features and Output, received {df.columns.tolist()}")
    if df[list(required)].isna().any().any():
        raise ValueError("Missing feature or label values require an explicit preprocessing protocol")
    if not set(df["Output"].unique()).issubset({0, 1, 2, 3}):
        raise ValueError("Expected labels 0 (legitimate) and 1-3 (simulated attacks)")
    df["binary"] = (df["Output"] != 0).astype(np.int8)
    return df


def ambiguity(df: pd.DataFrame, features: list[str]) -> dict:
    """Best possible in-sample accuracy of a deterministic one-row classifier.

    Identical feature vectors with opposite binary labels force at least the
    minority count of each vector to be wrong. This is a representation/label
    ambiguity bound on this table, not a bound on GNSS spoofing in nature.
    """
    counts = df.groupby(features + ["binary"], sort=False, dropna=False).size().unstack(fill_value=0)
    for label in (0, 1):
        if label not in counts:
            counts[label] = 0
    conflict = counts[(counts[0] > 0) & (counts[1] > 0)]
    errors = int(conflict[[0, 1]].min(axis=1).sum())
    return {
        "feature_set": f"{len(features)} features",
        "unique_feature_vectors": int(len(counts)),
        "conflicting_vectors": int(len(conflict)),
        "rows_with_conflicting_labels": int(conflict[[0, 1]].sum().sum()),
        "minimum_in_sample_errors": errors,
        "empirical_accuracy_ceiling": 1 - errors / len(df),
        "scope": "exact values in supplied table; one-row deterministic binary classifier",
    }


def ambiguity_by_window(df: pd.DataFrame, windows: np.ndarray) -> pd.DataFrame:
    rows = []
    for window in np.unique(windows):
        subset = df.iloc[np.flatnonzero(windows == window)]
        for name, columns in (("physical_9", PHYSICAL), ("all_13", ALL)):
            result = ambiguity(subset, columns)
            rows.append({**result, "window": int(window), "feature_set": name,
                         "n": len(subset)})
    return pd.DataFrame(rows)


def recording_windows(rx: np.ndarray, gap_seconds: float) -> tuple[np.ndarray, pd.DataFrame]:
    unique_rx = np.unique(rx)
    jumps = np.diff(unique_rx)
    gap_positions = np.flatnonzero(jumps > gap_seconds)
    edges = (unique_rx[gap_positions] + unique_rx[gap_positions + 1]) / 2
    groups = np.searchsorted(edges, rx).astype(np.int8)
    gaps = pd.DataFrame({
        "rx_before": unique_rx[gap_positions],
        "rx_after": unique_rx[gap_positions + 1],
        "gap_seconds": jumps[gap_positions],
    })
    return groups, gaps


def make_splits(y: np.ndarray, rx: np.ndarray, windows: np.ndarray, seed: int) -> dict:
    indices = np.arange(len(y))
    tv, random_test = train_test_split(indices, test_size=0.20, stratify=y, random_state=seed)
    random_train, random_val = train_test_split(
        tv, test_size=0.25, stratify=y[tv], random_state=seed
    )

    order = np.argsort(rx, kind="stable")
    sorted_rx = rx[order]
    # Keep all rows at an identical RX timestamp on the same side of each cut.
    cut1 = np.searchsorted(sorted_rx, sorted_rx[int(0.60 * len(y))], side="left")
    cut2 = np.searchsorted(sorted_rx, sorted_rx[int(0.80 * len(y))], side="left")
    if len(np.unique(windows)) != 4:
        raise ValueError(
            "This dataset-specific window protocol expects four RX-gap windows; "
            "inspect gap_inventory.csv and choose a suitable gap threshold."
        )
    protocols = {
        "random_rows": (random_train, random_val, random_test),
        "forward_rows": (order[:cut1], order[cut1:cut2], order[cut2:]),
        # Candidate recording windows, inferred from RX gaps rather than metadata.
        "forward_windows_early": (
            indices[windows == 0], indices[windows == 1], indices[windows == 2]
        ),
        "forward_windows": (
            indices[windows <= 1], indices[windows == 2], indices[windows == 3]
        ),
    }
    for name, parts in protocols.items():
        if any(len(np.unique(y[p])) != 2 for p in parts):
            raise ValueError(f"Both classes are needed in every {name} partition")
        selected = np.concatenate(parts)
        if len(np.unique(selected)) != len(selected):
            raise AssertionError(f"Partitions overlap in {name}")
        if name != "forward_windows_early" and len(selected) != len(y):
            raise AssertionError(f"Partitions omit rows in {name}")
    return protocols


def split_inventory(y4: np.ndarray, rx: np.ndarray, windows: np.ndarray, protocols: dict) -> pd.DataFrame:
    records = []
    for name, parts in protocols.items():
        for part_name, idx in zip(("train", "validation", "test"), parts):
            counts = np.bincount(y4[idx], minlength=4)
            records.append({
                "protocol": name, "part": part_name, "n": len(idx),
                "legitimate": int(counts[0]), "simplistic": int(counts[1]),
                "intermediate": int(counts[2]), "sophisticated": int(counts[3]),
                "attack": int(counts[1:].sum()),
                "attack_prevalence": float(counts[1:].sum() / len(idx)),
                "rx_min": float(rx[idx].min()), "rx_max": float(rx[idx].max()),
                "recording_windows": ",".join(map(str, np.unique(windows[idx]))),
            })
    return pd.DataFrame(records)


def xgb_model(y_train: np.ndarray, seed: int, max_trees: int, jobs: int) -> XGBClassifier:
    weight = float((y_train == 0).sum() / (y_train == 1).sum())
    return XGBClassifier(
        n_estimators=max_trees, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=weight, early_stopping_rounds=30,
        tree_method="hist", random_state=seed, n_jobs=jobs,
    )


def threshold_at_val_fpr(score_val: np.ndarray, y_val: np.ndarray, fpr: float) -> float:
    if not 0 < fpr < 1:
        raise ValueError("target validation FPR must be strictly between 0 and 1")
    return float(np.quantile(score_val[y_val == 0], 1 - fpr))


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "n_test": len(y), "n_legitimate": int((y == 0).sum()),
        "n_attack": int((y == 1).sum()), "attack_prevalence": float(y.mean()),
        "auroc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "ap_baseline": float(y.mean()), "threshold": threshold,
        "recall": float(recall_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def block_bootstrap_auc(y: np.ndarray, scores: np.ndarray, rx: np.ndarray,
                        block_seconds: float, repetitions: int, seed: int) -> dict:
    """Sensitivity interval within the observed period, not across new sessions."""
    if repetitions <= 0:
        return {"auc_lo": np.nan, "auc_hi": np.nan, "bootstrap_blocks": 0}
    group = np.floor(rx / block_seconds).astype(np.int64)
    _, codes = np.unique(group, return_inverse=True)
    blocks = [np.flatnonzero(codes == b) for b in range(codes.max() + 1)]
    if len(blocks) < 5:
        return {"auc_lo": np.nan, "auc_hi": np.nan, "bootstrap_blocks": len(blocks)}
    rng = np.random.default_rng(seed)
    sampled_aucs = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[b] for b in selected])
        if len(np.unique(y[idx])) == 2:
            sampled_aucs.append(roc_auc_score(y[idx], scores[idx]))
    if len(sampled_aucs) < repetitions // 2:
        raise ValueError("Too many block resamples have only one class")
    lo, hi = np.quantile(sampled_aucs, [0.025, 0.975])
    return {"auc_lo": float(lo), "auc_hi": float(hi), "bootstrap_blocks": len(blocks)}


def evaluate_scores(name: str, model_name: str, feature_set: str, seed: int,
                    y: np.ndarray, rx: np.ndarray, val_idx: np.ndarray,
                    test_idx: np.ndarray, val_scores: np.ndarray,
                    test_scores: np.ndarray, args: argparse.Namespace,
                    extra: dict | None = None) -> dict:
    threshold = threshold_at_val_fpr(val_scores, y[val_idx], args.target_val_fpr)
    result = {
        "protocol": name, "model": model_name, "features": feature_set, "seed": seed,
        **metrics(y[test_idx], test_scores, threshold),
    }
    if name != "random_rows":
        result.update(block_bootstrap_auc(
            y[test_idx], test_scores, rx[test_idx],
            args.bootstrap_block_seconds, args.bootstrap_reps, seed + 10000
        ))
    if extra:
        result.update(extra)
    print(f"{name:19s} {model_name:15s} {feature_set:12s} "
          f"AUROC={result['auroc']:.4f} AP={result['average_precision']:.4f} "
          f"recall={result['recall']:.4f} FPR={result['fpr']:.4f}", flush=True)
    return result


def run_ablation(df: pd.DataFrame, y: np.ndarray, rx: np.ndarray,
                 protocols: dict, args: argparse.Namespace, results: list[dict]) -> None:
    for protocol, (train, val, test) in protocols.items():
        for feature_set, columns in FEATURE_SETS.items():
            X = df[columns].to_numpy(dtype=np.float32)
            model = xgb_model(y[train], args.seed, args.max_trees, args.jobs)
            model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
            val_scores = model.predict_proba(X[val])[:, 1]
            test_scores = model.predict_proba(X[test])[:, 1]
            results.append(evaluate_scores(
                protocol, "XGBoost", feature_set, args.seed, y, rx,
                val, test, val_scores, test_scores, args,
                {"best_iteration": int(model.best_iteration)}
            ))
            pd.DataFrame(results).to_csv(args.out / "model_results.csv", index=False)
            if feature_set == "physical_9" and protocol != "random_rows":
                pd.DataFrame({
                    "row_index": test, "RX": rx[test], "label": y[test],
                    "score": test_scores,
                }).to_csv(args.out / f"scores_{protocol}_physical_9.csv", index=False)


def run_logistic(df: pd.DataFrame, y: np.ndarray, rx: np.ndarray,
                 protocols: dict, args: argparse.Namespace, results: list[dict]) -> None:
    X = df[PHYSICAL].to_numpy(dtype=np.float32)
    for protocol in ("forward_rows", "forward_windows_early", "forward_windows"):
        train, val, test = protocols[protocol]
        model = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=args.seed
        ))
        model.fit(X[train], y[train])
        results.append(evaluate_scores(
            protocol, "logistic", "physical_9", args.seed, y, rx,
            val, test, model.predict_proba(X[val])[:, 1],
            model.predict_proba(X[test])[:, 1], args,
        ))
        pd.DataFrame(results).to_csv(args.out / "model_results.csv", index=False)


def run_anomaly_baselines(df: pd.DataFrame, y: np.ndarray, rx: np.ndarray,
                          protocols: dict, args: argparse.Namespace,
                          results: list[dict]) -> None:
    train, val, test = protocols["forward_windows"]
    X = df[PHYSICAL].to_numpy(dtype=np.float32)
    legit_train = train[y[train] == 0]
    rng = np.random.default_rng(args.seed)

    # IF's contamination setting only shifts its built-in threshold. We use
    # continuous scores and choose one common threshold from validation normals.
    if_idx = rng.choice(legit_train, size=min(50_000, len(legit_train)), replace=False)
    forest = IsolationForest(n_estimators=200, max_samples="auto",
                             contamination="auto", random_state=args.seed, n_jobs=args.jobs)
    forest.fit(X[if_idx])
    results.append(evaluate_scores(
        "forward_windows", "isolation_forest", "physical_9", args.seed, y, rx,
        val, test, -forest.score_samples(X[val]), -forest.score_samples(X[test]),
        args, {"n_train_legitimate": len(if_idx), "n_estimators": 200},
    ))
    pd.DataFrame(results).to_csv(args.out / "model_results.csv", index=False)

    scaler = StandardScaler().fit(X[legit_train])
    X_val_scaled, X_test_scaled = scaler.transform(X[val]), scaler.transform(X[test])
    svm_idx = rng.choice(legit_train, size=min(8_000, len(legit_train)), replace=False)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(scaler.transform(X[svm_idx]))
    results.append(evaluate_scores(
        "forward_windows", "one_class_svm", "physical_9", args.seed, y, rx,
        val, test, -svm.decision_function(X_val_scaled).ravel(),
        -svm.decision_function(X_test_scaled).ravel(), args,
        {"n_train_legitimate": len(svm_idx), "kernel": "rbf", "gamma": "scale", "nu": 0.05},
    ))
    pd.DataFrame(results).to_csv(args.out / "model_results.csv", index=False)

    gmm_idx = rng.choice(legit_train, size=min(25_000, len(legit_train)), replace=False)
    X_gmm = scaler.transform(X[gmm_idx])
    candidates = []
    for k in (8, 32, 96, 128):
        gmm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-6,
                              n_init=2, max_iter=200, random_state=args.seed)
        gmm.fit(X_gmm)
        candidates.append((gmm.bic(X_gmm), k, gmm))
    bic, k, best = min(candidates, key=lambda item: item[0])
    results.append(evaluate_scores(
        "forward_windows", "GMM", "physical_9", args.seed, y, rx,
        val, test, -best.score_samples(X_val_scaled),
        -best.score_samples(X_test_scaled), args,
        {"n_train_legitimate": len(gmm_idx), "gmm_k": k, "train_bic": float(bic)},
    ))
    pd.DataFrame(results).to_csv(args.out / "model_results.csv", index=False)


def run_leave_attack_out(df: pd.DataFrame, y4: np.ndarray, y: np.ndarray,
                         rx: np.ndarray, protocols: dict,
                         args: argparse.Namespace, results: list[dict]) -> None:
    base_train, base_val, base_test = protocols["forward_windows"]
    X = df[PHYSICAL].to_numpy(dtype=np.float32)
    for held_out in (1, 2, 3):
        # The entire legitimate test period is disjoint from training and validation.
        train = base_train[y4[base_train] != held_out]
        val = base_val[y4[base_val] != held_out]
        unseen = base_test[(y4[base_test] == 0) | (y4[base_test] == held_out)]
        known = base_test[(y4[base_test] == 0) |
                          ((y4[base_test] != held_out) & (y4[base_test] != 0))]
        model = xgb_model(y[train], args.seed, args.max_trees, args.jobs)
        model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
        val_scores = model.predict_proba(X[val])[:, 1]
        for test_type, test in (("unseen", unseen), ("known_control", known)):
            results.append(evaluate_scores(
                "forward_windows", "XGBoost_LOAO", f"held_{held_out}_{test_type}",
                args.seed, y, rx, val, test, val_scores,
                model.predict_proba(X[test])[:, 1], args,
                {"held_out_attack": held_out, "test_type": test_type,
                 "best_iteration": int(model.best_iteration)},
            ))
        pd.DataFrame(results).to_csv(args.out / "model_results.csv", index=False)


def run_seed_checks(df: pd.DataFrame, y: np.ndarray, rx: np.ndarray,
                    protocols: dict, args: argparse.Namespace,
                    results: list[dict]) -> None:
    train, val, test = protocols["forward_windows"]
    X = df[PHYSICAL].to_numpy(dtype=np.float32)
    for seed in (args.seed + 1, args.seed + 2):
        model = xgb_model(y[train], seed, args.max_trees, args.jobs)
        model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
        results.append(evaluate_scores(
            "forward_windows_seed_check", "XGBoost", "physical_9", seed,
            y, rx, val, test, model.predict_proba(X[val])[:, 1],
            model.predict_proba(X[test])[:, 1], args,
            {"best_iteration": int(model.best_iteration)},
        ))
        pd.DataFrame(results).to_csv(args.out / "model_results.csv", index=False)


def run_train_only_selection(df: pd.DataFrame, y: np.ndarray, rx: np.ndarray,
                             protocols: dict, args: argparse.Namespace,
                             results: list[dict]) -> None:
    """Sensitivity check: choose tree count using only the training period.

    An internal random split may favor the training distribution, but the
    external validation and test periods remain disjoint and chronological.
    This checks whether near-chance transfer is just early stopping at tree 0.
    """
    X = df[PHYSICAL].to_numpy(dtype=np.float32)
    for protocol in ("forward_rows", "forward_windows_early", "forward_windows"):
        train, val, test = protocols[protocol]
        fit_idx, internal_val = train_test_split(
            train, test_size=0.20, stratify=y[train], random_state=args.seed
        )
        model = xgb_model(y[fit_idx], args.seed, args.max_trees, args.jobs)
        model.fit(X[fit_idx], y[fit_idx],
                  eval_set=[(X[internal_val], y[internal_val])], verbose=False)
        results.append(evaluate_scores(
            protocol, "XGBoost_train_tuned", "physical_9", args.seed,
            y, rx, val, test, model.predict_proba(X[val])[:, 1],
            model.predict_proba(X[test])[:, 1], args,
            {"best_iteration": int(model.best_iteration)},
        ))
        pd.DataFrame(results).to_csv(args.out / "model_results.csv", index=False)


def main() -> None:
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    y4 = df["Output"].to_numpy(dtype=np.int8)
    y = df["binary"].to_numpy(dtype=np.int8)
    rx = df["RX"].to_numpy(dtype=np.float64)
    windows, gaps = recording_windows(rx, args.gap_seconds)
    gaps.to_csv(args.out / "gap_inventory.csv", index=False)
    protocols = make_splits(y, rx, windows, args.seed)
    split_inventory(y4, rx, windows, protocols).to_csv(args.out / "split_inventory.csv", index=False)
    with (args.out / "ambiguity.json").open("w", encoding="utf-8") as out:
        json.dump({"physical_9": ambiguity(df, PHYSICAL),
                   "all_13": ambiguity(df, ALL)}, out, indent=2)
    ambiguity_by_window(df, windows).to_csv(args.out / "ambiguity_by_window.csv", index=False)
    with (args.out / "config.json").open("w", encoding="utf-8") as out:
        json.dump({
            "data_file": str(args.data.resolve()), "n_rows": len(df),
            "gap_seconds": args.gap_seconds,
            "gap_windows_are_verified_sessions": False,
            "target_validation_fpr": args.target_val_fpr,
            "bootstrap_block_seconds": args.bootstrap_block_seconds,
            "bootstrap_repetitions": args.bootstrap_reps,
            "bootstrap_interpretation": "within-period sensitivity, not across-session CI",
            "max_trees": args.max_trees, "seed": args.seed,
            "model_selection": "external validation early stopping; additional nine-feature XGBoost uses training-only internal validation",
            "gmm_component_candidates": [8, 32, 96, 128],
            "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                         "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
            "feature_sets": FEATURE_SETS,
        }, out, indent=2)

    print("Candidate RX-gap recording windows:", len(np.unique(windows)), flush=True)
    print((args.out / "split_inventory.csv").read_text(encoding="utf-8"), flush=True)
    results: list[dict] = []
    run_ablation(df, y, rx, protocols, args, results)
    if not args.core_only:
        run_train_only_selection(df, y, rx, protocols, args, results)
        run_logistic(df, y, rx, protocols, args, results)
        run_anomaly_baselines(df, y, rx, protocols, args, results)
        run_leave_attack_out(df, y4, y, rx, protocols, args, results)
        run_seed_checks(df, y, rx, protocols, args, results)
    print(f"Saved {len(results)} model rows to {args.out / 'model_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
