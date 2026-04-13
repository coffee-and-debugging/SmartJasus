"""
train.py — Model training entry point for CatchFish.

Loads data via preprocessing.py, engineers features via features.py,
trains a Logistic Regression classifier, and saves the result to
models/phishing_detection.pkl.

Usage:
    python train.py          # uses default threshold 0.60
    python train.py 0.55     # custom threshold
"""

import json
import os
import warnings

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from config import MODEL_PATH, MODELS_DIR
from features import build_preprocessor, extract_features
from preprocessing import discover_csv_files, load_dataset


# ── Model definition ───────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])


# ── Training ───────────────────────────────────────────────────────────────────

def train(threshold: float = 0.60) -> dict:
    """
    Full training run:
      1. Load & merge all CSVs from dataset/
      2. Engineer 19 numeric features
      3. 70 / 15 / 15 stratified split
      4. Fit Logistic Regression on train split
      5. Report validation metrics
      6. 5-fold CV AUC on train+val (overfitting check)
      7. Final metrics on held-out test split
      8. Save pipeline + metadata to models/phishing_detection.pkl
    """

    # ── Load data ──────────────────────────────────────────────────────────────
    df, per_dataset = load_dataset()
    df = extract_features(df)

    X = df.drop("label", axis=1)
    y = df["label"]

    n_phish = int(y.sum())
    n_legit = int((y == 0).sum())
    print(f"\n[Train] {len(df):,} rows  |  phishing={n_phish:,}  legit={n_legit:,}")

    # ── Split ──────────────────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.15 / 0.85,
        random_state=42, stratify=y_temp,
    )
    print(f"[Train] train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}\n")

    # ── Fit ────────────────────────────────────────────────────────────────────
    pipeline = build_pipeline()
    print("[Train] Fitting Logistic Regression …")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pipeline.fit(X_train, y_train)

    # ── Validation metrics ─────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    y_val_pred = (y_val_proba >= threshold).astype(int)
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_ap  = average_precision_score(y_val, y_val_proba)
    val_f1  = f1_score(y_val, y_val_pred, zero_division=0)
    _, fp_v, fn_v, _ = confusion_matrix(y_val, y_val_pred).ravel()
    print(
        f"[Train] VAL  AUC={val_auc:.4f}  AP={val_ap:.4f}"
        f"  F1@{threshold}={val_f1:.4f}  FP={fp_v}  FN={fn_v}"
    )

    # ── 5-fold CV AUC (overfitting check) ──────────────────────────────────────
    print("\n[Train] 5-fold CV AUC check …")
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_scores = cross_val_score(
            build_pipeline(), X_tv, y_tv, cv=5, scoring="roc_auc", n_jobs=-1
        )
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
    gap = abs(cv_mean - val_auc)
    status = (
        "✓ no overfitting" if gap < 0.01 else
        "⚠ minor gap"      if gap < 0.03 else
        "✗ significant gap"
    )
    print(f"  scores : {cv_scores.round(4)}")
    print(f"  mean   : {cv_mean:.4f} ± {cv_std:.4f}  |  val AUC={val_auc:.4f}  |  gap={gap:.4f}  {status}")

    # ── Test metrics ───────────────────────────────────────────────────────────
    print("\n[Train] Final metrics on held-out TEST set …")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        y_proba   = pipeline.predict_proba(X_test)[:, 1]
        y_pred_50 = pipeline.predict(X_test)

    y_pred_th = (y_proba >= threshold).astype(int)
    auc   = roc_auc_score(y_test, y_proba)
    ap    = average_precision_score(y_test, y_proba)
    acc   = accuracy_score(y_test, y_pred_th)
    prec  = precision_score(y_test, y_pred_th, zero_division=0)
    rec   = recall_score(y_test, y_pred_th, zero_division=0)
    f1_50 = f1_score(y_test, y_pred_50, zero_division=0)
    f1_th = f1_score(y_test, y_pred_th, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_th).ravel()

    print(f"  AUC={auc:.4f}  PR-AUC={ap:.4f}  Accuracy={acc:.4f}")
    print(f"  Precision={prec:.4f}  Recall={rec:.4f}  F1@0.5={f1_50:.4f}  F1@{threshold}={f1_th:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"\n{classification_report(y_test, y_pred_th, target_names=['legitimate', 'phishing'])}")
    print(f"  CV AUC : {cv_mean:.4f} ± {cv_std:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    meta = {
        "model_name":        "Logistic Regression",
        "threshold":         threshold,
        "f1":                round(float(f1_th), 4),
        "auc":               round(float(auc),   4),
        "average_precision": round(float(ap),    4),
        "cv_auc_mean":       round(cv_mean, 4),
        "cv_auc_std":        round(cv_std,  4),
        "split":             "70/15/15 train/val/test",
        "datasets_used":     discover_csv_files(),
        "total_rows":        len(df),
        "phishing_rows":     n_phish,
        "legitimate_rows":   n_legit,
        "per_dataset":       per_dataset,
    }
    joblib.dump({"pipeline": pipeline, "meta": meta}, MODEL_PATH, compress=3)
    print(f"\n[Train] Saved → {MODEL_PATH}")
    print(json.dumps({k: v for k, v in meta.items() if k != "per_dataset"}, indent=2))

    return meta


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
    train(threshold=threshold)
