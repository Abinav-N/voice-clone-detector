"""
Task 12: Classical Baseline
---------------------------
Train an SVM or Logistic Regression on mean MFCCs extracted from the training
manifest and evaluate on a test/validation manifest. Produces a comparison table
so the report can clearly show that the CNN beats simple linear classifiers.

Usage:
    python src/baseline.py \\
        --train  manifests/train.csv \\
        --test   manifests/eval.csv  \\
        --model  svm                 \\
        --feature mfcc               \\
        --out    results/baseline_svm.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Universal import hardening
_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from features import load_audio, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, WIN_LENGTH
    from metrics import compute_eer
except ImportError:
    from src.features import load_audio, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, WIN_LENGTH
    from src.metrics import compute_eer


# --------------------------------------------------------------------------- #
#  Feature extraction                                                         #
# --------------------------------------------------------------------------- #

def extract_mean_features(path: str, feature_type: str = "mfcc") -> np.ndarray:
    """
    Load one audio clip and return a 1-D mean+std feature vector.
    - MFCC   -> shape (2*N_MFCC,)
    - logmel -> shape (2*N_MFCC,)
    """
    import librosa

    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [WARNING] Could not load '{path}': {e}. Returning zeros.")
        return np.zeros(N_MFCC * 2, dtype=np.float32)

    if len(y) < WIN_LENGTH:
        y = np.pad(y, (0, WIN_LENGTH - len(y)))

    if feature_type == "logmel":
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MFCC, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        )
        feat = librosa.power_to_db(mel, ref=np.max)  # (N_MFCC, T)
    else:
        feat = librosa.feature.mfcc(
            y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        )  # (N_MFCC, T)

    mean_vec = feat.mean(axis=1)
    std_vec = feat.std(axis=1)
    return np.concatenate([mean_vec, std_vec], axis=0).astype(np.float32)


def build_feature_matrix(
    manifest_path: str,
    feature_type: str = "mfcc",
    max_samples: int = None,
    cache_path: str = None,
):
    """Extract mean features for all clips in a manifest CSV.

    Returns X (n_samples, 2*N_MFCC) and y (n_samples,) label array.
    Caches the result to a .npz file when cache_path is provided.
    """
    df = pd.read_csv(manifest_path)
    if "filepath" not in df.columns or "label_idx" not in df.columns:
        raise ValueError(
            f"Manifest '{manifest_path}' must have 'filepath' and 'label_idx' columns."
        )

    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    # Load from cache if available
    if cache_path and os.path.isfile(cache_path):
        print(f"  Loading cached baseline features from '{cache_path}' ...")
        data = np.load(cache_path)
        return data["X"], data["y"]

    print(f"  Extracting {feature_type} mean features for {len(df)} clips ...")
    X_list, y_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Baseline ({feature_type})"):
        vec = extract_mean_features(row["filepath"], feature_type=feature_type)
        X_list.append(vec)
        y_list.append(int(row["label_idx"]))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"  Cached to '{cache_path}'")

    return X, y


# --------------------------------------------------------------------------- #
#  Classifiers                                                                 #
# --------------------------------------------------------------------------- #

def build_classifier(model_name: str, seed: int = 42):
    """Return an sklearn pipeline: StandardScaler -> classifier."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()

    if model_name == "svm":
        clf = SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=seed, max_iter=5000
        )
    elif model_name == "logreg":
        clf = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            multi_class="auto", random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'svm' or 'logreg'.")

    return Pipeline([("scaler", scaler), ("clf", clf)])


# --------------------------------------------------------------------------- #
#  Evaluation helpers                                                          #
# --------------------------------------------------------------------------- #

def evaluate_classifier(pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute EER, AUC, accuracy and confusion matrix for a fitted pipeline."""
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

    y_proba = pipeline.predict_proba(X_test)[:, 1]   # P(spoof)
    y_pred = pipeline.predict(X_test)

    eer, eer_thresh = compute_eer(y_test, y_proba)

    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        auc = None

    acc = float((y_pred == y_test).mean())
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=["bonafide", "spoof"],
        output_dict=True, zero_division=0,
    )

    return {
        "eer": float(eer),
        "eer_pct": float(eer * 100),
        "eer_threshold": float(eer_thresh),
        "auc": auc,
        "accuracy": acc,
        "accuracy_pct": float(acc * 100),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# --------------------------------------------------------------------------- #
#  Main entry point                                                            #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Classical baseline: SVM or LogReg on mean MFCCs."
    )
    ap.add_argument("--train", required=True, help="Training manifest CSV")
    ap.add_argument("--test", required=True, help="Test/eval manifest CSV")
    ap.add_argument("--val", default=None, help="Optional validation manifest CSV")
    ap.add_argument(
        "--model", choices=["svm", "logreg", "both"], default="both",
        help="Classifier to train. 'both' trains SVM and LogReg and prints a comparison table."
    )
    ap.add_argument(
        "--feature", choices=["mfcc", "logmel"], default="mfcc",
        help="Feature type for mean-vector extraction."
    )
    ap.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of training samples (for smoke tests)."
    )
    ap.add_argument(
        "--out", default=None,
        help="Path to save results JSON (e.g. results/baseline_svm.json)."
    )
    ap.add_argument(
        "--cache-dir", default=".cache/baseline",
        help="Directory to cache extracted feature matrices."
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # ----------------------------------------------------------------------- #
    #  Extract features                                                        #
    # ----------------------------------------------------------------------- #
    os.makedirs(args.cache_dir, exist_ok=True)
    train_stem = Path(args.train).stem
    test_stem = Path(args.test).stem

    print(f"\n[Baseline] Feature: {args.feature} | Model: {args.model}")
    print("=" * 60)

    print(f"\n-> Building TRAIN feature matrix from: {args.train}")
    train_cache = os.path.join(args.cache_dir, f"train_{train_stem}_{args.feature}.npz")
    X_train, y_train = build_feature_matrix(
        args.train, args.feature, args.max_samples, cache_path=train_cache
    )

    print(f"\n-> Building TEST feature matrix from: {args.test}")
    test_cache = os.path.join(args.cache_dir, f"test_{test_stem}_{args.feature}.npz")
    X_test, y_test = build_feature_matrix(
        args.test, args.feature, cache_path=test_cache
    )

    X_val, y_val = None, None
    if args.val:
        val_stem = Path(args.val).stem
        print(f"\n-> Building VAL feature matrix from: {args.val}")
        val_cache = os.path.join(args.cache_dir, f"val_{val_stem}_{args.feature}.npz")
        X_val, y_val = build_feature_matrix(
            args.val, args.feature, cache_path=val_cache
        )

    print(f"\n  Train: {X_train.shape}, Test: {X_test.shape}")

    # ----------------------------------------------------------------------- #
    #  Train and evaluate classifiers                                          #
    # ----------------------------------------------------------------------- #
    models_to_run = ["svm", "logreg"] if args.model == "both" else [args.model]
    all_results = {}

    for m_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"  Training: {m_name.upper()} ({args.feature} features)")
        print(f"{'='*60}")

        pipeline = build_classifier(m_name, seed=args.seed)

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Fit time: {train_time:.1f}s")

        print(f"  Evaluating on TEST set ({len(y_test)} samples) ...")
        test_metrics = evaluate_classifier(pipeline, X_test, y_test)

        val_metrics = None
        if X_val is not None:
            print(f"  Evaluating on VAL set ({len(y_val)} samples) ...")
            val_metrics = evaluate_classifier(pipeline, X_val, y_val)

        all_results[m_name] = {
            "model": m_name,
            "feature_type": args.feature,
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_time_sec": float(train_time),
            "test": test_metrics,
            "val": val_metrics,
        }

        print(f"\n  +-- TEST RESULTS: {m_name.upper()}")
        print(f"  |  EER      : {test_metrics['eer_pct']:.2f}%  (threshold = {test_metrics['eer_threshold']:.3f})")
        auc_str = f"{test_metrics['auc']:.4f}" if test_metrics["auc"] is not None else "N/A"
        print(f"  |  ROC AUC  : {auc_str}")
        print(f"  |  Accuracy : {test_metrics['accuracy_pct']:.2f}%")
        print(f"  +----------------------------------------------")

        if val_metrics:
            print(f"\n  +-- VAL RESULTS: {m_name.upper()}")
            print(f"  |  EER      : {val_metrics['eer_pct']:.2f}%")
            auc_str_v = f"{val_metrics['auc']:.4f}" if val_metrics["auc"] is not None else "N/A"
            print(f"  |  ROC AUC  : {auc_str_v}")
            print(f"  |  Accuracy : {val_metrics['accuracy_pct']:.2f}%")
            print(f"  +----------------------------------------------")

    # ----------------------------------------------------------------------- #
    #  Comparison table                                                        #
    # ----------------------------------------------------------------------- #
    print(f"\n{'='*60}")
    print("  CLASSICAL BASELINE -- COMPARISON TABLE (Test Set)")
    print(f"{'='*60}")
    print(f"  {'Model':<12} {'Feature':<8} {'EER (%)':<10} {'AUC':<8} {'Accuracy (%)'}")
    print(f"  {'-'*54}")
    for m_name, res in all_results.items():
        t = res["test"]
        auc_d = f"{t['auc']:.4f}" if t["auc"] is not None else "N/A  "
        print(
            f"  {m_name.upper():<12} {res['feature_type']:<8} "
            f"{t['eer_pct']:<10.2f} {auc_d:<8} {t['accuracy_pct']:.2f}"
        )
    print(f"  {'='*54}\n")

    # ----------------------------------------------------------------------- #
    #  Save JSON results                                                       #
    # ----------------------------------------------------------------------- #
    if args.out:
        out_dir = os.path.dirname(args.out) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to: {args.out}")


if __name__ == "__main__":
    main()
