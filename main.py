"""
Heart Disease Identification using Machine Learning
====================================================

End-to-end pipeline:

1. Load the Cleveland heart disease dataset (297 samples, 13 features)
2. Exploratory data analysis (statistics, histograms, correlation heat-map)
3. Standard scaling and stratified 70/30 train-test split
4. Feature selection with ReliefF, mRMR, LASSO, LLBFS, FCMIM and the
   proposed EEMIFS algorithm
5. Train six classifiers on the top-k features of every selector
6. Evaluate (accuracy, precision, sensitivity, specificity, F1, MCC) on a
   held-out test set, plus stratified 10-fold cross-validation (feature
   selection repeated inside every fold to avoid selection bias)
7. Save tables and figures to ``results/``

Usage::

    python main.py                 # default: top 6 features
    python main.py --top-k 8
    python main.py --no-cv         # skip 10-fold cross-validation (faster)
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_selection import SELECTORS
from src.models import evaluate, get_classifiers

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "heart_cleveland.csv"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

FEATURE_NAMES = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type", "trestbps": "Resting BP",
    "chol": "Cholesterol", "fbs": "Fasting blood sugar", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise angina", "oldpeak": "ST depression",
    "slope": "ST slope", "ca": "Major vessels (ca)", "thal": "Thalassemia",
}


def exploratory_analysis(df: pd.DataFrame) -> None:
    df.describe().T.round(3).to_csv(RESULTS / "dataset_statistics.csv")

    df.drop(columns="target").hist(figsize=(14, 10), bins=20, color="#3b75af", edgecolor="white")
    plt.suptitle("Feature distributions - Cleveland dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS / "histograms.png", dpi=120)
    plt.close()

    plt.figure(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Correlation heat-map")
    plt.tight_layout()
    plt.savefig(FIGS / "correlation_heatmap.png", dpi=120)
    plt.close()


def plot_feature_ranking(name: str, scores: np.ndarray, ranking: np.ndarray,
                         features: list[str], top_k: int) -> None:
    order = ranking[::-1]
    colors = ["#d1495b" if i in ranking[:top_k] else "#9fb3c8" for i in order]
    plt.figure(figsize=(8, 5))
    plt.barh([FEATURE_NAMES[features[i]] for i in order], scores[order], color=colors)
    plt.xlabel("Importance score")
    plt.title(f"{name}: feature ranking (top {top_k} highlighted)")
    plt.tight_layout()
    fname = name.split()[0].lower()
    plt.savefig(FIGS / f"features_{fname}.png", dpi=120)
    plt.close()


def cross_validate(X: np.ndarray, y: np.ndarray, top_k: int, seed: int,
                   folds: int = 10) -> pd.DataFrame:
    """Stratified k-fold CV; scaling and feature selection are fitted on the
    training folds only."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_te = scaler.transform(X[tr]), scaler.transform(X[te])
        for sel_name, selector in SELECTORS.items():
            _, ranking = selector(X_tr, y[tr])
            top = ranking[:top_k]
            for clf_name, clf in get_classifiers(seed).items():
                clf.fit(X_tr[:, top], y[tr])
                m = evaluate(y[te], clf.predict(X_te[:, top]))
                m.update({"Feature Selection": sel_name, "Classifier": clf_name, "Fold": fold})
                rows.append(m)
    df = pd.DataFrame(rows)
    agg = df.groupby(["Feature Selection", "Classifier"], sort=False).agg(
        Accuracy_mean=("Accuracy", "mean"), Accuracy_std=("Accuracy", "std"),
        F1_mean=("F1-Score", "mean"), MCC_mean=("MCC", "mean")).reset_index()
    return agg


def main(top_k: int, seed: int, run_cv: bool = True) -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    df = pd.read_csv(DATA).dropna()
    features = [c for c in df.columns if c != "target"]
    print(f"Dataset: {df.shape[0]} samples, {len(features)} features, "
          f"class balance = {df.target.value_counts().to_dict()}")
    exploratory_analysis(df)

    X = df[features].to_numpy(float)
    y = df["target"].to_numpy(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    rows, selected_table = [], []
    for sel_name, selector in SELECTORS.items():
        t0 = time.perf_counter()
        scores, ranking = selector(X_tr, y_tr)
        fs_time = time.perf_counter() - t0
        top = ranking[:top_k]
        selected_table.append({"Method": sel_name, "Time (s)": round(fs_time, 3),
                               "Selected features": ", ".join(features[i] for i in top)})
        plot_feature_ranking(sel_name, np.asarray(scores, float), ranking, features, top_k)

        for clf_name, clf in get_classifiers(seed).items():
            t0 = time.perf_counter()
            clf.fit(X_tr[:, top], y_tr)
            pred = clf.predict(X_te[:, top])
            m = evaluate(y_te, pred)
            m.update({"Feature Selection": sel_name, "Classifier": clf_name,
                      "Train+Test time (ms)": 1000 * (time.perf_counter() - t0)})
            rows.append(m)
        print(f"  {sel_name:<18} -> {selected_table[-1]['Selected features']}")

    res = pd.DataFrame(rows)
    cols = ["Feature Selection", "Classifier", "Accuracy", "Precision", "Sensitivity",
            "Specificity", "F1-Score", "MCC", "Train+Test time (ms)"]
    res = res[cols]
    res.round(4).to_csv(RESULTS / "all_results.csv", index=False)
    pd.DataFrame(selected_table).to_csv(RESULTS / "selected_features.csv", index=False)

    # Baseline: all 13 features
    base_rows = []
    for clf_name, clf in get_classifiers(seed).items():
        clf.fit(X_tr, y_tr)
        m = evaluate(y_te, clf.predict(X_te))
        m["Classifier"] = clf_name
        base_rows.append(m)
    pd.DataFrame(base_rows).round(4).to_csv(RESULTS / "baseline_all_features.csv", index=False)

    # Best classifier per feature selection method
    best = res.loc[res.groupby("Feature Selection", sort=False)["Accuracy"].idxmax()]
    best.round(4).to_csv(RESULTS / "best_per_method.csv", index=False)

    # Accuracy heat-map
    pivot = res.pivot(index="Feature Selection", columns="Classifier", values="Accuracy")
    pivot = pivot.loc[list(SELECTORS)] * 100
    plt.figure(figsize=(11, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Accuracy (%)"})
    plt.title(f"Test accuracy (%) - top {top_k} features")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGS / "accuracy_heatmap.png", dpi=120)
    plt.close()

    # Best-model comparison
    metrics = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1-Score"]
    ax = (best.set_index("Feature Selection")[metrics] * 100).plot.bar(figsize=(11, 5), width=0.8,
                                                                      colormap="tab10")
    ax.set_ylabel("%")
    ax.set_ylim(50, 100)
    ax.set_title("Best classifier for each feature selection method")
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGS / "best_model_comparison.png", dpi=120)
    plt.close()

    print("\nBest classifier for each feature selection method:")
    show = best[["Feature Selection", "Classifier", "Accuracy", "Precision",
                 "Sensitivity", "Specificity", "F1-Score", "MCC"]].copy()
    for c in show.columns[2:]:
        show[c] = (show[c] * 100).round(2) if c != "MCC" else show[c].round(3)
    print(show.to_string(index=False))
    if run_cv:
        print("\nRunning stratified 10-fold cross-validation ...")
        cv = cross_validate(X, y, top_k, seed)
        cv.round(4).to_csv(RESULTS / "cv_results.csv", index=False)
        best_cv = cv.loc[cv.groupby("Feature Selection", sort=False)["Accuracy_mean"].idxmax()]
        best_cv.round(4).to_csv(RESULTS / "cv_best_per_method.csv", index=False)
        print("Best classifier per method (10-fold CV accuracy, mean +/- std):")
        for _, r in best_cv.iterrows():
            print(f"  {r['Feature Selection']:<18} {r['Classifier']:<20} "
                  f"{100 * r['Accuracy_mean']:.2f} +/- {100 * r['Accuracy_std']:.2f} %")

    print(f"\nResults saved to {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=6, help="number of features to keep")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split")
    parser.add_argument("--no-cv", action="store_true", help="skip 10-fold cross-validation")
    args = parser.parse_args()
    main(args.top_k, args.seed, not args.no_cv)
