import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, "..", "data", "clean")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

METADATA_PATH = os.path.join(CLEAN_DIR, "metadata_train.csv")
FEATURES_PATH = os.path.join(CLEAN_DIR, "features_train.csv")

TOP_N = 10

INPUT_SONGS = [
    "Bohemian Rhapsody",
    "Smells Like Teen Spirit",
    "Back In Black",
    "Enter Sandman",
    "Sweet Child O' Mine",
]


def normalize_title(s: str) -> str:
    return str(s).strip().lower()


def minmax_01(x: np.ndarray) -> np.ndarray:
    x_min = np.min(x)
    x_max = np.max(x)
    if np.isclose(x_max - x_min, 0.0):
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def _pick_best_match_index(
        metadata: pd.DataFrame,
        normalized_catalog: pd.Series,
        normalized_query: str,
) -> int | None:
    exact_idx = metadata[normalized_catalog == normalized_query].index
    if len(exact_idx) > 0:
        exact_rows = metadata.loc[exact_idx]
        if "popularity" in exact_rows.columns:
            return int(exact_rows["popularity"].astype(float).idxmax())
        return int(exact_idx[0])

    contains_mask = normalized_catalog.str.contains(normalized_query, regex=False, na=False)
    contains_idx = metadata[contains_mask].index
    if len(contains_idx) > 0:
        contains_rows = metadata.loc[contains_idx]
        if "popularity" in contains_rows.columns:
            return int(contains_rows["popularity"].astype(float).idxmax())
        return int(contains_idx[0])

    return None


def find_liked_indices(metadata: pd.DataFrame, input_songs: list[str]) -> pd.Index:
    """
    Zwraca maksymalnie 1 rekord na każdą piosenkę z INPUT_SONGS
    (żeby uniknąć wielu wersji/coverów tego samego tytułu jako pozytywów).
    """
    normalized_catalog = metadata["track_name"].astype(str).map(normalize_title)
    selected: list[int] = []

    for song in input_songs:
        q = normalize_title(song)
        idx = _pick_best_match_index(metadata, normalized_catalog, q)
        if idx is not None and idx not in selected:
            selected.append(idx)

    return pd.Index(selected)


def build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=700,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def compute_precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0

    k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1]
    topk = order[:k]

    tp_at_k = float(np.sum(y_true[topk] == 1))
    precision_at_k = tp_at_k / k
    recall_at_k = tp_at_k / max(float(np.sum(y_true == 1)), 1.0)
    return precision_at_k, recall_at_k


def evaluate_and_plot_rf(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> float:
    """
    Trenuje RF na train split, ocenia na test split, zapisuje wykresy i metryki.
    Zwraca najlepszy próg (max F1) z PR-curve.
    """
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    stratify = y if (n_pos >= 2 and n_neg >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    pos_weight = n_neg_train / max(n_pos_train, 1)

    sample_weight = np.ones(len(y_train), dtype=float)
    sample_weight[y_train == 1.0] = pos_weight

    reg = build_rf()
    reg.fit(X_train, y_train, sample_weight=sample_weight)

    y_score = np.clip(reg.predict(X_test), 0.0, 1.0)

    best_thr = 0.5
    pr_curve_p, pr_curve_r, pr_curve_t = precision_recall_curve(y_test, y_score)
    if len(pr_curve_t) > 0:
        f1_curve = 2 * pr_curve_p[:-1] * pr_curve_r[:-1] / (pr_curve_p[:-1] + pr_curve_r[:-1] + 1e-12)
        best_idx = int(np.nanargmax(f1_curve))
        best_thr = float(pr_curve_t[best_idx])

    y_pred = (y_score >= best_thr).astype(int)

    mae = mean_absolute_error(y_test, y_score)
    mse = mean_squared_error(y_test, y_score)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_score)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    roc_auc = np.nan
    pr_auc = np.nan
    unique_classes = np.unique(y_test)
    if len(unique_classes) == 2:
        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)

    p_at_10, r_at_10 = compute_precision_recall_at_k(y_test, y_score, k=10)

    metrics = {
        "best_threshold": float(best_thr),
        "regression_metrics": {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2": float(r2),
        },
        "classification_metrics": {
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "ROC_AUC": float(roc_auc) if not np.isnan(roc_auc) else None,
            "PR_AUC": float(pr_auc) if not np.isnan(pr_auc) else None,
        },
        "ranking_metrics": {
            "Precision@10": float(p_at_10),
            "Recall@10": float(r_at_10),
        }
    }

    print("\n=== RF METRYKI (hold-out test) ===")
    print(f"Best threshold (max F1): {best_thr:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"MSE:       {mse:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"R2:        {r2:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"P@10:      {p_at_10:.4f}")
    print(f"R@10:      {r_at_10:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC:   N/A (1 klasa w y_test)")
    print(f"PR-AUC:    {pr_auc:.4f}" if not np.isnan(pr_auc) else "PR-AUC:    N/A (1 klasa w y_test)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if len(unique_classes) == 2:
        RocCurveDisplay.from_predictions(y_test, y_score, ax=axes[0, 0])
        axes[0, 0].set_title("ROC Curve")

        PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=axes[0, 1])
        axes[0, 1].set_title("Precision-Recall Curve")
    else:
        axes[0, 0].text(0.5, 0.5, "Brak 2 klas w y_test", ha="center", va="center")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 1].text(0.5, 0.5, "Brak 2 klas w y_test", ha="center", va="center")
        axes[0, 1].set_title("Precision-Recall Curve")

    cm = confusion_matrix(y_test, y_pred, labels=[0.0, 1.0])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0])
    axes[1, 0].set_title(f"Confusion Matrix (threshold={best_thr:.3f})")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")

    importances = reg.feature_importances_
    top_k = min(15, len(importances))
    top_idx = np.argsort(importances)[-top_k:]
    axes[1, 1].barh([feature_names[i] for i in top_idx], importances[top_idx], color="teal")
    axes[1, 1].set_title(f"Top-{top_k} Feature Importances")
    axes[1, 1].set_xlabel("Importance")

    plt.tight_layout()
    os.makedirs(REPORTS_DIR, exist_ok=True)

    out_plot = os.path.join(REPORTS_DIR, "rf_metrics.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"Zapisano wykresy do: {out_plot}")

    backend = matplotlib.get_backend().lower()
    if "agg" in backend:
        plt.close(fig)
    else:
        plt.show()

    fig_pred, axes_pred = plt.subplots(1, 2, figsize=(14, 5))

    axes_pred[0].scatter(y_test, y_score, alpha=0.5, s=30, color="steelblue", edgecolors="black", linewidth=0.5)
    axes_pred[0].plot([0, 1], [0, 1], 'r--', lw=2, label="Perfect Prediction")
    axes_pred[0].set_xlabel("y_true")
    axes_pred[0].set_ylabel("y_pred (score)")
    axes_pred[0].set_title("y_true vs y_pred")
    axes_pred[0].legend()
    axes_pred[0].grid(True, alpha=0.3)

    residuals = y_test - y_score
    axes_pred[1].scatter(y_score, residuals, alpha=0.5, s=30, color="coral", edgecolors="black", linewidth=0.5)
    axes_pred[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes_pred[1].set_xlabel("y_pred (score)")
    axes_pred[1].set_ylabel("Residuals (y_true - y_pred)")
    axes_pred[1].set_title("Residual Plot")
    axes_pred[1].grid(True, alpha=0.3)

    plt.tight_layout()

    out_pred_plot = os.path.join(REPORTS_DIR, "rf_predictions_residuals.png")
    plt.savefig(out_pred_plot, dpi=150, bbox_inches="tight")
    print(f"Zapisano wykresy predykcji do: {out_pred_plot}")

    if "agg" not in backend:
        plt.show()
    plt.close(fig_pred)

    fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
    ax_metrics.axis("tight")
    ax_metrics.axis("off")

    metrics_data = [
        ["Metric", "Value"],
        ["MAE", f"{mae:.4f}"],
        ["RMSE", f"{rmse:.4f}"],
        ["R2", f"{r2:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1", f"{f1:.4f}"],
        ["ROC-AUC", f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"],
        ["PR-AUC", f"{pr_auc:.4f}" if not np.isnan(pr_auc) else "N/A"],
    ]

    table = ax_metrics.table(cellText=metrics_data, cellLoc="center", loc="center",
                             colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for i in range(2):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title("RF Model Metrics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    out_metrics_png = os.path.join(REPORTS_DIR, "rf_metrics_table.png")
    plt.savefig(out_metrics_png, dpi=150, bbox_inches="tight")
    print(f"Zapisano tabelę metryk do: {out_metrics_png}")

    if "agg" not in backend:
        plt.show()
    plt.close(fig_metrics)

    out_metrics = os.path.join(REPORTS_DIR, "rf_metrics.json")
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Zapisano metryki do: {out_metrics}")

    return best_thr


def main():
    metadata = pd.read_csv(METADATA_PATH)
    features = pd.read_csv(FEATURES_PATH)

    if "Unnamed: 0" in metadata.columns:
        metadata = metadata.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in features.columns:
        features = features.drop(columns=["Unnamed: 0"])

    if len(metadata) != len(features):
        raise ValueError(
            f"Niezgodna liczba wierszy: metadata={len(metadata)} vs features={len(features)}"
        )

    liked_idx = find_liked_indices(metadata, INPUT_SONGS)

    if len(liked_idx) == 0:
        raise ValueError("Nie znaleziono żadnej podanej piosenki w katalogu.")
    if len(liked_idx) < 3:
        print(f"Uwaga: znaleziono tylko {len(liked_idx)} z {len(INPUT_SONGS)} liked. Ranking może być słabszy.")

    y = np.zeros(len(metadata), dtype=float)
    y[liked_idx.to_numpy()] = 1.0

    X = features.to_numpy()
    feature_names = features.columns.tolist()

    _ = evaluate_and_plot_rf(X, y, feature_names)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    sample_weight = np.ones(len(y), dtype=float)
    sample_weight[y == 1.0] = pos_weight

    reg = build_rf()
    reg.fit(X, y, sample_weight=sample_weight)

    proba_like = np.clip(reg.predict(X), 0.0, 1.0)

    liked_vector = X[liked_idx.to_numpy()].mean(axis=0, keepdims=True)
    sim = cosine_similarity(X, liked_vector).ravel()

    proba_norm = minmax_01(proba_like)
    sim_norm = minmax_01(sim)
    final_score = 0.70 * proba_norm + 0.30 * sim_norm

    results = metadata.copy()
    results["proba_like"] = proba_like
    results["similarity"] = sim
    results["final_score"] = final_score
    results = results.drop(index=liked_idx, errors="ignore")

    recs = results.sort_values("final_score", ascending=False).head(TOP_N)

    print("\n=== FOUND LIKED SONGS ===")
    print(metadata.loc[liked_idx, ["track_name", "artists", "track_genre"]].to_string(index=False))

    print("\n=== TOP RECOMMENDATIONS ===")
    cols = ["track_name", "artists", "track_genre", "final_score", "proba_like", "similarity"]
    print(recs[cols].to_string(index=False))


if __name__ == "__main__":
    main()
