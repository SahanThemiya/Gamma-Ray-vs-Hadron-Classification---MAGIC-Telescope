import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from xgboost import XGBClassifier

PLOT_DIR = "plots"
LABELS   = ["hadron", "gamma"]


def build_models(y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,   # handles class imbalance
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
    }


def _plot_confusion_matrix(y_test, y_pred, name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=LABELS, ax=ax, colorbar=False
    )
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/cm_{name}.png", dpi=150)
    plt.close()


def _plot_feature_importance(model, feature_names, name):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"{name} — Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fi_{name}.png", dpi=150)
    plt.close()


def train_evaluate(x_train, x_test, y_train, y_test, feature_names):
    models  = build_models(y_train)
    results = {}

    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    fig_pr,  ax_pr  = plt.subplots(figsize=(7, 5))

    summary_rows = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc  = average_precision_score(y_test, y_prob)

        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        print(classification_report(y_test, y_pred, target_names=LABELS))
        print(f"ROC-AUC : {roc_auc:.4f}")
        print(f"PR-AUC  : {pr_auc:.4f}")

        results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob}
        summary_rows.append({"Model": name, "ROC-AUC": roc_auc, "PR-AUC": pr_auc})

        _plot_confusion_matrix(y_test, y_pred, name)

        if hasattr(model, "feature_importances_"):
            _plot_feature_importance(model, feature_names, name)

        RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_pr)

    # ROC curves
    ax_roc.set_title("ROC Curves")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    fig_roc.tight_layout()
    fig_roc.savefig(f"{PLOT_DIR}/roc_curves.png", dpi=150)
    plt.close(fig_roc)

    # Precision-Recall curves
    ax_pr.set_title("Precision-Recall Curves (Gamma Signal)")
    fig_pr.tight_layout()
    fig_pr.savefig(f"{PLOT_DIR}/pr_curves.png", dpi=150)
    plt.close(fig_pr)

    # Summary table
    summary = pd.DataFrame(summary_rows).set_index("Model")
    print(f"\n{'='*50}\n  Summary\n{'='*50}")
    print(summary.to_string())

    return results, summary