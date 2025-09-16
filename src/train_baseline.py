#leakage-safe baseline QSAR model for EGFR pActivity using scaffold splits
"""
This script:
  - Loads X (fingerprints), y (pActivity), and meta (with 'split' + SMILES).
  - Respects your precomputed scaffold split (train/val/test).
  - Trains XGBoost with early stopping on the validation set.
  - Evaluates RMSE/MAE/R2 on train/val/test.
  - Saves the model, metrics, predictions, and a few diagnostic plots.
  - Computes SHAP values to interpret feature importance.

"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# XGBoost is a strong baseline for QSAR on Morgan fingerprints
from xgboost import XGBRegressor

# SHAP will extract per-feature attributions from tree models
import shap
# shap._config.show_progress = True  # progress bars for clarity - removed due to API change


# 1) Paths and small utilities
DATA_DIR = Path("data_proc")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42  # ensure reproducibility


def rmse(y_true, y_pred):
    """Root Mean Squared Error: lower is better (penalizes large errors)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean Absolute Error: lower is better (robust to outliers)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    """Coefficient of determination: 1.0 is perfect, 0.0 means 'predict mean'."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")



# 2) Load data and splits
# X: shape (N_samples, N_bits) with 0/1 fingerprint bits
X = np.load(DATA_DIR / "X.npy")  # numpy array
y = np.load(DATA_DIR / "y.npy")  # shape (N,)
meta = pd.read_csv(DATA_DIR / "meta.csv")  # must include columns: smiles, assay_type, scaffold, split

assert X.shape[0] == y.shape[0] == meta.shape[0], "X, y, and meta must have same number of rows"

# Build boolean masks for the scaffold split 
train_mask = meta["split"] == "train"
val_mask   = meta["split"] == "val"
test_mask  = meta["split"] == "test"

X_train, y_train = X[train_mask.values], y[train_mask.values]
X_val,   y_val   = X[val_mask.values],   y[val_mask.values]
X_test,  y_test  = X[test_mask.values],  y[test_mask.values]

# sanity prints so you can see the sizes at a glance
print("Shapes:")
print("  train:", X_train.shape, y_train.shape)
print("  val:  ", X_val.shape,   y_val.shape)
print("  test: ", X_test.shape,  y_test.shape)



# 3) Define model with sensible defaults
# Notes:
#  - n_estimators is set high; early_stopping picks the right number.
#  - learning_rate small to allow fine-grained boosting steps.
#  - max_depth ~6 is a good starting point for 2048-bit fingerprints.
#  - subsample/colsample_bytree < 1.0 helps generalization.
#  - reg_lambda provides L2 regularization to reduce overfitting.

model = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.02,
    max_depth=6,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=0,  # use all CPU cores available
    tree_method="hist",  # fast histogram algorithm; good default on CPUs
)



# 4) Fit with early stopping on the VAL set

# Early stopping:
#   - watches validation RMSE (XGBoost uses the objective to compute eval metric).
#   - stops boosting after 'early_stopping_rounds' with no improvement.
#   - keeps the best iteration (best_ntree_limit) for subsequent predict().

eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=False,               # set True for per-iteration logs
)

print(f"Training completed with {model.n_estimators} boosting rounds")



# 5) Evaluate on train/val/test (no leakage)
def evaluate_split(name, Xs, ys):
    """Compute metrics for a split and return dict with predictions."""
    y_pred = model.predict(Xs)
    metrics = {
        "rmse": rmse(ys, y_pred),
        "mae": mae(ys, y_pred),
        "r2":  r2(ys, y_pred),
    }
    print(f"[{name}] RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R2={metrics['r2']:.3f}")
    return metrics, y_pred

metrics = {}
metrics["train"], y_pred_train = evaluate_split("train", X_train, y_train)
metrics["val"],   y_pred_val   = evaluate_split("val",   X_val,   y_val)
metrics["test"],  y_pred_test  = evaluate_split("test",  X_test,  y_test)

# Save metrics to a JSON so they're tracked over time
with open(REPORT_DIR / "metrics_xgb_baseline.json", "w") as f:
    json.dump(metrics, f, indent=2)



# 6) Save predictions with SMILES and split labels
# for error analysis, plotting, or comparing future models.
pred_df = meta.copy()
pred_df["y_true"] = y
pred_df.loc[train_mask, "y_pred_xgb"] = y_pred_train
pred_df.loc[val_mask,   "y_pred_xgb"] = y_pred_val
pred_df.loc[test_mask,  "y_pred_xgb"] = y_pred_test
pred_df.to_csv(REPORT_DIR / "predictions_xgb_baseline.csv", index=False)



# 7) Diagnostic plots: parity & residuals (test split)
def parity_plot(y_true, y_pred, title, out_path):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle="--")  # y=x reference line
    plt.xlabel("True pActivity")
    plt.ylabel("Predicted pActivity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def residual_hist(y_true, y_pred, title, out_path):
    res = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=30, alpha=0.9)
    plt.axvline(0.0, color="k", linestyle="--")
    plt.xlabel("Residual (pred - true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

parity_plot(y_test, y_pred_test, "XGB baseline — Test parity", REPORT_DIR / "parity_test_xgb.png")
residual_hist(y_test, y_pred_test, "XGB baseline — Test residuals", REPORT_DIR / "residuals_test_xgb.png")



# 8) Saves the model to disk for reuse
model.save_model(MODEL_DIR / "xgb_baseline.json")



# 9) SHAP: feature attributions for interpretability
# For 2048-bit fingerprints, SHAP tells you which bits (substructures) were helpful.
# computes SHAP on the validation set (kept small) 
# readable feature names: bit_0000..bit_2047
n_bits = X.shape[1]
feature_names = [f"bit_{i:04d}" for i in range(n_bits)]

# small background sample to speed up SHAP value computation
background_size = min(100, X_train.shape[0])
background = X_train[np.random.RandomState(RANDOM_STATE).choice(X_train.shape[0], size=background_size, replace=False)]

explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
shap_vals_val = explainer.shap_values(X_val)  # shape: (N_val, n_bits)

# SHAP summary plot (bar): which bits matter on average
plt.figure()
shap.summary_plot(shap_vals_val, X_val, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig(REPORT_DIR / "shap_summary_bar_val_xgb.png", dpi=200)
plt.close()

# Optional: detailed beeswarm plot (top 20 bits)
plt.figure()
shap.summary_plot(shap_vals_val, X_val, feature_names=feature_names, show=False, max_display=20)
plt.tight_layout()
plt.savefig(REPORT_DIR / "shap_beeswarm_val_xgb.png", dpi=200)
plt.close()

print("Done. Saved model, metrics, predictions, and plots in:", REPORT_DIR, "and", MODEL_DIR)
