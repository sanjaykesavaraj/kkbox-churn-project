from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

DATA_DIR = Path(r"C:\project")
OUT_DIR = DATA_DIR / "kkbox_processed"
SNAPSHOT = (OUT_DIR / "snapshot_features.parquet").as_posix()
RESULTS_DIR = OUT_DIR / "boosted_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "boosted_model.joblib"

numeric_features = [
    "age",
    "tenure_months_at_cutoff",
    "log_rows_last_month",
    "listening_secs_last3",
    "listening_secs_prev3",
    "listening_secs_last6",
    "completed_plays_last3",
    "daily_unique_song_sum_last3",
    "active_months_last3",
    "active_months_prev3",
    "log_rows_history",
    "transactions_last_month",
    "transactions_last3",
    "transactions_prev3",
    "transactions_history",
    "paid_amount_last3",
    "auto_renew_count_last3",
    "cancel_count_last3",
    "weighted_avg_plan_days_last3",
]
categorical_features = ["city", "gender", "registered_via"]
features = numeric_features + categorical_features

con = duckdb.connect()
model = joblib.load(MODEL_PATH)

def load_cohort(name):
    cols = ["is_churn"] + features
    return con.execute(f"""
        SELECT {", ".join(cols)}
        FROM read_parquet('{SNAPSHOT}')
        WHERE cohort = '{name}'
    """).df()

def prepare(X, category_levels=None):
    X = X.copy()
    X.loc[X["tenure_months_at_cutoff"] < 0, "tenure_months_at_cutoff"] = np.nan

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_features:
        values = (
            X[col].astype("string")
            .fillna("Unknown")
            .replace("", "Unknown")
        )
        if category_levels is None:
            X[col] = values.astype("category")
        else:
            X[col] = pd.Categorical(values, categories=category_levels[col])

    return X

print("Scoring the internal validation data...")
dev = load_cohort("train")
y_dev = dev.pop("is_churn").astype("int8").to_numpy()
X_dev = prepare(dev)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=42,
)
val_scores = model.predict_proba(X_val)[:, 1]

category_levels = {
    col: X_dev[col].cat.categories
    for col in categorical_features
}

print("Fitting isotonic calibration on internal validation scores...")
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_scores, y_val)

print("Scoring the later cohort...")
later = load_cohort("train_v2")
y_later = later.pop("is_churn").astype("int8").to_numpy()
X_later = prepare(later, category_levels)
raw_scores = model.predict_proba(X_later)[:, 1]
calibrated_scores = calibrator.predict(raw_scores)
calibrated_scores = np.clip(calibrated_scores, 1e-6, 1 - 1e-6)

def calibration_bins(y, scores, n_bins=10):
    frame = pd.DataFrame({"actual": y, "score": scores})
    frame["bin"] = pd.qcut(frame["score"], q=n_bins, duplicates="drop")
    result = (
        frame.groupby("bin", observed=True)
        .agg(
            customers=("actual", "size"),
            mean_score=("score", "mean"),
            observed_churn_rate=("actual", "mean"),
        )
        .reset_index(drop=True)
    )
    return result

raw_brier = brier_score_loss(y_later, raw_scores)
cal_brier = brier_score_loss(y_later, calibrated_scores)
raw_logloss = log_loss(y_later, np.clip(raw_scores, 1e-6, 1 - 1e-6))
cal_logloss = log_loss(y_later, calibrated_scores)

metrics = pd.DataFrame([
    {"score_type": "Raw model score", "brier_score": raw_brier, "log_loss": raw_logloss},
    {"score_type": "Isotonic calibrated", "brier_score": cal_brier, "log_loss": cal_logloss},
])
metrics.to_csv(RESULTS_DIR / "calibration_metrics.csv", index=False)

raw_bins = calibration_bins(y_later, raw_scores)
cal_bins = calibration_bins(y_later, calibrated_scores)
raw_bins.to_csv(RESULTS_DIR / "raw_score_calibration_bins.csv", index=False)
cal_bins.to_csv(RESULTS_DIR / "calibrated_score_bins.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
ax.plot(
    raw_bins["mean_score"],
    raw_bins["observed_churn_rate"],
    marker="o",
    label="Raw model score",
)
ax.plot(
    cal_bins["mean_score"],
    cal_bins["observed_churn_rate"],
    marker="o",
    label="Isotonic calibrated",
)
ax.set(
    xlabel="Mean predicted score",
    ylabel="Observed churn rate",
    title="Later cohort: calibration check",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "calibration_comparison.png", dpi=160)
plt.close(fig)

joblib.dump(calibrator, RESULTS_DIR / "isotonic_calibrator.joblib")

print("\nLater-cohort calibration metrics:")
print(metrics.to_string(index=False))
print("\nCalibrated-score bins:")
print(cal_bins.to_string(index=False))
print(f"\nChart and tables saved in: {RESULTS_DIR}")