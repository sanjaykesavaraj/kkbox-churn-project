from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(r"C:\project")
OUT_DIR = DATA_DIR / "kkbox_processed"
SNAPSHOT = (OUT_DIR / "snapshot_features.parquet").as_posix()
RESULTS_DIR = OUT_DIR / "boosted_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

con = duckdb.connect()

def load_cohort(cohort_name):
    df = con.execute(f"""
        SELECT
            msno,
            is_churn,
            {", ".join(numeric_features + categorical_features)}
        FROM read_parquet('{SNAPSHOT}')
        WHERE cohort = '{cohort_name}'
    """).df()

    y = df["is_churn"].astype("int8").to_numpy()
    ids = df["msno"].copy()
    X = df[numeric_features + categorical_features].copy()

    # Negative tenure is invalid; let the model treat it as missing.
    X.loc[X["tenure_months_at_cutoff"] < 0, "tenure_months_at_cutoff"] = np.nan

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_features:
        X[col] = (
            X[col]
            .astype("string")
            .fillna("Unknown")
            .replace("", "Unknown")
            .astype("category")
        )

    return ids, X, y

def make_lift_table(y, scores, n_groups=10):
    y = np.asarray(y)
    scores = np.asarray(scores)

    order = np.argsort(-scores)
    group = np.empty(len(scores), dtype=int)
    group[order] = np.minimum(
        (np.arange(len(scores)) * n_groups // len(scores)) + 1,
        n_groups,
    )

    table = pd.DataFrame({"group": group, "actual": y})
    result = (
        table.groupby("group")
        .agg(
            customers=("actual", "size"),
            churners=("actual", "sum"),
            churn_rate=("actual", "mean"),
        )
        .reset_index()
        .sort_values("group")
    )
    result["lift"] = result["churn_rate"] / y.mean()
    result["cumulative_customers"] = result["customers"].cumsum()
    result["cumulative_churners"] = result["churners"].cumsum()
    result["cumulative_churn_recall"] = result["cumulative_churners"] / y.sum()
    return result

def threshold_metrics(y, scores, threshold):
    y = np.asarray(y)
    scores = np.asarray(scores)
    flagged = scores >= threshold
    tp = int(((y == 1) & flagged).sum())
    n_flagged = int(flagged.sum())

    return {
        "threshold": float(threshold),
        "customers_flagged": n_flagged,
        "flagged_share": n_flagged / len(y),
        "precision": tp / n_flagged if n_flagged else 0.0,
        "recall": tp / int(y.sum()) if y.sum() else 0.0,
        "lift_at_threshold": (
            (tp / n_flagged) / y.mean()
            if n_flagged and y.mean() > 0 else 0.0
        ),
    }

print("Loading the earlier development cohort...")
_, X, y = load_cohort("train")

# Use the same split definition as the baseline.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

model = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.08,
    max_iter=150,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    categorical_features="from_dtype",
    class_weight="balanced",
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=15,
    random_state=42,
)

print("Training boosted trees...")
model.fit(X_train, y_train)
print(f"Iterations used: {model.n_iter_}")

print("Scoring internal validation data...")
val_scores = model.predict_proba(X_val)[:, 1]

# Select the threshold on validation only, using the same 10% outreach assumption.
top_k = max(1, int(np.ceil(0.10 * len(val_scores))))
validation_threshold = np.sort(val_scores)[-top_k]

val_metrics = {
    "cohort": "train_internal_validation",
    "customers": len(y_val),
    "churn_rate": float(y_val.mean()),
    "average_precision": float(average_precision_score(y_val, val_scores)),
    "roc_auc": float(roc_auc_score(y_val, val_scores)),
}
val_metrics.update(threshold_metrics(y_val, val_scores, validation_threshold))

pd.DataFrame([val_metrics]).to_csv(
    RESULTS_DIR / "validation_metrics.csv", index=False
)
make_lift_table(y_val, val_scores).to_csv(
    RESULTS_DIR / "validation_lift.csv", index=False
)

precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
pd.DataFrame({
    "precision": precision,
    "recall": recall,
    "threshold": np.append(thresholds, np.nan),
}).to_csv(RESULTS_DIR / "validation_precision_recall.csv", index=False)

print("\nValidation results:")
print(pd.DataFrame([val_metrics]).to_string(index=False))

print("\nLoading the later-period holdout cohort...")
_, X_holdout, y_holdout = load_cohort("train_v2")

# Align category levels to the development data; unseen categories become missing.
for col in categorical_features:
    X_holdout[col] = X_holdout[col].cat.set_categories(X[col].cat.categories)

print("Scoring later-period holdout at the unchanged validation threshold...")
holdout_scores = model.predict_proba(X_holdout)[:, 1]

holdout_metrics = {
    "cohort": "train_v2_temporal_holdout",
    "customers": len(y_holdout),
    "churn_rate": float(y_holdout.mean()),
    "average_precision": float(average_precision_score(y_holdout, holdout_scores)),
    "roc_auc": float(roc_auc_score(y_holdout, holdout_scores)),
}
holdout_metrics.update(
    threshold_metrics(y_holdout, holdout_scores, validation_threshold)
)

pd.DataFrame([holdout_metrics]).to_csv(
    RESULTS_DIR / "holdout_metrics.csv", index=False
)
make_lift_table(y_holdout, holdout_scores).to_csv(
    RESULTS_DIR / "holdout_lift.csv", index=False
)

precision_h, recall_h, thresholds_h = precision_recall_curve(
    y_holdout, holdout_scores
)
pd.DataFrame({
    "precision": precision_h,
    "recall": recall_h,
    "threshold": np.append(thresholds_h, np.nan),
}).to_csv(RESULTS_DIR / "holdout_precision_recall.csv", index=False)

joblib.dump(model, RESULTS_DIR / "boosted_model.joblib")

print("\nTemporal holdout results:")
print(pd.DataFrame([holdout_metrics]).to_string(index=False))
print(f"\nResults saved in: {RESULTS_DIR}")