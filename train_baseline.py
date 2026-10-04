from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(r"C:\project")
OUT_DIR = DATA_DIR / "kkbox_processed"
SNAPSHOT = (OUT_DIR / "snapshot_features.parquet").as_posix()
RESULTS_DIR = OUT_DIR / "baseline_results"
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

    # A negative tenure means the registration date is after the scoring
    # cutoff; treat those few records as missing.
    df.loc[df["tenure_months_at_cutoff"] < 0, "tenure_months_at_cutoff"] = np.nan

    # Keep missing categories explicit and make category types consistent.
    for col in categorical_features:
     df[col] = df[col].astype("string").fillna("Unknown")
    df[col] = df[col].replace("", "Unknown").astype(str)

    return df

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
        .agg(customers=("actual", "size"), churners=("actual", "sum"),
             churn_rate=("actual", "mean"))
        .reset_index()
        .sort_values("group")
    )
    overall_rate = y.mean()
    result["lift"] = result["churn_rate"] / overall_rate
    result["cumulative_customers"] = result["customers"].cumsum()
    result["cumulative_churners"] = result["churners"].cumsum()
    result["cumulative_churn_recall"] = result["cumulative_churners"] / y.sum()
    return result

def threshold_metrics(y, scores, threshold):
    y = np.asarray(y)
    scores = np.asarray(scores)
    flagged = scores >= threshold
    tp = int(((y == 1) & flagged).sum())
    total_flagged = int(flagged.sum())
    total_churners = int(y.sum())

    return {
        "threshold": float(threshold),
        "customers_flagged": total_flagged,
        "flagged_share": total_flagged / len(y),
        "precision": tp / total_flagged if total_flagged else 0.0,
        "recall": tp / total_churners if total_churners else 0.0,
        "lift_at_threshold": (
            (tp / total_flagged) / y.mean()
            if total_flagged and y.mean() > 0 else 0.0
        ),
    }

print("Loading the earlier development cohort...")
train_df = load_cohort("train")
y = train_df["is_churn"].astype(int).to_numpy()
customer_ids = train_df["msno"].copy()
X = train_df[numeric_features + categorical_features].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler(with_mean=False)),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=20,
        sparse_output=True,
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=0.0001,
    max_iter=40,
    tol=0.001,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

print("Training the baseline model...")
pipeline.fit(X_train, y_train)

print("Scoring internal validation data...")
val_scores = pipeline.predict_proba(X_val)[:, 1]

# Example operational assumption: the team can contact the top 10%.
top_k = max(1, int(np.ceil(0.10 * len(val_scores))))
validation_threshold = np.sort(val_scores)[-top_k]

validation_metrics = {
    "cohort": "train_internal_validation",
    "customers": len(y_val),
    "churn_rate": float(np.mean(y_val)),
    "average_precision": float(average_precision_score(y_val, val_scores)),
    "roc_auc": float(roc_auc_score(y_val, val_scores)),
}
validation_metrics.update(threshold_metrics(y_val, val_scores, validation_threshold))

pd.DataFrame([validation_metrics]).to_csv(
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
print(pd.DataFrame([validation_metrics]).to_string(index=False))

print("\nLoading the later-period holdout cohort...")
holdout_df = load_cohort("train_v2")
y_holdout = holdout_df["is_churn"].astype(int).to_numpy()
X_holdout = holdout_df[numeric_features + categorical_features]

print("Scoring later-period holdout using the unchanged validation threshold...")
holdout_scores = pipeline.predict_proba(X_holdout)[:, 1]

holdout_metrics = {
    "cohort": "train_v2_temporal_holdout",
    "customers": len(y_holdout),
    "churn_rate": float(np.mean(y_holdout)),
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

joblib.dump(pipeline, RESULTS_DIR / "baseline_model.joblib")

print("\nTemporal holdout results:")
print(pd.DataFrame([holdout_metrics]).to_string(index=False))
print(f"\nResults saved in: {RESULTS_DIR}")