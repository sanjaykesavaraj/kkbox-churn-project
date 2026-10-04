from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

DATA_DIR = Path(r"C:\project")
OUT_DIR = DATA_DIR / "kkbox_processed"
SNAPSHOT = (OUT_DIR / "snapshot_features.parquet").as_posix()
MODEL_PATH = OUT_DIR / "boosted_results" / "boosted_model.joblib"
RESULTS_DIR = OUT_DIR / "boosted_results"

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
feature_columns = numeric_features + categorical_features

con = duckdb.connect()
print("Loading the earlier cohort...")
df = con.execute(f"""
    SELECT
        is_churn,
        {", ".join(feature_columns)}
    FROM read_parquet('{SNAPSHOT}')
    WHERE cohort = 'train'
""").df()

y = df.pop("is_churn").astype("int8").to_numpy()

df.loc[df["tenure_months_at_cutoff"] < 0, "tenure_months_at_cutoff"] = np.nan

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

for col in categorical_features:
    df[col] = (
        df[col]
        .astype("string")
        .fillna("Unknown")
        .replace("", "Unknown")
        .astype("category")
    )

X_train, X_val, y_train, y_val = train_test_split(
    df,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Sample validation customers to keep the explanation step manageable.
val_frame = X_val.copy()
val_frame["_target"] = y_val

sample_size = min(30_000, len(val_frame))
sample, _, y_sample, _ = train_test_split(
    val_frame.drop(columns=["_target"]),
    val_frame["_target"],
    train_size=sample_size,
    stratify=val_frame["_target"],
    random_state=42,
)
y_sample = y_sample.to_numpy()

print(f"Computing permutation importance on {sample_size:,} validation customers...")
model = joblib.load(MODEL_PATH)

result = permutation_importance(
    model,
    sample,
    y_sample,
    scoring="average_precision",
    n_repeats=5,
    random_state=42,
    n_jobs=2,
)

importance = pd.DataFrame({
    "feature": sample.columns,
    "average_precision_drop_mean": result.importances_mean,
    "average_precision_drop_std": result.importances_std,
}).sort_values("average_precision_drop_mean", ascending=False)

importance.to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
print("\nTop features:")
print(importance.head(15).to_string(index=False))

top = importance.head(15).sort_values("average_precision_drop_mean")
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top["feature"], top["average_precision_drop_mean"])
ax.set(
    xlabel="Decrease in average precision when shuffled",
    title="Boosted model: permutation importance",
)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "permutation_importance.png", dpi=160)
plt.close(fig)

print(f"\nSaved importance table and chart in: {RESULTS_DIR}")