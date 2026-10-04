from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\project")
OUT_DIR = DATA_DIR / "kkbox_processed"
SNAPSHOT = (OUT_DIR / "snapshot_features.parquet").as_posix()
MODEL_PATH = OUT_DIR / "boosted_results" / "boosted_model.joblib"
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
features = numeric_features + categorical_features

con = duckdb.connect()

def load_features(cohort, include_label=False):
    columns = ["msno"]
    if include_label:
        columns.append("is_churn")
    columns += features

    return con.execute(f"""
        SELECT {", ".join(columns)}
        FROM read_parquet('{SNAPSHOT}')
        WHERE cohort = '{cohort}'
    """).df()

def prepare_numeric(df):
    df.loc[df["tenure_months_at_cutoff"] < 0, "tenure_months_at_cutoff"] = np.nan
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df

def prepare_categories(df, category_levels=None):
    for col in categorical_features:
        values = (
            df[col]
            .astype("string")
            .fillna("Unknown")
            .replace("", "Unknown")
        )
        if category_levels is None:
            df[col] = values.astype("category")
        else:
            df[col] = pd.Categorical(values, categories=category_levels[col])
    return df

print("Loading development categories and the later cohort...")
development = load_features("train", include_label=False)
category_levels = {}

for col in categorical_features:
    values = (
        development[col]
        .astype("string")
        .fillna("Unknown")
        .replace("", "Unknown")
    )
    category_levels[col] = values.astype("category").cat.categories

del development

holdout = load_features("train_v2", include_label=True)
actual_churn = holdout.pop("is_churn").astype("int8").to_numpy()
customer_ids = holdout.pop("msno")

X = prepare_numeric(holdout)
X = prepare_categories(X, category_levels)

print("Scoring the later cohort...")
model = joblib.load(MODEL_PATH)
scores = model.predict_proba(X)[:, 1]

# Rank customers so decile 1 is the highest-risk 10%.
n = len(scores)
order = np.argsort(-scores, kind="stable")
rank = np.empty(n, dtype=np.int64)
rank[order] = np.arange(n)
decile = np.minimum((rank * 10 // n) + 1, 10)

scored = pd.DataFrame({
    "msno": customer_ids,
    "is_churn": actual_churn,  # retained only for evaluation
    "risk_score": scores,
    "risk_decile": decile,
})

scored["risk_band"] = np.select(
    [
        scored["risk_decile"] == 1,
        scored["risk_decile"].between(2, 3),
        scored["risk_decile"].between(4, 6),
    ],
    ["High", "Elevated", "Moderate"],
    default="Low",
)

summary = (
    scored.groupby(["risk_decile", "risk_band"], observed=True)
    .agg(
        customers=("is_churn", "size"),
        churners=("is_churn", "sum"),
        observed_churn_rate=("is_churn", "mean"),
        mean_risk_score=("risk_score", "mean"),
    )
    .reset_index()
    .sort_values("risk_decile")
)
overall_churn_rate = scored["is_churn"].mean()
summary["lift_vs_overall"] = (
    summary["observed_churn_rate"] / overall_churn_rate
)

summary.to_csv(RESULTS_DIR / "later_cohort_risk_band_summary.csv", index=False)

# Queue features are used to explain routing, not to overwrite the model ranking.
queue = scored.loc[scored["risk_decile"] == 1].copy()

signals = prepare_numeric(holdout.copy())
signals = prepare_categories(signals, category_levels)

queue["cancel_count_last3"] = signals.loc[queue.index, "cancel_count_last3"]
queue["auto_renew_count_last3"] = signals.loc[queue.index, "auto_renew_count_last3"]
queue["transactions_last_month"] = signals.loc[queue.index, "transactions_last_month"]
queue["paid_amount_last3"] = signals.loc[queue.index, "paid_amount_last3"]
queue["listening_secs_last3"] = signals.loc[queue.index, "listening_secs_last3"]

def recommend(row):
    if row["cancel_count_last3"] > 0:
        return (
            "Review cancellation activity; use customer-success outreach "
            "to understand the issue and offer a relevant resolution."
        )
    if (
        row["auto_renew_count_last3"] == 0
        or row["transactions_last_month"] == 0
        or row["paid_amount_last3"] == 0
    ):
        return (
            "Check renewal timing and payment/auto-renew status; send a "
            "timely reminder and troubleshoot only if an issue is confirmed."
        )
    if row["listening_secs_last3"] == 0:
        return "Offer a product-adoption check-in; do not use inactivity alone as the trigger."
    return "Personalized retention check-in; review recent account context."

queue["suggested_action"] = queue.apply(recommend, axis=1)

# Exclude the actual churn label from the operational outreach file.
queue_columns = [
    "msno",
    "risk_score",
    "risk_decile",
    "risk_band",
    "cancel_count_last3",
    "auto_renew_count_last3",
    "transactions_last_month",
    "paid_amount_last3",
    "listening_secs_last3",
    "suggested_action",
]
queue[queue_columns].to_csv(
    RESULTS_DIR / "top_decile_outreach_queue.csv",
    index=False,
)

# Save the labeled scores separately for project evaluation only.
scored.to_parquet(
    RESULTS_DIR / "later_cohort_scored_for_evaluation.parquet",
    index=False,
    compression="zstd",
)

print("\nLater-cohort risk-band summary:")
print(summary.to_string(index=False))
print(f"\nTop-decile outreach queue: {len(queue):,} customers")
print(f"Saved outputs in: {RESULTS_DIR}")