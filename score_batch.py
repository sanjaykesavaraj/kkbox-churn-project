from pathlib import Path
from datetime import datetime, timezone
import argparse

import duckdb
import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\project")
OUT_DIR = DATA_DIR / "kkbox_processed"
MODEL_DIR = OUT_DIR / "boosted_results"

DEFAULT_INPUT = OUT_DIR / "snapshot_features.parquet"
MODEL_PATH = MODEL_DIR / "boosted_model.joblib"
SCHEMA_PATH = MODEL_DIR / "categorical_schema.joblib"
REFERENCE_PATH = DEFAULT_INPUT

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


def prepare_features(frame, category_schema):
    frame = frame.copy()

    frame.loc[
        frame["tenure_months_at_cutoff"] < 0,
        "tenure_months_at_cutoff",
    ] = np.nan

    for col in numeric_features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
        frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)

    for col in categorical_features:
        values = (
            frame[col]
            .astype("string")
            .fillna("Unknown")
            .replace("", "Unknown")
        )
        # Keep the category order identical to the model's training schema.
        frame[col] = pd.Categorical(
            values,
            categories=category_schema[col],
        )

    return frame


def recommend_action(row):
    if row["cancel_count_last3"] > 0:
        return (
            "Review cancellation activity; contact to understand the issue "
            "and offer a relevant resolution."
        )

    if (
        row["auto_renew_count_last3"] == 0
        or row["transactions_last_month"] == 0
        or row["paid_amount_last3"] == 0
    ):
        return (
            "Check expiry and renewal status; send a timely reminder. "
            "Troubleshoot payment only if an issue is confirmed."
        )

    if row["listening_secs_last3"] == 0:
        return (
            "Consider a product-adoption check-in; inactivity alone "
            "should not trigger an expensive intervention."
        )

    return "Personalized retention review; check recent account context."


parser = argparse.ArgumentParser(
    description="Batch-score a KKBox-format customer snapshot."
)
parser.add_argument(
    "--input",
    default=str(DEFAULT_INPUT),
    help="Input feature Parquet file. Must contain msno and the model features.",
)
parser.add_argument(
    "--cohort",
    default="train_v2",
    help="Optional cohort filter for files with a cohort column. Use 'none' for no filter.",
)
parser.add_argument(
    "--contact-share",
    type=float,
    default=0.10,
    help="Share of highest-risk customers to include in the outreach queue.",
)
parser.add_argument(
    "--output-dir",
    default=str(MODEL_DIR / "batch_demo"),
    help="Folder for scored Parquet and outreach CSV outputs.",
)
args = parser.parse_args()

if not 0 < args.contact_share <= 1:
    raise ValueError("--contact-share must be greater than 0 and at most 1.")

input_path = Path(args.input)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Create the category schema once, using the same conversion/order as training.
if not SCHEMA_PATH.exists():
    print("Creating the model's categorical feature schema...")
    con = duckdb.connect()
    ref_path = REFERENCE_PATH.as_posix()
    ref_cols = ", ".join(categorical_features)
    reference = con.execute(f"""
        SELECT {ref_cols}
        FROM read_parquet('{ref_path}')
        WHERE cohort = 'train'
    """).df()

    category_schema = {}
    for col in categorical_features:
        values = (
            reference[col]
            .astype("string")
            .fillna("Unknown")
            .replace("", "Unknown")
        )
        category_schema[col] = values.astype("category").cat.categories

    joblib.dump(category_schema, SCHEMA_PATH)
else:
    category_schema = joblib.load(SCHEMA_PATH)

# Read only the scoring ID and required features. Never select the label.
selected_columns = ["msno"] + feature_columns
columns_sql = ", ".join(selected_columns)
input_sql = input_path.resolve().as_posix().replace("'", "''")

if args.cohort.lower() == "none":
    query = f"""
        SELECT {columns_sql}
        FROM read_parquet('{input_sql}')
    """
    df = duckdb.connect().execute(query).df()
else:
    query = f"""
        SELECT {columns_sql}
        FROM read_parquet('{input_sql}')
        WHERE cohort = ?
    """
    df = duckdb.connect().execute(query, [args.cohort]).df()

if df.empty:
    raise ValueError("No scoring rows found. Check the input path and cohort filter.")

missing = sorted(set(selected_columns) - set(df.columns))
if missing:
    raise ValueError(f"Input is missing required columns: {missing}")

if df["msno"].duplicated().any():
    raise ValueError(
        "Input contains duplicate customer IDs. Score one snapshot per customer at a time."
    )

print(f"Scoring {len(df):,} customers...")
ids = df.pop("msno")
X = prepare_features(df, category_schema)

model = joblib.load(MODEL_PATH)
scores = model.predict_proba(X)[:, 1]

# Rank scores; decile 1 is the highest-risk group.
n = len(scores)
order = np.argsort(-scores, kind="stable")
rank = np.empty(n, dtype=np.int64)
rank[order] = np.arange(n)
decile = np.minimum((rank * 10 // n) + 1, 10)

risk_band = np.select(
    [
        decile == 1,
        decile == 2,
        (decile >= 3) & (decile <= 6),
    ],
    ["High", "Elevated", "Moderate"],
    default="Low",
)

scored = pd.DataFrame({
    "msno": ids,
    "risk_score": scores,
    "risk_decile": decile,
    "risk_band": risk_band,
    "run_time_utc": datetime.now(timezone.utc).isoformat(),
})

scored_path = output_dir / "customer_risk_scores.parquet"
scored.to_parquet(scored_path, index=False, compression="zstd")

# The queue is capacity-based: take the highest-ranked customers this run.
n_contact = max(1, int(np.ceil(args.contact_share * n)))
selected_rows = order[:n_contact]

queue = df.iloc[selected_rows].copy()
queue.insert(0, "msno", ids.iloc[selected_rows].to_numpy())
queue["risk_score"] = scores[selected_rows]
queue["risk_decile"] = decile[selected_rows]
queue["risk_band"] = risk_band[selected_rows]
queue["suggested_action"] = queue.apply(recommend_action, axis=1)

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
queue_path = output_dir / "outreach_queue.csv"
queue[queue_columns].to_csv(queue_path, index=False)

print(f"Scored file: {scored_path}")
print(f"Outreach queue: {queue_path}")
print(f"Customers selected for outreach: {n_contact:,} ({args.contact_share:.1%})")
print("\nRisk-band counts:")
print(scored["risk_band"].value_counts().to_string())