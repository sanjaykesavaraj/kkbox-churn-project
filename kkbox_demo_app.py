"""Interactive portfolio demo for the historical KKBox churn model.

This app uses synthetic example profiles only. It does not contact customers,
accept uploads, or store visitor data. Scores are ranking scores, not calibrated
individual churn probabilities.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="KKBox Churn Risk Demo",
    page_icon="🎧",
    layout="wide",
)

NUMERIC_FEATURES = [
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
CATEGORICAL_FEATURES = ["city", "gender", "registered_via"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = Path(
    os.environ.get(
        "KKBOX_ARTIFACT_DIR",
        str(PROJECT_ROOT / "kkbox_processed" / "boosted_results"),
    )
)
MODEL_PATH = DEFAULT_ARTIFACT_DIR / "boosted_model.joblib"
SCHEMA_PATH = DEFAULT_ARTIFACT_DIR / "categorical_schema.joblib"

st.title("🎧 KKBox Customer Churn — Interactive Demo")
st.caption(
    "Portfolio demonstration using a historical 2015–2017 KKBox model. "
    "All profiles below are synthetic; no real customer data is used."
)

st.warning(
    "This is not a live retention service. The model score is an uncalibrated "
    "ranking score—not an individual probability—and the model must not be used "
    "for real customer outreach."
)

@st.cache_resource
def load_artifacts(model_path, schema_path):
    model = joblib.load(model_path)
    schema = joblib.load(schema_path)
    return model, schema

if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
    st.error(
        "Model artifacts were not found. Set `KKBOX_ARTIFACT_DIR` to the folder "
        "containing `boosted_model.joblib` and `categorical_schema.joblib`."
    )
    st.code(
        '$env:KKBOX_ARTIFACT_DIR = "C:\\project\\kkbox_processed\\boosted_results"\n'
        'python -m streamlit run C:\\project\\kkbox_demo_app.py',
        language="powershell",
    )
    st.stop()

model, category_schema = load_artifacts(str(MODEL_PATH), str(SCHEMA_PATH))

# Build plausible synthetic profiles. These are illustrative inputs, not
# sampled or copied from any particular KKBox customer.
def profile(name):
    base = {
        "age": 32.0,
        "tenure_months_at_cutoff": 18.0,
        "log_rows_last_month": 80.0,
        "listening_secs_last3": 180000.0,
        "listening_secs_prev3": 210000.0,
        "listening_secs_last6": 390000.0,
        "completed_plays_last3": 250.0,
        "daily_unique_song_sum_last3": 420.0,
        "active_months_last3": 3.0,
        "active_months_prev3": 3.0,
        "log_rows_history": 900.0,
        "transactions_last_month": 1.0,
        "transactions_last3": 1.0,
        "transactions_prev3": 1.0,
        "transactions_history": 12.0,
        "paid_amount_last3": 149.0,
        "auto_renew_count_last3": 1.0,
        "cancel_count_last3": 0.0,
        "weighted_avg_plan_days_last3": 30.0,
        "city": "1",
        "gender": "Unknown",
        "registered_via": "7",
    }

    if name == "Renewal friction example":
        base.update({
            "auto_renew_count_last3": 0.0,
            "transactions_last_month": 0.0,
            "transactions_last3": 0.0,
            "paid_amount_last3": 0.0,
            "weighted_avg_plan_days_last3": np.nan,
        })
    elif name == "Cancellation example":
        base.update({
            "auto_renew_count_last3": 0.0,
            "cancel_count_last3": 1.0,
            "transactions_last_month": 1.0,
            "paid_amount_last3": 149.0,
            "listening_secs_last3": 90000.0,
        })
    elif name == "Lower-risk comparison example":
        base.update({
            "auto_renew_count_last3": 3.0,
            "cancel_count_last3": 0.0,
            "transactions_last_month": 1.0,
            "transactions_last3": 3.0,
            "paid_amount_last3": 447.0,
            "listening_secs_last3": 240000.0,
            "listening_secs_prev3": 210000.0,
            "active_months_last3": 3.0,
        })
    return base

with st.sidebar:
    st.header("Synthetic profile")
    scenario = st.selectbox(
        "Choose an example",
        [
            "Renewal friction example",
            "Cancellation example",
            "Lower-risk comparison example",
        ],
    )
    st.markdown("These controls change only the synthetic example.")
    preset = profile(scenario)

    auto_options = [0, 1, 2, 3]
    cancel_options = [0, 1, 2]
    transaction_options = [0, 1, 2]
    payment_options = [0, 149, 298, 447]
    listening_options = [0, 30000, 90000, 180000, 300000]

    auto_default = int(preset["auto_renew_count_last3"])
    cancel_default = int(preset["cancel_count_last3"])
    transaction_default = int(preset["transactions_last_month"])
    payment_default = int(preset["paid_amount_last3"])
    listening_default = int(preset["listening_secs_last3"])

    auto_renew = st.selectbox(
        "Auto-renew transactions (last 3 months)",
        auto_options,
        index=auto_options.index(auto_default),
        key=f"auto_renew_{scenario}",
    )
    cancellations = st.selectbox(
        "Cancellation transactions (last 3 months)",
        cancel_options,
        index=cancel_options.index(cancel_default),
        key=f"cancellations_{scenario}",
    )
    recent_transaction = st.selectbox(
        "Transactions last month",
        transaction_options,
        index=transaction_options.index(transaction_default),
        key=f"transactions_{scenario}",
    )
    paid_last3 = st.selectbox(
        "Recorded paid amount (last 3 months)",
        payment_options,
        index=payment_options.index(payment_default),
        key=f"payment_{scenario}",
    )
    listening = st.select_slider(
        "Listening seconds (last 3 months)",
        options=listening_options,
        value=listening_default,
        key=f"listening_{scenario}",
    )

row = profile(scenario)
row.update({
    "auto_renew_count_last3": float(auto_renew),
    "cancel_count_last3": float(cancellations),
    "transactions_last_month": float(recent_transaction),
    "paid_amount_last3": float(paid_last3),
    "listening_secs_last3": float(listening),
})

# Transform one synthetic record to the exact training feature schema.
X = pd.DataFrame([{feature: row[feature] for feature in FEATURES}])
for col in NUMERIC_FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
for col in CATEGORICAL_FEATURES:
    values = X[col].astype("string").fillna("Unknown").replace("", "Unknown")
    X[col] = pd.Categorical(values, categories=category_schema[col])

score = float(model.predict_proba(X)[:, 1][0])

left, right = st.columns([1, 1])
with left:
    st.subheader("Model output")
    st.metric("Raw ranking score", f"{score:.3f}")
    st.caption(
        "Use this score to compare synthetic profiles in the demo only. "
        "It is not calibrated and must not be interpreted as a churn probability."
    )
with right:
    st.subheader("Suggested follow-up")
    if cancellations > 0:
        action = "Review cancellation activity; understand the reason and offer a relevant resolution."
    elif auto_renew == 0 or recent_transaction == 0 or paid_last3 == 0:
        action = "Check expiry and renewal status; send a timely reminder. Troubleshoot payment only if an issue is confirmed."
    elif listening == 0:
        action = "Consider a product-adoption check-in; inactivity alone should not trigger an expensive intervention."
    else:
        action = "Personalized retention review; check recent account context."
    st.info(action)

st.subheader("Historical model evidence")
metric_cols = st.columns(4)
metric_cols[0].metric("Later-cohort average precision", "0.574")
metric_cols[1].metric("Later-cohort ROC-AUC", "0.851")
metric_cols[2].metric("Top-decile churn rate", "53.7%")
metric_cols[3].metric("Overall later-cohort churn", "9.0%")
st.caption(
    "Historical evaluation only. The top decile captured 59.7% of observed "
    "churners in the 2017 later cohort; this does not show that outreach prevents churn."
)

st.subheader("Evaluation visuals")
reports = PROJECT_ROOT / "reports"
chart_files = [
    ("holdout_precision_recall_comparison.png", "Precision–recall"),
    ("holdout_lift_comparison.png", "Lift by decile"),
    ("permutation_importance.png", "Permutation importance"),
]
for filename, caption in chart_files:
    image = reports / filename
    if image.exists():
        st.image(str(image), caption=caption, width="stretch")

st.divider()
st.caption(
    "Demo safeguards: synthetic inputs only; no file uploads; no customer IDs; "
    "no retention action is sent or stored. Historical KKBox model, not for production use."
)
