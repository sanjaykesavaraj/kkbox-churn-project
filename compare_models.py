from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"C:\project\kkbox_processed")
BASE = ROOT / "baseline_results"
BOOST = ROOT / "boosted_results"
OUT = ROOT / "model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

models = {
    "Logistic baseline": BASE,
    "Boosted trees": BOOST,
}

# Overall metric comparison
metric_frames = []
for model_name, folder in models.items():
    for period, filename in [
        ("Internal validation", "validation_metrics.csv"),
        ("Later cohort", "holdout_metrics.csv"),
    ]:
        df = pd.read_csv(folder / filename)
        df["model"] = model_name
        df["period"] = period
        metric_frames.append(df)

metrics = pd.concat(metric_frames, ignore_index=True)
metrics.to_csv(OUT / "model_metrics_comparison.csv", index=False)

# Top-decile operating results. Group 1 is the highest-risk decile.
top_decile_rows = []
for model_name, folder in models.items():
    for period, filename in [
        ("Internal validation", "validation_lift.csv"),
        ("Later cohort", "holdout_lift.csv"),
    ]:
        lift = pd.read_csv(folder / filename)
        top = lift.loc[lift["group"] == 1].iloc[0]
        top_decile_rows.append({
            "model": model_name,
            "period": period,
            "customers_in_top_decile": int(top["customers"]),
            "observed_churn_rate": top["churn_rate"],
            "lift": top["lift"],
            "share_of_all_churners_captured": top["cumulative_churn_recall"],
        })

top_decile = pd.DataFrame(top_decile_rows)
top_decile.to_csv(OUT / "top_decile_comparison.csv", index=False)

print("Top-decile comparison:")
print(top_decile.to_string(index=False))

# Precision-recall comparison: one chart per evaluation period.
for period, prefix in [
    ("Internal validation", "validation"),
    ("Later cohort", "holdout"),
]:
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, folder in models.items():
        curve = pd.read_csv(folder / f"{prefix}_precision_recall.csv")
        ap = metrics.loc[
            (metrics["model"] == model_name) & (metrics["period"] == period),
            "average_precision",
        ].iloc[0]

        ax.plot(
            curve["recall"],
            curve["precision"],
            label=f"{model_name} (AP={ap:.3f})",
        )

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision–recall: {period}",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_precision_recall_comparison.png", dpi=160)
    plt.close(fig)

# Lift comparison on the later cohort.
fig, ax = plt.subplots(figsize=(8, 6))
for model_name, folder in models.items():
    lift = pd.read_csv(folder / "holdout_lift.csv")
    ax.plot(lift["group"], lift["lift"], marker="o", label=model_name)

ax.axhline(1, color="gray", linestyle="--", label="Overall churn rate")
ax.set(
    xlabel="Risk decile (1 = highest risk)",
    ylabel="Lift over overall churn rate",
    title="Later cohort: lift by risk decile",
    xticks=range(1, 11),
)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "holdout_lift_comparison.png", dpi=160)
plt.close(fig)

print(f"\nComparison files saved in: {OUT}")