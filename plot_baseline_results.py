from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path(r"C:\project\kkbox_processed\baseline_results")

# 1. Precision-recall curves
val_pr = pd.read_csv(RESULTS / "validation_precision_recall.csv")
holdout_pr = pd.read_csv(RESULTS / "holdout_precision_recall.csv")
val_metrics = pd.read_csv(RESULTS / "validation_metrics.csv").iloc[0]
holdout_metrics = pd.read_csv(RESULTS / "holdout_metrics.csv").iloc[0]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(
    val_pr["recall"], val_pr["precision"],
    label=f"Earlier validation (AP={val_metrics['average_precision']:.3f})"
)
ax.plot(
    holdout_pr["recall"], holdout_pr["precision"],
    label=f"Later holdout (AP={holdout_metrics['average_precision']:.3f})"
)
ax.set(
    xlabel="Recall",
    ylabel="Precision",
    title="Churn model: precision–recall",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "precision_recall_comparison.png", dpi=160)
plt.close(fig)

# 2. Lift by decile. Group 1 is the highest-risk decile.
val_lift = pd.read_csv(RESULTS / "validation_lift.csv")
holdout_lift = pd.read_csv(RESULTS / "holdout_lift.csv")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(val_lift["group"], val_lift["lift"], marker="o", label="Earlier validation")
ax.plot(holdout_lift["group"], holdout_lift["lift"], marker="o", label="Later holdout")
ax.axhline(1, color="gray", linestyle="--", label="Overall churn rate")
ax.set(
    xlabel="Risk decile (1 = highest risk)",
    ylabel="Lift over overall churn rate",
    title="Churn model: lift by risk decile",
    xticks=range(1, 11),
)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "lift_by_decile.png", dpi=160)
plt.close(fig)

# 3. Cumulative gains: what share of churners is captured as outreach expands?
fig, ax = plt.subplots(figsize=(8, 6))

for table, label in [
    (val_lift, "Earlier validation"),
    (holdout_lift, "Later holdout"),
]:
    customers = table["cumulative_customers"] / table["customers"].sum()
    churn_recall = table["cumulative_churn_recall"]
    ax.plot(customers, churn_recall, marker="o", label=label)

ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random targeting")
ax.set(
    xlabel="Share of customers contacted (highest risk first)",
    ylabel="Share of churners captured",
    title="Churn model: cumulative gains",
    xlim=(0, 1),
    ylim=(0, 1),
)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "cumulative_gains.png", dpi=160)
plt.close(fig)

print("Charts saved in:")
print(RESULTS)
for filename in [
    "precision_recall_comparison.png",
    "lift_by_decile.png",
    "cumulative_gains.png",
]:
    print(" -", filename)