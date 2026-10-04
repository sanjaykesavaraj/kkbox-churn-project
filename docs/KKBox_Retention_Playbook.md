# KKBox Churn Prediction — Operational Retention Playbook

**Version:** 1.0 — evidence-based draft from the current modeling run  
**Purpose:** Translate churn-risk rankings into measurable retention actions.  
**Important:** This is a prioritization and test plan, not proof that any intervention prevents churn.

## Executive summary

A boosted-tree classifier currently provides the strongest ranking performance. On the later March-expiry cohort, its highest-risk 10% contained **53.7% observed churn**, approximately **5.97× the cohort-wide churn rate**, and captured **59.7% of all observed churners**. The next decile had an 11.8% churn rate; deciles 3–10 were below the cohort-wide 8.99% rate.

Recommended operating approach:

1. Prioritize the top risk decile for human review and targeted contact.
2. Use the second decile for lower-cost, automated renewal-readiness actions.
3. Do not spend broad discount budgets on everyone the model flags. Route outreach using observed cancellation, payment/renewal, and engagement signals.
4. Test each intervention against a randomized holdout before claiming retention impact.

The raw model output is a **ranking score**, not a calibrated individual probability of churn. An isotonic calibrator fitted on the earlier internal validation data improved later-cohort Brier score from 0.114 to 0.055 and log loss from 0.386 to 0.209. However, the later-cohort calibration bins still show underprediction in the highest-risk bin (mean calibrated score 45.2% versus 54.1% observed churn). Treat calibrated scores as provisional; use rankings and observed risk bands for prioritization until recalibration is checked on recent labeled data.

## Project and data summary

- Source: KKBox subscription churn competition dataset.
- Historical listening logs span January 2015–March 2017; transactions span January 2015–March 2017.
- Daily listening records were processed in chunks and summarized to customer-month; customer timelines were retained.
- The monthly listening table contains 27 months and 15.83 million customer-month rows. The cleaned Parquet output is approximately 0.68 GB, versus tens of GB of raw logs.
- Listening seconds were cleaned by setting negative/non-finite daily values to zero and capping daily values at 86,400 seconds. Invalid and capped counts were retained in the processing audit.
- Original and refreshed transaction files were combined, exact duplicate events removed, and transaction events summarized monthly.
- The model snapshot contains 992,931 February-expiry (`train`) rows and 970,960 March-expiry (`train_v2`) rows. All have an expiry anchor after combining transaction sources.
- Features are point-in-time: February-cohort features use data through January 2017; March-cohort features use data through February 2017. Target-month data is excluded.

## Model and evaluation

### Validation design

- A stratified 80/20 split of the earlier `train` cohort was used for model development and internal validation.
- The later `train_v2` cohort was evaluated as an out-of-time cohort.
- Two models were compared: a logistic/SGD baseline and `HistGradientBoostingClassifier`.
- The boosted model performed better on internal validation and on the later cohort.
- The later cohort has a higher churn prevalence (8.99% versus 6.39% in the earlier cohort), so average precision should be interpreted alongside the cohort prevalence.

### Boosted-model results

| Evaluation set | Churn prevalence | Average precision | ROC-AUC | Precision at validation-selected threshold | Recall | Lift at threshold |
|---|---:|---:|---:|---:|---:|---:|
| Earlier internal validation | 6.39% | 0.562 | 0.893 | 42.7% | 66.8% | 6.68× |
| Later out-of-time cohort | 8.99% | 0.574 | 0.851 | 51.0% | 61.5% | 5.67× |

The threshold was selected on internal validation assuming a 10% contact capacity. It flagged 10.85% of the later cohort, showing that a fixed score threshold does not guarantee a fixed contact volume as score distributions shift. For a hard capacity constraint, rank customers each cycle and contact the chosen top share.

### Later-cohort risk deciles

Decile 1 is the highest-risk 10%; each decile contains 97,096 customers.

| Risk decile | Observed churn rate | Lift vs. later-cohort average |
|---:|---:|---:|
| 1 | 53.71% | 5.97× |
| 2 | 11.78% | 1.31× |
| 3 | 5.07% | 0.56× |
| 4 | 4.09% | 0.46× |
| 5 | 3.57% | 0.40× |
| 6 | 3.28% | 0.37× |
| 7 | 2.83% | 0.31× |
| 8 | 2.52% | 0.28× |
| 9 | 2.11% | 0.24× |
| 10 | 0.96% | 0.11× |

The top 10% captures 59.7% of observed churners. The top 20% captures approximately 72.8%. Decile 2 is modestly above the overall cohort rate; decile 3 is below it. Therefore, the initial recommended bands are **High = decile 1**, **Elevated = decile 2**, **Moderate = deciles 3–6**, and **Low = deciles 7–10**. These are cohort-relative priority bands, not calibrated probability categories.

## Signals and their observed associations

Permutation importance ranked `auto_renew_count_last3`, `cancel_count_last3`, recent log activity, recent transaction activity, and recent payment amount highly. The following later-cohort group rates help describe direction:

- No auto-renew transactions in the previous three months: **42.4% churn**; one: 7.8%; two or more: 4.4%.
- One or more cancellation transactions in the previous three months: **22.8% churn**, versus 8.4% with none.
- No transaction in the previous month: **65.8% churn**, versus 6.5% with a transaction.
- No recorded payment in the previous three months: **80.3% churn**, versus 7.0% with a payment.
- No listening logs in the previous month: 7.4% churn, below the cohort average of 9.0%; this simple inactivity flag is not a consistent standalone risk trigger.

These are unadjusted associations. No recent transaction/payment does **not** necessarily mean billing failure: plan duration, renewal timing, free/promotional arrangements, and data coverage can affect whether an event appears in the window. Feature importance and group rates are not causal effects.

## Recommended retention routing

| Priority | Entry rule | Initial action to test | Guardrail |
|---|---|---|---|
| **High** | Top risk decile | Customer-success or retention-team review. Route by cancellation/renewal signals; offer a relevant resolution. | Do not automatically issue a discount. Confirm the customer’s context first. |
| **Elevated** | Second risk decile | Low-cost renewal-readiness reminder; check upcoming expiry and auto-renew/payment status. | A missing transaction is a prompt to verify status, not evidence of failed payment. |
| **Moderate** | Deciles 3–6 | Standard lifecycle messaging or a low-cost product-adoption prompt when supported by usage history. | Avoid expensive outbound effort based on model score alone. |
| **Low** | Deciles 7–10 | Standard communications; no special retention incentive by default. | Continue monitoring; do not infer zero risk. |

### Signal-based routing within High risk

1. **Cancellation recorded:** review the event and available support history; ask the customer what prompted cancellation; resolve the service issue or offer a suitable pause/flexible option if available.
2. **No recent auto-renew, payment, or transaction event:** verify expiry and renewal status; send a timely reminder; troubleshoot payment only if a real issue is confirmed.
3. **Usage signal:** use recent listening trend and product-adoption information as supporting context. Do not target solely on a one-month no-log flag.
4. **No clear signal:** use a personalized check-in and gather the reason for risk; do not assume price sensitivity.

## Experiment plan

The model estimates risk; it does not estimate treatment uplift. To measure whether the playbook works:

1. For eligible High-risk customers, randomly assign customers to a retention action or business-as-usual holdout. Stratify randomization by action route and, if practical, risk decile.
2. Keep a no-intervention control group. Avoid comparing contacted high-risk customers against all customers, because the groups have different baseline risk.
3. Predefine the outcome window based on the renewal/expiry process—such as renewal within 30 days of expiry—and use the same definition for treatment and control.
4. Report renewal/churn difference, confidence interval, contact and response rates, incentive cost, and net retained contribution margin if revenue data is available.
5. Roll out only actions with positive, economically meaningful incremental impact. Re-test offers rather than assuming a discount is effective.

## Monitoring and production considerations

- Track precision and recall at the actual contact capacity, average precision, ROC-AUC, decile lift, and observed churn by risk band.
- Monitor feature coverage, category changes, score distribution, and monthly drift. Revalidate on newer periods before expanding use.
- Keep risk scores labeled as **scores** until probability calibration is separately assessed on a representative, later validation period.
- Set a fixed contact capacity by ranking customers each cycle; do not rely on a static threshold to produce exactly 10% volume.
- Review outcomes and model performance across relevant customer groups. Demographic fields should not be used to justify differential treatment without a documented fairness and legal review.

## Limitations

- The data is historical (2015–2017) and reflects a Taiwan-based music subscription service; results may not transfer to a current SaaS business or another market.
- There are only two labeled cohort periods. The later cohort is a useful time-based evaluation, but after comparing models and reviewing results it should not be described as a completely untouched final test set.
- Many customers have missing age/gender information; those fields are not suitable as reliable operational explanations.
- The model does not include intervention history, customer lifetime value, support case detail, or intervention costs. It cannot determine who is persuadable or which action has positive ROI.
- Risk decile churn rates are observed historical cohort rates, not guaranteed future outcomes or calibrated individual probabilities.
