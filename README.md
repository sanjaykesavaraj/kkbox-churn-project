# KKBox Churn Prediction Project

A time-aware churn-classification and retention-prioritization project using the WSDM/KKBox subscription dataset. The workflow reduces large event files to customer-month summaries, builds point-in-time cohort snapshots, compares a linear baseline with boosted trees, evaluates imbalanced-data performance, and translates risk rankings into a testable retention plan.

## Current project status

* Raw listening logs were chunk-processed and reduced from tens of GB to monthly Parquet summaries while retaining 27 months of customer history.
* Transactions from the original and refreshed files were combined and exact duplicate records removed.
* Features use only records from months before each cohort’s expiry month.
* A logistic/SGD baseline and a HistGradientBoosting model were compared.
* The boosted model has stronger internal validation and out-of-time ranking performance.
* Risk bands, action-queue output, calibration diagnostics, and a proposed A/B test plan were produced.
* No live retention experiment has been run; the data is historical and does not include randomized outreach outcomes.

## Data and target

Inputs expected in `C:\\project`:

* `train.csv` — earlier labeled cohort, 992,931 customers.
* `train\_v2.csv` — later labeled cohort, 970,960 customers.
* `transactions.csv` — transaction history through February 2017.
* `transactions\_v2.csv` — refreshed transaction records through March 2017.
* `user\_logs.csv` — daily listening history January 2015–February 2017.
* `user\_logs\_v2.csv` — March 2017 listening activity.
* `members\_v3.csv` — member attributes.

The target is `is\_churn`. For this project the earlier cohort is used for model development and the later cohort is treated as an out-of-time evaluation cohort. The feature cutoff is the first day of the target expiry month: earlier-cohort features use data through January 2017; later-cohort features use data through February 2017. Target-month information is excluded from features.

## Environment

Python was run from PowerShell. The project used pandas, PyArrow, DuckDB, scikit-learn, joblib, and Matplotlib. Install missing packages in the same Python environment used to run the scripts:

```powershell
python -m pip install pandas pyarrow duckdb scikit-learn matplotlib joblib
```

## Processing and analysis sequence

Run the local scripts from `C:\\project`. Scripts and raw files should remain outside the generated output folder; raw source files are never overwritten.

1. Inspect input names, sizes, headers, and sample rows.
2. Audit row counts and date coverage for each dataset.
3. Chunk-read daily listening logs, filter to the union of labeled customer IDs, aggregate by `msno` and month, and save Parquet.
4. Validate month coverage and customer-month uniqueness.
5. Inspect raw `total\_secs` anomalies; rebuild only that feature using the documented daily cleaning rule: negative/non-finite values set to zero and daily values capped at 86,400 seconds.
6. Audit expiry-date matches across both transaction files.
7. Deduplicate exact transaction rows, produce monthly transaction summaries, and create cohort expiry anchors.
8. Build one point-in-time row per customer per labeled cohort, joining prior-window listening, transaction, and member features.
9. Validate duplicates, labels, expiry anchors, ranges, and missing values.
10. Train and evaluate the baseline and boosted models; generate precision–recall and lift/gains tables and charts.
11. Analyze feature importance and descriptive churn rates for selected signals.
12. Produce later-cohort risk bands and an outreach queue; evaluate calibrated scores separately.

### Main processed files

* `kkbox\_processed/logs\_monthly.parquet` — initial monthly listening aggregates; contains uncleaned `total\_secs` and is retained for audit.
* `kkbox\_processed/logs\_monthly\_clean.parquet` — monthly listening features with cleaned seconds.
* `kkbox\_processed/transactions\_monthly.parquet` — customer-month transaction summaries.
* `kkbox\_processed/cohort\_anchors.parquet` — labels and target expiry anchors by cohort/customer.
* `kkbox\_processed/snapshot\_features.parquet` — model-ready point-in-time cohort snapshots.

### Model and analysis outputs

* `kkbox\_processed/baseline\_results/` — baseline estimator, metrics, lift tables, and PR curves.
* `kkbox\_processed/boosted\_results/` — boosted estimator, metrics, lift tables, permutation importance, retention-signal rates, calibration outputs, and later-cohort queue.
* `kkbox\_processed/model\_comparison/` — model comparisons and charts.
* `top\_decile\_outreach\_queue.csv` excludes the churn label; labeled evaluation scores are stored separately for analysis only.

## Model results

The boosted classifier was trained with class balancing and outperformed the baseline on the internal validation split and the later cohort.

|Metric|Earlier internal validation|Later out-of-time cohort|
|-|-:|-:|
|Churn rate|6.39%|8.99%|
|Boosted average precision|0.562|0.574|
|Boosted ROC-AUC|0.893|0.851|
|Precision at validation-selected threshold|42.7%|51.0%|
|Recall at validation-selected threshold|66.8%|61.5%|
|Lift at threshold|6.68×|5.67×|

On the later cohort, the model’s top decile had 53.7% observed churn, around 5.97× the overall cohort rate, and contained 59.7% of observed churners. The second decile had 11.8% churn; decile 3 had 5.1%, below the 8.99% cohort average. Current descriptive bands are therefore: High = decile 1, Elevated = decile 2, Moderate = deciles 3–6, Low = deciles 7–10.

A fixed validation threshold flagged 10.85% of the later cohort rather than exactly 10%. If outreach capacity is fixed, rank current customers and take the chosen top share rather than rely on a static threshold.

## Calibration status

The raw class-weighted model score is a ranking score, not automatically a probability. An isotonic calibrator fitted on internal validation scores improved later-cohort Brier score from 0.114 to 0.055 and log loss from 0.386 to 0.209. However, the highest-risk calibration bin still underpredicted churn (45.2% mean calibrated score versus 54.1% observed). Treat calibration as provisional; use risk ranking and cohort-observed band rates for prioritization until recent labels are available for recalibration.

Key results



On the later KKBox cohort, the boosted model achieved \*\*0.574 average precision\*\*

and \*\*0.851 ROC-AUC\*\*. Its highest-risk 10% had \*\*53.7% observed churn\*\*

versus \*\*9.0% overall\*\*, capturing \*\*59.7% of churners\*\*.



> Historical KKBox data from 2015–2017; this is a batch-scoring demonstration,

> not a live retention system.



Model performance



!\[Precision-recall comparison](reports/holdout\_precision\_recall\_comparison.png)



!\[Lift by risk decile](reports/holdout\_lift\_comparison.png)



!\[Cumulative gains](reports/cumulative\_gains.png)



What drives risk ranking?



!\[Permutation importance](reports/permutation\_importance.png)

## Retention interpretation

Strong descriptive signals in the later cohort included no recent auto-renew activity, a recent cancellation record, and no recent transaction/payment event. These do not establish causes: no transaction may reflect plan length or renewal timing, and a cancellation indicator does not identify the customer’s reason. A simple no-listening-logs flag was not consistently higher-risk than the cohort average, so do not target inactivity alone.

Initial actions to test:

* Cancellation signal: service-recovery follow-up and issue resolution.
* No recent renewal/payment event: verify expiry and renewal status; send a timely reminder; troubleshoot only when a problem is confirmed.
* Other high-risk customers: customer-success or product-adoption check-in based on available context.
* Lower-risk bands: standard lifecycle communication, without default incentives.

The companion `KKBox\_Retention\_Playbook.md` and `KKBox\_Retention\_Experiment\_Plan.md` document the operational recommendations and proposed randomized evaluation.

## Important limitations

* KKBox data is historical (2015–2017) and may not represent a current SaaS or another market.
* Only two labeled cohort periods are available. The later cohort is an out-of-time evaluation, but results have already been reviewed for model comparison and are not a completely untouched final test.
* Model explanations and group churn rates are associations, not causal effects.
* The data has no randomized intervention history, intervention costs, or customer lifetime value; the model cannot establish which retention action works or its ROI.
* Missing demographic fields and the age distribution limit demographic interpretation. Review fairness and applicable policies before any operational use.
* The proposed top-decile queue is historical and must not be used as a real-world contact list.

