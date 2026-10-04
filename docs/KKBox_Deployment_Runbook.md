# KKBox Churn Batch-Scoring Deployment Runbook

**Deployment mode:** Scheduled batch scoring  
**Current status:** Historical offline demo validated; not approved for live customer outreach.  
**Model artifact:** `boosted_model.joblib` (HistGradientBoosting, trained on 80% of the earlier cohort).  
**Scoring output:** Risk ranking score, not a calibrated individual probability.

## 1. Purpose and safety boundary

This runbook describes how a subscription business could move from an offline churn model to controlled operation. The current KKBox data is from 2015–2017 and must only be used for a portfolio demonstration. Do not use the historical scored list for real outreach. A live deployment requires current data, permissioned use, new validation, and an approved experiment.

## 2. Versioned model package

Create an immutable package for each approved model version containing:

- Model file and serialized categorical feature schema.
- Feature names, definitions, units, cutoff rules, and missing-value handling.
- Training data window, label definition, prediction horizon, cohort split, and model parameters.
- Validation, temporal evaluation, calibration, and lift reports.
- Inference/scoring code and package versions.
- SHA-256 checksums and model owner/approval date.
- A rollback pointer to the prior approved version.

Do not put raw customer data or customer-level score files in the model package or source-control repository.

## 3. Input contract for a scoring batch

Require one row per eligible customer and scoring cycle, containing:

- Stable customer key (`msno` or approved internal ID).
- The exact feature columns used at training, with documented units and types.
- A feature cutoff date and batch/run ID.
- No target label in the scoring input.

Reject a batch if customer IDs are duplicated, required features are missing, cutoff dates are inconsistent, or feature coverage/ranges depart materially from training expectations. Unknown categories must follow the stored training schema.

## 4. Batch-scoring stages

### Offline validation (completed for the historical cohort)

- Score the later cohort without using its churn label as a feature.
- Confirm row count, unique IDs, missing scores, risk-band sizes, and that the outreach queue excludes labels.
- Reconcile observed churn by score decile against the evaluation report.

### Shadow mode (required before live outreach)

- Generate scores on current eligible customers, but do not act on them.
- Compare model rankings with existing retention prioritization.
- Check ingestion freshness, feature coverage, score distribution, duplicate counts, category drift, and score volume at the planned contact capacity.
- Store run metadata and aggregate monitoring data; restrict access to customer-level scores.
- Run for enough cycles to observe the normal renewal cadence and resolve pipeline/data defects.

### Controlled experiment

- After current-data validation, select eligible customers by capacity-based ranking.
- Randomize within pre-defined action routes to treatment and business-as-usual control.
- Measure churn/renewal in the pre-registered post-expiry window, plus contact response, complaints, opt-outs, and intervention cost.
- Do not infer intervention impact from contacted-versus-uncontacted observational comparisons.

### Gradual rollout

- Expand only after the experiment shows a worthwhile effect and guardrails pass.
- Start with a limited share of eligible volume, retain a control group where practical, and increase only with review.

## 5. Monitoring dashboard

### Data and pipeline

- Batch row count, duplicate customer IDs, missing required fields, invalid dates, and feature freshness.
- Share of customers with missing/imputed features and unknown categories.
- Feature distributions and drift indicators versus the training reference.
- Job success/failure, output row count, execution time, and input/output checksums.

### Model and business

- Score distribution, contact volume, and risk-band/decile sizes.
- Once outcomes mature: churn rate, precision, recall, average precision, lift, and calibration by cohort and risk band.
- Treatment-versus-control churn difference, confidence interval, contact/response rate, complaints/opt-outs, and incremental economics.

Do not report predictive performance for a cohort until its outcome window has matured and labels are available.

## 6. Alert and rollback conditions

Pause scoring or outreach and investigate if any of the following occur:

- Feature schema, cutoff, or label definition changes unexpectedly.
- Input freshness/coverage fails a pre-agreed threshold.
- Large unexplained drift, score collapse, or risk-band size anomaly.
- Matured-cohort lift/precision falls below the agreed floor.
- Calibration is materially poor when probabilities are being used.
- Privacy, fairness, customer-contact, or experiment guardrails are breached.

Rollback to the prior approved model or to business-as-usual prioritization. Do not silently retrain or replace production artifacts.

## 7. Current KKBox demonstration findings

- Later-cohort top decile: 97,096 of 970,960 customers; 53.7% observed churn; 5.97× cohort-average lift; 59.7% of observed churners captured.
- Batch-scoring checks: 970,960 unique customers, no missing scores, 97,096 queue rows, no label in the outreach CSV.
- Risk scores are rankings. An isotonic calibration fit on the earlier validation cohort improved later-cohort Brier score and log loss, but the highest-risk bin still underpredicted observed churn; probability estimates need current-label recalibration.
- Historical signals and risk rates are observational and do not show that an action will retain customers.

## 8. Go-live checklist

- [ ] Current, permissioned data approved for the intended use.
- [ ] Current churn definition and prediction horizon signed off.
- [ ] Point-in-time feature pipeline tested and leakage-audited.
- [ ] Versioned model/schema/code bundle checksum recorded.
- [ ] Current temporal validation and calibration reviewed.
- [ ] Shadow-mode monitoring passed.
- [ ] Experiment protocol, sample size, treatment, control, outcomes, and guardrails pre-registered.
- [ ] Customer-success operations trained; queue access restricted.
- [ ] Rollback owner and business-as-usual fallback documented.
