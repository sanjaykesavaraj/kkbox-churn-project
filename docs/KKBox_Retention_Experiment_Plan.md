# KKBox Churn Retention — A/B Experiment Plan

**Status:** Proposed protocol; no intervention experiment has been run.  
**Purpose:** Determine whether a retention action causes a reduction in churn, beyond the model’s ability to identify high-risk customers.

## 1. What this experiment can answer

The churn model ranks risk. Historical KKBox labels and behavioral data do not record randomized retention actions, so they cannot establish whether a reminder, service-recovery contact, or offer prevents churn. This experiment is designed to estimate the incremental effect of an action.

## 2. Proposed first experiment

### Population

Customers who:

- Are in the currently eligible paid-subscriber population.
- Have an upcoming subscription expiry within the campaign window.
- Fall in the top 10% of the current scored cohort, ranked by risk score.
- Are contactable under the company’s communication and consent rules.

Do not use the historical 2017 customer list as a live outreach list. The KKBox files are historical project data; a real experiment requires a current eligible population and current model inputs.

### Stratified action routes

Assign each eligible customer an action route before randomization, using available signals:

1. **Cancellation signal:** cancellation transaction present in the recent window → service-recovery conversation; diagnose and resolve the issue.
2. **Renewal-readiness signal:** no recent auto-renew/transaction/payment event → verify the actual expiry and renewal status; send an appropriate reminder or troubleshoot only a confirmed issue.
3. **Other high-risk customers:** personalized customer-success check-in or low-cost product-adoption support.

Do not make a discount the default treatment. If testing an incentive, run it as a separate treatment arm with its own cost and margin analysis.

### Randomization

Within each action route, randomly assign eligible customers 1:1:

- **Treatment:** the specified, standardized action.
- **Control:** business-as-usual communications and support.

Randomize at the customer/account level, preserve assignment for repeat scoring, and prevent customers from receiving conflicting treatment arms. If one organization has multiple users, randomize at the organization level where account-level spillover is possible.

## 3. Outcomes and analysis

### Primary outcome

**Churn within 30 days after subscription expiry**, using the business’s documented churn definition. Report the treatment-minus-control absolute churn difference and a 95% confidence interval. A negative difference indicates lower churn in treatment.

### Secondary outcomes

- Renewal within 30 days of expiry.
- Contact delivery, response, and resolution rates.
- Customer complaints or opt-outs.
- Incentive cost and incremental retained contribution margin, when available.

Analyze customers in their assigned group (**intention-to-treat**), even if the contact was not delivered or the customer did not respond. Report delivered-contact results separately as descriptive diagnostics, not as the primary causal estimate.

### Sample-size planning

For an illustrative high-risk baseline churn rate near 54%, a two-sided 5% significance level and 80% power require approximately:

- **1,600 customers per arm** to detect a 5 percentage-point absolute reduction.
- **4,400 customers per arm** to detect a 3 percentage-point absolute reduction.

These are normal-approximation planning figures, not a guarantee. Recalculate using the current eligible population, the chosen minimum detectable effect, expected attrition/non-delivery, route sizes, and any multiple-arm correction before launch. Do not assume the historical 2017 cohort size equals live monthly volume.

## 4. Decision rules

Before launch, register:

- Eligibility rules and the model version/scoring date.
- The exact treatment content, channel, timing, and route logic.
- Primary outcome window and exclusions.
- Minimum detectable effect and sample size.
- Guardrails and stopping criteria.

Expand an intervention only if it reduces churn by a practically meaningful amount and does not breach complaint, opt-out, or cost guardrails. A statistically significant result with negative unit economics is not a successful retention action.

## 5. Monitoring and guardrails

- Monitor randomization balance by risk decile, action route, and key non-sensitive operational characteristics.
- Check contact delivery, duplicate contacts, opt-outs, and customer complaints.
- Keep control customers on normal service; never withhold required support or billing notices.
- Avoid repeated discounting or treatment leakage between groups.
- Review outcomes by route, but do not over-interpret small subgroups.
- Reassess model ranking, feature coverage, and churn calibration on contemporary data before deployment.

## 6. Project interpretation

The historical later cohort’s top decile had 53.7% observed churn, about 5.97 times its cohort average. This supports prioritizing a high-risk group for a live experiment; it does **not** imply that contacting those customers will retain them. Only randomized treatment-versus-control results can estimate intervention impact.
