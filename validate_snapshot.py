from pathlib import Path
import duckdb

path = Path(r"C:\project\kkbox_processed\snapshot_features.parquet")
con = duckdb.connect()
p = path.as_posix()

print("Cohort checks:")
print(con.execute(f"""
    SELECT
        cohort,
        COUNT(*) AS rows,
        COUNT(DISTINCT msno) AS unique_customers,
        COUNT(*) FILTER (WHERE is_churn IS NULL) AS missing_labels,
        COUNT(*) FILTER (WHERE anchor_expiry_date IS NULL) AS missing_anchors,
        COUNT(*) FILTER (
            WHERE SUBSTR(anchor_expiry_date, 1, 6) <> target_expiry_month
        ) AS anchor_month_mismatches,
        MIN(is_churn) AS min_label,
        MAX(is_churn) AS max_label,
        AVG(is_churn) AS churn_rate
    FROM read_parquet('{p}')
    GROUP BY cohort
    ORDER BY cohort
""").df().to_string(index=False))

print("\nDuplicate cohort/customer rows:")
print(con.execute(f"""
    SELECT COUNT(*) AS duplicate_groups
    FROM (
        SELECT cohort, msno
        FROM read_parquet('{p}')
        GROUP BY cohort, msno
        HAVING COUNT(*) > 1
    )
""").df().to_string(index=False))

print("\nFeature-range checks:")
print(con.execute(f"""
    SELECT
        MIN(age) AS min_age,
        MAX(age) AS max_age,
        MIN(tenure_months_at_cutoff) AS min_tenure_months,
        MAX(tenure_months_at_cutoff) AS max_tenure_months,
        MIN(listening_secs_last3) AS min_listening_secs_last3,
        MAX(listening_secs_last3) AS max_listening_secs_last3,
        MIN(paid_amount_last3) AS min_paid_amount_last3,
        MAX(paid_amount_last3) AS max_paid_amount_last3,
        COUNT(*) FILTER (WHERE tenure_months_at_cutoff < 0) AS negative_tenure_rows,
        COUNT(*) FILTER (WHERE listening_secs_last3 < 0) AS negative_listening_rows,
        COUNT(*) FILTER (WHERE paid_amount_last3 < 0) AS negative_payment_rows
    FROM read_parquet('{p}')
""").df().to_string(index=False))

print("\nSelected feature missingness:")
print(con.execute(f"""
    SELECT
        cohort,
        COUNT(*) FILTER (WHERE age IS NULL) AS age_missing,
        COUNT(*) FILTER (WHERE city IS NULL) AS city_missing,
        COUNT(*) FILTER (WHERE gender IS NULL OR gender = '') AS gender_missing,
        COUNT(*) FILTER (
            WHERE weighted_avg_plan_days_last3 IS NULL
        ) AS plan_days_missing
    FROM read_parquet('{p}')
    GROUP BY cohort
    ORDER BY cohort
""").df().to_string(index=False))