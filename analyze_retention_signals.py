from pathlib import Path
import duckdb

DATA = Path(r"C:\project\kkbox_processed\snapshot_features.parquet")
OUT = Path(r"C:\project\kkbox_processed\boosted_results\retention_signal_rates.csv")
p = DATA.as_posix()

con = duckdb.connect()

query = f"""
WITH d AS (
    SELECT
        cohort,
        is_churn,
        CASE
            WHEN auto_renew_count_last3 = 0 THEN '0'
            WHEN auto_renew_count_last3 = 1 THEN '1'
            ELSE '2+'
        END AS auto_renew_band,
        CASE
            WHEN cancel_count_last3 = 0 THEN '0'
            ELSE '1+'
        END AS cancel_band,
        CASE
            WHEN log_rows_last_month = 0 THEN 'No listening logs last month'
            ELSE 'Listening logs last month'
        END AS activity_band,
        CASE
            WHEN transactions_last_month = 0 THEN 'No transaction last month'
            ELSE 'Transaction last month'
        END AS transaction_band,
        CASE
            WHEN paid_amount_last3 = 0 THEN 'No recorded payment last 3 months'
            ELSE 'Payment recorded last 3 months'
        END AS payment_band,
        CASE
            WHEN listening_secs_last3 = 0 THEN 'No listening seconds last 3 months'
            ELSE 'Listening seconds last 3 months'
        END AS listening_band
    FROM read_parquet('{p}')
),
grouped AS (
    SELECT cohort, 'Auto-renew transactions, last 3 months' AS signal,
           auto_renew_band AS band, COUNT(*) AS customers,
           SUM(is_churn) AS churners, AVG(is_churn) AS churn_rate
    FROM d GROUP BY cohort, band

    UNION ALL
    SELECT cohort, 'Cancellation transactions, last 3 months',
           cancel_band, COUNT(*), SUM(is_churn), AVG(is_churn)
    FROM d GROUP BY cohort, cancel_band

    UNION ALL
    SELECT cohort, 'Listening activity, last month',
           activity_band, COUNT(*), SUM(is_churn), AVG(is_churn)
    FROM d GROUP BY cohort, activity_band

    UNION ALL
    SELECT cohort, 'Transaction activity, last month',
           transaction_band, COUNT(*), SUM(is_churn), AVG(is_churn)
    FROM d GROUP BY cohort, transaction_band

    UNION ALL
    SELECT cohort, 'Payment recorded, last 3 months',
           payment_band, COUNT(*), SUM(is_churn), AVG(is_churn)
    FROM d GROUP BY cohort, payment_band

    UNION ALL
    SELECT cohort, 'Listening seconds, last 3 months',
           listening_band, COUNT(*), SUM(is_churn), AVG(is_churn)
    FROM d GROUP BY cohort, listening_band
)
SELECT *
FROM grouped
ORDER BY signal, cohort, band
"""

result = con.execute(query).df()
result.to_csv(OUT, index=False)

print(result.to_string(index=False))
print(f"\nSaved: {OUT}")