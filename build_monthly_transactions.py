from pathlib import Path
import duckdb

DATA_DIR = Path(r"C:\project")
OUT_DIR = DATA_DIR / "kkbox_processed"
TEMP_DIR = OUT_DIR / "duckdb_temp"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

monthly_output = OUT_DIR / "transactions_monthly.parquet"
anchors_output = OUT_DIR / "cohort_anchors.parquet"

def sql_path(path):
    # DuckDB SQL paths use forward slashes.
    return path.as_posix().replace("'", "''")

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET threads=4")
con.execute(f"SET temp_directory='{sql_path(TEMP_DIR)}'")

train_path = sql_path(DATA_DIR / "train.csv")
train_v2_path = sql_path(DATA_DIR / "train_v2.csv")
tx_path = sql_path(DATA_DIR / "transactions.csv")
tx_v2_path = sql_path(DATA_DIR / "transactions_v2.csv")

print("Reading label cohorts...")
con.execute(f"""
    CREATE TEMP TABLE cohort_labels AS
    SELECT
        msno,
        'train' AS cohort,
        '201702' AS target_expiry_month,
        TRY_CAST(is_churn AS INTEGER) AS is_churn
    FROM read_csv_auto('{train_path}', all_varchar=true)

    UNION ALL

    SELECT
        msno,
        'train_v2' AS cohort,
        '201703' AS target_expiry_month,
        TRY_CAST(is_churn AS INTEGER) AS is_churn
    FROM read_csv_auto('{train_v2_path}', all_varchar=true)
""")

print("Combining transaction files and removing exact duplicate events...")
con.execute(f"""
    CREATE TEMP TABLE transactions_filtered AS
    SELECT DISTINCT
        t.msno,
        t.payment_method_id,
        t.payment_plan_days,
        t.plan_list_price,
        t.actual_amount_paid,
        t.is_auto_renew,
        t.transaction_date,
        t.membership_expire_date,
        t.is_cancel
    FROM (
        SELECT * FROM read_csv_auto('{tx_path}', all_varchar=true)
        UNION ALL BY NAME
        SELECT * FROM read_csv_auto('{tx_v2_path}', all_varchar=true)
    ) AS t
    WHERE EXISTS (
        SELECT 1
        FROM cohort_labels AS c
        WHERE c.msno = t.msno
    )
""")

print("Creating monthly transaction features...")
con.execute(f"""
    COPY (
        SELECT
            msno,
            STRFTIME(
                TRY_STRPTIME(transaction_date, '%Y%m%d'),
                '%Y%m'
            ) AS month,
            COUNT(*) AS transaction_count,
            SUM(TRY_CAST(actual_amount_paid AS DOUBLE)) AS paid_amount_sum,
            SUM(TRY_CAST(plan_list_price AS DOUBLE)) AS list_price_sum,
            AVG(TRY_CAST(payment_plan_days AS DOUBLE)) AS avg_plan_days,
            SUM(
                CASE WHEN TRY_CAST(is_auto_renew AS INTEGER) = 1
                     THEN 1 ELSE 0 END
            ) AS auto_renew_count,
            SUM(
                CASE WHEN TRY_CAST(is_cancel AS INTEGER) = 1
                     THEN 1 ELSE 0 END
            ) AS cancel_count,
            COUNT(DISTINCT payment_method_id) AS payment_method_count
        FROM transactions_filtered
        WHERE TRY_STRPTIME(transaction_date, '%Y%m%d') IS NOT NULL
        GROUP BY msno, month
    )
    TO '{sql_path(monthly_output)}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("Creating expiry anchors for the labeled cohorts...")
con.execute(f"""
    COPY (
        SELECT
            c.msno,
            c.cohort,
            c.target_expiry_month,
            c.is_churn,
            MAX(
                CASE
                    WHEN STRFTIME(
                        TRY_STRPTIME(t.membership_expire_date, '%Y%m%d'),
                        '%Y%m'
                    ) = c.target_expiry_month
                    THEN t.membership_expire_date
                END
            ) AS anchor_expiry_date
        FROM cohort_labels AS c
        LEFT JOIN transactions_filtered AS t
            ON c.msno = t.msno
        GROUP BY
            c.msno,
            c.cohort,
            c.target_expiry_month,
            c.is_churn
    )
    TO '{sql_path(anchors_output)}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nMonthly transaction output:")
print(con.execute(f"""
    SELECT
        COUNT(*) AS customer_month_rows,
        COUNT(DISTINCT msno) AS customers,
        MIN(month) AS first_month,
        MAX(month) AS last_month
    FROM read_parquet('{sql_path(monthly_output)}')
""").df().to_string(index=False))

print("\nAnchor coverage:")
print(con.execute(f"""
    SELECT
        cohort,
        COUNT(*) AS customers,
        COUNT(anchor_expiry_date) AS customers_with_anchor,
        COUNT(*) - COUNT(anchor_expiry_date) AS customers_without_anchor
    FROM read_parquet('{sql_path(anchors_output)}')
    GROUP BY cohort
    ORDER BY cohort
""").df().to_string(index=False))

print(f"\nCreated: {monthly_output}")
print(f"Created: {anchors_output}")