from pathlib import Path
import duckdb

DATA_DIR = Path(r"C:\project")
OUT_DIR = DATA_DIR / "kkbox_processed"
TEMP_DIR = OUT_DIR / "duckdb_temp"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

logs_path = (OUT_DIR / "logs_monthly_clean.parquet").as_posix()
tx_path = (OUT_DIR / "transactions_monthly.parquet").as_posix()
anchors_path = (OUT_DIR / "cohort_anchors.parquet").as_posix()
members_path = (DATA_DIR / "members_v3.csv").as_posix()
output_path = (OUT_DIR / "snapshot_features.parquet").as_posix()
temp_path = TEMP_DIR.as_posix()

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET threads=4")
con.execute(f"SET temp_directory='{temp_path}'")

print("Building point-in-time customer snapshots...")

con.execute(f"""
    COPY (
        WITH c AS (
            SELECT
                msno,
                cohort,
                target_expiry_month,
                anchor_expiry_date,
                is_churn,
                STRPTIME(target_expiry_month || '01', '%Y%m%d') AS cutoff_date
            FROM read_parquet('{anchors_path}')
        ),

        logs AS (
            SELECT
                c.msno,
                c.cohort,

                SUM(CASE
                    WHEN l.month = STRFTIME(c.cutoff_date - INTERVAL 1 MONTH, '%Y%m')
                    THEN l.log_rows ELSE 0 END
                ) AS log_rows_last_month,

                SUM(CASE
                    WHEN l.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND l.month < c.target_expiry_month
                    THEN l.total_secs_clean ELSE 0 END
                ) AS listening_secs_last3,

                SUM(CASE
                    WHEN l.month >= STRFTIME(c.cutoff_date - INTERVAL 6 MONTH, '%Y%m')
                     AND l.month < STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                    THEN l.total_secs_clean ELSE 0 END
                ) AS listening_secs_prev3,

                SUM(CASE
                    WHEN l.month >= STRFTIME(c.cutoff_date - INTERVAL 6 MONTH, '%Y%m')
                     AND l.month < c.target_expiry_month
                    THEN l.total_secs_clean ELSE 0 END
                ) AS listening_secs_last6,

                SUM(CASE
                    WHEN l.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND l.month < c.target_expiry_month
                    THEN l.num_100 ELSE 0 END
                ) AS completed_plays_last3,

                SUM(CASE
                    WHEN l.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND l.month < c.target_expiry_month
                    THEN l.num_unq ELSE 0 END
                ) AS daily_unique_song_sum_last3,

                COUNT(DISTINCT CASE
                    WHEN l.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND l.month < c.target_expiry_month
                     AND l.log_rows > 0
                    THEN l.month END
                ) AS active_months_last3,

                COUNT(DISTINCT CASE
                    WHEN l.month >= STRFTIME(c.cutoff_date - INTERVAL 6 MONTH, '%Y%m')
                     AND l.month < STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND l.log_rows > 0
                    THEN l.month END
                ) AS active_months_prev3,

                SUM(CASE
                    WHEN l.month < c.target_expiry_month
                    THEN l.log_rows ELSE 0 END
                ) AS log_rows_history

            FROM c
            LEFT JOIN read_parquet('{logs_path}') AS l
                ON l.msno = c.msno
               AND l.month < c.target_expiry_month
            GROUP BY c.msno, c.cohort
        ),

        transactions AS (
            SELECT
                c.msno,
                c.cohort,

                SUM(CASE
                    WHEN t.month = STRFTIME(c.cutoff_date - INTERVAL 1 MONTH, '%Y%m')
                    THEN t.transaction_count ELSE 0 END
                ) AS transactions_last_month,

                SUM(CASE
                    WHEN t.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND t.month < c.target_expiry_month
                    THEN t.transaction_count ELSE 0 END
                ) AS transactions_last3,

                SUM(CASE
                    WHEN t.month >= STRFTIME(c.cutoff_date - INTERVAL 6 MONTH, '%Y%m')
                     AND t.month < STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                    THEN t.transaction_count ELSE 0 END
                ) AS transactions_prev3,

                SUM(CASE
                    WHEN t.month < c.target_expiry_month
                    THEN t.transaction_count ELSE 0 END
                ) AS transactions_history,

                SUM(CASE
                    WHEN t.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND t.month < c.target_expiry_month
                    THEN t.paid_amount_sum ELSE 0 END
                ) AS paid_amount_last3,

                SUM(CASE
                    WHEN t.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND t.month < c.target_expiry_month
                    THEN t.auto_renew_count ELSE 0 END
                ) AS auto_renew_count_last3,

                SUM(CASE
                    WHEN t.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND t.month < c.target_expiry_month
                    THEN t.cancel_count ELSE 0 END
                ) AS cancel_count_last3,

                SUM(CASE
                    WHEN t.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                     AND t.month < c.target_expiry_month
                    THEN t.avg_plan_days * t.transaction_count ELSE 0 END
                ) / NULLIF(
                    SUM(CASE
                        WHEN t.month >= STRFTIME(c.cutoff_date - INTERVAL 3 MONTH, '%Y%m')
                         AND t.month < c.target_expiry_month
                        THEN t.transaction_count ELSE 0 END
                    ), 0
                ) AS weighted_avg_plan_days_last3

            FROM c
            LEFT JOIN read_parquet('{tx_path}') AS t
                ON t.msno = c.msno
               AND t.month < c.target_expiry_month
            GROUP BY c.msno, c.cohort
        ),

        member_attributes AS (
            SELECT
                msno,
                MAX(TRY_CAST(city AS INTEGER)) AS city,
                MAX(TRY_CAST(bd AS INTEGER)) AS bd,
                MAX(gender) AS gender,
                MAX(registered_via) AS registered_via,
                MAX(registration_init_time) AS registration_init_time
            FROM read_csv_auto('{members_path}', all_varchar=true)
            GROUP BY msno
        )

        SELECT
            c.msno,
            c.cohort,
            c.target_expiry_month,
            c.anchor_expiry_date,
            c.is_churn,

            CASE
                WHEN m.bd BETWEEN 13 AND 100 THEN m.bd
                ELSE NULL
            END AS age,

            m.city,
            m.gender,
            m.registered_via,

            DATE_DIFF(
                'month',
                TRY_STRPTIME(m.registration_init_time, '%Y%m%d'),
                c.cutoff_date
            ) AS tenure_months_at_cutoff,

            COALESCE(l.log_rows_last_month, 0) AS log_rows_last_month,
            COALESCE(l.listening_secs_last3, 0) AS listening_secs_last3,
            COALESCE(l.listening_secs_prev3, 0) AS listening_secs_prev3,
            COALESCE(l.listening_secs_last6, 0) AS listening_secs_last6,
            COALESCE(l.completed_plays_last3, 0) AS completed_plays_last3,
            COALESCE(l.daily_unique_song_sum_last3, 0)
                AS daily_unique_song_sum_last3,
            COALESCE(l.active_months_last3, 0) AS active_months_last3,
            COALESCE(l.active_months_prev3, 0) AS active_months_prev3,
            COALESCE(l.log_rows_history, 0) AS log_rows_history,

            COALESCE(t.transactions_last_month, 0) AS transactions_last_month,
            COALESCE(t.transactions_last3, 0) AS transactions_last3,
            COALESCE(t.transactions_prev3, 0) AS transactions_prev3,
            COALESCE(t.transactions_history, 0) AS transactions_history,
            COALESCE(t.paid_amount_last3, 0) AS paid_amount_last3,
            COALESCE(t.auto_renew_count_last3, 0) AS auto_renew_count_last3,
            COALESCE(t.cancel_count_last3, 0) AS cancel_count_last3,
            t.weighted_avg_plan_days_last3

        FROM c
        LEFT JOIN logs AS l USING (msno, cohort)
        LEFT JOIN transactions AS t USING (msno, cohort)
        LEFT JOIN member_attributes AS m USING (msno)
    )
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nSnapshot dataset created.")
print(con.execute(f"""
    SELECT
        cohort,
        COUNT(*) AS customers,
        SUM(is_churn) AS churned_customers,
        AVG(is_churn) AS churn_rate
    FROM read_parquet('{output_path}')
    GROUP BY cohort
    ORDER BY cohort
""").df().to_string(index=False))

print(f"\nOutput: {output_path}")