from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(r"C:\project\kkbox_processed")
scores = (ROOT / "boosted_results" / "batch_demo" /
          "customer_risk_scores.parquet").as_posix()
snapshot = (ROOT / "snapshot_features.parquet").as_posix()
queue = ROOT / "boosted_results" / "batch_demo" / "outreach_queue.csv"

con = duckdb.connect()

print("Scored-file checks:")
print(con.execute(f"""
    SELECT
        COUNT(*) AS scored_customers,
        COUNT(DISTINCT msno) AS unique_customers,
        COUNT(*) FILTER (WHERE risk_score IS NULL) AS missing_scores,
        MIN(risk_score) AS min_score,
        MAX(risk_score) AS max_score
    FROM read_parquet('{scores}')
""").df().to_string(index=False))

print("\nObserved churn by scored risk band:")
print(con.execute(f"""
    SELECT
        s.risk_band,
        COUNT(*) AS customers,
        AVG(t.is_churn) AS observed_churn_rate
    FROM read_parquet('{scores}') AS s
    JOIN read_parquet('{snapshot}') AS t
      ON s.msno = t.msno
    WHERE t.cohort = 'train_v2'
    GROUP BY s.risk_band
    ORDER BY CASE s.risk_band
        WHEN 'High' THEN 1
        WHEN 'Elevated' THEN 2
        WHEN 'Moderate' THEN 3
        ELSE 4
    END
""").df().to_string(index=False))

print("\nOutreach queue checks:")
queue_df = pd.read_csv(queue, nrows=5)
print(f"File exists: {queue.exists()}")
print(f"Header columns: {list(queue_df.columns)}")
print(f"Has churn label column: {'is_churn' in queue_df.columns}")
print(f"Queue rows: {sum(1 for _ in open(queue, encoding='utf-8')) - 1:,}")