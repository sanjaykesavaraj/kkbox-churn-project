from pathlib import Path
import duckdb

file_path = Path(r"C:\project\kkbox_processed\logs_monthly.parquet")
con = duckdb.connect()

print("Overall total_secs diagnostics:")
print(con.execute(f"""
    SELECT
        MIN(total_secs) AS min_monthly_secs,
        MAX(total_secs) AS max_monthly_secs,
        SUM(total_secs) AS total_secs_sum,
        COUNT(*) FILTER (WHERE NOT isfinite(total_secs)) AS non_finite_rows,
        COUNT(*) FILTER (WHERE total_secs < 0) AS negative_rows,
        COUNT(*) FILTER (WHERE ABS(total_secs) > 1000000000) AS extreme_rows
    FROM read_parquet('{file_path.as_posix()}')
""").df().to_string(index=False))

print("\nMonthly diagnostics:")
print(con.execute(f"""
    SELECT
        month,
        MIN(total_secs) AS min_monthly_secs,
        MAX(total_secs) AS max_monthly_secs,
        SUM(total_secs) AS total_secs_sum,
        COUNT(*) FILTER (WHERE NOT isfinite(total_secs)) AS non_finite_rows,
        COUNT(*) FILTER (WHERE total_secs < 0) AS negative_rows
    FROM read_parquet('{file_path.as_posix()}')
    GROUP BY month
    ORDER BY month
""").df().to_string(index=False))