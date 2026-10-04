from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import duckdb

DATA_DIR = Path(r"C:\project")
WORK_DIR = DATA_DIR / "kkbox_processed"
PARTS_DIR = WORK_DIR / "seconds_parts"
LOGS_MONTHLY = WORK_DIR / "logs_monthly.parquet"
OUTPUT_FILE = WORK_DIR / "logs_monthly_clean.parquet"

TRAIN_FILES = ["train.csv", "train_v2.csv"]
LOG_FILES = ["user_logs.csv", "user_logs_v2.csv"]
CHUNK_SIZE = 250_000
FLUSH_EVERY = 4
DAILY_SECONDS_CAP = 86_400

if not LOGS_MONTHLY.exists():
    raise FileNotFoundError(LOGS_MONTHLY)

# Start with a fresh temporary parts folder.
if PARTS_DIR.exists():
    shutil.rmtree(PARTS_DIR)
PARTS_DIR.mkdir(parents=True)

# Restrict processing to the labeled customer cohorts.
customer_ids = set()
for filename in TRAIN_FILES:
    labels = pd.read_csv(
        DATA_DIR / filename,
        usecols=["msno"],
        dtype={"msno": "string"},
    )
    customer_ids.update(labels["msno"].dropna().tolist())

print(f"Labeled-cohort customers: {len(customer_ids):,}")

pending = []
part_number = 0
total_rows = 0
cohort_rows = 0
capped_rows = 0
invalid_rows = 0

def flush_parts():
    global pending, part_number

    if not pending:
        return

    partial = pd.concat(pending, ignore_index=True)
    partial.to_parquet(
        PARTS_DIR / f"part_{part_number:05d}.parquet",
        index=False,
        compression="zstd",
    )
    part_number += 1
    pending = []

for filename in LOG_FILES:
    path = DATA_DIR / filename
    print(f"\nProcessing {filename}...")

    file_rows = 0
    reader = pd.read_csv(
        path,
        usecols=["msno", "date", "total_secs"],
        dtype={"msno": "string", "date": "string", "total_secs": "float64"},
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        file_rows += len(chunk)
        total_rows += len(chunk)

        chunk = chunk[chunk["msno"].isin(customer_ids)].copy()
        cohort_rows += len(chunk)

        if chunk.empty:
            continue

        values = chunk["total_secs"]
        finite = np.isfinite(values)

        # Invalid negatives/non-finite values contribute zero.
        chunk["invalid_secs_rows"] = ((~finite) | (values < 0)).astype("int32")

        # Values above one day are capped at one day.
        chunk["capped_secs_rows"] = (finite & (values > DAILY_SECONDS_CAP)).astype("int32")

        chunk["secs_clean"] = (
            values.where(finite & (values >= 0), 0)
            .clip(upper=DAILY_SECONDS_CAP)
        )

        chunk["month"] = chunk["date"].str.slice(0, 6)

        monthly = (
            chunk.groupby(["msno", "month"], observed=True)
            .agg(
                total_secs_clean=("secs_clean", "sum"),
                invalid_secs_rows=("invalid_secs_rows", "sum"),
                capped_secs_rows=("capped_secs_rows", "sum"),
            )
            .reset_index()
        )

        pending.append(monthly)

        if len(pending) >= FLUSH_EVERY:
            flush_parts()

        if chunk_number % 20 == 0:
            print(f"  chunks: {chunk_number:,} | rows read: {file_rows:,}")

    flush_parts()
    print(f"Finished {filename}: {file_rows:,} rows")

print(f"\nRows scanned: {total_rows:,}")
print(f"Labeled-cohort rows: {cohort_rows:,}")
print(f"Temporary monthly parts: {part_number:,}")

# Consolidate partial monthly summaries, then join to the existing
# monthly log features. Keep the old file unchanged.
con = duckdb.connect()
parts_glob = (PARTS_DIR / "*.parquet").as_posix()
logs_path = LOGS_MONTHLY.as_posix()
output_path = OUTPUT_FILE.as_posix()

con.execute(f"""
    COPY (
        WITH seconds AS (
            SELECT
                msno,
                month,
                SUM(total_secs_clean) AS total_secs_clean,
                SUM(invalid_secs_rows) AS invalid_secs_rows,
                SUM(capped_secs_rows) AS capped_secs_rows
            FROM read_parquet('{parts_glob}')
            GROUP BY msno, month
        )
        SELECT
            logs.* EXCLUDE (total_secs),
            COALESCE(seconds.total_secs_clean, 0) AS total_secs_clean,
            COALESCE(seconds.invalid_secs_rows, 0) AS invalid_secs_rows,
            COALESCE(seconds.capped_secs_rows, 0) AS capped_secs_rows
        FROM read_parquet('{logs_path}') AS logs
        LEFT JOIN seconds
        USING (msno, month)
    )
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print(f"\nCreated: {OUTPUT_FILE}")
print(f"File size: {OUTPUT_FILE.stat().st_size / 1024**3:.2f} GB")

print("\nClean-seconds validation:")
print(con.execute(f"""
    SELECT
        MIN(total_secs_clean) AS minimum_monthly_seconds,
        MAX(total_secs_clean) AS maximum_monthly_seconds,
        SUM(total_secs_clean) AS total_clean_seconds,
        SUM(invalid_secs_rows) AS invalid_daily_rows,
        SUM(capped_secs_rows) AS capped_daily_rows
    FROM read_parquet('{output_path}')
""").df().to_string(index=False))