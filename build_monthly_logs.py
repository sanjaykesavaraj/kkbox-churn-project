from pathlib import Path
import pandas as pd
import duckdb

# CHANGE this to the folder containing your KKBox CSV files.
DATA_DIR = Path("C:/project")

WORK_DIR = DATA_DIR / "kkbox_processed"
PARTS_DIR = WORK_DIR / "log_parts"
OUTPUT_FILE = WORK_DIR / "logs_monthly.parquet"

TRAIN_FILES = ["train.csv", "train_v2.csv"]
LOG_FILES = ["user_logs.csv", "user_logs_v2.csv"]
CHUNK_SIZE = 250_000
FLUSH_EVERY = 4

WORK_DIR.mkdir(parents=True, exist_ok=True)
PARTS_DIR.mkdir(parents=True, exist_ok=True)

# Collect the labeled customer cohort. The union avoids dropping customers
# who appear in either labeled period.
customer_ids = set()

for filename in TRAIN_FILES:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    labels = pd.read_csv(path, usecols=["msno"], dtype={"msno": "string"})
    customer_ids.update(labels["msno"].dropna().tolist())
    print(f"Loaded IDs from {filename}: {len(labels):,} rows")

print(f"Unique customers across both cohorts: {len(customer_ids):,}")

log_columns = [
    "msno", "date", "num_25", "num_50", "num_75",
    "num_985", "num_100", "num_unq", "total_secs"
]

numeric_columns = [
    "num_25", "num_50", "num_75",
    "num_985", "num_100", "num_unq", "total_secs"
]

dtypes = {
    "msno": "string",
    "date": "string",
    "num_25": "int32",
    "num_50": "int32",
    "num_75": "int32",
    "num_985": "int32",
    "num_100": "int32",
    "num_unq": "int32",
    "total_secs": "float32",
}

part_number = 0
pending = []
total_input_rows = 0
total_kept_rows = 0

def write_pending_parts():
    global part_number, pending

    if not pending:
        return

    partial = pd.concat(pending, ignore_index=True)
    part_path = PARTS_DIR / f"part_{part_number:05d}.parquet"
    partial.to_parquet(part_path, index=False, compression="zstd")

    part_number += 1
    pending = []

for filename in LOG_FILES:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    print(f"\nProcessing {filename}...")
    file_rows = 0
    file_kept = 0

    reader = pd.read_csv(
        path,
        usecols=log_columns,
        dtype=dtypes,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        file_rows += len(chunk)
        total_input_rows += len(chunk)

        chunk = chunk[chunk["msno"].isin(customer_ids)].copy()
        file_kept += len(chunk)
        total_kept_rows += len(chunk)

        if chunk.empty:
            continue

        # Dates are YYYYMMDD strings, so the first six characters are YYYYMM.
        chunk["month"] = chunk["date"].str.slice(0, 6)

        monthly = (
            chunk.groupby(["msno", "month"], observed=True)
            .agg(
                num_25=("num_25", "sum"),
                num_50=("num_50", "sum"),
                num_75=("num_75", "sum"),
                num_985=("num_985", "sum"),
                num_100=("num_100", "sum"),
                num_unq=("num_unq", "sum"),
                total_secs=("total_secs", "sum"),
                log_rows=("date", "size"),
            )
            .reset_index()
        )

        pending.append(monthly)

        if len(pending) >= FLUSH_EVERY:
            write_pending_parts()

        if chunk_number % 20 == 0:
            print(
                f"  chunks: {chunk_number:,} | "
                f"rows read: {file_rows:,} | "
                f"cohort rows kept: {file_kept:,}"
            )

    write_pending_parts()
    print(f"  Finished {filename}: {file_rows:,} rows; {file_kept:,} cohort rows kept")

if part_number == 0:
    raise RuntimeError("No matching listening-log rows were found.")

print(f"\nCombining {part_number:,} partial Parquet files...")

parts_glob = (PARTS_DIR / "*.parquet").as_posix()
output_path = OUTPUT_FILE.as_posix()

con = duckdb.connect()
con.execute(f"""
    COPY (
        SELECT
            msno,
            month,
            SUM(num_25) AS num_25,
            SUM(num_50) AS num_50,
            SUM(num_75) AS num_75,
            SUM(num_985) AS num_985,
            SUM(num_100) AS num_100,
            SUM(num_unq) AS num_unq,
            SUM(total_secs) AS total_secs,
            SUM(log_rows) AS log_rows
        FROM read_parquet('{parts_glob}')
        GROUP BY msno, month
    )
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\nDone.")
print(f"Input log rows scanned: {total_input_rows:,}")
print(f"Rows belonging to labeled customers: {total_kept_rows:,}")
print(f"Monthly output: {OUTPUT_FILE}")
print(f"Output size: {OUTPUT_FILE.stat().st_size / 1024**3:.2f} GB")