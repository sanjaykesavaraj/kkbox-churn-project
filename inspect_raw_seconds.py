from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\project")
TRAIN_FILES = ["train.csv", "train_v2.csv"]
LOG_FILES = ["user_logs.csv", "user_logs_v2.csv"]
CHUNK_SIZE = 250_000

customer_ids = set()
for filename in TRAIN_FILES:
    labels = pd.read_csv(
        DATA_DIR / filename,
        usecols=["msno"],
        dtype={"msno": "string"},
    )
    customer_ids.update(labels["msno"].dropna().tolist())

print(f"Labeled-cohort customers: {len(customer_ids):,}")

for filename in LOG_FILES:
    path = DATA_DIR / filename
    print(f"\nScanning {filename}...")

    rows = 0
    kept_rows = 0
    negative_count = 0
    over_86400_count = 0
    nonfinite_count = 0
    min_value = float("inf")
    max_value = float("-inf")
    examples = []

    reader = pd.read_csv(
        path,
        usecols=["msno", "date", "total_secs"],
        dtype={"msno": "string", "date": "string", "total_secs": "float64"},
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        rows += len(chunk)
        chunk = chunk[chunk["msno"].isin(customer_ids)]
        kept_rows += len(chunk)

        values = chunk["total_secs"]
        finite = values.notna() & values.map(lambda x: abs(x) != float("inf"))
        finite_values = values[finite]

        if not finite_values.empty:
            min_value = min(min_value, finite_values.min())
            max_value = max(max_value, finite_values.max())

        negative_count += int((finite_values < 0).sum())
        over_86400_count += int((finite_values > 86400).sum())
        nonfinite_count += int((~finite).sum())

        bad = chunk.loc[
            (~finite) | (values < 0) | (values > 86400),
            ["msno", "date", "total_secs"],
        ]
        if not bad.empty:
            examples.extend(bad.head(5).to_dict("records"))
            examples = examples[:20]

        if rows % 5_000_000 < CHUNK_SIZE:
            print(f"  Rows scanned: {rows:,}")

    print(f"Rows scanned: {rows:,}")
    print(f"Cohort rows: {kept_rows:,}")
    print(f"Finite-value minimum: {min_value}")
    print(f"Finite-value maximum: {max_value}")
    print(f"Negative daily values: {negative_count:,}")
    print(f"Daily values above 86,400 seconds: {over_86400_count:,}")
    print(f"Missing/non-finite values: {nonfinite_count:,}")
    print("Examples of suspicious daily values:")
    for row in examples:
        print(" ", row)