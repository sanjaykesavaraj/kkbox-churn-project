from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\project")

COHORTS = {
    "train.csv": "201702",
    "train_v2.csv": "201703",
}
TRANSACTION_FILES = ["transactions.csv", "transactions_v2.csv"]

cohort_ids = {}
all_ids = set()

for filename in COHORTS:
    labels = pd.read_csv(
        DATA_DIR / filename,
        usecols=["msno"],
        dtype={"msno": "string"},
    )
    ids = set(labels["msno"].dropna().tolist())
    cohort_ids[filename] = ids
    all_ids.update(ids)

print(f"Unique customers across cohorts: {len(all_ids):,}")

for tx_filename in TRANSACTION_FILES:
    path = DATA_DIR / tx_filename
    if not path.exists():
        print(f"\nMISSING: {tx_filename}")
        continue

    print(f"\nScanning {tx_filename}...")
    row_count = 0
    min_transaction_date = None
    max_transaction_date = None
    min_expiry_date = None
    max_expiry_date = None

    matched_ids = {name: set() for name in COHORTS}
    matched_rows = {name: 0 for name in COHORTS}

    reader = pd.read_csv(
        path,
        usecols=["msno", "transaction_date", "membership_expire_date"],
        dtype={
            "msno": "string",
            "transaction_date": "string",
            "membership_expire_date": "string",
        },
        chunksize=200_000,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        row_count += len(chunk)

        for col, min_key, max_key in [
            ("transaction_date", "min_transaction_date", "max_transaction_date"),
            ("membership_expire_date", "min_expiry_date", "max_expiry_date"),
        ]:
            values = chunk[col].dropna()
            values = values[values.str.len() == 8]
            if not values.empty:
                local_min, local_max = values.min(), values.max()
                current_min = locals()[min_key]
                current_max = locals()[max_key]
                if current_min is None or local_min < current_min:
                    if col == "transaction_date":
                        min_transaction_date = local_min
                    else:
                        min_expiry_date = local_min
                if current_max is None or local_max > current_max:
                    if col == "transaction_date":
                        max_transaction_date = local_max
                    else:
                        max_expiry_date = local_max

        chunk = chunk[chunk["msno"].isin(all_ids)].copy()
        chunk["expiry_month"] = chunk["membership_expire_date"].str.slice(0, 6)

        for cohort_file, target_month in COHORTS.items():
            relevant = chunk[
                chunk["msno"].isin(cohort_ids[cohort_file])
                & (chunk["expiry_month"] == target_month)
            ]
            matched_rows[cohort_file] += len(relevant)
            matched_ids[cohort_file].update(relevant["msno"].dropna().tolist())

        if chunk_number % 5 == 0:
            print(f"  chunks scanned: {chunk_number}")

    print(f"Rows: {row_count:,}")
    print(f"Transaction date range: {min_transaction_date} to {max_transaction_date}")
    print(f"Expiry date range: {min_expiry_date} to {max_expiry_date}")

    for cohort_file, target_month in COHORTS.items():
        print(
            f"  {cohort_file}, expiry month {target_month}: "
            f"{len(matched_ids[cohort_file]):,} customers matched "
            f"({matched_rows[cohort_file]:,} transaction rows)"
        )