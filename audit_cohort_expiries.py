from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\project")

COHORTS = {
    "train.csv": "201702",
    "train_v2.csv": "201703",
}

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
    print(f"{filename}: {len(ids):,} customers")

matched_ids = {name: set() for name in COHORTS}
matching_rows = {name: 0 for name in COHORTS}
expiry_dates = {name: [] for name in COHORTS}

reader = pd.read_csv(
    DATA_DIR / "transactions_v2.csv",
    usecols=["msno", "membership_expire_date"],
    dtype={"msno": "string", "membership_expire_date": "string"},
    chunksize=200_000,
)

for chunk_number, chunk in enumerate(reader, start=1):
    chunk = chunk[chunk["msno"].isin(all_ids)].copy()
    chunk["expiry_month"] = chunk["membership_expire_date"].str.slice(0, 6)

    for filename, target_month in COHORTS.items():
        relevant = chunk[
            chunk["msno"].isin(cohort_ids[filename])
            & (chunk["expiry_month"] == target_month)
        ]

        matching_rows[filename] += len(relevant)
        matched_ids[filename].update(relevant["msno"].dropna().tolist())
        expiry_dates[filename].extend(
            relevant["membership_expire_date"].dropna().tolist()
        )

    if chunk_number % 5 == 0:
        print(f"Transaction chunks scanned: {chunk_number}")

print("\nExpiry matches:")
for filename, target_month in COHORTS.items():
    cohort_size = len(cohort_ids[filename])
    matched = len(matched_ids[filename])
    dates = expiry_dates[filename]

    print(f"\n{filename} — expiry month {target_month}")
    print(f"  Matching transaction rows: {matching_rows[filename]:,}")
    print(f"  Customers with a matching expiry: {matched:,}")
    print(f"  Customers without a matching expiry: {cohort_size - matched:,}")
    if dates:
        print(f"  Expiry-date range: {min(dates)} to {max(dates)}")