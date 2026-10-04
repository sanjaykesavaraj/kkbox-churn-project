import csv
from pathlib import Path
from collections import Counter

FILES = {
    "members_v3.csv": None,
    "train_v2.csv": None,
    "transactions_v2.csv": "transaction_date",
    "user_logs_v2.csv": "date",
}

for filename, date_column in FILES.items():
    path = Path(filename)

    if not path.exists():
        print(f"\nMISSING: {filename}")
        continue

    print(f"\nScanning {filename} ({path.stat().st_size / 1024**3:.2f} GB)...")

    row_count = 0
    date_counts = Counter()
    min_date = None
    max_date = None

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)

        if date_column and date_column not in (reader.fieldnames or []):
            print(f"  Expected date column '{date_column}' not found.")
            continue

        for row in reader:
            row_count += 1

            if date_column:
                value = row.get(date_column, "").strip()
                if len(value) == 8 and value.isdigit():
                    min_date = value if min_date is None else min(min_date, value)
                    max_date = value if max_date is None else max(max_date, value)
                    date_counts[value[:6]] += 1

    print(f"  Rows: {row_count:,}")

    if date_column:
        print(f"  {date_column} range: {min_date} to {max_date}")
        print("  Rows by month:")
        for month, count in sorted(date_counts.items()):
            print(f"    {month}: {count:,}")