import csv
from pathlib import Path
from collections import Counter

FILES = {
    "user_logs.csv": "date",
    "train.csv": None,
}

for filename, date_column in FILES.items():
    path = Path(filename)

    if not path.exists():
        print(f"\nMISSING: {filename}")
        continue

    print(f"\nScanning {filename} ({path.stat().st_size / 1024**3:.2f} GB)...")

    row_count = 0
    monthly_rows = Counter()
    min_date = None
    max_date = None

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)

        print("Columns:", reader.fieldnames)

        for row in reader:
            row_count += 1

            if date_column:
                value = row.get(date_column, "").strip()
                if len(value) == 8 and value.isdigit():
                    min_date = value if min_date is None else min(min_date, value)
                    max_date = value if max_date is None else max(max_date, value)
                    monthly_rows[value[:6]] += 1

    print(f"Rows: {row_count:,}")

    if date_column:
        print(f"{date_column} range: {min_date} to {max_date}")
        print("Rows by month:")
        for month, count in sorted(monthly_rows.items()):
            print(f"  {month}: {count:,}")