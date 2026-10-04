from pathlib import Path
import csv

for path in sorted(Path(".").glob("*.csv")):
    print(f"\n--- {path.name} | {path.stat().st_size / 1024**3:.2f} GB ---")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        sample = [next(reader, None) for _ in range(3)]
    print("Columns:", header)
    print("First rows:", sample)