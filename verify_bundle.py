from pathlib import Path
import hashlib
import json
import zipfile

zip_path = Path(r"C:\project\deployment_bundle_v1.zip")

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

with zipfile.ZipFile(zip_path, "r") as archive:
    bad_file = archive.testzip()
    if bad_file:
        raise RuntimeError(f"ZIP integrity check failed at: {bad_file}")

    manifest = json.loads(archive.read("manifest.json"))
    names = set(archive.namelist())

    failed = []
    for name, expected_hash in manifest["sha256_by_file"].items():
        if name not in names:
            failed.append(f"Missing from ZIP: {name}")
            continue

        actual_hash = sha256_bytes(archive.read(name))
        if actual_hash != expected_hash:
            failed.append(f"Checksum mismatch: {name}")

    required = [
        "score_batch.py",
        "kkbox_processed/boosted_results/boosted_model.joblib",
        "kkbox_processed/boosted_results/categorical_schema.joblib",
        "kkbox_processed/boosted_results/isotonic_calibrator.joblib",
        "manifest.json",
    ]

    for name in required:
        if name not in names:
            failed.append(f"Required package file missing: {name}")

    forbidden_names = [
        "user_logs.csv",
        "user_logs_v2.csv",
        "transactions.csv",
        "transactions_v2.csv",
        "outreach_queue.csv",
        "customer_risk_scores.parquet",
    ]
    for forbidden in forbidden_names:
        if any(forbidden.lower() in name.lower() for name in names):
            failed.append(f"Unexpected sensitive/raw file in ZIP: {forbidden}")

    if failed:
        print("\n".join(failed))
        raise SystemExit("Bundle verification FAILED.")

    print("Bundle ZIP integrity: PASS")
    print(f"Manifest checksums verified: {len(manifest['sha256_by_file'])}")
    print("Required model and scoring files: present")
    print("Raw data and customer-level scoring files: absent")
    print(f"Model version: {manifest['model_version']}")