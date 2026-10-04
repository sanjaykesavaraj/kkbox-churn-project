from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile

PROJECT = Path(r"C:\project")
RESULTS = PROJECT / "kkbox_processed" / "boosted_results"
BUNDLE = PROJECT / "deployment_bundle_v1"
ZIP_FILE = PROJECT / "deployment_bundle_v1.zip"

if BUNDLE.exists():
    shutil.rmtree(BUNDLE)

# Keep the same folder layout expected by score_batch.py.
(BUNDLE / "kkbox_processed" / "boosted_results").mkdir(
    parents=True, exist_ok=True
)

required_files = {
    RESULTS / "boosted_model.joblib":
        BUNDLE / "kkbox_processed" / "boosted_results" / "boosted_model.joblib",
    RESULTS / "categorical_schema.joblib":
        BUNDLE / "kkbox_processed" / "boosted_results" / "categorical_schema.joblib",
    RESULTS / "isotonic_calibrator.joblib":
        BUNDLE / "kkbox_processed" / "boosted_results" / "isotonic_calibrator.joblib",
    PROJECT / "score_batch.py":
        BUNDLE / "score_batch.py",
}

optional_reports = [
    "validation_metrics.csv",
    "holdout_metrics.csv",
    "validation_lift.csv",
    "holdout_lift.csv",
    "validation_precision_recall.csv",
    "holdout_precision_recall.csv",
    "calibration_metrics.csv",
    "calibrated_score_bins.csv",
    "permutation_importance.csv",
    "retention_signal_rates.csv",
    "calibration_comparison.png",
    "permutation_importance.png",
]

for source, destination in required_files.items():
    if not source.exists():
        raise FileNotFoundError(f"Required file missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

reports_dir = BUNDLE / "reports"
reports_dir.mkdir(exist_ok=True)

for filename in optional_reports:
    source = RESULTS / filename
    if source.exists():
        shutil.copy2(source, reports_dir / filename)

# Record exact library versions from the Python environment used here.
packages = [
    "pandas",
    "pyarrow",
    "duckdb",
    "scikit-learn",
    "matplotlib",
    "joblib",
    "numpy",
]
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = "not installed"

requirements = [
    f"{name}=={version}"
    for name, version in versions.items()
    if version != "not installed"
]
(BUNDLE / "requirements.txt").write_text(
    "\n".join(requirements) + "\n",
    encoding="utf-8",
)

readme = """KKBox churn model bundle — v1 demo

This is a historical batch-scoring demonstration, not an approved production model.
Extract this bundle into C:\\project to preserve the paths used by score_batch.py.
The model-ready input Parquet file is intentionally not included.

Run from PowerShell after extraction and after providing a compatible input:
  python C:\\project\\score_batch.py --input <feature_snapshot.parquet> --cohort none --contact-share 0.10

The risk_score is a ranking score, not a calibrated individual probability.
The model was trained on 2015–2017 KKBox data; do not use for live outreach.
Review reports/ and the project documentation for evaluation, calibration,
experiment design, and limitations.

Raw customer data, customer-level risk scores, and outreach queues are
intentionally excluded from this bundle.
"""
(BUNDLE / "BUNDLE_README.txt").write_text(readme, encoding="utf-8")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = {}
for path in sorted(BUNDLE.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        files[path.relative_to(BUNDLE).as_posix()] = sha256(path)

manifest = {
    "model_version": "kkbox_churn_hgb_v1_demo",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version,
    "model_type": "sklearn HistGradientBoostingClassifier",
    "target": "is_churn",
    "cohort_development": "train.csv; stratified internal validation",
    "temporal_evaluation": "train_v2.csv; later cohort",
    "feature_cutoff": "First day of target expiry month; target-month events excluded",
    "score_semantics": (
        "Raw ranking score. Isotonic calibrator is included for analysis, "
        "but later-cohort bins show residual underprediction."
    ),
    "historical_metrics": {
        "internal_validation_average_precision": 0.561741,
        "internal_validation_roc_auc": 0.893340,
        "later_cohort_average_precision": 0.574364,
        "later_cohort_roc_auc": 0.851047,
        "later_cohort_top_decile_churn_rate": 0.537128,
        "later_cohort_top_decile_lift": 5.971945,
    },
    "library_versions": versions,
    "sha256_by_file": files,
    "excluded": [
        "Raw KKBox CSVs",
        "Customer-level score files",
        "Customer outreach queue",
        "Model-ready customer feature Parquet",
    ],
}

(BUNDLE / "manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

if ZIP_FILE.exists():
    ZIP_FILE.unlink()

with zipfile.ZipFile(ZIP_FILE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(BUNDLE.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(BUNDLE).as_posix())

print(f"Bundle folder: {BUNDLE}")
print(f"ZIP package:   {ZIP_FILE}")
print(f"Files packaged: {len(files) + 1:,}")
print("Raw data and customer-level score outputs were not included.")