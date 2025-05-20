from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "samples"

SAMPLE_DATASETS = {
    "Iris Sample": SAMPLE_DIR / "iris_sample.csv",
    "Tips Sample": SAMPLE_DIR / "tips_sample.csv",
}
