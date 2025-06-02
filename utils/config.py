"""Configuration paths and sample dataset locations."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "samples"
STATIC_DIR = BASE_DIR / "static"
LOGO_PATH = STATIC_DIR / "logo.png"

SAMPLE_DATASETS = {
    "Titanic (Classification)": SAMPLE_DIR / "titanic.csv",
    "California Housing (Regression)": SAMPLE_DIR / "california_housing.csv", 
    "Customer Churn (Classification)": SAMPLE_DIR / "telco_churn.csv",
    "Airline Passengers (Time Series)": SAMPLE_DIR / "airline_passengers.csv",
    "Wine Quality (Multi-class)": SAMPLE_DIR / "wine_quality.csv",
}
