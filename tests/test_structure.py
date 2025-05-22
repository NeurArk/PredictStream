from importlib import import_module

from utils import config


def test_sample_files_exist():
    for path in config.SAMPLE_DATASETS.values():
        assert path.exists(), f"Sample dataset missing: {path}"


def test_app_importable():
    app = import_module("app")
    assert hasattr(app, "main")


def test_logo_exists():
    assert config.LOGO_PATH.exists(), "Logo file missing"
