from importlib import import_module

import pytest

PAGES = ["pages.data_explorer", "pages.modeling"]

@pytest.mark.parametrize("mod_name", PAGES)
def test_page_importable(mod_name):
    mod = import_module(mod_name)
    assert hasattr(mod, "main")
