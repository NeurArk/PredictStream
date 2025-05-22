import inspect

from utils import ui


def test_get_theme_css():
    light = ui.get_theme_css("Light")
    dark = ui.get_theme_css("Dark")
    for css in (light, dark):
        assert "stSidebar" in css
        assert "plotly-chart" in css


def test_getting_started_markdown():
    text = ui.getting_started_markdown()
    assert "Getting Started" in text
    assert "Upload" in text


def test_use_container_width():
    with open("pages/data_explorer.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "use_container_width=True" in content
