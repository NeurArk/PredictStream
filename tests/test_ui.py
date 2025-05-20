import inspect

from utils import ui


def test_get_theme_css():
    css = ui.get_theme_css("Dark")
    assert "background-color" in css


def test_getting_started_markdown():
    text = ui.getting_started_markdown()
    assert "Getting Started" in text
    assert "Upload" in text


def test_use_container_width():
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "use_container_width=True" in content
