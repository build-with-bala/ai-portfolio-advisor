"""Smoke test: the Streamlit app renders every tab against the real artifact."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data" / "artifacts.pkl").exists(), reason="artifact not built")
def test_app_renders_without_errors():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not app.exception, [e.value for e in app.exception]
