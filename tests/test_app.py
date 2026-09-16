from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.slow
def test_dashboard_renders_offline_and_runs_calibration():
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    assert not app.exception
    assert any("Volatility Surface Lab" in t.value for t in app.title)

    app.button[0].click().run()
    assert not app.exception
    assert "calibration" in app.session_state
    assert app.session_state["calibration"].rmse_vol < 0.01
