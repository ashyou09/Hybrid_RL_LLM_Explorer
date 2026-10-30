"""Drive the Streamlit Lab headlessly: load, train small, then use every tab."""
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception


def test_lab_end_to_end():
    at = st_testing.AppTest.from_file("../lab.py", default_timeout=240)
    at.run()
    assert not at.exception
    at.sidebar.select_slider[1].set_value(500)       # training episodes
    at.sidebar.slider[2].set_value(1)                 # seeds
    at.sidebar.select_slider[2].set_value(20)         # episodes per seed
    at.sidebar.selectbox[0].set_value("fake")         # offline stand-in, no API key needed
    at.run()
    click(at, "Train Agent A")
    assert len(at.metric) == 3
    click(at, "▶ Play episode")
    click(at, "Compare")
    assert any("Perfect knowledge" in m.value for m in at.markdown)
    click(at, "Run sweep")
