import pytest

from research.experience import build_log, to_text, validate

HISTORY = [
    {"outcome": "died", "events": [{"event": "walked_on", "tile": "grass"},
                                   {"event": "died", "tile": "red lava"}]},
    {"outcome": "goal", "events": [{"event": "walked_on", "tile": "grass"}]},
    {"outcome": "died", "events": [{"event": "died", "tile": "campfire"}]},
    {"outcome": "died", "events": [{"event": "died", "tile": "red lava"}]},
]


def test_log_stops_at_experience_budget():
    log = build_log(HISTORY, experience_budget=2)
    died = {e["tile"]: e["count"] for e in log if e["event"] == "died_on"}
    assert died == {"red lava": 1, "campfire": 1}


def test_log_aggregates_and_validates():
    log = build_log(HISTORY, experience_budget=10)
    validate(log)
    by = {(e["event"], e["tile"]): e["count"] for e in log}
    assert by[("died_on", "red lava")] == 2
    assert by[("walked_safely_on", "grass")] == 2
    assert by[("reached_goal", None)] == 1
    assert [e["id"] for e in log] == [f"E{i}" for i in range(1, len(log) + 1)]


def test_validate_rejects_coordinates_and_bad_events():
    good = build_log(HISTORY, 10)[0]
    with pytest.raises(ValueError):
        validate([{**good, "description": "lava at (3, 4)"}])
    with pytest.raises(ValueError):
        validate([{**good, "event": "teleported"}])
    with pytest.raises(ValueError):
        validate([{**good, "extra": 1}])


def test_text_for_prompt_cites_ids():
    text = to_text(build_log(HISTORY, 10))
    assert "[E1] DIED after stepping on red lava" in text
