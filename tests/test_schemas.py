import pytest

from agents.triage import paths
from agents.triage.io import load_alerts, save_alerts
from common.schemas import Alert

DATA = paths.ALERTS          # P1's real export: logs/alerts.csv (team repo) or data/alerts.csv
needs_data = pytest.mark.skipif(not DATA.exists(), reason=f"{DATA} not found: run P1's scripts.04_run_monitor first")


@needs_data
def test_real_alerts_load_and_respect_contract():
    alerts = load_alerts(DATA)
    assert len(alerts) > 0
    for a in alerts:
        assert a.source in {"ci", "logs", "metrics", "model"}
        assert a.kind in {"anomaly", "drift"}
        assert a.threshold == 1.0 and a.family in {"series", "logs", "ci"}


def test_episode_duration_from_details():
    a = Alert.from_dict({"detector": "vote(seasonal)", "source": "model", "feature": "trip_count",
                         "score": 6.3, "threshold": 1.0, "value": 7629,
                         "timestamp": "2026-01-01 00:00:00+00:00", "dataset": "nyc",
                         "details": '{"start": "2026-01-01 00:00:00+00:00", '
                                    '"end": "2026-01-01 05:00:00+00:00", "n_points": 6}'})
    assert a.duration_hours == 6 and a.family == "series"


def test_monitor_node_dict_is_accepted():
    # what P1's monitor_node sends: "signal" instead of "feature", no detector/threshold/value
    a = Alert.from_dict({"id": "alert-1", "timestamp": "2016-08-17T06:29:14+00:00", "source": "ci",
                         "signal": "build_failed", "score": 0.5, "kind": "anomaly",
                         "dataset": "travis", "details": {"project": "a/b"}})
    assert a.feature == "build_failed" and a.ratio == 0.5 and a.family == "ci"


@needs_data
def test_round_trip_json(tmp_path):
    a = load_alerts(DATA)[:50]
    save_alerts(a, tmp_path / "x.json")
    b = load_alerts(tmp_path / "x.json")
    assert [x.id for x in a] == [x.id for x in b] and b[0].ratio == a[0].ratio