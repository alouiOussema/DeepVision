"""Small hand-made alerts, one test per rule, so a rule change that breaks something is caught."""
from common.schemas import Alert, Decision, Severity
from agents.triage.agent import RuleTriage
from agents.triage.config import TriageConfig
from agents.triage.grouping import group_all

CFG = TriageConfig()


def series(feature, start, end, score, n=1, dataset="nyc", kind="anomaly"):
    return Alert.from_dict(dict(
        detector="vote(seasonal)", source="model" if dataset == "nyc" else "metrics",
        feature=feature, score=score, threshold=1.0, timestamp=start, value=0.0,
        dataset=dataset, kind=kind,
        details={"voters": ["seasonal"], "start": start, "end": end, "n_points": n}))


def block(i, events, score):
    return Alert.from_dict(dict(
        detector="block_pca", source="logs", feature="block_events", score=score,
        threshold=1.0, timestamp="2026-10-20T11:00:00+00:00", value=score, dataset="hdfs",
        details={"block_id": f"blk_{i}", "top_events": [{"event": e, "share": .3} for e in events]}))


def build(i, project, day, score, fail_rate_before=0.05):
    return Alert.from_dict(dict(
        detector="flaky_model", source="ci", feature="build_failed", score=score,
        threshold=1.0, timestamp=f"2016-08-{day:02d}T{8 + i:02d}:00:00+00:00", value=score / 40,
        dataset="travis", details={"project": project, "build_id": i, "commit": f"c{i}",
                                   "evidence": {"recent_fail_rate": fail_rate_before}}))


def one(alerts):
    inc = RuleTriage(CFG).process(alerts)
    assert len(inc) == 1, [i.summary for i in inc]
    return inc[0]


# ------------------------------------------------------------------ series
def test_overlapping_features_are_one_incident():
    i = one([series("trip_count", "2026-01-01T00:00+00:00", "2026-01-01T05:00+00:00", 6.3, 6),
             series("avg_fare", "2026-01-01T01:00+00:00", "2026-01-01T02:00+00:00", 1.3, 2)])
    assert i.features == ["avg_fare", "trip_count"] and i.decision == Decision.INCIDENT


def test_far_apart_episodes_are_separate():
    inc = RuleTriage(CFG).process([
        series("avg_fare", "2026-01-01T00:00+00:00", "2026-01-01T00:00+00:00", 1.2),
        series("avg_fare", "2026-01-01T05:00+00:00", "2026-01-01T05:00+00:00", 1.2)])
    assert len(inc) == 2


def test_drift_does_not_swallow_anomalies_and_is_only_watched():
    inc = RuleTriage(CFG).process([
        series("avg_fare", "2026-01-01T00:00+00:00", "2026-03-01T00:00+00:00", 1.8, 1400, kind="drift"),
        series("trip_count", "2026-02-01T10:00+00:00", "2026-02-01T10:00+00:00", 5.0)])
    assert len(inc) == 2
    drift = next(i for i in inc if i.kind == "drift")
    assert drift.decision == Decision.WATCH


def test_weak_short_single_feature_is_noise():
    assert one([series("avg_fare", "2026-01-01T00:00+00:00", "2026-01-01T00:00+00:00", 1.1)]
               ).decision == Decision.NOISE


def test_very_strong_is_critical():
    i = one([series("trip_count", "2026-01-01T00:00+00:00", "2026-01-01T00:00+00:00", 4.5)])
    assert i.severity == Severity.CRITICAL and i.decision == Decision.INCIDENT


def test_deploy_latency_and_errors_together_is_paged():
    i = one([series("latency_ms", "2026-01-15T02:00+00:00", "2026-01-15T02:05+00:00", 1.8, 6, "deploy"),
             series("error_rate", "2026-01-15T02:03+00:00", "2026-01-15T02:06+00:00", 1.6, 4, "deploy")])
    assert i.decision == Decision.INCIDENT and len(i.features) == 2


# ------------------------------------------------------------------ logs
def test_blocks_grouped_by_pattern():
    alerts = [block(k, ["E2", "E3", "E4"], 5.0) for k in range(12)] + [block(99, ["E9", "E1", "E5"], 1.5)]
    groups = group_all(alerts, CFG)
    assert sorted(len(g.alerts) for g in groups) == [1, 12]


def test_recurring_pattern_is_incident_rare_weak_is_noise():
    inc = RuleTriage(CFG).process([block(k, ["E2", "E3", "E4"], 5.0) for k in range(12)]
                                     + [block(99, ["E9", "E1", "E5"], 1.5)])
    by_size = {len(i.alert_ids): i.decision for i in inc}
    assert by_size == {12: Decision.INCIDENT, 1: Decision.NOISE}


def test_rare_but_strong_pattern_is_incident():
    assert one([block(1, ["E10", "E14", "E7"], 6.0)]).decision == Decision.INCIDENT


# ------------------------------------------------------------------ ci
def test_one_normal_failed_build_is_noise():
    assert one([build(1, "a/b", 2, 0.3)]).decision == Decision.NOISE


def test_likely_flaky_goes_to_rerun_with_low_severity():
    i = one([build(1, "a/b", 2, 5.0)])
    assert i.decision == Decision.INCIDENT and i.severity == Severity.LOW


def test_usually_green_project_broken_is_paged():
    i = one([build(k, "a/b", 2, 0.3, fail_rate_before=0.05) for k in range(3)])
    assert i.decision == Decision.INCIDENT and i.severity == Severity.HIGH


def test_always_red_project_is_only_watched():
    assert one([build(k, "a/b", 2, 0.3, fail_rate_before=0.9) for k in range(3)]
               ).decision == Decision.WATCH


def test_different_days_are_different_incidents():
    inc = RuleTriage(CFG).process([build(1, "a/b", 2, 0.3), build(2, "a/b", 3, 0.3)])
    assert len(inc) == 2