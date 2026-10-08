import pandas as pd
import pytest

from agents.monitor.agent import MonitorAgent
from common.schemas import Alert


@pytest.fixture
def builds():
    return pd.DataFrame({
        "build_id":     [100, 100, 101, 102],
        "job_id":       [1, 2, 3, 4],
        "commit":       ["abc", "abc", "def", "ghi"],
        "project":      ["rails/rails"] * 4,
        "started_at":   ["2026-10-08 09:00", "2026-10-08 09:00",
                         "2026-10-08 09:20", "2026-10-08 09:40"],
        "status":       ["failed", "failed", "passed", "errored"],
        "failed_tests": ["test_login", "test_login", "", ""],
        "tests_failed": [1, 1, 0, 0],
    })

def test_only_failed_jobs_create_alerts(builds):
    alerts = MonitorAgent().check_ci(builds)
    assert len(alerts) == 3

def test_every_alert_is_valid(builds):
    alerts = MonitorAgent().check_ci(builds)
    for a in alerts:
        assert isinstance(a, Alert)
        assert a.source == "ci"
        assert a.signal == "build_failed"


def test_passed_job_creates_no_alert(builds):
    alerts = MonitorAgent().check_ci(builds)
    job_ids = [a.details["job_id"] for a in alerts]
    assert 3 not in job_ids          # job 3 is "passed"


def test_details_has_commit_and_job(builds):
    alerts = MonitorAgent().check_ci(builds)
    first = alerts[0]
    assert first.details["commit"] == "abc"
    assert first.details["job_id"] == 1


def test_errored_job_counts_as_failure(builds):
    alerts = MonitorAgent().check_ci(builds)
    job4 = [a for a in alerts if a.details["job_id"] == 4]
    assert len(job4) == 1
    assert job4[0].details["ci_status"] == "errored"


def test_empty_table_gives_empty_list(builds):
    empty = pd.DataFrame(columns=builds.columns)
    assert MonitorAgent().check_ci(empty) == []