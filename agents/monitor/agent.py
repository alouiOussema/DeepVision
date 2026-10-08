from __future__ import annotations

import pandas as pd

from common.schemas import Alert


class MonitorAgent:
    """Premier maillon de la chaîne. Produit des Alert, consommées par Triage."""

    def __init__(self, thresholds: dict | None = None) -> None:
        # seuils par défaut, surchargeables
        self.thresholds = {
            "psi": 0.2,
            **(thresholds or {}),
        }

    # ---------- Détecteur 1 : CI (scénario 1, flaky test) ----------
    def check_ci(self, builds: pd.DataFrame) -> list[Alert]:
        """One Alert per failed job. Input columns: build_id, job_id, commit,
        project, started_at, status, failed_tests, tests_failed."""
        alerts = []

        failed = builds[builds["status"].isin(["failed", "errored"])]

        for _, row in failed.iterrows():
            alerts.append(
                Alert(
                    source="ci",
                    signal="build_failed",
                    score=1.0,
                    threshold=1.0,
                    details={
                        "build_id": row["build_id"],
                        "job_id": row["job_id"],
                        "commit": row["commit"],
                        "project": row["project"],
                        "started_at": row["started_at"],
                        "ci_status": row["status"],
                        "failed_tests": row["failed_tests"],
                        "tests_failed": row["tests_failed"],
                    },
                )
            )

        return alerts
    

    # ---------- Détecteur 2 : logs (scénarios 2 & 4) ----------
    def check_logs(self, windows: pd.DataFrame) -> list[Alert]:
        # plus tard
        ...

    # ---------- Détecteur 3 : metrics / Isolation Forest ----------
    def check_metrics(self, df: pd.DataFrame) -> list[Alert]:
        # plus tard
        ...

    # ---------- Détecteur 4 : drift / PSI (scénario 3) ----------
    def check_drift(self, ref: pd.Series, cur: pd.Series) -> list[Alert]:
        # plus tard
        ...


