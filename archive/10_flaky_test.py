"""Step 1 of the flaky-test detector: build the labelled builds table and COUNT what we have.

Run from the project root:  python -m scripts.10_flaky_labels
Writes data/builds.parquet (used by the next step) and reports/flaky_census.csv.
"""
import pandas as pd
from config import settings
from ingest.travis_build import build_builds

BUILDS_FILE = settings.ROOT / "data" / "builds.parquet"


def main():
    b = build_builds()
    b.to_parquet(BUILDS_FILE, index=False)
    pd.set_option("display.width", 140)

    print(f"\nbuilds: {len(b):,}   projects: {b.project.nunique():,}   "
          f"from {b.started.min().date()} to {b.started.max().date()}")
    print("\nbuild status / label:")
    print(b["label"].value_counts().to_string())

    failed = b[b["label"].isin(["flaky_fail", "real_fail"])]
    n_flaky = int((failed["label"] == "flaky_fail").sum())
    print(f"\nfailed builds: {len(failed):,}   of which flaky (same code passed elsewhere): "
          f"{n_flaky:,} ({n_flaky / max(len(failed), 1):.1%})")

    per_year = (b[b["label"] == "flaky_fail"].groupby(b["started"].dt.year).size()
                .rename("flaky_fail").to_frame()
                .join(failed.groupby(failed["started"].dt.year).size().rename("all_failed")))
    print("\nper year:")
    print(per_year.to_string())

    by_proj = failed.groupby("project")["label"].apply(lambda s: (s == "flaky_fail").sum())
    print(f"\nprojects with at least one flaky failure: {int((by_proj > 0).sum()):,} "
          f"of {b.project.nunique():,}")
    top5 = by_proj.sort_values(ascending=False).head(5)
    print(f"the 5 projects with most flaky failures hold {top5.sum() / max(n_flaky, 1):.0%} of them:")
    print(top5.to_string())

    if "tests_failed" in b:
        known = failed["tests_failed"].notna().mean()
        print(f"\nfailed builds with test-count information: {known:.0%}")
        print(failed.groupby("label")["tests_failed"].median().rename("median tests_failed").to_string())

    per_year.to_csv(settings.REPORTS_DIR / "flaky_census.csv")
    print(f"\nsaved {BUILDS_FILE}")
    if n_flaky < 500:
        print("WARNING: fewer than 500 flaky failures: a learned model will be unreliable.")


if __name__ == "__main__":
    main()