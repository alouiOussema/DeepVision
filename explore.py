"""Look at P1's real alerts BEFORE writing the triage rules.

    py explore.py                      # reads data/alerts.csv
    py explore.py other_file.csv       # or any other export (csv, json, jsonl)

Three families of alerts, each needs its own grouping rule:
  A. series : nyc (model) + deploy (metrics) -- already merged into episodes by P1
  B. logs   : one alert per HDFS block
  C. ci     : one alert per failed Travis build
"""
import sys
from collections import Counter

import pandas as pd

from agents.triage import paths
from agents.triage.io import load_alerts

alerts = load_alerts(sys.argv[1] if len(sys.argv) > 1 else paths.ALERTS)
print(f"\n=== {len(alerts)} alerts")
print(Counter(a.family for a in alerts))

# ------------------------------------------------------------------ A. series
s = pd.DataFrame([{"dataset": a.dataset, "feature": a.feature, "kind": a.kind, "ratio": a.ratio,
                   "start": a.start, "end": a.end, "hours": a.duration_hours}
                  for a in alerts if a.family == "series"])
print("\n--- A. SERIES (nyc + deploy) ---")
print("A1. Alerts per dataset / feature / kind")
print(s.groupby(["dataset", "feature", "kind"]).size().to_string())
print("\nA2. Strength (ratio)")
print(s["ratio"].describe(percentiles=[.25, .5, .75, .9]).round(2).to_string())
print(f"   borderline (ratio < 1.3): {(s['ratio'] < 1.3).mean():.0%}")
print("\nA3. Episode length in hours")
print(s.groupby("dataset")["hours"].describe(percentiles=[.5, .9]).round(2).to_string())

print("\nA4. Episodes of DIFFERENT features that overlap in time (same dataset)")
pairs = Counter()
for ds, g in s.groupby("dataset"):
    rows = g.sort_values("start").to_dict("records")
    for i, x in enumerate(rows):
        for y in rows[i + 1:]:
            if y["start"] > x["end"]:
                break
            if x["feature"] != y["feature"]:
                pairs[(ds, *sorted((x["feature"], y["feature"])))] += 1
for k, c in pairs.most_common(8):
    print(f"   {k[0]}: {k[1]} + {k[2]}: {c} time(s)")
print(f"   episodes overlapping another feature: {sum(pairs.values())} pairs out of {len(s)} episodes")

# ------------------------------------------------------------------ B. logs
b = [a for a in alerts if a.family == "logs"]
print(f"\n--- B. LOGS (hdfs blocks): {len(b)} alerts ---")
if b:
    print(f"B1. distinct timestamps: {len({a.timestamp for a in b})}  (time is useless for grouping)")
    r = pd.Series([a.ratio for a in b])
    print("B2. Strength:", r.describe(percentiles=[.5, .9]).round(2).to_dict())
    top1 = Counter(a.details["top_events"][0]["event"] for a in b)
    top3 = Counter(tuple(sorted(e["event"] for e in a.details["top_events"])) for a in b)
    print(f"B3. distinct main event (top-1): {len(top1)}  -> {top1.most_common(6)}")
    print(f"    distinct top-3 patterns     : {len(top3)}  -> {top3.most_common(4)}")

# ------------------------------------------------------------------ C. ci
c = pd.DataFrame([{"project": a.details.get("project"), "commit": a.details.get("commit"),
                   "ratio": a.ratio, "p_flaky": a.details.get("flaky_probability"),
                   "day": a.timestamp.date()} for a in alerts if a.family == "ci"])
print(f"\n--- C. CI (travis failed builds): {len(c)} alerts ---")
if len(c):
    print(f"C1. projects: {c['project'].nunique()}, commits: {c['commit'].nunique()}, "
          f"days: {c['day'].nunique()}")
    per_proj = c.groupby("project").size()
    print("C2. failed builds per project:", per_proj.describe(percentiles=[.5, .9]).round(1).to_dict())
    per_pd = c.groupby(["project", "day"]).size()
    print(f"    (project, day) groups: {len(per_pd)}, mean {per_pd.mean():.1f} builds, max {per_pd.max()}")
    print("C3. ratio = flaky likelihood vs a typical failure (HIGH = probably flaky = re-run)")
    print(f"    ratio < 1 (typical / real failure): {(c['ratio'] < 1).mean():.0%}")
    print(f"    ratio > 3 (very likely flaky)    : {(c['ratio'] > 3).mean():.0%}")
    print(f"    flaky probability: median {c['p_flaky'].median():.3f}, max {c['p_flaky'].max():.3f}")