"""py demo.py -> Triage on P1's real alerts: before (every alert pages) vs after, plus quality."""
from collections import Counter

from agents.triage import paths
from agents.triage.agent import RuleTriage
from agents.triage.evaluate import eval_ci, eval_hdfs, eval_planted
from agents.triage.io import load_alerts
from common.schemas import Decision

alerts = load_alerts(paths.ALERTS)
incidents = RuleTriage().process(alerts)
sent = [i for i in incidents if i.decision == Decision.INCIDENT]
reruns = [i for i in sent if i.source == "ci" and i.severity.value == "low"]
human = len(sent) - len(reruns)

print("=== 1. Volume: P1's real alerts ===")
print(f"{'':10s}{'alerts':>8s}{'groups':>8s}{'incident':>10s}{'watch':>7s}{'noise':>7s}")
for src in ("model", "metrics", "logs", "ci"):
    a = sum(x.source == src for x in alerts)
    inc = [i for i in incidents if i.source == src]
    d = Counter(i.decision.value for i in inc)
    print(f"{src:10s}{a:8d}{len(inc):8d}{d['incident']:10d}{d['watch']:7d}{d['noise']:7d}")
print(f"\nBefore triage : {len(alerts)} pages (every alert)")
print(f"After triage  : {human} pages to a human + {len(reruns)} automatic CI re-runs "
      f"-> {100 * (1 - human / len(alerts)):.1f}% fewer pages")

print("\n=== 2. Quality against ground truth ===")
rs = [eval_planted(f) for f in ("trip_count", "avg_fare", "avg_distance")]
caught = sum(r["caught_by_monitor"] for r in rs)
paged, watch = sum(r["paged"] for r in rs), sum(r["watch"] for r in rs)
print(f"NYC planted : {caught} problems caught by the monitor -> paged {paged / caught:.0%}, "
      f"kept in WATCH {watch / caught:.0%}")
h = eval_hdfs(alerts)
print(f"HDFS        : {h['recall_of_failing']:.0%} of failing blocks reach an incident, "
      f"precision {h['precision_paged']:.0%}")
c = eval_ci(alerts)
print(f"CI re-runs  : {c['rerun_precision']:.0%} really flaky (random would be {c['flaky_base_rate']:.1%})")

print("\n=== 3. Three example incidents ===")
for src in ("model", "logs", "ci"):
    i = next(x for x in sent if x.source == src and x.severity.value != "low")
    print(f"\n[{i.id}] {i.severity.value.upper()}  {i.summary}")
    print(f"   why: {i.reasons[-1]}")