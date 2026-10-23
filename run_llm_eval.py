"""py run_llm_eval.py -> rules alone vs rules + LLM agent on ground truth (~15-20 min with qwen2.5:7b).

Results are also saved to data/llm_eval_results.json so you do not need to run it twice.
"""
import json
import time

from agents.triage import paths
from agents.triage.evaluate_llm import compare

MODEL = "qwen2.5:7b"

t = time.time()
paths.OWN.mkdir(parents=True, exist_ok=True)
r = compare(model=MODEL)
minutes = (time.time() - t) / 60

print("\n================ RESULTS ================")
print(f"model {MODEL}: {r['reviews']} reviews in {minutes:.0f} min | "
      f"errors {r['errors']} | guardrail used {r['guardrail_used']}")

print("\nA. NYC planted: recall = share of caught problems that are paged")
print(r["nyc"].to_string(index=False))
nyc = r["nyc"].groupby("system")[["caught", "paged", "pages"]].sum()
for s, row in nyc.iterrows():
    print(f"   {s:10s} total recall {row.paged / row.caught:.0%}  pages {row.pages}")

print(f"\nB. HDFS ({r['feedback_given']} patterns checked by the operator, 5 blocks each)")
print(r["hdfs"].to_string(index=False))
for k, v in r["memory"].items():
    print(f"   memory {k}: {v['verdict']} ({v['note']})")

print(f"\nC. Reading accuracy: {r['misread_reviews']} of {r['reviews']} reviews had at least one misread "
      "field (corrected by the fact-check)")
for k, v in r["reading_accuracy"].items():
    print(f"   {k:18s} {v:.0%}")

json.dump({"model": MODEL, "minutes": round(minutes, 1), "nyc": r["nyc"].to_dict("records"),
           "hdfs": r["hdfs"].to_dict("records"), "reading_accuracy": r["reading_accuracy"],
           "errors": r["errors"], "guardrail_used": r["guardrail_used"]},
          open(paths.LLM_RESULTS, "w"), indent=2)
print(f"\nsaved -> {paths.LLM_RESULTS}")