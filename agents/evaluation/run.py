"""Run the evaluation: inject failures, run the five agents, grade them.

    python -m agents.evaluation.run           # full run  -> data/evaluation/latest/
    python -m agents.evaluation.run --demo    # one deploy seed -> data/evaluation/demo/
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .config import EvalConfig
from .injector import episodes
from .report import incident_report
from .runner import run_episode
from .score import records, summarize, table
from .trace import timeline


def evaluate(cfg: EvalConfig, out: Path, injections_dir: Path | None = paths.INJECTIONS_DIR,
             diagnosis_node=None, verbose: bool = True) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    per_episode, all_events = [], []
    for ep in episodes(cfg):
        if injections_dir is not None:
            ep.save(injections_dir)
        tracer, state, files = run_episode(ep, out / paths.AGENT_FILES / ep.name, cfg, diagnosis_node)
        recs = records(ep, state, files.inbox.all(), key=tracer.key)
        per_episode.append((ep, recs))
        all_events += tracer.events
        tracer.save(out / paths.TRACE, append=True)
        for r in recs:
            page = out / paths.REPORTS / ep.name / f"{r.incident_id}.md"
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(incident_report(r, timeline(tracer.events, r.key)), encoding="utf-8")
        if verbose:
            print(f"  {ep.name:14s} {len(ep.alerts):4d} alerts  {sum(i.is_fault for i in ep.injections):3d} faults  "
                  f"{len(recs):3d} incidents  {sum(r.sent for r in recs):3d} sent  "
                  f"{len(state.get('actions_done', [])):3d} actions")
    summary = summarize(per_episode)
    result = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": {"seeds": list(cfg.seeds), "families": list(cfg.families),
                         "human_delay_min": cfg.human_delay_min},
              "simulated": True, "kpis": summary,
              "records": [r.to_dict() for _, recs in per_episode for r in recs],
              "injections": [i.to_dict() for ep, _ in per_episode for i in ep.injections]}
    (out / paths.RESULTS).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="seed 4 only, written to data/evaluation/demo/")
    ap.add_argument("--seeds", type=int, nargs="+", help="deploy simulation seeds (default 4 5: never used to train)")
    ap.add_argument("--families", nargs="+", choices=["deploy", "ci", "logs", "model"])
    args = ap.parse_args()
    cfg = EvalConfig()
    if args.demo:
        cfg.seeds = (4,)
    if args.seeds:
        cfg.seeds = tuple(args.seeds)
    if args.families:
        cfg.families = tuple(args.families)
    out = paths.DEMO_DIR if args.demo else paths.RUN_DIR
    print(f"Evaluation [simulation]: seeds {list(cfg.seeds)}, families {list(cfg.families)}\n")
    result = evaluate(cfg, out)
    print("\n" + table(result["kpis"]))
    print(f"\nresults: {out / paths.RESULTS}\nreports: {out / paths.REPORTS}\n"
          "dashboard: python -m streamlit run ui/evaluation_app.py")


if __name__ == "__main__":
    main()
