# DeepVision: working notes for Claude Code

Living file: update it whenever a decision is taken, a step is finished, or something non-obvious is learned.
The product description is in `system_presentation.pdf` (slides 10-16 = architecture + 4 scenarios).

## What this project is
A multi-agent system for autonomous MLOps pipelines. Five agents run as one LangGraph graph (`pipeline.py`):

```
Monitor (P1) ─> Triage (P2) ─> Diagnosis (P3) ─> Decision (P4) ─┬─ safe + confident ─> Action (P5) ─> pipeline fixed
 detects        merges alerts   finds the WHY     risk x confidence │
 anomalies      into incidents  (classifier+LLM)  policy table      └─ risky / uncertain ─> human approval page ─> Action
```

Every agent writes its input, reason, action and result to a log (slide 10). Principle of the deck: *trust before
autonomy*: no action without a visible reason, and a human gate on risky ones.

| Agent | Owner | Code | Output contract (`common/schemas.py`) |
|---|---|---|---|
| Monitor | P1 | `agents/monitor/` | `Alert` (do NOT change it here: change it with P1) |
| Triage | P2 | `agents/triage/` | `Incident` |
| Diagnosis | P3 | `agents/diagnosis/` | `Diagnosis` |
| **Decision** | **P4 (us)** | `agents/decision/` | `ActionDecision` |
| Action | P5 | `agents/action/` | consumes `ActionDecision`, reports back through `ActionInbox.report` |

## Repo map
- `common/schemas.py`: all shared contracts. Append, never reshape a contract someone else produces.
- `knowledge_base/`: runbooks, past incidents, `action_risk_table.md` (the Decision policy, shared P3 + P4).
- `agents/<name>/`: `paths.py` (the ONLY place a path is written), `config.py` (all limits in one dataclass), `agent.py`,
  `node.py` (`make_*_node` for LangGraph), `README.md` (agent card), tests in `tests/test_<name>.py`.
- `pipeline.py`: the graph. `ui/`: Streamlit pages (`approval_app.py` reviews diagnoses, `decision_app.py` gates actions,
  `action_app.py` shows what P5 did).
- `scripts/`, `archive/`: P1's scripts and old experiments. `data/`, `logs/*.csv`, `models/*.pkl` are not in git.

## Conventions
- Python 3.12, dataclasses, type hints, short docstrings that say WHY. English everywhere (code, docs, UI).
- Rules live in code, not in a prompt. Decision has no LLM.
- Tests need no Ollama and no data files: hand-made objects, `tmp_path` for files.
- Never silently fall back to defaults when a policy or contract cannot be read: raise.
- Keep demos away from real data (`data/decision/demo/` is separate from `data/decision/`).

## Commands
```bash
python3 -m venv .venv && .venv/bin/pip install pytest pandas numpy scikit-learn pyarrow langgraph streamlit ollama
.venv/bin/python -m pytest tests/ --continue-on-collection-errors  # whole suite (test_alert.py cannot be collected: P1)
.venv/bin/python -m agents.decision.demo                # the deck's scenarios through Decision
.venv/bin/python pipeline.py sim                        # Monitor -> Triage -> Diagnosis -> Decision, no data files
.venv/bin/python -m streamlit run ui/decision_app.py    # approve / reject pending actions
.venv/bin/python -m agents.decision.evaluate            # policy vs simpler policies on the Diagnosis benchmark
.venv/bin/python -m agents.action.demo                  # the deck's scenarios through Decision AND Action (simulation)
.venv/bin/python pipeline.py sim --action               # the whole graph ending with the Action agent
.venv/bin/python -m streamlit run ui/action_app.py      # what P5 executed / failed / reverted (read-only)
```
On Windows use `venv/Scripts/python` instead of `.venv/bin/python`.
`requirement.txt` has a typo (`pytest=9.1.1`) and duplicate pins (pyarrow, pylance): not touched, tell the team.

## Decision agent (P4): the policy
Input `Diagnosis`, output `ActionDecision` (route `act | approval | escalate | none`). Rule chain, first match wins
(`agents/decision/agent.py`, bars read from `knowledge_base/action_risk_table.md`):

1. recognised no-fault (recovery / known false alarm, confidence >= 0.70) -> `none`, nobody paged
2. confidence < 0.50 -> `escalate`, no action proposed
3. never-alone causes (code_regression, infra, data_quality, unknown) -> `escalate`
4. preconditions: rollback needs a `suspect`; rerun only for CI; retrain only for model inputs -> else `escalate`
5. `needs_human`, warning "LLM argues for ...", or confidence under the cause's bar -> `approval`
6. risk gate: rerun -> act alone; **rollback -> always approval** (team decision, see below); retrain -> act alone by day
   (07-19 UTC) but a human approves the deploy of the new model
7. guards (they use the history): rerun budget 1 per CI project per 24 h (second failure = real bug -> escalate);
   same action on the same target already open or done within 60 min -> closed as duplicate
8. valves: kill switch `enabled=False`, `dry_run`, circuit breaker (> 5 automatic actions per hour) turn `act` into `approval`

### Decisions taken (do not re-litigate without the team)
- **2026-11-03: human gate on rollback.** The table says rollback is low-risk and can run alone at >= 0.80; slide 14 says
  it is risky and needs approval, and slide 10 puts auto-rollback under "if time allowed". We follow the deck. Switch:
  `DecisionConfig.auto_rollback = True`. The table file got a "P4 rules" note (P3's owner should read it).
- Decision is deterministic (no LLM): "Tools: policy table" on slide 11.
- Threshold re-tuning from logged mistakes is a REPORT (`kpis.suggest_thresholds`), never automatic.

### Hand-off contract to the Action agent (P5)
Everything goes through `ActionInbox` (`agents/decision/inbox.py`, file `data/decision/action_inbox.json`), or in the
graph through `state["to_action"]`.
```python
from agents.decision.inbox import ActionInbox
inbox = ActionInbox()
for order in inbox.ready():            # status == "approved": policy or a human said yes. ONLY these may be executed
    order.action                       # rerun | rollback | retrain | escalate     (never "none": those are closed)
    order.target                       # release / config id to undo (rollback), else ""
    order.constraints                  # max_retries, verify_after_min, revert_if_worse, human_before_deploy, notify
    order.summary, order.reasons       # what to tell a human when escalating; the decision trail
    ...                                # act, then VERIFY, then report:
    inbox.report(order.id, "executed" | "failed" | "reverted", "what happened")
```
- `escalate` is an action too: P5 notifies a person (`constraints["notify"] == "author"` for code_regression) with `summary`.
- retrain: P5 trains a challenger and reports `executed`; it must NOT deploy it (`human_before_deploy`): needs its own approval step
  (open question: reuse `ActionInbox` with a follow-up order, or P5's own gate?).
- `report` feeds the rerun budget, the KPIs and the "thresholds from mistakes" report. Report honestly: `failed`/`reverted` count as false actions.
- Graph: `build_graph(..., action_node=p5_node)` adds the node after Decision (conditional on `to_action`).

## Action agent (P5)
Input: approved `ActionDecision` (`state["to_action"]`, or `ActionInbox().ready()` standalone). Output: `ActionInbox.report(...)`,
`state["actions_done"]` (list of `ActionRun`), decision-log lines with `agent="action"`, notifications. No LLM, no new decision.
Loop (`agents/action/agent.py`): gate (inbox status `approved` NOW + O_EXCL claim file in `data/action/claims/`) -> constraints
(missing -> failed, never guessed) -> precheck -> execute -> wait `verify_after_min` (injected clock) -> verify -> retry (max 2,
capped by `ActionConfig.max_retries_cap`) -> undo if worse and `revert_if_worse` -> report -> tell a human on failed / reverted.
- Retries only where safe: an API that did not answer (any action), a message not delivered (escalate). A red re-run is NOT re-run
  (rerun budget 1, runbook), a rollback that did not help is not repeated, a losing challenger is not retrained again.
- Simulation by default (`ActionConfig.simulate=True`): every backend must have `touches_production = False` (unknown = True).
  `simulate=False` requires real backends passed explicitly. The repo has no CI API / deploy API / MLflow: all backends are
  simulated (`backends.py`), plus `retrain_sklearn.py` (real local training on `data/hourly/nyc.parquet`, never deploys).
- Retrain never deploys: `ModelBackend` has no deploy method; an order with `human_before_deploy != True` is refused. A better
  challenger -> `data/action/deploy_requests.jsonl` (`awaiting_human`) + message to the ML engineer.
- Rerun build id: NOT in the `ActionDecision`; read from the incident's CI alert (`Alert.details.build_id`, P1) via
  `state["to_diagnosis"]`, or from `snapshot["job"]` if P4 ever adds it. Missing -> failed + author told.
- Simulated rollback world: the graph node takes the live changes from P3's `ContextStore` (`state["context"]`) and assumes the blamed
  change is the faulty one; `agents.action.run` assumes each approved target is live and faulty (printed).
- Demo `python -m agents.action.demo`: 9 cases (4 deck scenarios x success/failure + rejected rollback), all through the real
  Decision agent, files in `data/action/demo/`. 9/9 as the deck says.

## Status
- [x] `ActionDecision` contract in `common/schemas.py`
- [x] policy parser, config, rule chain, log + history, inbox, KPIs
- [x] LangGraph node + `pipeline.py` wiring (`action_node` hook for P5)
- [x] approval page `ui/decision_app.py` (smoke-tested with Streamlit AppTest)
- [x] demo of the 4 deck scenarios (+3), `tests/test_decision.py` (40 tests)
- [x] evaluation on the Diagnosis benchmark (numbers in `agents/decision/README.md`)
- [ ] commit / PR (waiting for the user)
- [x] P5 reads the inbox and calls `report`: `agents/action/` (agent, executors, simulated backends, node, KPIs, demo, README)
- [x] P5 wired into the graph: `python pipeline.py sim --action`; `actions_done` appended to `PipelineState`
- [x] `tests/test_action.py` (46 tests), results page `ui/action_app.py` (read-only, smoke-tested with AppTest)
- [ ] P5 real adapters (CI API, deploy, MLflow, Slack): nothing to run them against in this repo
- [ ] commit / PR of `action-agent` (waiting for the user)

## Session log
- 2026-11-03: read the deck + repo; chose "human gate on rollback"; built the Decision agent end to end on branch
  `Decision-Agent`. Found and fixed a double-count of grants when a log is attached (caught by the circuit-breaker test).
  `.venv/` created locally (gitignored) because the system Python had no dependencies.
  Edited P3's `tests/test_diagnosis.py::test_pipeline_graph_...` (graph now ends with `decision`; it gets a tmp-dir Decision node
  so tests never write to the real inbox). Pre-existing failures NOT ours: `tests/test_alert.py` (imports the missing
  `agents.monitor.core`) and `tests/test_zscore.py` (`ZScoreDetector.is_anomaly` is gone): tell P1.
  Evaluation: the human gate on rollback costs ~14 points of autonomy and prevents 2 wrong automatic actions in 200 test incidents.

- 2026-11-04: built the Action agent (P5) on branch `action-agent` (from `decision-agent-seif`). Windows machine: only Python
  3.11, the `.venv/` is a Linux venv, so a Windows venv `venv/` (gitignored) was made: `venv/Scripts/python -m pytest tests/`.
  Suite: 125 passed, 2 skipped; only the known `test_zscore` failure and `test_alert` collection error remain.
  Found: the job id for a rerun is not in `ActionDecision` (only `action_key="ci:<project>"`); Decision's 60-min duplicate guard
  closes demo cases that share a target (each demo case has its own release / feature). `pipeline.py sim --action` on the real
  inbox does nothing new because yesterday's orders are still open (duplicates): expected.

- 2026-11-05: the user ran the standalone runner on the real inbox: 6 approved rollbacks shared ONE simulated service, so 5 of
  them "failed" (the other releases' simulated damage was still live). Fixed: `run.run_ready` gives each rollback its own
  simulated service (+ test). `pytest tests/` stops on test_alert's collection error: use `--continue-on-collection-errors`.

## Open questions / watch out
- P5 -> P4: please append `snapshot["job"] = {"project", "build_id", "commit"}` to rerun orders (from the incident's CI alert),
  so a rerun approved from the inbox alone (outside the graph) can run. Append-only, P5 already reads it.
- P5 -> P4: who approves the DEPLOY of a retrained model? P5 writes `data/action/deploy_requests.jsonl`; options: a follow-up
  order in `ActionInbox` (needs a new `Action`, e.g. `deploy_model`: contract change, Seifcha) or a gate page of P5's own.
- `system_presentation.pdf` shows as untracked: ask whether it should be committed.
- Diagnosis only sets `needs_human` below 0.60 confidence; between 0.50 and 0.60 Decision therefore asks a human for a
  proposed action, below 0.50 it escalates without a proposal. Revisit if P3 changes `human_below`.
- The stateful guards are not part of the benchmark evaluation (they need a real history).
- Time zone of "day time" is UTC. If operations run in Tunis time (UTC+1), shift `day_start_hour` / `day_end_hour`.
