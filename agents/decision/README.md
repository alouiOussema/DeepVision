# Decision agent (P4)

> Diagnosis says **why**. Decision says **what happens next**: the Action agent (P5) may do it alone, a human approves it first,
> or a human takes over. One `Diagnosis` in, one `ActionDecision` out, with the reasons in plain words.

## Agent card

| | |
|---|---|
| **Goal** | Never take an action the system cannot justify; hand every safe, confident action to P5 and every other case to a person (KPI: false action rate) |
| **Input** | `Diagnosis` from P3 (cause, calibrated confidence, suspect release, `needs_human`, warnings, evidence) + the `Incident` it came from |
| **Output** | `ActionDecision` (`common/schemas.py`): `action`, `route`, `status`, `target`, `risk`, `rule`, `reasons[]` (decision trail), `summary`, `constraints`, diagnosis `snapshot` |
| **Tools** | the action risk table (`knowledge_base/action_risk_table.md`), the decision history (guards), the approval page |
| **Rules in code** | no LLM. First matching rule wins and every rule writes one readable line. A table that cannot be read raises instead of guessing |
| **When unsure** | confidence < 0.50 → escalate with no proposed action; below the cause's bar, `needs_human` or an LLM/classifier disagreement → a human approves the proposed action |

## The rule chain

```
Diagnosis ─► A no-fault? ─► B conf < 0.50? ─► C never-alone cause? ─► D can it be done? ─► E needs a human? ─► F risk gate ─► G guards ─► H valves ─► ActionDecision
              none           escalate          escalate                escalate            approval             see below     see below   act → approval
```

| Cause | Action | Alone when | What actually happens by default |
|---|---|---|---|
| flaky_test | rerun (once) | confidence ≥ 0.60 | **act alone** |
| bad_deploy / config_change | rollback | table: ≥ 0.80 | **approval always** (team decision, deck scenario 2). `auto_rollback=True` follows the table |
| data_drift | retrain | ≥ 0.70, day time | **act alone** by day, at night approval; the new model's **deploy always needs a human** |
| code_regression, infra, data_quality, unknown | escalate | never | **escalate** (code_regression: notify the author) |
| unknown, recognised recovery / known false alarm | none | ≥ 0.70 | **closed**, nobody paged, logged |

Guards that remember (`log.py`): a flaky test is re-run **once** per CI project per day (failing again = a real bug, so it escalates, deck scenario 1);
the same action on the same release / project that is still open or was done in the last hour is closed as a duplicate; more than 5 automatic actions
in one hour trips a circuit breaker; `enabled=False` is a kill switch and `dry_run` shows what would happen without anything running alone.

A decision trail (real output of `python -m agents.decision.demo`, deck scenario 2):

```
2 · Bad deployment
  ASK A HUMAN -> rollback release-v2.3   confidence 0.93  risk low  [gate:rollback]  status: pending_approval
    - rollback needs a human (team rule, deck scenario 2) although 0.93 >= 0.80 would be enough by the table; `auto_rollback` switches this off
```

## Run it

```bash
python -m agents.decision.demo [--night] [--auto-rollback]   # the 4 scenarios of the deck + 3 more
python pipeline.py sim                                      # Monitor -> Triage -> Diagnosis -> Decision, no data files
python -m streamlit run ui/decision_app.py                  # approve / reject (tick "Demo data" for the demo orders)
python -m agents.decision.evaluate                          # policy vs simpler policies on the Diagnosis benchmark
python -m pytest tests/test_decision.py                     # 40 tests, only pytest needed
```

In the graph (`pipeline.py`): `diagnosis → decision → [action → ] END`. The Decision node reads `state["diagnoses"]` (+ `to_diagnosis` for the incident source),
writes `decisions`, `to_action`, `pending_approval`, `log`, and persists every order to the inbox and the decision log.

## Hand-off contract for the Action agent (P5)

P5 only ever touches **`ActionInbox`** (`agents/decision/inbox.py`, file `data/decision/action_inbox.json`), or `state["to_action"]` inside the graph.

```python
from agents.decision.inbox import ActionInbox
inbox = ActionInbox()
for order in inbox.ready():        # status "approved": the policy or a human said yes. Nothing else may be executed
    ...                            # do order.action on order.target, respect order.constraints, then VERIFY
    inbox.report(order.id, "executed", "rolled back release-v2.3, error rate back in band after 12 min")
    # or "failed" (the problem is still there) / "reverted" (it got worse and you undid it)
```

| `ActionDecision` field | Meaning for P5 |
|---|---|
| `action` | `rerun` re-run the failed CI job once · `rollback` undo `target` (release / config id) · `retrain` train a challenger, **do not deploy it** · `escalate` notify a person with `summary` and `constraints["notify"]` |
| `status` | `approved` → execute · `pending_approval` / `rejected` / `closed` → do nothing · `executed` / `failed` / `reverted` are set by `report` |
| `constraints` | `max_retries` (2), `verify_after_min` (15), `revert_if_worse` (true for rerun/rollback/retrain), `human_before_deploy` (retrain), `reruns` (rerun), `notify` (escalate) |
| `reasons`, `summary`, `snapshot` | the decision trail, a paragraph for the human you notify, the diagnosis evidence |

`report` is not optional: it feeds the re-run budget, the KPIs (`kpis.py`) and the "thresholds from mistakes" report. Reporting `failed` honestly is what makes the policy better.
`ActionInbox` is safe to use while the pipeline and the approval page write to it (lock file + atomic replace).

## Results

200 test incidents never used for training (Diagnosis benchmark, hybrid diagnoses). `wrong_auto_actions` = rerun / rollback / retrain run alone on the
wrong cause or wrong release; `silenced_faults` = "nothing to do" said alone on a real fault.

| Policy | Act alone | Right and alone | Wrong auto actions | Silenced faults | Reaches a person | Fixable incidents that still need a person |
|---|---|---|---|---|---|---|
| suggested (no Decision agent: do what Diagnosis says) | 31.0 % | 30.0 % | 2 | 1 | 58.5 % | 9.0 % |
| deck rule (≥ 0.8 safe → act, risky → human, < 0.5 → escalate) | 17.0 % | 17.0 % | 0 | 0 | 83.0 % | 49.3 % |
| **ours (table + human gate on rollback)** | 17.0 % | 17.0 % | **0** | 1 | 72.5 % | 47.8 % |
| ours with `auto_rollback=True` | 31.0 % | 30.0 % | 2 | 1 | 58.5 % | 9.0 % |

On the 78 incidents Triage actually sends: no wrong automatic action and no silenced fault under any policy (ours: 14.1 % alone, 83.3 % reach a person; with `auto_rollback` 26.9 % / 70.5 %).

How to read this: **the human gate on rollback costs about 14 points of autonomy** (31 % → 17 %) **and prevents 2 wrong automatic actions in 200**; both of them were on incidents Triage had already filed as watch or noise.
Most of what P5 would do alone is the flaky-test re-run and the daytime retrain. The table's own bars add nothing on top of Diagnosis' calibrated `needs_human` at 0.60 on this benchmark
(suggested and ours-with-auto-rollback are identical): confidence is either high or low, rarely in between. The one silenced fault of "ours" is the one P3 already reports in its README.

## What we tried (and kept or dropped)

| Experiment | Result | Decision |
|---|---|---|
| Follow the table (rollback alone at ≥ 0.80) vs the deck (rollback needs a human) | table: +14 points autonomy but 2 wrong automatic actions in 200; deck: none | deck by default, one config flag to switch |
| Count guard grants in memory and in the history file | with a log attached every grant counted twice (circuit breaker tripped at 2 instead of 5): caught by a test | memory only when there is no log |
| Table row for each cause hard-coded in `agent.py` vs read from `action_risk_table.md` | hard-coding would let the two drift apart; P3 and P4 share that file | read it, raise on a malformed table |
| LLM in Decision | slide 11 says "policy table"; an LLM could not be tested rule by rule or audited | no LLM |

## Limits and next steps

- The stateful guards (re-run budget, cooldown, circuit breaker) are tested but not part of the benchmark numbers: the benchmark has no time order.
- "Day time" is UTC (07:00-19:00); shift `DecisionConfig.day_start_hour/day_end_hour` for local time.
- Retrain's deploy gate is a constraint P5 must respect; P5 still needs a way to ask for that second approval (follow-up order in the same inbox, or its own).
- Threshold suggestions (`kpis.suggest_thresholds`) are a report: they need real outcomes reported by P5 before they say anything.
- The inbox is a JSON file with a lock: right for one machine. A second machine needs a real queue or database behind the same four methods (`add`, `ready`, `resolve`, `report`).
