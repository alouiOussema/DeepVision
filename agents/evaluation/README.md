# Evaluation & observability layer

> The five agents were each tested on their own. This layer tests **the whole chain**: it breaks the pipeline on purpose
> (with the right answer written down), runs Monitor → Triage → Diagnosis → Decision → Action, records what every agent
> did and why, and grades the deck's KPIs against the truth.

## Card

| | |
|---|---|
| **Goal** | Show, with numbers, whether the system does what the deck promises (slide 10: "Measured KPIs", "Decision log + dashboard") |
| **Input** | Injected failures with their truth (deck data source 04, "failures we inject on purpose"): `data/injections/` |
| **Output** | `results.json` (KPIs + one graded record per incident), `trace.jsonl` (every agent's input, reason, action, result), one Markdown incident report per incident, `ui/evaluation_app.py` |
| **Changes to other agents** | none. It calls `pipeline.build_graph` and the real `DecisionAgent` / `ActionAgent`; all their files go to the run's own directory |
| **Rules** | no LLM; the same train / test split as Diagnosis' benchmark (it is never graded on what it trained on) |

## How it works

```
injector ──► episode (alerts + context + truth) ──► the team's graph ──────────────────────────────► tracer ──► score ──► results.json
             deploy: P1's simulator → P1's real      monitor → triage → diagnosis → decision          every        KPIs vs     reports/*.md
             detector (fault times known → MTTD)                                    (replayed:        agent's      the truth   dashboard
             ci / model / logs: P1's Alert format                                    Decision →       step
                                                                                     simulated human → Action
                                                                                     on a world built from the truth)
```

| Part | File | What it does |
|---|---|---|
| Injector | `injector.py` | Reuses Diagnosis' scenarios (`simulate_deploy`, `ci_cases`, `log_cases`, `model_cases`): each fault becomes an `Injection` (cause, release to undo, start time) and every alert points back to one. Deck mapping: flaky test → ci, bad deploy → deploy, data drift → model, unknown failure → logs |
| World | `world.py` | Builds P5's simulated CI / deployment / model backends **from the truth**: only the release we broke hurts, so rolling back the wrong one fails verification |
| Runner | `runner.py` | Runs `pipeline.build_graph`. Decision, the approval step and Action are replayed **one incident at a time, at the incident's own time** (in production they arrive hours apart; deciding a batch at one instant would trip the circuit breaker and close later incidents as duplicates of orders not run yet). The human approves every pending order after 5 virtual minutes |
| Tracer | `trace.py` | Reads each node's state output and writes one event per agent per incident: `at, agent, input, reason, action, result` |
| Score | `score.py` | Grades every incident and every injected fault; the deck's KPIs per family |
| Report | `report.py` | Auto-drafted incident report (the deck's "keep simple" idea; Yassine's "incident timelines") |

## Run it

```bash
python -m agents.evaluation.run                 # deploy seeds 4 + 5, ci, model, logs (~20 s) -> data/evaluation/latest/
python -m agents.evaluation.run --demo          # seed 4 only -> data/evaluation/demo/
python -m streamlit run ui/evaluation_app.py    # KPIs, incidents (wrong or missed first), timelines, reports (read-only)
python -m pytest tests/test_evaluation.py       # 13 tests, no Ollama, no data files
```

Diagnosis' classifier must exist (`data/diagnosis/classifier.pkl`): `python -m agents.diagnosis.train` once.

## Results (simulation, 2026-11-10)

200 graded incidents (Diagnosis' held-out set: deploy seeds 4 + 5 and 40 % of the synthetic families), 532 alerts, 154 injected faults.

| | deploy | ci | model | logs | **all** |
|---|---|---|---|---|---|
| noise reduction (1 − incidents sent / alerts) | 81.2 % | 78.3 % | 71.4 % | 92.8 % | **85.3 %** |
| MTTD, median (fault injected → first alert) | 4 min | - | - | - | **4 min** |
| faults missed by the Monitor | 12 / 44 | - | - | - | 12 |
| faults kept as watch / noise by Triage | 19 | 44 | 20 | 0 | 83 |
| diagnosis accuracy (incidents sent) | 93.3 % | 96.7 % | 100 % | 94.7 % | **96.2 %** |
| false action rate (fixes run) | 0 % | 8.3 % | - | - | **4.5 %** (1 of 22) |
| wrong automatic actions | 0 | 1 | 0 | 0 | **1** |
| silenced faults | 0 | 0 | 0 | 0 | **0** |
| faults closed as a duplicate | 0 | 0 | 0 | 7 | 7 |
| fixable faults resolved (with / without a human) | 26 % / 0 % | 65 % / 65 % | 0 % / 0 % | - | 28 % / 15 % |
| MTTR, median (injected or detected → verified fix) | 21 min | 15 min | - | - | **15 min** |

Cross-check: the graph sends exactly **78** incidents to Diagnosis and grades **200**, the same numbers as Diagnosis'
own benchmark, and diagnosis accuracy (96.2 %) matches its README (97 % on those 78). The injector reproduces the benchmark.

### What the numbers say (to discuss with the team)

1. **Drift never reaches Diagnosis without the LLM** (P2). Triage files every drift alert as `watch` ("check before
   retraining"); only Triage's LLM layer can promote it, and with Ollama off it keeps the rule decision. All 20 drift
   faults stop there: deck scenario 3 is never reached in this run. (The deck's flow assumes drift is sent on.)
2. **The duplicate guard merges different HDFS failures** (P4). For log incidents the guard's key is
   `logs:hdfs:block_events`, the same for every pattern: once one HDFS failure is escalated, a *different* one in the
   same hour (datanode network vs replication) is closed as a duplicate, and its evidence reaches nobody. All 7 graded
   HDFS faults were closed this way. A key that includes the pattern would fix it.
3. **Most lost faults are lost before Diagnosis** (P1 / P2). Of 44 deploy faults, 12 produced no alert (slow leaks,
   small regressions: P1's own finding, "62 % of bad deploys caught") and 19 were kept as watch / noise by Triage.
   On what they do receive, Diagnosis is right 14 times out of 15 and no deploy fix ran on a wrong release.
4. **One wrong automatic action**: a CI runner outage diagnosed as flaky was re-run alone. P5's verification caught it
   (the re-run stayed red, the author was told), so it was reported `failed`, not hidden.
5. **Rollbacks are never automatic** (team decision): 0 % of deploy faults resolved with no human, by design.

## Limits

- Only the deploy family goes through P1's real detector, so MTTD is measured there only. CI, model and log alerts are
  synthetic (P1's format) and enter at Triage.
- The human is simulated (approves everything after 5 min), times are virtual (the Action agent's `SimClock`), and
  "detected" = the first abnormal point the detector flagged (P1 merges an episode before sending it: real delivery is later).
- Triage's LLM layer is off (no Ollama). Findings 1 and 2 may change with it on.
- These numbers describe our injected failures, not production.
