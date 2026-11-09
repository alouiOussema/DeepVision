# Action agent (P5)

> Decision says **what happens next**. Action **does it, checks that it worked, undoes it if it made things worse, and says honestly how it went**.
> One approved `ActionDecision` in, one report (`executed | failed | reverted`) out, and a person is told whenever the machine could not fix it.

## Agent card

| | |
|---|---|
| **Goal** | Turn every approved order into a *verified* fix, fast, without ever making things worse silently (KPIs: MTTR, % resolved with no human) |
| **Input** | `ActionDecision` with status `approved` (by the Decision policy or a human on `ui/decision_app.py`): `state["to_action"]` in the graph, or `ActionInbox().ready()` on its own. For a rerun, the failed build id comes from the incident's CI alert (`Alert.details.build_id`, P1) |
| **Output** | `ActionInbox.report(id, executed \| failed \| reverted, detail)`, one `ActionRun` per order in `state["actions_done"]`, a line per step in the decision log (agent `action`), notifications in `data/action/notifications.jsonl` |
| **Tools** | pluggable backends (`backends.py`): CI, deployments, model registry, notifications. **All simulated** in this repo, plus one local scikit-learn retrain (`retrain_sklearn.py`) |
| **Rules in code** | no LLM, no new decision. Precheck → execute → verify → retry (max 2, only where safe) → undo if worse → report → tell a human |
| **When unsure** | a missing target, build id or constraint is never guessed: `failed` + a human is told. A result that cannot be measured is `unknown` = not fixed |
| **Safety** | only `approved`, never twice (O_EXCL claim file), the inbox wins over a stale copy, never deploys a model, simulation by default |

## The loop

```
approved order ─► gate ─► constraints ─► precheck ─► execute ─► wait verify_after_min ─► verify ─┬─ fixed ──────────────► report executed
                  status   max_retries   target?     once       (injected clock:              ├─ not fixed ─ retry? ─► report failed  + tell a human
                  claim    verify...     build id?               no sleep in tests)           ├─ worse ─── undo ────► report reverted + tell a human
                                         deploy gate                                          └─ unknown ───────────► report failed  + tell a human
```

| Action | Execute | Verify | Retry when | Undo |
|---|---|---|---|---|
| `rerun` | re-run the incident's failed build (CI backend) | the re-run passed | the CI API did not answer. **Not** when the re-run is red: rerun budget 1 (`constraints["reruns"]`, flaky-test runbook), red twice = real bug | - (a re-run changes nothing) |
| `rollback` | undo `order.target`, only if it is live | error rate and latency back in the normal band after `verify_after_min` (bad-deploy runbook: 15 min) | the deploy API did not answer. Not when still bad: we are already on the previous version | redeploy the target if the rollback made things **worse** (`revert_if_worse`) |
| `retrain` | train a challenger, score it against the champion on recent data neither has seen | challenger at least 2 % better | training crashed | - : the live model was never touched. A losing challenger is thrown away |
| `escalate` | notify `constraints["notify"]` (or on-call) with the summary + decision trail | message delivered | message not delivered | - (changes nothing) |

Who is told after a failure (`ActionConfig.escalate_to`, deck scenarios): red re-run → the commit **author**, rollback that did not help or was undone → **on-call**,
retrain not better → **ML engineer** (drift report), better challenger → **ML engineer** asked to approve the deploy.

## A real trail (python -m agents.action.demo, deck scenario 1)

```
1 · Flaky test: the re-run passes
  Decision : act -> rerun  [table:flaky_test]
  Action   : DONE + VERIFIED  attempts 1  MTTR 20 min
    - rerun on ACT-1 (approved by policy, rule table:flaky_test) [simulation]
    - attempt 1/3: re-ran build 48211 of shop-api (commit a1b2c3d) as run-04f9ec
    - waited 15 min before verifying
    - attempt 1/3: fixed: run-04f9ec passed: build green, the failure was flaky (test tagged flaky, nobody paged)
```

## Results of the demo (simulation, virtual clock)

Nine cases: the deck's four scenarios, each with its success and failure branches, plus a rejected rollback. Every case first goes
through the real Decision agent (and the approval step when Decision asks for a human). **9/9 behave as the deck says.**

| Case | Decision | Action | Told |
|---|---|---|---|
| 1 flaky, re-run passes | act | executed | - |
| 1b flaky, re-run red again | act | failed (not re-run) | author |
| 2 bad deploy, rollback works | human approves | executed (back in band after 15 min) | - |
| 2b bad deploy, still bad after rollback | human approves | failed | on-call |
| 2c bad deploy, rollback makes it worse | human approves | reverted (redeployed) | on-call |
| 3 drift, challenger better (−19 % MAE) | act | executed, **not deployed** | ML engineer (approve the deploy?) |
| 3b drift, challenger worse | act | failed, champion kept | ML engineer (drift report) |
| 4 unknown failure | escalate | executed (notification only) | on-call |
| 5 rollback rejected by a human | human rejects | not run | - |

KPIs of that run: MTTR median 20 min (order → verified fix, includes the 4 min a human took to approve and the 15 min verification wait),
25 % of orders resolved with no human at all, 3 of 7 fixes worked, 4 of 7 failed or were reverted. **These percentages come from the scenario mix
(half the cases are failure branches on purpose): they show that failures are reported honestly, not how good the system is.**

In the full graph (`monitor → triage → diagnosis → decision → action`, simulated deploy run, files in a temp dir): the 2 escalations were delivered,
the 6 rollbacks waited for a human; after one was approved, P5 rolled `release-064` back and verified it: error rate 6 % → 1 %, latency 180 → 120 ms.

## Run it

```bash
python -m agents.action.demo                    # the 9 cases above, data/action/demo/ (reset each run)
python -m agents.action.demo --real-retrain     # scenario 3 with scikit-learn on data/hourly/nyc.parquet (fails honestly if absent)
python pipeline.py sim --action                 # the whole graph ending with Action (simulation)
python -m agents.action.run                     # run what the inbox says is approved (e.g. after ui/decision_app.py)
python -m streamlit run ui/action_app.py        # results, who was told, models waiting for a deploy, the trail (read-only)
python -m pytest tests/test_action.py           # 46 tests, no Ollama, no data files, no sleep
```

In the graph: `build_graph(..., action_node=make_action_node())`. Decision's `route_after_decision` only calls it when `to_action` is not empty.

## What is simulated, what is real

| Part | Status |
|---|---|
| Rule loop, retries, undo, claims, reporting through `ActionInbox`, decision log, KPIs | real code, tested |
| CI re-run (`SimCI`) | **simulated**: no CI API in this repo. The build id is the real one from P1's alert |
| Rollback + metrics (`SimDeploy`) | **simulated**: a service whose health = base + effect of each live change. In the graph, the change log comes from P3's `ContextStore`; the simulation assumes the blamed change is the faulty one |
| Retrain (`SimModel`) | **simulated** scores by default. `SklearnNYCModel` really trains and compares two models on the NYC hourly table, but there is no live model or registry: the champion is "a model trained before the recent window" |
| Notifications (`FileNotifier`) | a JSON-lines file: nobody is really paged |
| Time | virtual in simulation (`SimClock`): 15 min of waiting take 0 s, so MTTR numbers are virtual |

A real adapter implements the same few methods (`backends.py` Protocols), sets `touches_production = True`, and is only accepted with `ActionConfig(simulate=False)`.
A backend that does not declare `touches_production` is assumed to touch production.

## Limits and next steps

- **Build id for reruns**: the `ActionDecision` does not carry it; P5 reads it from the incident in the graph (`state["to_diagnosis"]`). From the inbox alone
  (`python -m agents.action.run`), a rerun fails with "no build id". Ask P4 to append `snapshot["job"] = {"project", "build_id", "commit"}`: P5 already reads it.
- **Deploy of a retrained model**: P5 writes `data/action/deploy_requests.jsonl` (status `awaiting_human`) and tells the ML engineer. Who approves it,
  and through which page, is an open question with P4 (a follow-up order in the inbox needs a new action, e.g. `deploy_model`: a contract change).
- Only one service and one model; no database-migration check before a rollback (the runbook says to escalate then).
- A crash between execute and report leaves the order `approved` with a claim: it is never re-run, but only a human can close it (no `in_progress` status in the contract).
- Real adapters (GitHub Actions re-run API, `kubectl rollout undo`, MLflow registry, Slack) are not written: nothing to test them against here.
