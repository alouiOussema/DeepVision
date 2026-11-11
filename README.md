# DeepVision

**A multi-agent system that keeps an MLOps pipeline running on its own, and asks a human before doing anything risky.**

When a test is flaky, a release breaks the service or the model's data drifts, someone usually has to notice it,
find the cause and fix it by hand. DeepVision chains five agents that do this loop: detect, group, diagnose,
decide, fix and verify. Our rule is *trust before autonomy*: no action without a visible reason, and a person
approves every risky one.

```
Monitor ──> Triage ──> Diagnosis ──> Decision ──┬── safe and sure ───────────────> Action ──> fixed and verified
detects     groups      finds the     risk x     │
anomalies   alerts,     cause, with   confidence └── risky or unsure ──> human approval ──> Action
            cuts noise  evidence
```

The agents run as one [LangGraph](https://github.com/langchain-ai/langgraph) graph (`pipeline.py`) and pass each
other typed messages defined in `common/schemas.py`: `Alert` → `Incident` → `Diagnosis` → `ActionDecision`.

## The agents

| Agent | What it does | Code | Author |
|---|---|---|---|
| **Monitor** | Watches CI builds, service metrics, model inputs and logs, and raises alerts (seasonal z-score, PSI drift, flaky-test model, PCA on logs, rolling baseline) | `agents/monitor/` | Oussema Aloui |
| **Triage** | Groups alerts into incidents and decides which ones deserve attention (rules, plus an LLM layer for unsure cases) | `agents/triage/` | Hajer Abdelkefi |
| **Diagnosis** | Names the root cause, the release to undo and a calibrated confidence, citing its evidence (rules + classifier + RAG + LLM explanation) | `agents/diagnosis/` | Eya Jmaa |
| **Decision** | Chooses what happens next from a policy table: act alone, ask a human, escalate, or do nothing | `agents/decision/` | Seifcha |
| **Action** | Executes approved orders, verifies the fix, undoes it if things got worse, and reports honestly | `agents/action/` | Hanine Attia |
| **Evaluation & demo** | Injects failures with known answers, runs the whole chain, traces every agent and grades the KPIs; the final demo app | `agents/evaluation/`, `ui/demo_app.py` | Hedi Brahmi |

Diagnosis, Decision, Action and Evaluation have their own README with details and results.

## The four scenarios

| | Scenario | What goes wrong | Right reaction |
|---|---|---|---|
| 1 | Flaky test in CI | A build turns red; the test passed on the same code before | Re-run once, alone. Red again → tell the author |
| 2 | Bad deployment | Errors or latency jump right after a release | Roll back the release, after a human approves |
| 3 | Data drift | A model input slowly changes | Retrain a challenger; a human approves its deploy |
| 4 | Unknown failure | A log pattern never seen before | Never act alone: escalate with the evidence |

## The project we operate, and its data

The pipeline DeepVision looks after is a **taxi-fare model**: it predicts the average fare of the coming hour in New
York. Around it sit a CI, a web service and storage. Its data comes from public datasets we brought in, not from
something we built:

| Data | Source | Real or not |
|---|---|---|
| Model data | [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page): 22.7 M taxi trips turned into one row per hour | real |
| CI builds | [TravisTorrent](https://travistorrent.testroots.org) | real dataset; the evaluation uses examples in its format |
| Storage logs | [Loghub HDFS](https://github.com/logpai/loghub) | real dataset; the evaluation uses examples in its format |
| Web service metrics | the Monitor's simulator (`ingest/deploy.py`) | simulated |

On top of it we **inject failures** with known answers to grade the agents. Samples of every source, the list of
injected faults and the download links are in [`demo_data/`](demo_data/README.md) (rebuild with
`python -m scripts.make_demo_data`). The real data shows a genuine drift: the average fare per trip went from $17.95
(Jan 2025) to $21.02 (Jan 2026), +17 %.

## Getting started

Python 3.12.

```bash
python3 -m venv .venv
.venv/bin/pip install pytest pandas numpy scikit-learn pyarrow langgraph streamlit ollama
```

Ollama is optional: without it, Triage and Diagnosis use their rules and template explanations.

```bash
.venv/bin/python -m streamlit run ui/demo_app.py       # the demo: project, scenarios, every step, one incident at a time
.venv/bin/python pipeline.py sim --action              # the whole chain on a simulated deployment, in the terminal
.venv/bin/python -m agents.evaluation.run              # inject failures, run the five agents, grade them (~20 s)
.venv/bin/python -m pytest tests/                      # tests: no Ollama and no data files needed
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python`.

Other pages, one per agent:

```bash
.venv/bin/python -m streamlit run app.py                  # Triage playground
.venv/bin/python -m streamlit run ui/approval_app.py      # review diagnoses
.venv/bin/python -m streamlit run ui/decision_app.py      # approve or reject risky actions
.venv/bin/python -m streamlit run ui/action_app.py        # what the Action agent did
```

## Results

We broke the pipeline on purpose (154 real faults plus harmless controls, across the four scenarios), ran the five
agents and graded them against the truth. Only cases the Diagnosis classifier never trained on are graded.

| KPI | Result |
|---|---|
| Noise reduction (alerts → incidents sent on) | 532 → 78, **85 %** |
| Time to detect a bad release (median) | **4 min** |
| Diagnosis accuracy | **96 %** |
| False action rate | **4.5 %** (1 of 22 fixes, caught and reported) |
| Real faults silenced | **0** |
| Time to a verified fix (median) | **15 min** |

What we learned: the middle of the chain is reliable, but most faults are lost early. Some slow faults raise no
alert, and Triage holds back drift and weak episodes. Details, findings and limits are in
[`agents/evaluation/README.md`](agents/evaluation/README.md).

These numbers come from a simulation (injected failures, a simulated human, virtual time), not from production.

## Repository layout

```
agents/          one folder per agent (+ evaluation)
demo_data/       samples of every data source and the list of injected faults
common/          shared message contracts (schemas.py)
knowledge_base/  runbooks, past incidents, action risk table
ui/              Streamlit pages; demo_app.py is the main one
pipeline.py      the LangGraph graph
ingest/, scripts/, config/   data loading and the Monitor's training scripts
tests/           one test file per agent
```

`data/`, `logs/*.csv` and `models/*.pkl` are not in git.
