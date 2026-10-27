# Diagnosis agent (P3)

> Triage says **what** is abnormal. Diagnosis says **why**, which change to undo, how sure it is,
> and when a human must look. One `Incident` in, one `Diagnosis` out, to the Decision agent (P4).

## Agent card

| | |
|---|---|
| **Goal** | Name the root cause of an incident with cited evidence and a confidence P4 can multiply by the action risk |
| **Input** | `Incident` from Triage (P2) + what is around it: change log, raw minute metrics, HDFS log templates, the other incidents of the batch, the knowledge base |
| **Output** | `Diagnosis` (`common/schemas.py`): `category`, `sub_cause`, `suspect` (release / config to undo), `confidence` (calibrated), `evidence[]`, ranked `hypotheses`, `explanation`, `suggested_action`, `needs_human`, `warnings` |
| **Causes** | `flaky_test` · `code_regression` · `bad_deploy` · `config_change` · `infra` · `data_drift` · `data_quality` · `unknown` |
| **Tools** | 7 evidence collectors, rule scorer, gradient-boosting classifier, hybrid RAG (BM25 + embeddings), local LLM (Ollama) |
| **Rules in code** | the LLM explains but never picks the cause; every citation and number it writes is checked; disagreement only lowers confidence |
| **When unsure** | best cause < 0.40 → `unknown`; confidence < 0.60 → `needs_human`; never-seen pattern → confidence × 0.7; `unknown` always escalates, except a confident *recovery* or a known false alarm (action `none`) |

## How it works

```
Incident ──► 1 COLLECT evidence ─────────► 2 RANK causes ─────────► 3 RETRIEVE ──► 4 EXPLAIN ──► 5 JUDGE ──► Diagnosis
             incident    (what Triage saw)   rules: prior + Σ log-odds   runbooks +     LLM, fact-     confidence,
             changes     (which release?)    classifier: boosted trees   past incidents checked;       needs_human,
             metrics     (step/ramp/blip/    blend 50/50, then           BM25 + nomic-  template if    suggested
                          recovery)          calibrated (isotonic)       embed, RRF     rejected       action, suspect
             ci / logs / data / history
```

Every piece of evidence is a sentence with its numbers, an id (`E3`) and the causes it supports.
Real output of the demo (held-out seed 4, incident DEP4-00028):

```
diagnosis : config_change (bad_config)  suspect=config-003  confidence=0.99  -> rollback
[E2] Config push config-003 went out at 01-25 15:28; the incident started 0 min right after;
     metrics started to degrade at 15:28.                                 supports config_change +2.0
[E3] After config-003, only the error rate jumped (latency_ms +4%, error_rate +372% (step)):
     requests fail fast, which is typical of a wrong setting or a broken code path.
                                                                          supports config_change +0.6, bad_deploy +0.3
[E4] Past incident PI-002 ('Config push pointed the service to a wrong endpoint') looks like
     this one (similarity 1.00); its cause was config_change (bad_config).
```

### What makes it more than a classifier

| Feature | Why it matters |
|---|---|
| **Onset-based blame** (`evidence/changes.py`) | finds the minute metrics started to degrade and blames the last change *before* it, not the latest one: a harmless release landing on top of a leak is cleared, so P5 rolls back the right one |
| **Metric shape** (`evidence/metrics.py`) | step in latency → latency regression; errors only → error regression / bad config; both ramping → leak (after a release) or saturation (no release); short bump → blip |
| **Recovery detection** | the detector also fires when metrics go *back* to normal after a rollback; recognised as no fault, so nobody is paged twice. Must return to the healthy level, so a dip inside a live fault is not silenced |
| **Cross-incident reasoning** (`evidence/ci.py`) | 5 projects red in the same hour = CI infrastructure, not 5 code bugs |
| **Log templates** (`evidence/logs.py`) | turns `E7+E10+E12` into "writeBlock received exception, PacketResponder Exception, ..." and into a fault class |
| **Case-based memory** (`evidence/history.py`, `feedback.py`) | an approved or corrected diagnosis becomes a confirmed past incident; the next similar one finds it. Learning without retraining |
| **Calibrated confidence** (`classifier.py`, `train.py`) | isotonic curve fitted on out-of-fold diagnoses: 0.9 means right about 90 % of the time |
| **Fact-checked LLM** (`llm.py`) | invented `[E9]` or a number found nowhere in the evidence → the answer is rejected and the template is used |

## Run it

```bash
python -m agents.diagnosis.train                 # benchmark (first time ~2 min) + classifier
python -m agents.diagnosis.evaluate              # baseline vs rules vs ml vs hybrid
python -m agents.diagnosis.evaluate --llm llama3.1:8b,mistral:instruct --n-llm 16   # + compare 2 LLMs
python -m agents.diagnosis.demo [--llm llama3.1:8b]   # Scenario 2 end to end, fills the approval inbox
python -m streamlit run ui/approval_app.py       # human approval page
python -m pytest tests/test_diagnosis.py         # 18 tests, no Ollama needed
```

Ollama is optional: `ollama pull nomic-embed-text` for embeddings, `llama3.1:8b` or `mistral:instruct` for explanations.
Without it the agent uses BM25 and template explanations.

### In the orchestrator (P4)

```python
from agents.diagnosis.node import make_diagnosis_node
graph.add_node("diagnosis", make_diagnosis_node(llm_model="llama3.1:8b"))   # or None: no LLM
graph.add_edge("to_diagnosis", "diagnosis")
# reads state["to_diagnosis"] (and optional state["context"]: ContextStore) -> writes state["diagnoses"], state["log"]
```

## Data

| Source | What it gives | Status |
|---|---|---|
| **P1's deploy simulator → P1's rolling detector → P2's Triage** | bad deploys (slow / errors / leak) and harmless deploys, with the truth; we inject config pushes and infra saturation so causes must be told apart | used, 6 seeds (4 train, 2 held out) |
| **P1's labelled exports** (`ci_labels.csv`, `hdfs_labels.csv`) | real TravisTorrent flaky vs real failures, real Loghub HDFS failing blocks | loaded automatically when present (`data/triage_eval/`) |
| **Synthetic CI / logs / model inputs** in P1's exact Alert format | causes no public dataset labels: CI outage, replication failure, data outage; overlapping distributions and hard cases | used until P5's injections exist |
| **Knowledge base** (`knowledge_base/`) | 9 runbooks, 5 past incidents (2 from real team findings), action risk table | used |

## Results

Test split: 200 incidents never used for training (2 held-out simulation seeds + 40 % of the other families).

| Method | Accuracy | Macro-F1 | Right release | Handled alone | ...of which right | Wrong auto actions | Silenced faults | Calibration error |
|---|---|---|---|---|---|---|---|---|
| Baseline (source → fixed cause) | 0.35 | 0.17 | 0.00 | 100 % | 0.35 | 57 | 0 | 0.12 |
| Rules only | 0.88 | 0.89 | 0.87 | 72 % | 0.93 | 2 | 1 | 0.15 |
| Classifier only | 0.87 | 0.90 | 0.87 | 77 % | 0.91 | 3 | 1 | 0.06 |
| **Hybrid (deployed)** | **0.93** | **0.94** | **0.87** | **78 %** | **0.96** | 2 | 1 | **0.05** |
| **Hybrid, on the 78 incidents Triage sends** | **0.97** | **0.98** | **0.91** | 77 % | **0.98** | **0** | **0** | 0.07 |

- Rules and classifier fail on different cases (rules better on logs, classifier on deploys): the blend beats both.
- The remaining wrong actions and the silenced fault are all on incidents Triage had already filed as *watch* or *noise*.

**Two LLMs** (16 stratified held-out incidents each; the cause is never changed by the LLM, so accuracy is identical):

| | llama3.1:8b | mistral:instruct |
|---|---|---|
| answers that passed the fact-check | **16 / 16** | 14 / 16 (2 replaced by the template) |
| invented evidence ids / numbers | 0 / 0 | 0 / 0 |
| seconds per incident (CPU) | ~79 (incl. cold start, shared CPU) | ~18 |

## What we tried (and kept or dropped)

| Experiment | Result | Decision |
|---|---|---|
| Blame the latest change vs onset-based blame | first version blamed a harmless release landing on a leak | onset-based |
| Onset: threshold crossing vs change-point (max mean-after − mean-before) | right release on train deploys: **90 / 96** vs 69 / 96; noisy error minutes pull the change-point early | threshold crossing (simple wins) |
| Confidence: raw probability vs isotonic calibration | calibration error 0.23 → **0.04**; cases handled without a human 57 % → 70 % at the same safety | calibrated |
| Recovery = "metrics went down" vs "went down **and** back to the healthy level" | the first silenced 2 live faults | second |
| Retrieval without vs with source filter | without: an HDFS runbook advised for a web-service incident | filter by source |
| ChromaDB vs in-memory numpy index | 73 short chunks fit in memory; no server, no dependency (not benchmarked against Chroma) | numpy (`Index` is the swap point) |

## Limits and next steps

- The deploy truth comes from P1's simulator, and three families are synthetic: the real test is P5's injected failures (`data/injections/`) and live pipeline data, and the benchmark loader is the one place to plug them in.
- Config and infra faults are injected by us (6 config cases in the test split): too few to be precise about `config_change`.
- CPU / memory faults barely move latency (P1's RCAEval finding): a host-metrics collector (CPU, memory) would make `infra` sub-causes sharper.
- Raw build logs do not exist for TravisTorrent, so CI diagnosis relies on P1's flaky model and history, not on the failing test's output.
