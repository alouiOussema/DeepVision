---
id: TB-action-risk
title: Action risk table (P3 + P4)
kind: table
---
What P4's Decision agent may do for each cause named by Diagnosis. Diagnosis only SUGGESTS the action;
P4 multiplies the risk of the action by the confidence of the diagnosis and decides.

## Table

| Cause           | Suggested action  | Risk   | Reversible | Act alone if confidence >= |
|-----------------|-------------------|--------|------------|----------------------------|
| flaky_test      | rerun (once)      | none   | yes        | 0.60                       |
| bad_deploy      | rollback          | low    | yes        | 0.80                       |
| config_change   | rollback (config) | low    | yes        | 0.80                       |
| data_drift      | retrain           | low    | yes        | 0.70 (day time only)       |
| code_regression | escalate (author) | medium | -          | never alone                |
| infra           | escalate          | medium | -          | never alone                |
| data_quality    | escalate          | medium | -          | never alone                |
| unknown         | escalate          | high   | -          | never alone                |

## Rules
- needs_human = true in a Diagnosis always means: ask a human, whatever the table says.
- A diagnosis with a warning "LLM argues for ..." is shown with both causes on the approval page.

## P4 rules on top of the table (Decision agent, team decision of Nov 2026)
- Confidence below 0.50 always escalates: no action is proposed, whatever the cause.
- Rollback (bad_deploy, config_change) always waits for a human on the approval page, as in the deck (scenario 2),
  even when confidence is above its bar. `DecisionConfig.auto_rollback = True` switches to the table's 0.80 rule.
- Retrain may start alone (bar above, day time only) but the new model is never deployed without a human.
- A re-run is granted once per CI project per day: failing again means a real bug, so it escalates.
- The bars in the table are read by agents/decision/policy.py: edit them here, not in code.
