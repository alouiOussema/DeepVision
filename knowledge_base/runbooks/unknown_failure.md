---
id: RB-unknown
title: Unknown failure
kind: runbook
category: unknown
tags: [evidence:novel_pattern, evidence:metric_blip, evidence:log_no_error]
---
The evidence does not point to any known cause, or the pattern was never seen before.

## Signs
- No change before the incident and no clear metric shape.
- A log pattern that no runbook and no past incident mentions.
- Hypotheses with close probabilities: the agent cannot tell them apart.

## Check
- Look at the raw data around the incident: logs, metrics, recent changes.
- Ask the owner of the service.

## Fix
- Do NOT act automatically. Escalate to a human with the evidence collected so far.
- Once the cause is known, approve or correct the diagnosis on the approval page: it becomes a past incident and the agent recognises it next time.

## Risk
- Any automatic action on an unknown cause: high risk.
