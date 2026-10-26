---
id: RB-data-drift
title: Data drift on model inputs
kind: runbook
category: data_drift
tags: [source:model, kind:drift, evidence:psi_drift]
---
The distribution of the model inputs moved away from the training data (PSI above 0.25), slowly, over days.

## Signs
- PSI alert on one input over a 7 to 28 day window, kind drift.
- Example: average fares went up after a price change; distances changed after a new zone policy.

## Check
- Compare the recent window with the training year for the drifted input.
- Check if model quality dropped on recent labelled data (champion metrics in MLflow).

## Fix
- Retrain on recent data and compare challenger against champion before promoting.
- Never retrain at night on a drift alert alone: it is a planned action, not an emergency.

## Risk
- Retraining: low risk while the champion is kept until the challenger wins.
