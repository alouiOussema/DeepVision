---
id: RB-data-quality
title: Data quality problem or external event
kind: runbook
category: data_quality
tags: [source:model, kind:anomaly, evidence:inputs_jump_together]
---
Several model inputs jump together in the same hours: one cause upstream of all of them.

## Signs
- Two or more inputs leave their weekday-hour band together, for a few hours only.
- A collapse to almost zero trips is usually missing data (the ingestion job failed).
- A holiday, a storm or a big event moves trip counts, fares and distances together: real, but not a bug.

## Check
- Row counts per hour in the ingestion job; late or missing files.
- The calendar: public holidays, weather alerts, events.

## Fix
- Missing data: re-run the ingestion job for the missing hours.
- Real-world event: no action; label the hours so they do not trigger a retraining.

## Risk
- Re-running ingestion: low risk.
