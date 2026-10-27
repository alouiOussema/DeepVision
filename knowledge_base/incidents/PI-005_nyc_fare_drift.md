---
id: PI-005
title: NYC taxi fares shifted in January 2026 (+7 %)
kind: incident
category: data_drift
sub_cause: avg_fare
confirmed: true
tags: [source:model, dataset:nyc, kind:drift, feature:avg_fare, evidence:psi_drift]
---
Found by the Monitor's PSI detector on real NYC TLC data (team presentation, slide 26).

## What happened
The PSI of avg_fare over a 7-day window crossed 0.25 in January 2026 and stayed above it.

## Cause
A real change of the fares in January 2026 (about +7 %): the inputs changed, the model did not.

## Resolution
Retraining planned on data that includes 2026, champion against challenger before promotion.

## Lesson
A drift that stays is a retraining question for the day team, never a night page.
