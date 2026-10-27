---
id: PI-003
title: Shared CI runners lost network, many projects red in the same hour
kind: incident
category: infra
sub_cause: ci_runners
confirmed: false
tags: [source:ci, dataset:travis, kind:anomaly, feature:build_failed, evidence:flaky_probability, evidence:many_projects_fail]
---
Example, not a real outage.

## What happened
Builds of six unrelated projects failed within 40 minutes. The flaky model was undecided on most of them.

## Cause
The package mirror used by the shared runners was down: dependency downloads timed out.

## Resolution
Re-runs were paused until the mirror came back, then all failed builds were re-run and passed.

## Lesson
Many projects failing at once is the machines, not the code: do not page each project owner.
