---
id: PI-004
title: HDFS pattern E2+E3+E4 is a known detector false alarm
kind: incident
category: unknown
sub_cause: known_false_alarm
confirmed: true
tags: [source:logs, dataset:hdfs, kind:anomaly, feature:block_events, evidence:log_errors, event:E2, event:E3, event:E4]
---
From the Triage agent's operator feedback (P2's memory, scenario 4b): thousands of HDFS blocks with this pattern were flagged by the PCA detector, and the operator marked them as a false alarm.

## What happened
Blocks with Verification succeeded (E2), Served block (E3) and Got exception while serving block (E4) look strange to the detector because E4 is rare, but the blocks are healthy.

## Cause
A reader closed its connection early: the datanode logs an exception while serving, the block itself is fine.

## Resolution
No action. The pattern is not a failure.

## Lesson
E2+E3+E4 alone is not a fault. Escalate only if other error events appear with it.
