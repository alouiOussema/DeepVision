---
id: PI-001
title: Release leaked connections, latency and errors climbed for 2 hours
kind: incident
category: bad_deploy
sub_cause: resource_leak
confirmed: false
tags: [source:metrics, dataset:deploy, kind:anomaly, feature:latency_ms, feature:error_rate, evidence:deploy_before, evidence:metric_ramp]
---
Example written from P1's deploy simulation (a "leak" deploy), not a real outage.

## What happened
A release went out and nothing alerted for about 40 minutes. Then latency and the error rate crossed their band together and kept climbing: each minute was worse than the previous one.

## Cause
The new release opened a database connection per request and never closed it. The pool filled up slowly.

## Resolution
Rollback to the previous release. Latency was back in its band 10 minutes later.

## Lesson
A slow climb of both metrics after a deploy is a leak: roll back early, it will not recover alone.
