---
id: PI-002
title: Config push pointed the service to a wrong endpoint
kind: incident
category: config_change
sub_cause: bad_config
confirmed: false
tags: [source:metrics, dataset:deploy, kind:anomaly, feature:error_rate, evidence:config_before, evidence:metric_step]
---
Example, not a real outage.

## What happened
Two minutes after a configuration push, the error rate jumped and stayed high. Latency did not move.

## Cause
The push replaced the payment endpoint with the staging one; every call failed fast with an authentication error.

## Resolution
Config revert. The error rate was back in its band within 3 minutes.

## Lesson
Errors only, as a step, right after a config push: revert the config, do not roll back the code.
