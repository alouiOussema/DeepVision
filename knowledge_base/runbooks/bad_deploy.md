---
id: RB-bad-deploy
title: Bad deployment
kind: runbook
category: bad_deploy
tags: [source:metrics, evidence:deploy_before]
---
A new release degraded the service: latency, error rate or both left their normal band shortly after a deploy.

## Signs
- The incident starts within about 2 hours after a release, often within 20 minutes.
- Latency only jumps: latency regression (slower code path, missing index, chatty dependency call).
- Error rate only jumps: error regression (broken code path, incompatible API, missing migration).
- Latency and errors climb slowly together: resource leak (memory, connections, threads), worse every minute.
- A 2-3 minute bump that goes back to normal is a restart blip, not a bad deploy.

## Check
- Compare the metrics of the hour before the release with the time after it.
- Look at the release diff: new dependencies, queries, timeouts, pool sizes.
- For a leak, check memory and open connections of the new pods: they grow and never go down.

## Fix
- Roll back to the previous release. Rollback is safe and fast for stateless services.
- Do not roll back if the release contains a database migration that is not backwards compatible: escalate.
- After the rollback, verify that latency and error rate are back in their band within 15 minutes.

## Risk
- Rollback: low risk, reversible. A wrong rollback only delays a feature.
