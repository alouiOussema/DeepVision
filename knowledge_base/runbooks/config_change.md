---
id: RB-config-change
title: Bad configuration push
kind: runbook
category: config_change
tags: [source:metrics, evidence:config_before]
---
A configuration change (feature flag, endpoint, credentials, limits) broke the service without any new code.

## Signs
- A config push happened just before, and it is the most recent change.
- The error rate jumps at once (a step, not a ramp): requests fail fast on a wrong endpoint, expired secret or bad flag.
- Latency often stays normal or even drops, because failing requests are fast.

## Check
- Diff the configuration versions: endpoints, timeouts, secrets, feature flags, rate limits.
- Look for authentication or connection errors in the application logs.

## Fix
- Revert the configuration to the previous version. Faster and safer than a code rollback.
- Verify the error rate is back in its band within 5 minutes.

## Risk
- Config revert: low risk, reversible.
