---
id: RB-flaky-test
title: Flaky test
kind: runbook
category: flaky_test
tags: [source:ci, evidence:flaky_probability, evidence:project_history]
---
A test failed for a reason unrelated to the code change: timing, network, random order, shared state.

## Signs
- High flaky probability from the Monitor's flaky model (above 0.5).
- The project has a history of flaky failures.
- Different commits fail on different tests; a re-run of the same commit usually passes.

## Check
- Re-run the failed build once. If it passes, the failure was flaky.
- Note the failing test name: the same test failing on unrelated commits is a known flaky test.

## Fix
- Re-run the build automatically, one time only. Nobody needs to be paged.
- If the same test is flaky 3 times in a week, open a ticket to quarantine or fix it.

## Risk
- Re-run: no risk. A real failure simply fails again and comes back with more evidence.
