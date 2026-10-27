---
id: RB-code-regression
title: Code regression in CI
kind: runbook
category: code_regression
tags: [source:ci, evidence:same_commit_fails, evidence:usually_green]
---
A commit really broke the build or the tests.

## Signs
- Low flaky probability (below 0.2).
- The same commit fails again after a re-run.
- The project usually passes, so the failure is news.

## Check
- Find the first failing commit and the tests it breaks.

## Fix
- Notify the author of the commit; block the merge until it is fixed or reverted.
- Do not re-run in a loop: it wastes CI time and hides the regression.

## Risk
- Reverting a commit on main: medium risk, the author or a reviewer decides.
