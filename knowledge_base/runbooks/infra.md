---
id: RB-infra
title: Infrastructure fault (machines, network, storage)
kind: runbook
category: infra
tags: [source:metrics, evidence:no_change]
---
Something below our code failed: hosts, network, disks, shared CI runners. Nothing we shipped is to blame.

## Signs
- No deploy and no config push before the incident.
- Latency climbs slowly with errors and nothing changed: capacity saturation (CPU, memory pressure, noisy neighbour).
- Many CI projects fail in the same hour: the shared runners, the network or a package mirror are down.
- HDFS logs with exceptions while writing or transferring blocks: a datanode or the network between datanodes.

## Check
- Host metrics: CPU, memory, disk, network errors of the machines behind the service.
- Status page of the cloud provider and of the CI provider.
- For HDFS: which datanodes appear in the failing blocks; one bad node is the usual cause.

## Fix
- Do not roll back: the code is not the cause and a rollback adds risk.
- Escalate to the infrastructure on-call. Scale out or move load away from the bad hosts.
- For CI infrastructure: pause automatic re-runs until the runners are healthy, then re-run the failed builds.

## Risk
- Restarting or draining a node: medium risk. Needs a human.
