---
id: RB-hdfs-datanode
title: HDFS block write and transfer failures
kind: runbook
category: infra
sub_cause: datanode_network
tags: [source:logs, dataset:hdfs, evidence:log_errors, event:E7, event:E10, event:E12, event:E14, event:E17, event:E29]
---
HDFS blocks whose log events are exceptions while data moves between datanodes.

## Signs
- writeBlock received exception (E7), PacketResponder Exception (E10), Exception writing block to mirror (E12).
- Exception in receiveBlock (E14), Failed to transfer block (E17): the chain of datanodes broke during a write.
- PendingReplicationMonitor timed out block (E29) and ask to replicate (E25): the namenode cannot get enough replicas.

## Check
- Group the failing blocks by datanode address: one node in most of them means a bad disk or network card.
- Look at the datanode logs around the same time for disk errors or long GC pauses.

## Fix
- Decommission the bad datanode; HDFS re-replicates its blocks.
- If many nodes are involved, check the network (switch, rack) before touching nodes.

## Risk
- Decommissioning a datanode: medium risk, needs a human.
