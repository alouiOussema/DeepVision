"""What do the log events of a strange HDFS block SAY? Event ids become readable templates.

P1's PCA detector names the 3 events that made a block look strange (details.top_events), but
"E7+E10+E12" means nothing to a human. The Loghub template file turns them into text, and the
text says what kind of fault it is:

    exceptions while writing / receiving / transferring   -> infra: datanode or network fault
    replication asked / timed out                          -> infra: under-replicated storage
    only normal events (served, verified, received)        -> no fault visible: unknown
    a pattern no past incident or runbook mentions         -> novel: a human should look
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

import pandas as pd

from common.schemas import Category as C, Incident

from .. import paths
from ..config import DiagnosisConfig

# Loghub HDFS_v1 templates. data/raw/hdfs/HDFS.log_templates.csv wins when it is present.
BUILTIN_TEMPLATES = {
    "E1": "Adding an already existing block",
    "E2": "Verification succeeded for block",
    "E3": "Served block to host",
    "E4": "Got exception while serving block to host",
    "E5": "Receiving block src dest",
    "E6": "Received block src dest of size",
    "E7": "writeBlock received exception",
    "E8": "PacketResponder for block Interrupted",
    "E9": "Received block of size from host",
    "E10": "PacketResponder Exception",
    "E11": "PacketResponder for block terminating",
    "E12": "Exception writing block to mirror",
    "E13": "Receiving empty packet for block",
    "E14": "Exception in receiveBlock for block",
    "E15": "Changing block file offset of block meta file offset",
    "E16": "Transmitted block to host",
    "E17": "Failed to transfer block to host got exception",
    "E18": "Starting thread to transfer block to host",
    "E19": "Reopen Block",
    "E20": "Unexpected error trying to delete block, BlockInfo not found in volumeMap",
    "E21": "Deleting block file",
    "E22": "BLOCK* NameSystem allocateBlock",
    "E23": "BLOCK* NameSystem delete: block is added to invalidSet",
    "E24": "BLOCK* Removing block from neededReplications as it does not belong to any file",
    "E25": "BLOCK* ask host to replicate block to host",
    "E26": "BLOCK* NameSystem addStoredBlock: blockMap updated",
    "E27": "BLOCK* NameSystem addStoredBlock: Redundant addStoredBlock request received",
    "E28": "BLOCK* NameSystem addStoredBlock request received but it does not belong to any file",
    "E29": "PendingReplicationMonitor timed out block",
}
ERROR = re.compile(r"exception|error|fail|interrupted|timed out|unexpected", re.I)
REPLICATION = re.compile(r"replicat", re.I)


@lru_cache(maxsize=1)
def templates() -> dict[str, str]:
    if paths.HDFS_TEMPLATES.exists():
        df = pd.read_csv(paths.HDFS_TEMPLATES)
        return dict(zip(df["EventId"].astype(str), df["EventTemplate"].astype(str).str.replace("[*]", "", regex=False)))
    return BUILTIN_TEMPLATES


def collect(inc: Incident, cfg: DiagnosisConfig, f, kb=None) -> None:
    t = templates()
    weights = Counter()
    for a in inc.alerts:
        for e in a.details.get("top_events", []):
            weights[e["event"]] += e.get("share", 1 / 3)
    total = sum(weights.values()) or 1.0
    events = [e for e, _ in weights.most_common()]
    err = sum(w for e, w in weights.items() if ERROR.search(t.get(e, ""))) / total
    rep = sum(w for e, w in weights.items() if REPLICATION.search(t.get(e, ""))) / total
    pattern = "+".join(sorted(events[:3]))
    f.novel = kb is not None and not kb.mentions(pattern)
    f.features.update(log_n_blocks=math.log1p(len(inc.alerts)), share_error_events=err,
                      share_replication_events=rep, novel_pattern=float(f.novel))

    text = "; ".join(f"{e} '{t.get(e, '?')}'" for e in events[:3])
    if err >= 0.4:
        sub = "replication" if rep >= 0.3 else "datanode_network"
        f.sub_causes[C.INFRA] = sub
        f.add("log_errors", f"{len(inc.alerts)} block(s) share the events {text}: {err:.0%} of the weight is "
                            f"exceptions / failures while moving data between datanodes.", "logs",
              {C.INFRA: 1.8, C.UNKNOWN: -0.8}, pattern=pattern, share_error=round(err, 2))
    elif rep >= 0.3:
        f.sub_causes[C.INFRA] = "replication"
        f.add("log_replication", f"{len(inc.alerts)} block(s) share the events {text}: the namenode keeps asking "
                                 "for replicas, a storage / datanode capacity problem.", "logs",
              {C.INFRA: 1.4, C.UNKNOWN: -0.4}, pattern=pattern, share_replication=round(rep, 2))
    else:
        f.add("log_no_error", f"{len(inc.alerts)} block(s) share the events {text}: mostly normal events "
                              f"(errors are {err:.0%} of the weight), the block is unusual but little says it "
                              "failed.", "logs", {C.INFRA: -0.6, C.UNKNOWN: 1.0}, pattern=pattern,
              share_error=round(err, 2))
    if f.novel:
        f.add("novel_pattern", f"No runbook or past incident mentions the pattern {pattern}: never seen before.",
              "knowledge_base", {C.UNKNOWN: 0.6}, pattern=pattern)
