"""Follow one incident from the first alarm to the verified fix: the deck's scenarios, success and failure branches."""
import streamlit as st

from demo_pages.common import FAMILIES, ORDER, need_data, run_dir
from agents.evaluation import paths
from agents.evaluation.report import verdict
from agents.evaluation.score import Record

AGENT = {"monitor": ("📡", "Monitor"), "triage": ("🧹", "Triage"), "diagnosis": ("🔍", "Diagnosis"),
         "decision": ("⚖️", "Decision"), "human": ("🙋", "Human"), "action": ("🛠️", "Action")}

st.title("Follow one incident, step by step")
st.markdown("Every agent wrote down **what it received, why it decided, what it did and what came out**. Pick a story: "
            "each one is a real incident of this run, next to the truth we injected.")

d = need_data()
recs = d["scored"]


def first(mask, why):
    hit = recs[mask]
    return (hit.iloc[0], why) if len(hit) else None


fam, sent = recs["family"], recs["sent"]
STORIES = {
    "Scenario 1 · Flaky test: re-run alone, fixed": first(
        (fam == "ci") & (recs["truth"] == "flaky_test") & recs["fixed_alone"],
        "The deck's success branch: nobody is paged for a test that is not really broken."),
    "Scenario 1 · Failure branch: a real bug goes to its author": first(
        (fam == "ci") & (recs["truth"] == "code_regression") & (recs["action"] == "escalate"),
        "A real regression is never re-run in a loop: its author is told, with the evidence."),
    "Scenario 1 · The one wrong automatic action, caught": first(
        (fam == "ci") & recs["wrong_fix"] & (recs["route"] == "act"),
        "A CI outage looked flaky; the re-run stayed red, the agent reported 'failed' and told a person."),
    "Scenario 2 · Bad deploy: a human approves the rollback, fixed": first(
        (fam == "deploy") & recs["fixed"],
        "The deck's scenario 2: the right release is blamed, a human approves, the metrics are checked after 15 min."),
    "Scenario 2 · Wrong diagnosis, but no harm done": first(
        (fam == "deploy") & sent & (recs["diagnosis_right"] == False) & (recs["action"] == "escalate"),  # noqa: E712
        "Diagnosis was wrong, but its answer (infra) is never handled alone: the mistake reached a person, not production."),
    "Scenario 2 · Metrics coming back after a fix: nobody paged": first(
        (fam == "deploy") & sent & (recs["truth_sub"] == "recovery") & (recs["action"] == "none"),
        "The detector also fires when metrics go back to normal; the system recognises it and stays quiet."),
    "Scenario 3 · Data drift held back by Triage": first(
        (fam == "model") & (recs["truth"] == "data_drift") & ~sent,
        "Our finding: drift stays in 'watch' unless Triage's LLM layer sends it on (off in this demo)."),
    "Scenario 3 · Broken data handed to a person": first(
        (fam == "model") & (recs["truth"] == "data_quality") & (recs["action"] == "escalate"),
        "Broken data is not fixed by retraining: a person is told."),
    "Scenario 4 · Storage failure: a person takes over": first(
        (fam == "logs") & (recs["truth"] == "infra") & (recs["action"] == "escalate"),
        "The deck's scenario 4: the system knows its limits and escalates with the evidence."),
    "Scenario 4 · Storage failure closed as a duplicate": first(
        (fam == "logs") & recs["closed_duplicate"],
        "Our finding: a different HDFS failure in the same hour is closed as a copy of an earlier one."),
    "Scenario 4 · Normal blocks: nobody paged": first(
        (fam == "logs") & (recs["truth"] == "unknown") & (recs["action"] == "none"),
        "Strange-looking but healthy log blocks: recognised, logged, nobody woken up."),
}
STORIES = {k: v for k, v in STORIES.items() if v is not None}

mode = st.segmented_control("Choose", ["A ready-made story", "Any incident"], default="A ready-made story") \
    or "A ready-made story"
if mode == "A ready-made story":
    name = st.selectbox("Story", list(STORIES))
    r, why = STORIES[name]
    st.caption(why)
else:
    f = st.selectbox("Scenario", ORDER, format_func=lambda x: f"{FAMILIES[x]['scenario']} · {FAMILIES[x]['name']}")
    sub = recs[recs["family"] == f]
    key = st.selectbox("Incident", sub["key"].tolist(), format_func=lambda k: (
        lambda x: f"{k} · truth {x.truth or '-'} · {x.triage} · {x.action or '-'} {x.outcome or ''}")(
        sub.set_index("key").loc[k]))
    r = sub.set_index("key").loc[key]
    r["key"] = key

rec =Record(**{k: (v.to_pydatetime() if hasattr(v, "to_pydatetime") else v)
                for k, v in r.to_dict().items() if k in Record.__dataclass_fields__})
rec.finished_at = None
v = verdict(rec)
(st.success if v.startswith(("fixed", "closed, nobody", "filtered", "handed to a person (right")) else
 st.error if v.startswith(("WRONG", "missed")) else st.info)(f"**Outcome:** {v}")
c = st.columns(4)
c[0].markdown(f"**Injected truth**  \n{r['truth'].replace('_', ' ') or '–'}"
              f"{' · ' + r['truth_sub'].replace('_', ' ') if r['truth_sub'] else ''}"
              f"{' · ' + r['truth_suspect'] if r['truth_suspect'] else ''}")
c[1].markdown(f"**Diagnosis said**  \n{(r['category'] or '–').replace('_', ' ')}"
              f"{' · ' + r['suspect'] if r['suspect'] else ''}"
              f"{f' · {r.confidence:.2f}' if r['category'] else ''}")
c[2].markdown(f"**Decision**  \n{(r['route'] + ': ' + r['action']) if r['action'] else '–'}"
              f"{' ' + r['target'] if r['target'] else ''}")
c[3].markdown(f"**Result**  \n{r['outcome'] or r['status'] or r['triage']}")

st.subheader("The timeline")
events = sorted((e for e in d["events"] if e["incident_id"] == r["key"]),
                key=lambda e: (e["at"] or "", list(AGENT).index(e["agent"])))
for e in events:
    icon, name = AGENT[e["agent"]]
    with st.container(border=True):
        left, right = st.columns([1, 5])
        left.markdown(f"### {icon}\n**{name}**  \n{(e['at'] or '')[5:16].replace('T', ' ')}")
        right.markdown(f"**Received:** {e['input']}  \n**Did:** {e['action']}  \n**Result:** {e['result']}")
        if e["reason"]:
            with right.expander("Why (in the agent's own words)"):
                st.markdown(e["reason"].replace(" -> ", "\n\n→ ").replace(" | ", "\n\n• "))
if not r["sent"] and r["is_fault"]:
    st.warning("The story stops here: Triage kept this real fault as *" + r["triage"] + "*, so no agent after it saw "
               "the problem.")

episode, incident_id = r["key"].split(":", 1)
page = run_dir() / paths.REPORTS / episode / f"{incident_id}.md"
if page.exists():
    st.download_button("Download the incident report (auto-drafted)", page.read_text(encoding="utf-8"),
                       file_name=r["key"].replace(":", "_") + ".md")
