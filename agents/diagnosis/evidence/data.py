"""Model inputs (NYC taxi): did the data slowly CHANGE (drift), or is it BROKEN / strange right now?

    PSI drift alert                          -> data_drift (a retraining question)
    several inputs jump in the same hour     -> data_quality (pipeline bug, missing data, external event)
    one input, short, weak                   -> unknown (probably a real-world oddity)
"""
from __future__ import annotations

from common.schemas import Category as C, Incident

from ..config import DiagnosisConfig


def collect(inc: Incident, cfg: DiagnosisConfig, f) -> None:
    if inc.kind == "drift":
        psi = max((a.details.get("peak_raw_scores", {}).get("psi", a.value) for a in inc.alerts), default=0.0)
        f.features["psi"] = float(psi)
        f.sub_causes[C.DATA_DRIFT] = "+".join(inc.features)
        f.add("psi_drift", f"The distribution of {', '.join(inc.features)} moved away from the training year "
                           f"(PSI {psi:.2f}, limit 0.25) for {inc.duration_hours:g} h: the inputs changed, slowly.",
              "alert", {C.DATA_DRIFT: 2.5, C.DATA_QUALITY: -0.5}, psi=round(float(psi), 3))
        return
    if len(inc.features) >= 2 and inc.duration_hours <= 6:
        f.sub_causes[C.DATA_QUALITY] = "burst_on_" + "+".join(inc.features)
        f.add("inputs_jump_together", f"{len(inc.features)} inputs ({', '.join(inc.features)}) left their usual "
                                      f"weekday-hour band together for {inc.duration_hours:g} h: one cause upstream "
                                      "of all of them (data pipeline, or an event like a holiday).", "alert",
              {C.DATA_QUALITY: 1.2, C.DATA_DRIFT: -0.8})
    else:
        f.add("single_input", f"Only {inc.features[0]} is abnormal, for {inc.duration_hours:g} h (strength "
                              f"{inc.peak_ratio:.1f}): too little to name a cause.", "alert",
              {C.UNKNOWN: 0.8, C.DATA_DRIFT: -0.5})
