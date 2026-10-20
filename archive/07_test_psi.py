"""Does the PSI drift detector work, and which window size is best?

QUIET test : on the real test months, how often does it cry wolf?
DRIFT test : we inject a known change (a sudden step or a 30-day ramp) at 3 different dates
             and check whether PSI notices, and after how many days.
The test is repeated for several window sizes (a longer window is steadier but slower).

Cell format: "3/3 5d" = caught 3 times out of 3, median delay 5 days.
Run from the project root:  python -m scripts.07_test_psi
"""
import numpy as np
import pandas as pd
from config import settings
from agents.monitor.factory import make_detector, fit_detector

SOURCE = "nyc"
WINDOWS_DAYS = (7, 14, 28)
CASES = [("step", 1.05), ("step", 1.10), ("step", 1.25), ("ramp", 1.10), ("ramp", 1.25)]
START_POINTS = (0.20, 0.45, 0.70)        # where in the test period the drift begins
RAMP_HOURS = 24 * 30


def between(df, start, end):
    return df[(df.timestamp >= pd.Timestamp(start, tz="UTC")) &
              (df.timestamp <  pd.Timestamp(end,   tz="UTC"))]


def apply_drift(values, start, factor, kind):
    v = values.copy()
    if kind == "step":
        v[start:] *= factor
    else:
        ramp = np.clip((np.arange(len(v)) - start) / RAMP_HOURS, 0, 1)
        v *= 1 + (factor - 1) * ramp
    return v


def episodes(flag):
    return int(np.sum(np.diff(np.r_[0, flag.astype(int)]) == 1))


def main():
    df = pd.read_parquet(settings.UNIFIED_FILE)
    df = df[df.source == SOURCE]
    sp = settings.SPLITS[SOURCE]
    train, test = between(df, *sp["train"]), between(df, *sp["test"])
    pd.set_option("display.width", 200)

    for feature in settings.DRIFT[SOURCE]:
        if feature not in df.columns:
            continue
        tr = train[["timestamp", feature]].dropna()
        te = test[["timestamp", feature]].dropna().reset_index(drop=True)

        rows = []
        for days in WINDOWS_DAYS:
            det = make_detector("psi")
            det.window = days * 24
            fit_detector(det, tr[feature].values, tr["timestamp"])
            clean_flag = det.score(te[feature].values) > det.threshold
            row = {"window": f"{days} d", "limit": round(det.threshold, 2),
                   "quiet_%": f"{clean_flag.mean():.1%}",
                   "false_alarms": f"{episodes(clean_flag)} in {len(te) // 24} d"}
            for kind, factor in CASES:
                delays = []
                for frac in START_POINTS:
                    start = int(frac * len(te))
                    mod = apply_drift(te[feature].values, start, factor, kind)
                    new = (det.score(mod) > det.threshold) & ~clean_flag
                    hit = np.flatnonzero(new[start:])
                    if len(hit):
                        delays.append(hit[0] / 24)
                label = f"{kind} +{factor - 1:.0%}"
                row[label] = (f"{len(delays)}/{len(START_POINTS)} {np.median(delays):.0f}d"
                              if delays else f"0/{len(START_POINTS)}")
            rows.append(row)
        print(f"\n=== {SOURCE} / {feature} ===")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()