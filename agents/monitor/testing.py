"""Test helpers for the planted-anomaly measurement (used only by scripts/03_test.py).

PART 1 - plant(): inject KNOWN anomalies into a time series and remember where they are.

Why: NYC and Travis have no labels, so without this nobody can tell whether a
detector works. After planting we know exactly which hours are abnormal.

Size is expressed in "local sigmas": how far the change is, compared with what
is normal FOR THAT weekday-hour. A 5-sigma spike at 4 am is small in absolute
terms but just as abnormal as a 5-sigma spike at 6 pm.
"""
import numpy as np
import pandas as pd

KINDS = {          # kind -> duration in hours
    "spike": 1,    # one hour too high
    "drop":  1,    # one hour too low
    "shift": 12,   # level moves up or down for half a day
    "drift": 48,   # slow ramp over two days
}
SIZES = (3, 5, 10)  # in local sigmas


def _slot(ts) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    return (ts.dayofweek * 24 + ts.hour).to_numpy()


def _local_sigma(train_values, train_ts, test_ts) -> np.ndarray:
    """Robust sigma (MAD) of each weekday-hour slot, looked up for the test hours."""
    s = pd.Series(np.asarray(train_values, float), index=_slot(train_ts))
    med = s.groupby(level=0).transform("median")
    mad = (s - med).abs().groupby(level=0).median() * 1.4826
    glob = float((s - s.median()).abs().median() * 1.4826) + 1e-9
    sigma = pd.Series(_slot(test_ts)).map(mad).fillna(glob).to_numpy()
    return np.maximum(sigma, 0.1 * glob)      # floor so tiny slots still get a real change


def plant(test_df, feature, train_values, train_ts, n_per_cell=4, seed=0, gap=24):
    """Return (series_with_anomalies, labels).

    series_with_anomalies: timestamp, <feature>, planted (bool), event_id (-1 if none)
    labels: one row per event: event_id, feature, kind, size, sign, start, end, n_hours
    """
    rng = np.random.default_rng(seed)
    df = test_df[["timestamp", feature]].reset_index(drop=True)
    clean = df[feature].to_numpy(float)
    values = clean.copy()
    sigma = _local_sigma(train_values, train_ts, df["timestamp"])

    n = len(df)
    taken = np.zeros(n, bool)
    event_id = np.full(n, -1)
    labels = []

    cells = [(k, s) for k in KINDS for s in SIZES for _ in range(n_per_cell)]
    rng.shuffle(cells)
    cells.sort(key=lambda c: -KINDS[c[0]])    # place long events first so they fit

    for kind, size in cells:
        dur = KINDS[kind]
        for _ in range(300):                   # try random positions
            start = int(rng.integers(gap, n - dur - gap))
            lo, hi = start - gap, start + dur + gap
            if taken[lo:hi].any() or np.isnan(clean[start:start + dur]).any():
                continue
            w = slice(start, start + dur)
            sign = {"spike": 1, "drop": -1}.get(kind) or int(rng.choice([-1, 1]))
            ramp = np.linspace(0, 1, dur) if kind == "drift" else 1.0
            values[w] = np.maximum(clean[w] + sign * ramp * size * sigma[w], 0)
            taken[w] = True
            event_id[w] = len(labels)
            labels.append({
                "event_id": len(labels), "feature": feature, "kind": kind,
                "size": size, "sign": sign,
                "start": df["timestamp"].iloc[start],
                "end": df["timestamp"].iloc[start + dur - 1],
                "n_hours": dur,
            })
            break

    out = df.copy()
    out[feature] = values
    out["planted"] = taken
    out["event_id"] = event_id
    return out, pd.DataFrame(labels)


# ---------------------------------------------------------------------------------------
# PART 2 - evaluate(): score a detector against the planted anomalies. No guessing: we know the truth.
# ---------------------------------------------------------------------------------------
def evaluate(timestamps, planted, flagged, labels):
    """timestamps: hours; planted: bool mask; flagged: bool mask (detector fired);
    labels: DataFrame from plant().  Returns (table_by_kind_and_size, summary)."""
    ts = pd.DatetimeIndex(timestamps)
    planted = np.asarray(planted, bool)
    flagged = np.asarray(flagged, bool)

    rows = []
    for e in labels.itertuples():
        m = np.asarray((ts >= e.start) & (ts <= e.end))
        hit = flagged[m]
        found = bool(hit.any())
        delay = (ts[m][int(hit.argmax())] - e.start) / pd.Timedelta("1h") if found else np.nan
        rows.append({"kind": e.kind, "size": e.size, "n_hours": e.n_hours,
                     "detected": found, "delay_h": delay})
    ev = pd.DataFrame(rows)

    table = (ev.groupby(["kind", "size"])
               .agg(events=("detected", "size"),
                    recall=("detected", "mean"),
                    median_delay_h=("delay_h", "median"),
                    n_hours=("n_hours", "first"))
               .reset_index())

    clean = ~planted
    weeks = clean.sum() / 168
    # recall a detector with this false-alert rate would get by pure luck
    rate = float(flagged[clean].mean())
    table["chance"] = 1 - (1 - rate) ** table["n_hours"]
    table = table.drop(columns="n_hours")

    summary = {
        "recall_overall": float(ev["detected"].mean()),
        "false_alerts": int(flagged[clean].sum()),
        "false_alerts_per_week": float(flagged[clean].sum() / weeks) if weeks else None,
        "alert_rate_on_clean_hours": float(flagged[clean].mean()),
    }
    return table, summary
