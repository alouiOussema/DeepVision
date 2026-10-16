"""
Generate a PDF report of training/evaluation metrics.
Run: python -m scripts.make_report_pdf
Output: reports/training_report.pdf
"""
import json
import textwrap
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from config import settings


# ---------------- helpers ----------------

def _section(pdf, title, lines):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.06, 0.95, title, fontsize=15, fontweight="bold", va="top")
    body = "\n".join(textwrap.fill(l, 95) if l else "" for l in lines)
    fig.text(0.06, 0.90, body, fontsize=10, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _histogram_page(pdf, title, source, feature, det_name, m):
    hist = m.get("score_histogram")
    if not hist:
        return
    counts = np.array(hist["counts"])
    edges  = np.array(hist["edges"])
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8.27, 5))
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9)
    ax.set_title(f"{title}\n{source} / {feature} / {det_name}", fontsize=12)
    ax.set_xlabel("score")
    ax.set_ylabel("number of hours")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _hour_dist_page(pdf, title, source, feature, det_name, m):
    by_hour = m.get("alerts_by_hour", {})
    if not by_hour:
        return
    hours = [int(h) for h in by_hour.keys()]
    counts = [by_hour[str(h)] if str(h) in by_hour else by_hour.get(h, 0)
              for h in hours]
    all_hours = list(range(24))
    all_counts = [by_hour.get(str(h), by_hour.get(h, 0)) for h in all_hours]

    fig, ax = plt.subplots(figsize=(8.27, 5))
    ax.bar(all_hours, all_counts)
    ax.set_title(f"{title}\n{source} / {feature} / {det_name}", fontsize=12)
    ax.set_xlabel("hour of day (0–23)")
    ax.set_ylabel("number of alerts")
    ax.set_xticks(all_hours)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _month_dist_page(pdf, title, source, feature, det_name, m):
    by_month = m.get("alerts_by_month", {})
    if not by_month:
        return
    months = sorted(by_month.keys())
    counts = [by_month[k] for k in months]

    fig, ax = plt.subplots(figsize=(8.27, 5))
    ax.bar(range(len(months)), counts)
    ax.set_title(f"{title}\n{source} / {feature} / {det_name}", fontsize=12)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m[:7] for m in months], rotation=45, ha="right")
    ax.set_ylabel("number of alerts")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------- main report ----------------

def main():
    with open(settings.MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    out = settings.REPORTS_DIR / "training_report.pdf"

    with PdfPages(out) as pdf:
        # cover
        _section(pdf, "Monitor — Training & Evaluation Report",
                 [f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                  "",
                  "This report describes the detectors' behaviour on the test window.",
                  "It covers three sources: NYC taxi, HDFS logs, TravisTorrent CI.",
                  "",
                  "Two categories of metrics appear:",
                  "  Supervised   (HDFS only): precision, recall, F1, AUC",
                  "  Unsupervised (NYC, Travis): alert rate, separation ratio,",
                  "                              score histogram, hour/month distribution",
                  "",
                  "Every section that follows begins with a definition of the",
                  "metrics it contains, then reports the observed values."])

        # metric definitions
        _section(pdf, "Metric definitions (1/2)",
                 ["ALERT RATE",
                  "  Fraction of test hours flagged as anomalous.",
                  "  Target range for the Monitor: 1% to 5%.",
                  "  Below 0.5%: too quiet, missing real anomalies.",
                  "  Above 10%:  too noisy, drowns the downstream agents.",
                  "",
                  "SEPARATION RATIO",
                  "  mean(score | flagged) / mean(score | not flagged).",
                  "  > 1 means flagged hours score higher than unflagged ones (good).",
                  "  < 1 means flagged hours score lower (the detector is inverted).",
                  "  Typical good values: 2 to 10.",
                  "",
                  "SCORE HISTOGRAM",
                  "  Distribution of every hour's anomaly score.",
                  "  Long right tail or bimodal: detector separates two populations.",
                  "  Flat or single tight cluster: detector sees no structure.",
                  "",
                  "PRECISION / RECALL / F1  (requires labels)",
                  "  Precision: of flagged hours, fraction that were truly anomalous.",
                  "  Recall:    of truly anomalous hours, fraction that were flagged.",
                  "  F1:        harmonic mean of the two.",
                  "",
                  "AUC",
                  "  Ranking quality across all thresholds.",
                  "  0.5 = random, 1.0 = perfect separation."])

        _section(pdf, "Metric definitions (2/2)",
                 ["ALERTS BY HOUR OF DAY",
                  "  Counts alerts per hour (0–23).",
                  "  Uniform distribution: detector alerts randomly (suspicious).",
                  "  Peaked distribution: matches domain patterns (good).",
                  "  Single dominant hour: either a real incident or a detector bug.",
                  "",
                  "ALERTS BY MONTH",
                  "  Counts alerts per calendar month in the test window.",
                  "  Stable counts: reliable detector.",
                  "  Bursty counts: sensitive to a factor unrelated to anomalies.",
                  "  Trending counts: drift, retrain the detector.",
                  "",
                  "MONTHLY STABILITY",
                  "  Std of monthly alert counts divided by their mean.",
                  "  < 0.5 stable, 0.5–1.0 acceptable, > 1.0 unstable.",
                  "",
                  "EPOCHS USED (autoencoder only)",
                  "  How many training epochs ran before early stopping.",
                  "  Should be less than the maximum (300). If it equals the max,",
                  "  increase the max or check that loss is decreasing.",
                  "",
                  "RECONSTRUCTION ERROR MEAN / STD (autoencoder only)",
                  "  Distribution of |input − reconstruction| on the training set.",
                  "  Smaller and tighter means the autoencoder compresses normal data well."])

        # per source / feature / detector
        for source, features in metrics.items():
            if not features:
                _section(pdf, f"Source: {source}",
                         ["No metrics produced for this source.",
                          "Possible causes:",
                          "  - training window too narrow",
                          "  - feature columns missing from the unified dataset",
                          "  - fewer than MIN_TRAIN_ROWS rows available",
                          "Fix and re-run scripts.02_train."])
                continue

            for feature, detectors in features.items():
                for det_name, m in detectors.items():
                    title = f"{source} / {feature} / {det_name}"
                    lines = [
                        f"ALERT RATE         : {m.get('alert_rate', 'n/a')}",
                        f"SCORE MEAN         : {m.get('score_mean', 'n/a')}",
                        f"SCORE STD          : {m.get('score_std', 'n/a')}",
                        f"SEPARATION RATIO   : {m.get('separation_ratio', 'n/a')}",
                        f"MONTHLY STABILITY  : {m.get('monthly_stability', 'n/a')}",
                        "",
                    ]
                    if "precision" in m:
                        lines += [
                            f"PRECISION          : {m['precision']}",
                            f"RECALL             : {m['recall']}",
                            f"F1                 : {m['f1']}",
                            f"AUC                : {m.get('auc', 'n/a')}",
                            f"CONFUSION MATRIX   : {m['confusion_matrix']}",
                            "",
                        ]
                    if "model" in m:
                        lines.append("MODEL INTERNALS")
                        for k, v in m["model"].items():
                            lines.append(f"  {k:15s}= {v}")
                    _section(pdf, f"Metrics — {title}", lines)

                    _histogram_page(pdf, "Score histogram", source,
                                    feature, det_name, m)
                    _hour_dist_page(pdf, "Alerts by hour of day", source,
                                    feature, det_name, m)
                    _month_dist_page(pdf, "Alerts by month", source,
                                     feature, det_name, m)

    print(f"Wrote → {out}")


if __name__ == "__main__":
    main()