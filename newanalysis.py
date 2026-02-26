"""
================================================================================
  INCIDENT INTELLIGENCE SYSTEM — analysis.py
  Production-Ready Prototype for TasksMind (AI Engineer for On-Call)
  Data Source: Loghub HDFS (https://github.com/logpai/loghub) [ISSRE'23]
================================================================================

  BUSINESS VALUE:
  ───────────────
  On-call engineers are drowning in noise. A single infrastructure incident can
  generate thousands of log lines, yet the *root cause* is usually just one or
  two unique failure modes buried in the flood.

  This system delivers three measurable wins:
    1. NOISE REDUCTION  — Clusters thousands of raw logs into a handful of
       meaningful groups. Instead of reading 5,000 lines, the engineer reads 5
       cluster summaries. Estimated time savings: 80-90% per incident.
    2. ANOMALY SURFACING — Isolation Forest flags the statistically unusual log
       patterns that are most likely the *cause* rather than the *symptom*.
       Engineers stop chasing cascading effects and go straight to the source.
    3. AGENT-READY OUTPUT — The structured JSON report is designed to be consumed
       by a downstream AI agent (e.g., an LLM-based remediation bot) that can
       autonomously open tickets, suggest runbooks, or trigger auto-healing.

  ARCHITECTURE:
  ─────────────
  Loghub Download -> Raw Log Parsing -> Regex Preprocessing -> TF-IDF
    -> K-Means Clustering -> Isolation Forest Anomaly Detection
    -> JSON Incident Report + Dashboard

  DATA CITATION (required by Loghub license):
  ────────────────────────────────────────────
  Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.
  "Loghub: A Large Collection of System Log Datasets for AI-driven
  Log Analytics." IEEE ISSRE, 2023. https://github.com/logpai/loghub

  Author:  TasksMind AI Engineering Team
  License: MIT
================================================================================
"""

from __future__ import annotations

# IMPORTANT: Set non-interactive backend BEFORE any other matplotlib imports
import matplotlib
matplotlib.use("Agg")

import json
import re
import sys
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
from urllib.error import URLError

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Output directory: same folder as this script (works on Windows/Mac/Linux)
OUTPUT_DIR = Path(__file__).resolve().parent

# ── Loghub HDFS dataset URL ─────────────────────────────────────────────────
LOGHUB_HDFS_2K_URL = (
    "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"
)


# =============================================================================
# 1. DATA LOADING — Download real Loghub HDFS logs
# =============================================================================

def _parse_hdfs_line(line: str) -> dict[str, Any] | None:
    """
    Parse a single line from the Loghub HDFS_2k.log dataset.

    The HDFS log format is:
        <Date> <Time> <Pid> <Level> <Component>: <Content>

    Example:
        081109 203518 148 INFO dfs.DataNode$PacketResponder: ...
    """
    pattern = re.compile(
        r"^(\d{6})\s+(\d{6})\s+(\d+)\s+(INFO|WARN|ERROR|FATAL)\s+"
        r"([\w.$]+):\s+(.*)$"
    )
    match = pattern.match(line.strip())
    if not match:
        return None

    date_str, time_str, pid, level, component, content = match.groups()

    try:
        ts = datetime.strptime(f"20{date_str} {time_str}", "%Y%m%d %H%M%S")
    except ValueError:
        ts = datetime(2008, 11, 9, 0, 0, 0)

    return {
        "timestamp": ts,
        "level": level,
        "component": component,
        "message": content,
        "raw": line.strip(),
    }


def download_loghub_hdfs(cache_path: Path | None = None) -> list[dict[str, Any]]:
    """
    Download and parse the Loghub HDFS_2k.log dataset (2,000 real HDFS logs).

    The file is cached locally after the first download so subsequent runs
    don't hit the network. Exits with a clear error if download fails.
    """
    if cache_path is None:
        cache_path = OUTPUT_DIR / "HDFS_2k.log"

    if cache_path.exists() and cache_path.stat().st_size > 1000:
        print(f"      Using cached dataset: {cache_path}")
    else:
        print(f"      Downloading from Loghub...")
        print(f"      URL: {LOGHUB_HDFS_2K_URL}")
        try:
            urlretrieve(LOGHUB_HDFS_2K_URL, str(cache_path))
            print(f"      Saved to: {cache_path}")
        except (URLError, OSError, Exception) as e:
            print(f"\n      [ERROR] Download failed: {e}")
            print(f"      Please check your internet connection and try again.")
            print(f"      Or manually download the file from:")
            print(f"        {LOGHUB_HDFS_2K_URL}")
            print(f"      and place it at:")
            print(f"        {cache_path}")
            sys.exit(1)

    logs: list[dict[str, Any]] = []
    with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_hdfs_line(line)
            if parsed:
                logs.append(parsed)

    if len(logs) < 100:
        print(f"\n      [ERROR] Only {len(logs)} logs parsed from {cache_path}.")
        print(f"      The file may be corrupted. Delete it and re-run.")
        sys.exit(1)

    return logs


# =============================================================================
# 2. PREPROCESSING PIPELINE
# =============================================================================

_RE_TIMESTAMP    = re.compile(r"^\d{6}\s+\d{6}\s+\d+\s+")
_RE_HEX_BLOCK    = re.compile(r"blk_[-]?\d{10,}")
_RE_IP_PORT      = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?")
_RE_PURE_NUMBERS = re.compile(r"(?<![a-zA-Z])\d{4,}(?![a-zA-Z])")
_RE_FILE_PATHS   = re.compile(r"/[\w\-./]+")
_RE_MULTI_SPACE  = re.compile(r"\s+")


def preprocess_log(raw_message: str) -> str:
    """
    Clean a raw log message for NLP/ML processing.

    Transformations:
      1. Strip timestamps & PIDs   -> removes temporal/process noise
      2. Normalize block IDs       -> "blk_12345..." -> "BLK_ID"
      3. Normalize IP addresses    -> "10.x.x.x:port" -> "IP_ADDR"
      4. Normalize long numbers    -> large integers -> "NUM"
      5. Normalize file paths      -> "/mnt/hadoop/..." -> "FILEPATH"
      6. Collapse whitespace & lowercase
    """
    text = _RE_TIMESTAMP.sub("", raw_message)
    text = _RE_HEX_BLOCK.sub("BLK_ID", text)
    text = _RE_IP_PORT.sub("IP_ADDR", text)
    text = _RE_PURE_NUMBERS.sub("NUM", text)
    text = _RE_FILE_PATHS.sub("FILEPATH", text)
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    return text.lower()


# =============================================================================
# 3. VECTORIZATION & CLUSTERING
# =============================================================================

def vectorize_logs(cleaned_logs: list[str]) -> tuple[np.ndarray, TfidfVectorizer]:
    """Convert cleaned log strings to TF-IDF feature vectors."""
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned_logs)
    return tfidf_matrix.toarray(), vectorizer


def cluster_logs(
    tfidf_matrix: np.ndarray, n_clusters: int = 8, seed: int = 42,
) -> tuple[np.ndarray, KMeans]:
    """Group similar logs using K-Means clustering."""
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    labels = model.fit_predict(tfidf_matrix)
    return labels, model


# =============================================================================
# 4. ANOMALY DETECTION
# =============================================================================

def detect_anomalies(
    tfidf_matrix: np.ndarray, contamination: float = 0.08, seed: int = 42,
) -> np.ndarray:
    """Flag anomalous log entries using Isolation Forest."""
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=seed, n_jobs=-1,
    )
    predictions = model.fit_predict(tfidf_matrix)
    return predictions == -1


# =============================================================================
# 5. AGENT-READY INCIDENT REPORT (JSON)
# =============================================================================

def _suggest_action(keywords: list[str], severity: str) -> str:
    """Map cluster keywords to actionable remediation suggestions."""
    kw = " ".join(keywords).lower()
    # Check most specific patterns first, then broader ones
    if "slow" in kw or "blockreceiver" in kw or "cost" in kw:
        return "Slow I/O detected. Profile disk write latency, check for degraded drives or noisy neighbors."
    elif "runtime" in kw or ("thread" in kw and "exception" in kw):
        return "Runtime exception in monitor thread. Check ReplicationMonitor and DataNode health."
    elif "writeblock" in kw or ("exception" in kw and "io" in kw):
        return "Write pipeline error. Check DataNode logs for disk I/O errors and network resets."
    elif "disk" in kw or "space" in kw:
        return "URGENT: Check disk utilization on DataNodes. Run `df -h` and clear temp blocks."
    elif "refused" in kw or "connection" in kw:
        return "Check if target DataNode is running. Verify firewall rules and port availability."
    elif "timeout" in kw or "sockettimeout" in kw:
        return "Network latency issue. Check NIC errors, switch health, inter-rack bandwidth."
    elif "exception" in kw and "transfer" in kw:
        return "Block transfer exception. Check network between DataNodes and block integrity."
    elif "replicate" in kw or "replica" in kw:
        return "Under-replication detected. Verify cluster has sufficient healthy DataNodes."
    elif "delet" in kw:
        return "Block deletion activity. Verify expected cleanup vs. data loss from failed node."
    elif "allocate" in kw:
        return "High block allocation rate. Check write-heavy jobs and NameNode heap usage."
    elif "packetresponder" in kw or "terminat" in kw:
        return "PacketResponder terminations. Check for slow/failing DataNodes in write pipeline."
    elif "stored" in kw or "blockmap" in kw or "updated" in kw:
        return "Block metadata updates. Check NameNode heap if this cluster is unusually large."
    elif "served" in kw or "receiving" in kw or "dataxceiver" in kw:
        return "Read/write traffic cluster. Monitor for latency spikes or connection drops."
    else:
        return f"Investigate keywords: {', '.join(keywords[:4])}. Correlate with recent changes."


def _classify_severity(
    level_counts: dict, anomaly_ratio: float, total_logs: int,
    dataset_has_errors: bool,
) -> str:
    """
    Adaptive severity classification that works with ANY log distribution.

    When the dataset contains ERROR logs (e.g., HDFS_v1):
      - CRITICAL = high ERROR ratio + high anomaly ratio
      - HIGH     = moderate ERROR ratio or high anomaly ratio

    When the dataset has NO ERROR logs (e.g., HDFS_v2, some HDFS_v1 variants):
      - WARN becomes the highest-severity signal
      - CRITICAL = high WARN ratio + high anomaly ratio
      - HIGH     = moderate WARN ratio or high anomaly ratio

    This makes the system robust to whichever Loghub HDFS version is used.
    """
    err_r = level_counts.get("ERROR", 0) / total_logs if total_logs > 0 else 0
    wrn_r = level_counts.get("WARN", 0) / total_logs if total_logs > 0 else 0

    if dataset_has_errors:
        # Standard mode: ERROR logs are the primary severity signal
        if err_r > 0.5 and anomaly_ratio > 0.2:
            return "CRITICAL"
        elif err_r > 0.3 or anomaly_ratio > 0.15:
            return "HIGH"
        elif wrn_r > 0.3 or anomaly_ratio > 0.08:
            return "MEDIUM"
        else:
            return "LOW"
    else:
        # Adaptive mode: WARN logs are the highest-severity signal available
        if wrn_r > 0.5 and anomaly_ratio > 0.2:
            return "CRITICAL"
        elif wrn_r > 0.2 or anomaly_ratio > 0.15:
            return "HIGH"
        elif wrn_r > 0.05 or anomaly_ratio > 0.05:
            return "MEDIUM"
        else:
            return "LOW"


def _get_cluster_summary(
    logs, cluster_labels, anomaly_flags, vectorizer, tfidf_matrix, kmeans_model,
) -> list[dict[str, Any]]:
    """Build a summary for each cluster with top keywords and severity."""
    feature_names = vectorizer.get_feature_names_out()
    summaries = []

    # Detect whether this dataset has any ERROR logs at all
    global_levels = Counter(log["level"] for log in logs)
    dataset_has_errors = global_levels.get("ERROR", 0) > 0

    for cid in range(kmeans_model.n_clusters):
        mask = cluster_labels == cid
        c_logs = [logs[i] for i in range(len(logs)) if mask[i]]
        c_anom = anomaly_flags[mask]
        if not c_logs:
            continue

        centroid = kmeans_model.cluster_centers_[cid]
        top_idx = centroid.argsort()[-8:][::-1]
        top_kw = [feature_names[i] for i in top_idx if centroid[i] > 0]

        lc = Counter(log["level"] for log in c_logs)
        ts_list = [log["timestamp"] for log in c_logs]
        anom_r = float(c_anom.sum()) / len(c_logs)

        sev = _classify_severity(lc, anom_r, len(c_logs), dataset_has_errors)

        summaries.append({
            "cluster_id": int(cid), "log_count": len(c_logs), "severity": sev,
            "anomaly_count": int(c_anom.sum()), "anomaly_ratio": round(anom_r, 3),
            "level_distribution": dict(lc), "top_keywords": list(top_kw[:6]),
            "time_window": {"start": min(ts_list).isoformat(), "end": max(ts_list).isoformat()},
            "sample_log": c_logs[0]["raw"][:250],
        })

    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    summaries.sort(key=lambda x: sev_order.get(x["severity"], 99))
    return summaries


def generate_incident_report(
    logs, cluster_labels, anomaly_flags, vectorizer, tfidf_matrix, kmeans_model,
) -> dict[str, Any]:
    """Generate a structured JSON incident report (agent-ready output)."""
    summaries = _get_cluster_summary(
        logs, cluster_labels, anomaly_flags, vectorizer, tfidf_matrix, kmeans_model
    )
    total_anom = int(anomaly_flags.sum())
    lc = Counter(log["level"] for log in logs)
    ts_list = [log["timestamp"] for log in logs]

    root_causes = [
        {
            "cause_id": i + 1, "cluster_id": c["cluster_id"],
            "description": (
                f"Cluster of {c['log_count']} logs dominated by "
                f"keywords: {', '.join(c['top_keywords'][:4])}. "
                f"{c['anomaly_count']} anomalies detected."
            ),
            "severity": c["severity"],
            "suggested_action": _suggest_action(c["top_keywords"], c["severity"]),
        }
        for i, c in enumerate(summaries) if c["severity"] in ("CRITICAL", "HIGH")
    ]

    return {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "system": "TasksMind Incident Intelligence v1.0",
            "pipeline": "TF-IDF -> K-Means -> Isolation Forest",
            "data_source": "Loghub HDFS_2k (https://github.com/logpai/loghub)",
            "total_logs_analyzed": len(logs),
            "citation": (
                "Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. "
                "Loghub: A Large Collection of System Log Datasets for AI-driven "
                "Log Analytics. IEEE ISSRE, 2023."
            ),
        },
        "executive_summary": {
            "total_logs": len(logs), "total_anomalies": total_anom,
            "anomaly_rate": round(total_anom / len(logs), 3),
            "unique_clusters": len(summaries),
            "root_causes_identified": len(root_causes),
            "overall_severity": root_causes[0]["severity"] if root_causes else "LOW",
            "time_range": {"start": min(ts_list).isoformat(), "end": max(ts_list).isoformat()},
            "level_distribution": dict(lc),
        },
        "root_causes": root_causes,
        "cluster_details": summaries,
        "recommended_next_steps": [
            "Investigate CRITICAL/HIGH clusters immediately.",
            "Cross-reference top keywords with known runbooks.",
            "Check if WARN clusters (slow I/O, replication) precede larger failures.",
            "Review block replication status for clusters with transfer errors.",
            "After resolution, feed this report back to retrain the anomaly model.",
        ],
    }


# =============================================================================
# 6. VISUALIZATION DASHBOARD
# =============================================================================

def create_dashboard(
    logs, cluster_labels, anomaly_flags, tfidf_matrix,
    output_path: str = "incident_dashboard.png",
) -> str:
    """Create a 2x2 Matplotlib dashboard with timeline, donut, PCA, and bar."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    fig.suptitle(
        "TasksMind Incident Intelligence Dashboard  |  Data: Loghub HDFS",
        fontsize=17, fontweight="bold", color="#1a1a2e", y=0.98,
    )
    fig.patch.set_facecolor("#fafbfc")

    C = {
        "primary": "#2563eb", "danger": "#dc2626", "warning": "#f59e0b",
        "success": "#10b981", "bg": "#f8fafc",
    }
    cluster_palette = plt.cm.Set2(np.linspace(0, 1, len(set(cluster_labels))))

    # ── Panel 1: Timeline ───────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor(C["bg"])
    timestamps = [log["timestamp"] for log in logs]
    min_ts, max_ts = min(timestamps), max(timestamps)
    total_sec = (max_ts - min_ts).total_seconds()
    bin_min = 1 if total_sec < 7200 else (5 if total_sec < 86400 else 30)

    edges = []
    cur = min_ts.replace(second=0, microsecond=0)
    while cur <= max_ts + timedelta(minutes=bin_min):
        edges.append(cur)
        cur += timedelta(minutes=bin_min)

    all_c = np.zeros(len(edges) - 1)
    anom_c = np.zeros(len(edges) - 1)
    for i, log in enumerate(logs):
        for b in range(len(edges) - 1):
            if edges[b] <= log["timestamp"] < edges[b + 1]:
                all_c[b] += 1
                if anomaly_flags[i]:
                    anom_c[b] += 1
                break

    centers = [edges[i] + (edges[i+1] - edges[i]) / 2 for i in range(len(edges) - 1)]
    ax1.fill_between(centers, all_c, alpha=0.3, color=C["primary"], label="All Logs")
    ax1.plot(centers, all_c, color=C["primary"], linewidth=1.5)

    am = anom_c > 0
    if any(am):
        ax1.bar([c for c, m in zip(centers, am) if m], anom_c[am],
                width=timedelta(minutes=max(1, bin_min - 1)),
                color=C["danger"], alpha=0.7, label="Anomalies", zorder=3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.set_xlabel("Time", fontsize=10)
    ax1.set_ylabel(f"Log Count ({bin_min}-min bins)", fontsize=10)
    ax1.set_title("Log Frequency & Anomaly Timeline", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)

    # ── Panel 2: Donut ──────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor(C["bg"])
    lc = Counter(log["level"] for log in logs)
    labels_p, sizes = list(lc.keys()), list(lc.values())
    pcols = [C["success"] if l == "INFO" else C["warning"] if l == "WARN" else C["danger"] for l in labels_p]
    w, t, at = ax2.pie(sizes, labels=labels_p, colors=pcols, autopct="%1.1f%%",
                        startangle=90, pctdistance=0.75,
                        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))
    for a in at:
        a.set_fontsize(11); a.set_fontweight("bold")
    ax2.set_title("Log Level Distribution", fontsize=13, fontweight="bold", pad=12)
    ax2.text(0, 0, f"{len(logs)}\nTotal", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#1a1a2e")

    # ── Panel 3: PCA Scatter ────────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor(C["bg"])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix)
    for cid in sorted(set(cluster_labels)):
        m = (cluster_labels == cid) & (~anomaly_flags)
        if m.any():
            ax3.scatter(coords[m, 0], coords[m, 1], c=[cluster_palette[cid]],
                        s=15, alpha=0.5, label=f"Cluster {cid}" if cid < 5 else None)
    if anomaly_flags.any():
        ax3.scatter(coords[anomaly_flags, 0], coords[anomaly_flags, 1],
                    c=C["danger"], s=40, alpha=0.8, marker="x", linewidths=1.5,
                    label="Anomaly", zorder=5)
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=10)
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=10)
    ax3.set_title("Log Clusters (PCA Projection)", fontsize=13, fontweight="bold", pad=12)
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax3.grid(alpha=0.2)

    # ── Panel 4: Anomaly Bar ────────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor(C["bg"])
    cids = sorted(set(cluster_labels))
    csz = [int(np.sum(cluster_labels == c)) for c in cids]
    can = [int(np.sum(anomaly_flags[cluster_labels == c])) for c in cids]
    ar = [a / s if s > 0 else 0 for a, s in zip(can, csz)]
    bcols = [C["danger"] if r > 0.25 else C["warning"] if r > 0.10 else C["success"] for r in ar]
    bars = ax4.bar([f"C{c}" for c in cids], ar, color=bcols, edgecolor="white", linewidth=1.5)
    for bar, cnt, tot in zip(bars, can, csz):
        if cnt > 0:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{cnt}/{tot}", ha="center", va="bottom", fontsize=8,
                     fontweight="bold", color="#374151")
    ax4.set_xlabel("Cluster ID", fontsize=10)
    ax4.set_ylabel("Anomaly Ratio", fontsize=10)
    ax4.set_title("Anomaly Density by Cluster", fontsize=13, fontweight="bold", pad=12)
    ax4.set_ylim(0, max(ar) * 1.3 + 0.05 if ar else 0.5)
    ax4.axhline(y=0.10, color=C["warning"], linestyle="--", alpha=0.5, label="Warning")
    ax4.axhline(y=0.25, color=C["danger"], linestyle="--", alpha=0.5, label="Critical")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


# =============================================================================
# 7. INTERACTIVE HTML DASHBOARD GENERATOR
# =============================================================================

def _build_viz_data(
    logs, cluster_labels, anomaly_flags, tfidf_matrix,
) -> dict[str, Any]:
    """
    Build all visualization data needed for the interactive HTML dashboard.
    This avoids needing matplotlib — everything is rendered client-side with Chart.js.
    """
    timestamps = [log["timestamp"] for log in logs]
    min_ts, max_ts = min(timestamps), max(timestamps)
    total_sec = (max_ts - min_ts).total_seconds()
    bin_min = 1 if total_sec < 7200 else (5 if total_sec < 86400 else 30)

    # ── Timeline bins ───────────────────────────────────────────────────────
    edges = []
    cur = min_ts.replace(second=0, microsecond=0)
    while cur <= max_ts + timedelta(minutes=bin_min):
        edges.append(cur)
        cur += timedelta(minutes=bin_min)

    all_c = [0] * (len(edges) - 1)
    anom_c = [0] * (len(edges) - 1)
    for i, log in enumerate(logs):
        for b in range(len(edges) - 1):
            if edges[b] <= log["timestamp"] < edges[b + 1]:
                all_c[b] += 1
                if anomaly_flags[i]:
                    anom_c[b] += 1
                break

    timeline_labels = [
        (edges[i] + (edges[i+1] - edges[i]) / 2).strftime("%H:%M")
        for i in range(len(edges) - 1)
    ]

    # ── PCA coordinates ─────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(tfidf_matrix)

    pca_points = []
    for i in range(len(logs)):
        pca_points.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster": int(cluster_labels[i]),
            "anomaly": bool(anomaly_flags[i]),
            "level": logs[i]["level"],
        })

    # ── Cluster bar data ────────────────────────────────────────────────────
    cids = sorted(set(cluster_labels))
    cluster_bars = []
    for c in cids:
        sz = int(np.sum(cluster_labels == c))
        an = int(np.sum(anomaly_flags[cluster_labels == c]))
        cluster_bars.append({
            "id": int(c), "size": sz, "anomalies": an,
            "ratio": round(an / sz, 3) if sz > 0 else 0,
        })

    # ── Level distribution ──────────────────────────────────────────────────
    lc = Counter(log["level"] for log in logs)

    return {
        "timeline": {
            "labels": timeline_labels,
            "all_counts": all_c,
            "anomaly_counts": anom_c,
            "bin_minutes": bin_min,
        },
        "level_distribution": dict(lc),
        "pca": {
            "points": pca_points,
            "variance": [
                round(float(pca.explained_variance_ratio_[0]) * 100, 1),
                round(float(pca.explained_variance_ratio_[1]) * 100, 1),
            ],
        },
        "cluster_bars": cluster_bars,
    }


def generate_html_dashboard(report: dict, viz_data: dict, output_path: str) -> str:
    """Generate a self-contained interactive HTML dashboard (Linear/Vercel style)."""

    report_json = json.dumps(report, default=str)
    viz_json = json.dumps(viz_data, default=str)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Incident Intelligence — TasksMind</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #ffffff;
    --bg-subtle: #fafafa;
    --bg-muted: #f4f4f5;
    --border: #e4e4e7;
    --border-hover: #d4d4d8;
    --text: #18181b;
    --text-secondary: #52525b;
    --text-muted: #a1a1aa;
    --accent: #18181b;
    --red: #e11d48;
    --red-bg: #fff1f2;
    --red-border: #fecdd3;
    --orange: #ea580c;
    --orange-bg: #fff7ed;
    --orange-border: #fed7aa;
    --green: #16a34a;
    --green-bg: #f0fdf4;
    --blue: #2563eb;
    --blue-bg: #eff6ff;
    --mono: 'DM Mono', 'SF Mono', 'Consolas', monospace;
    --sans: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --radius: 8px;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: var(--sans); background: var(--bg); color: var(--text); -webkit-font-smoothing: antialiased; }}

  .page {{ max-width: 1120px; margin: 0 auto; padding: 48px 24px 64px; }}

  /* ── Nav ──────────────────────────────────────── */
  nav {{
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: 48px; border-bottom: 1px solid var(--border); margin-bottom: 40px;
  }}
  nav .left {{ display: flex; align-items: center; gap: 12px; }}
  nav .logo {{
    width: 28px; height: 28px; background: var(--accent); border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 700; font-size: 13px; font-family: var(--mono);
  }}
  nav .wordmark {{ font-size: 15px; font-weight: 600; letter-spacing: -0.3px; }}
  nav .meta {{
    font-size: 12px; color: var(--text-muted); font-family: var(--mono);
  }}

  /* ── Section titles ──────────────────────────── */
  .section-label {{
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 16px;
  }}
  h2 {{
    font-size: 22px; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 6px;
  }}
  .page-desc {{
    font-size: 14px; color: var(--text-secondary); line-height: 1.5; margin-bottom: 32px;
  }}

  /* ── KPI strip ───────────────────────────────── */
  .kpi-strip {{
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px;
    background: var(--border); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden; margin-bottom: 40px;
  }}
  .kpi-cell {{
    background: var(--bg); padding: 20px 16px; text-align: center;
  }}
  .kpi-cell .val {{
    font-family: var(--mono); font-size: 24px; font-weight: 500;
    letter-spacing: -0.5px; margin-bottom: 4px;
  }}
  .kpi-cell .lbl {{ font-size: 11px; color: var(--text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-cell .val.sev {{
    display: inline-block; font-size: 12px; font-family: var(--mono); font-weight: 600;
    padding: 3px 10px; border-radius: 4px; letter-spacing: 0.3px;
  }}
  .sev-CRITICAL {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red-border); }}
  .sev-HIGH {{ background: var(--orange-bg); color: var(--orange); border: 1px solid var(--orange-border); }}
  .sev-MEDIUM {{ background: var(--blue-bg); color: var(--blue); }}
  .sev-LOW {{ background: var(--green-bg); color: var(--green); }}

  /* ── Root cause cards ────────────────────────── */
  .causes {{ display: flex; flex-direction: column; gap: 12px; margin-bottom: 48px; }}
  .cause {{
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 20px 24px; transition: border-color 0.15s;
  }}
  .cause:hover {{ border-color: var(--border-hover); }}
  .cause-top {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .cause-sev {{
    font-size: 10px; font-family: var(--mono); font-weight: 600;
    padding: 2px 8px; border-radius: 4px; letter-spacing: 0.3px;
  }}
  .cause-id {{ font-size: 13px; color: var(--text-muted); font-family: var(--mono); }}
  .cause-desc {{ font-size: 14px; color: var(--text-secondary); line-height: 1.6; margin-bottom: 12px; }}
  .cause-tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }}
  .cause-tags span {{
    font-size: 11px; font-family: var(--mono); color: var(--text-secondary);
    background: var(--bg-muted); padding: 3px 8px; border-radius: 4px;
  }}
  .cause-action {{
    font-size: 13px; font-family: var(--mono); color: var(--text-secondary);
    background: var(--bg-subtle); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px 16px; line-height: 1.5;
  }}

  /* ── Chart grid ──────────────────────────────── */
  .chart-grid {{ display: grid; grid-template-columns: 5fr 3fr; gap: 24px; margin-bottom: 24px; }}
  .chart-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }}
  @media (max-width: 800px) {{ .chart-grid, .chart-pair {{ grid-template-columns: 1fr; }} }}

  .card {{
    border: 1px solid var(--border); border-radius: var(--radius); padding: 24px;
  }}
  .card-title {{
    font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  .card-title .hint {{ font-size: 11px; color: var(--text-muted); font-weight: 400; font-family: var(--mono); }}
  .chart-box {{ position: relative; }}
  .chart-box.h280 {{ height: 260px; }}
  .chart-box.h300 {{ height: 300px; }}
  .chart-box.h240 {{ height: 230px; }}

  /* ── Table ────────────────────────────────────── */
  .tbl-wrap {{ overflow-x: auto; margin-bottom: 48px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{
    text-align: left; font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-muted); padding: 10px 12px;
    border-bottom: 1px solid var(--border); white-space: nowrap;
  }}
  td {{
    padding: 12px; border-bottom: 1px solid var(--bg-muted);
    color: var(--text-secondary); vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--bg-subtle); }}
  .mono {{ font-family: var(--mono); }}
  .dot {{
    width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px;
    vertical-align: middle;
  }}
  .dot-CRITICAL {{ background: var(--red); }}
  .dot-HIGH {{ background: var(--orange); }}
  .dot-MEDIUM {{ background: var(--blue); }}
  .dot-LOW {{ background: #d4d4d8; }}
  .bar-track {{
    width: 64px; height: 4px; background: var(--bg-muted); border-radius: 2px;
    display: inline-block; vertical-align: middle; margin-left: 6px;
  }}
  .bar-fill {{ height: 100%; border-radius: 2px; }}

  /* ── Footer ──────────────────────────────────── */
  footer {{
    padding-top: 32px; border-top: 1px solid var(--border);
    font-size: 12px; color: var(--text-muted); display: flex;
    justify-content: space-between; flex-wrap: wrap; gap: 8px;
  }}
  footer a {{ color: var(--text-secondary); text-decoration: none; }}
  footer a:hover {{ color: var(--text); }}
</style>
</head>
<body>
<div class="page">

  <nav>
    <div class="left">
      <div class="logo">T</div>
      <span class="wordmark">TasksMind</span>
    </div>
    <span class="meta" id="metaTime"></span>
  </nav>

  <p class="section-label">Incident Intelligence</p>
  <h2>Anomaly Report</h2>
  <p class="page-desc" id="pageDesc"></p>

  <div class="kpi-strip" id="kpiStrip"></div>

  <p class="section-label">Root Causes</p>
  <div class="causes" id="causes"></div>

  <p class="section-label">Visualizations</p>

  <div class="chart-grid">
    <div class="card">
      <div class="card-title">Log frequency<span class="hint" id="binHint"></span></div>
      <div class="chart-box h280"><canvas id="timelineChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Level breakdown</div>
      <div class="chart-box h280"><canvas id="donutChart"></canvas></div>
    </div>
  </div>

  <div class="chart-pair">
    <div class="card">
      <div class="card-title">Cluster map<span class="hint" id="pcaHint"></span></div>
      <div class="chart-box h300"><canvas id="scatterChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Anomaly density</div>
      <div class="chart-box h300"><canvas id="barChart"></canvas></div>
    </div>
  </div>

  <p class="section-label">All Clusters</p>
  <div class="card tbl-wrap">
    <table id="clusterTable"></table>
  </div>

  <footer>
    <span>TasksMind Incident Intelligence v1.0 &mdash; TF-IDF &rarr; K-Means &rarr; Isolation Forest</span>
    <a href="https://github.com/logpai/loghub" target="_blank">Data: Loghub HDFS (ISSRE 2023)</a>
  </footer>

</div>

<script>
const R = {report_json};
const V = {viz_json};
const S = R.executive_summary;

document.getElementById('metaTime').textContent = new Date(R.report_metadata.generated_at).toLocaleDateString('en-US', {{year:'numeric',month:'short',day:'numeric'}}) + ' ' + new Date(R.report_metadata.generated_at).toLocaleTimeString('en-US',{{hour:'2-digit',minute:'2-digit'}});
document.getElementById('pageDesc').textContent = S.total_logs.toLocaleString() + ' HDFS logs analyzed. ' + S.total_anomalies + ' anomalies flagged across ' + S.unique_clusters + ' clusters. ' + S.root_causes_identified + ' root cause' + (S.root_causes_identified !== 1 ? 's' : '') + ' identified.';
document.getElementById('binHint').textContent = V.timeline.bin_minutes + '-min bins';
document.getElementById('pcaHint').textContent = 'PCA 2-component';

// KPIs
const strip = document.getElementById('kpiStrip');
[
  [S.total_logs.toLocaleString(), 'Logs'],
  [S.total_anomalies, 'Anomalies'],
  [(S.anomaly_rate*100).toFixed(1)+'%', 'Anomaly rate'],
  [S.unique_clusters, 'Clusters'],
  ['__SEV__', 'Severity'],
].forEach(([v,l]) => {{
  const d = document.createElement('div');
  d.className = 'kpi-cell';
  if (v === '__SEV__') {{
    d.innerHTML = '<div class="val sev sev-'+S.overall_severity+'">'+S.overall_severity+'</div><div class="lbl">'+l+'</div>';
  }} else {{
    d.innerHTML = '<div class="val">'+v+'</div><div class="lbl">'+l+'</div>';
  }}
  strip.appendChild(d);
}});

// Root causes
const cDiv = document.getElementById('causes');
R.root_causes.forEach(rc => {{
  const cl = R.cluster_details.find(c => c.cluster_id === rc.cluster_id);
  const kws = cl ? cl.top_keywords.slice(0,5) : [];
  const sevCls = rc.severity === 'CRITICAL' ? 'sev-CRITICAL' : rc.severity === 'HIGH' ? 'sev-HIGH' : 'sev-MEDIUM';
  const d = document.createElement('div');
  d.className = 'cause';
  d.innerHTML = `
    <div class="cause-top">
      <span class="cause-sev ${{sevCls}}">${{rc.severity}}</span>
      <span class="cause-id">Cluster ${{rc.cluster_id}} &middot; ${{cl ? cl.log_count : '?'}} logs &middot; ${{rc.cause_id === 1 ? '1st' : rc.cause_id === 2 ? '2nd' : rc.cause_id+'th'}} priority</span>
    </div>
    <div class="cause-tags">${{kws.map(k=>'<span>'+k+'</span>').join('')}}</div>
    <div class="cause-desc">${{rc.description}}</div>
    <div class="cause-action">&rarr; ${{rc.suggested_action}}</div>`;
  cDiv.appendChild(d);
}});

// Chart.js config
Chart.defaults.color = '#a1a1aa';
Chart.defaults.borderColor = '#e4e4e7';
Chart.defaults.font.family = "'DM Sans', sans-serif";
Chart.defaults.font.size = 11;
const tip = {{ backgroundColor:'#18181b', titleColor:'#fafafa', bodyColor:'#d4d4d8', borderWidth:0, cornerRadius:6, padding:10, titleFont:{{size:12,weight:'600'}}, bodyFont:{{size:11}} }};

// Colors
const cPal = ['#6366f1','#f59e0b','#ec4899','#10b981','#06b6d4','#8b5cf6','#84cc16','#f97316'];

// Timeline
new Chart(document.getElementById('timelineChart'), {{
  type:'line',
  data: {{
    labels: V.timeline.labels,
    datasets: [
      {{ label:'Logs', data:V.timeline.all_counts, borderColor:'#18181b', backgroundColor:'rgba(24,24,27,0.04)', fill:true, tension:0.35, pointRadius:0, borderWidth:1.5 }},
      {{ label:'Anomalies', data:V.timeline.anomaly_counts, type:'bar', backgroundColor:'rgba(225,29,72,0.65)', borderRadius:2, borderSkipped:false, barPercentage:0.6 }}
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    interaction:{{ mode:'index', intersect:false }},
    plugins:{{ legend:{{ position:'top', align:'end', labels:{{usePointStyle:true, pointStyle:'circle', boxWidth:6, padding:16 }} }}, tooltip:tip }},
    scales:{{
      x:{{ grid:{{display:false}}, ticks:{{maxTicksLimit:12}} }},
      y:{{ grid:{{color:'#f4f4f5'}}, beginAtZero:true }}
    }}
  }}
}});

// Donut
const lL=Object.keys(V.level_distribution), lV=Object.values(V.level_distribution);
const lC=lL.map(l=>l==='INFO'?'#18181b':l==='WARN'?'#f59e0b':'#e11d48');
new Chart(document.getElementById('donutChart'), {{
  type:'doughnut',
  data:{{ labels:lL, datasets:[{{ data:lV, backgroundColor:lC, borderColor:'#fff', borderWidth:2.5, hoverOffset:4 }}] }},
  options:{{
    responsive:true, maintainAspectRatio:false, cutout:'72%',
    plugins:{{ legend:{{ position:'bottom', labels:{{usePointStyle:true, padding:12, font:{{size:12}}}} }}, tooltip:tip }}
  }}
}});

// Scatter
const norm={{}}, anom=[];
V.pca.points.forEach(p => {{
  if(p.anomaly) anom.push({{x:p.x,y:p.y}});
  else {{ if(!norm[p.cluster]) norm[p.cluster]=[]; norm[p.cluster].push({{x:p.x,y:p.y}}); }}
}});
const sDS=Object.keys(norm).map(c => ({{
  label:'C'+c, data:norm[c],
  backgroundColor:cPal[c%8]+'44', borderColor:cPal[c%8], borderWidth:1,
  pointRadius:2.5, pointHoverRadius:5
}}));
sDS.push({{ label:'Anomaly', data:anom, backgroundColor:'rgba(225,29,72,0.7)', borderColor:'#e11d48', pointRadius:4, pointStyle:'crossRot', pointHoverRadius:7 }});
new Chart(document.getElementById('scatterChart'), {{
  type:'scatter', data:{{datasets:sDS}},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{position:'top',align:'end',labels:{{usePointStyle:true,boxWidth:6,padding:10,font:{{size:10}}}}}}, tooltip:tip }},
    scales:{{
      x:{{ title:{{display:true,text:'PC1 ('+V.pca.variance[0]+'%)',font:{{size:11}}}}, grid:{{color:'#f4f4f5'}} }},
      y:{{ title:{{display:true,text:'PC2 ('+V.pca.variance[1]+'%)',font:{{size:11}}}}, grid:{{color:'#f4f4f5'}} }}
    }}
  }}
}});

// Bar
const bL=V.cluster_bars.map(c=>'C'+c.id), bV=V.cluster_bars.map(c=>c.ratio);
const bC=bV.map(v=>v>0.25?'#e11d48':v>0.1?'#f59e0b':'#d4d4d8');
new Chart(document.getElementById('barChart'), {{
  type:'bar',
  data:{{ labels:bL, datasets:[{{ data:bV, backgroundColor:bC, borderRadius:4, barPercentage:0.5 }}] }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{display:false}}, tooltip:{{ ...tip, callbacks:{{ label:ctx=>{{ const c=V.cluster_bars[ctx.dataIndex]; return c.anomalies+'/'+c.size+' ('+(c.ratio*100).toFixed(1)+'%)'; }} }} }} }},
    scales:{{
      x:{{ grid:{{display:false}} }},
      y:{{ grid:{{color:'#f4f4f5'}}, ticks:{{callback:v=>(v*100).toFixed(0)+'%'}} }}
    }}
  }}
}});

// Table
const tbl=document.getElementById('clusterTable');
let h='<thead><tr><th>ID</th><th>Severity</th><th>Logs</th><th>Anomalies</th><th>Rate</th><th>Levels</th><th>Keywords</th></tr></thead><tbody>';
R.cluster_details.forEach(c => {{
  const pct=(c.anomaly_ratio*100).toFixed(1);
  const lvl=Object.entries(c.level_distribution).map(([k,v])=>k+' '+v).join(', ');
  const kw=c.top_keywords.slice(0,3).join(', ');
  const bw=Math.min(c.anomaly_ratio/0.6*100,100);
  const bc=c.anomaly_ratio>0.25?'var(--red)':c.anomaly_ratio>0.1?'var(--orange)':'#d4d4d8';
  h+=`<tr>
    <td class="mono" style="font-weight:500">C${{c.cluster_id}}</td>
    <td><span class="dot dot-${{c.severity}}"></span>${{c.severity}}</td>
    <td>${{c.log_count}}</td>
    <td>${{c.anomaly_count}}</td>
    <td class="mono">${{pct}}%<div class="bar-track"><div class="bar-fill" style="width:${{bw}}%;background:${{bc}}"></div></div></td>
    <td class="mono" style="font-size:11px">${{lvl}}</td>
    <td style="font-size:12px;color:var(--text-muted)">${{kw}}</td>
  </tr>`;
}});
h+='</tbody>';
tbl.innerHTML=h;
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# =============================================================================
# 8. MAIN ORCHESTRATOR
# =============================================================================

def main() -> None:
    """Execute the full Incident Intelligence pipeline."""
    print("=" * 72)
    print("  TASKSMIND INCIDENT INTELLIGENCE SYSTEM")
    print("  Data: Loghub HDFS (https://github.com/logpai/loghub)")
    print("=" * 72)

    # ── Stage 1 ─────────────────────────────────────────────────────────────
    print("\n[1/7] Loading HDFS log dataset from Loghub...")
    logs = download_loghub_hdfs()
    print(f"      Loaded {len(logs):,} log entries")
    print(f"      Time range: {logs[0]['timestamp']} to {logs[-1]['timestamp']}")
    level_dist = Counter(l["level"] for l in logs)
    for level, count in sorted(level_dist.items()):
        print(f"        {level:>5}: {count:>5} ({count/len(logs):.1%})")

    if "ERROR" not in level_dist:
        print(f"      Note: No ERROR logs in dataset. Using WARN as primary severity signal.")

    # ── Stage 2 ─────────────────────────────────────────────────────────────
    print("\n[2/7] Preprocessing logs (regex normalization)...")
    cleaned = [preprocess_log(log["raw"]) for log in logs]
    print(f"      Cleaned {len(cleaned):,} messages")
    print(f"      Sample: \"{cleaned[0][:90]}...\"")

    # ── Stage 3 ─────────────────────────────────────────────────────────────
    print("\n[3/7] Vectorizing with TF-IDF (500 features, bigrams)...")
    tfidf_matrix, vectorizer = vectorize_logs(cleaned)
    print(f"      Matrix shape: {tfidf_matrix.shape}")
    print(f"      Vocabulary: {len(vectorizer.vocabulary_):,} terms")

    # ── Stage 4 ─────────────────────────────────────────────────────────────
    n_clusters = 8
    print(f"\n[4/7] Clustering with K-Means (k={n_clusters})...")
    cluster_labels, kmeans_model = cluster_logs(tfidf_matrix, n_clusters=n_clusters)
    print(f"      Inertia: {kmeans_model.inertia_:.2f}")
    for cid in range(n_clusters):
        c = int(np.sum(cluster_labels == cid))
        print(f"        Cluster {cid}: {c:>4} logs ({c/len(logs):.1%})")

    # ── Stage 5 ─────────────────────────────────────────────────────────────
    print("\n[5/7] Running Isolation Forest anomaly detection...")
    anomaly_flags = detect_anomalies(tfidf_matrix, contamination=0.08)
    n_anom = int(anomaly_flags.sum())
    print(f"      Detected {n_anom:,} anomalies ({n_anom/len(logs):.1%})")
    print(f"      Breakdown: {dict(Counter(logs[i]['level'] for i in range(len(logs)) if anomaly_flags[i]))}")

    # ── Stage 6 ─────────────────────────────────────────────────────────────
    print("\n[6/7] Generating agent-ready incident report...")
    report = generate_incident_report(
        logs, cluster_labels, anomaly_flags, vectorizer, tfidf_matrix, kmeans_model
    )
    rp = OUTPUT_DIR / "incident_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"      Report saved: {rp}")

    s = report["executive_summary"]
    print(f"\n      {'='*54}")
    print(f"               EXECUTIVE SUMMARY")
    print(f"      {'='*54}")
    print(f"       Total Logs Analyzed:   {s['total_logs']:>6,}")
    print(f"       Anomalies Detected:    {s['total_anomalies']:>6,}  ({s['anomaly_rate']:.1%})")
    print(f"       Unique Clusters:       {s['unique_clusters']:>6}")
    print(f"       Root Causes Found:     {s['root_causes_identified']:>6}")
    print(f"       Overall Severity:      {s['overall_severity']:<10}")
    print(f"      {'='*54}")

    if report["root_causes"]:
        print(f"\n      ROOT CAUSES (prioritized for triage):")
        for rc in report["root_causes"]:
            print(f"        [{rc['severity']}] {rc['description']}")
            print(f"           -> Action: {rc['suggested_action']}")

    # ── Stage 7 ─────────────────────────────────────────────────────────────
    print("\n[7/7] Creating interactive dashboard...")

    # Build visualization data for HTML dashboard
    viz_data = _build_viz_data(logs, cluster_labels, anomaly_flags, tfidf_matrix)

    # Generate interactive HTML dashboard
    html_path = generate_html_dashboard(
        report, viz_data,
        output_path=str(OUTPUT_DIR / "incident_dashboard.html"),
    )
    print(f"      Interactive dashboard saved: {html_path}")

    # Also keep the static PNG dashboard
    dp = create_dashboard(
        logs, cluster_labels, anomaly_flags, tfidf_matrix,
        output_path=str(OUTPUT_DIR / "incident_dashboard.png"),
    )
    print(f"      Static dashboard saved: {dp}")

    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs:")
    print(f"    - incident_report.json       (agent-ready structured data)")
    print(f"    - incident_dashboard.html    (interactive — open in browser)")
    print(f"    - incident_dashboard.png     (static image)")
    print("=" * 72)


if __name__ == "__main__":
    main()
