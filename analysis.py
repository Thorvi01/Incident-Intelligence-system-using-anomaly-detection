"""
================================================================================
  INCIDENT INTELLIGENCE SYSTEM — analysis.py
  Production-Ready Prototype for Real-World Log Analysis & Root Cause Identification
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

  Author:  TDishaa Bornare
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
# 7. MAIN ORCHESTRATOR
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

    # Inform user about adaptive mode
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
    print("\n[7/7] Creating visualization dashboard...")
    dp = create_dashboard(
        logs, cluster_labels, anomaly_flags, tfidf_matrix,
        output_path=str(OUTPUT_DIR / "incident_dashboard.png"),
    )
    print(f"      Dashboard saved: {dp}")

    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs: incident_report.json, incident_dashboard.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
