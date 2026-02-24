# Incident Intelligence System

**Turn thousands of messy server logs into one clear answer: "Here's what broke and how to fix it."**

This is an end-to-end ML pipeline that reads real system logs, finds the anomalies hiding in the noise, groups them into root causes, and hands you a structured report that a human *or* an AI agent can act on.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## The Problem

A single infrastructure incident can generate **thousands of log lines** in minutes. On-call engineers have to:

1. Read through walls of text at 3am
2. Figure out which logs are symptoms vs. the actual cause
3. Decide what to do about it

Most of those logs are noise. The real root cause is usually buried in 1–2% of the data.

## What This Project Does

It automates that entire triage process in **7 steps**:

```
Download real HDFS logs
        ↓
Parse raw log lines (regex)
        ↓
Clean & normalize text (remove IPs, hex codes, paths)
        ↓
Convert text → numbers (TF-IDF vectorization)
        ↓
Group similar logs (K-Means clustering)
        ↓
Flag the unusual ones (Isolation Forest anomaly detection)
        ↓
Output: JSON report + interactive HTML dashboard
```

**The key insight:** Instead of reading 2,000 log lines, you read 2–3 root cause summaries with suggested actions.

---

## Quick Start

### 1. Install dependencies

```bash
pip install scikit-learn matplotlib numpy
```

### 2. Run it

```bash
python analysis.py
```

That's it. The script automatically downloads the dataset on first run.

### 3. Check the outputs

The pipeline creates three files in the same folder:

| File | What it is |
|------|-----------|
| `incident_report.json` | Structured report an AI agent can read and act on |
| `incident_dashboard.html` | Interactive dashboard — open in any browser |
| `incident_dashboard.png` | Static chart image for docs/presentations |

---

## Sample Output

When you run the pipeline, you'll see something like this:

```
[1/7] Loading HDFS log dataset from Loghub...
      Loaded 2,000 log entries
         INFO:  1920 (96.0%)
         WARN:    80 (4.0%)

[5/7] Running Isolation Forest anomaly detection...
      Detected 108 anomalies (5.4%)
      Breakdown: {'INFO': 28, 'WARN': 80}

      ======================================================
               EXECUTIVE SUMMARY
      ======================================================
       Total Logs Analyzed:    2,000
       Anomalies Detected:       108  (5.4%)
       Unique Clusters:            8
       Root Causes Found:          1
       Overall Severity:      HIGH
      ======================================================

      ROOT CAUSES (prioritized for triage):
        [HIGH] Cluster of 187 logs dominated by keywords:
               dataxceiver, blk_id filepath, datanode dataxceiver.
               106 anomalies detected.
           -> Action: Read/write traffic cluster. Monitor for
              latency spikes or connection drops.
```

The system found that 106 out of 108 anomalies belong to **one cluster** — that's your root cause. Instead of reading 2,000 lines, you read one summary.

---

## How Each Step Works

### Step 1 — Data Loading

The pipeline downloads **HDFS_2k.log** from the [Loghub](https://github.com/logpai/loghub) dataset — 2,000 real log lines from a Hadoop Distributed File System. This is a standard benchmark dataset used in log analysis research.

The file is cached locally after the first download, so you only need internet once.

Each raw log line looks like this:
```
081109 203518 148 INFO dfs.DataNode$PacketResponder: Received block blk_3382592750766595 of size 37047819 from /10.235.63.250
```

The parser extracts the timestamp, log level (INFO/WARN/ERROR), component name, and message content.

### Step 2 — Preprocessing (Regex Cleaning)

Raw logs are full of noise that would confuse an ML model. The preprocessing step normalizes everything:

| Before | After |
|--------|-------|
| `blk_3382592750766595` | `BLK_ID` |
| `10.235.63.250:50010` | `IP_ADDR` |
| `/mnt/hadoop/dfs/data/current/blk_...` | `FILEPATH` |
| `37047819` | `NUM` |

**Why this matters:** Without this step, the model would think `blk_123` and `blk_456` are completely different things. They're not — they're both "some block." Normalization lets the model focus on *what happened* rather than *which specific block it happened to*.

### Step 3 — TF-IDF Vectorization

ML models can't read text — they need numbers. TF-IDF (Term Frequency–Inverse Document Frequency) converts each cleaned log message into a vector of 500 numbers.

- **TF** = How often a word appears in *this* log line
- **IDF** = How rare the word is across *all* log lines
- **TF × IDF** = Words that are frequent in this log but rare overall get the highest scores

For example, the word `"exception"` gets a high score because it only appears in error logs, while `"block"` gets a low score because it appears everywhere.

We also use **bigrams** (two-word pairs like `"connection refused"`) because they carry more meaning than individual words.

### Step 4 — K-Means Clustering

K-Means groups the 2,000 log vectors into 8 clusters based on similarity. Logs that say similar things end up in the same cluster.

**Why this matters for on-call:** If 300 logs are all saying "connection timeout" in slightly different ways, K-Means collapses them into one group. You don't need to read 300 lines — you just need to know "there's a connection timeout cluster with 300 entries."

### Step 5 — Isolation Forest (Anomaly Detection)

Isolation Forest finds the logs that *don't fit the normal pattern*. It works by building random decision trees — anomalous points get isolated quickly (in few splits), while normal points take many splits to isolate.

**Key properties:**
- **Unsupervised** — No labeled training data needed
- **Contamination parameter** — We set it to 8%, meaning "flag the top 8% most unusual logs"
- **The output** — A boolean flag for each log: anomaly or normal

### Step 6 — Incident Report (JSON)

The pipeline combines clustering + anomaly detection into a structured JSON report:

```json
{
  "root_causes": [
    {
      "severity": "HIGH",
      "description": "Cluster of 187 logs dominated by keywords: dataxceiver...",
      "suggested_action": "Monitor for latency spikes or connection drops."
    }
  ],
  "cluster_details": [...],
  "recommended_next_steps": [...]
}
```

**Why JSON?** Because a downstream AI agent (like an LLM-based runbook bot) can parse this and take automatic action — open a ticket, page the right team, or trigger auto-remediation.

### Step 7 — Interactive Dashboard

The HTML dashboard visualizes everything with four interactive charts:

- **Timeline** — Log volume over time with anomaly bars highlighted
- **Level breakdown** — Donut chart showing INFO/WARN/ERROR distribution
- **Cluster map** — PCA scatter plot showing how clusters separate in 2D space
- **Anomaly density** — Bar chart showing which clusters have the most anomalies

All rendered client-side with Chart.js. Just open the `.html` file in a browser — no server needed.

---

## Adaptive Severity

The system works with **any** version of the HDFS dataset:

- **If the dataset has ERROR logs:** Errors are the primary severity signal. A cluster with 50%+ errors and high anomaly ratio = CRITICAL.
- **If the dataset only has INFO + WARN:** The system automatically switches to using WARN as the highest severity signal, with adjusted thresholds.

This makes it robust regardless of which Loghub HDFS variant you download.

---

## Project Structure

```
├── analysis.py              # The entire pipeline (single file, one command)
├── HDFS_2k.log              # Auto-downloaded Loghub dataset (cached)
├── incident_report.json     # Generated: structured JSON report
├── incident_dashboard.html  # Generated: interactive browser dashboard
├── incident_dashboard.png   # Generated: static chart image
└── README.md                # This file
```

Everything is in **one Python file** by design. No complex package structure, no config files, no Docker — just `python analysis.py`.

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Language | Python 3.10+ | Industry standard for ML pipelines |
| Text vectorization | TF-IDF (scikit-learn) | Simple, fast, interpretable — no GPU needed |
| Clustering | K-Means (scikit-learn) | Well-understood, deterministic, maps to "root cause groups" |
| Anomaly detection | Isolation Forest (scikit-learn) | Unsupervised, scales well, no labels needed |
| Dimensionality reduction | PCA (scikit-learn) | For 2D visualization of clusters |
| Dashboard | Chart.js + HTML | Self-contained, no server, portfolio-friendly |
| Static charts | Matplotlib | Fallback PNG for docs and presentations |
| Data source | Loghub HDFS | Research benchmark, real production logs |

---

## Why These Algorithms?

**"Why not use a neural network / transformer / deep learning?"**

For this problem, classical ML is the right choice:

1. **TF-IDF + K-Means** is interpretable. You can look at the cluster centroids and immediately see *why* logs were grouped together (top keywords). A neural embedding would be a black box.

2. **Isolation Forest** works without labeled data. In real incidents, you don't have time to label thousands of logs as "normal" vs. "anomaly" before running detection.

3. **Speed matters.** This pipeline processes 2,000 logs in under 2 seconds. An on-call engineer waiting at 3am doesn't want to wait for a GPU to warm up.

4. **Explainability matters.** When the system says "this cluster is CRITICAL," the engineer can look at the keywords and sample logs to verify it makes sense. Trust is everything in incident response.

---

## Business Value

For a startup like TasksMind building AI for on-call teams:

| Metric | Before | After |
|--------|--------|-------|
| Time to identify root cause | 30–60 min reading logs | < 5 seconds |
| Logs an engineer must read | All of them (thousands) | 2–3 cluster summaries |
| Format of output | Raw text in a terminal | Structured JSON an AI agent can act on |
| Skill required | Senior SRE with domain expertise | Any engineer can read the dashboard |

---

## Configuration

You can tune these parameters in `analysis.py`:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `n_clusters` | 8 | Number of log groups. Increase if you have more distinct log types |
| `contamination` | 0.08 | Fraction of logs flagged as anomalies. Lower = fewer, stricter alerts |
| `max_features` | 500 | TF-IDF vocabulary size. Higher = more detail, slower processing |
| `ngram_range` | (1, 2) | Word and bigram features. (1,3) adds trigrams at cost of speed |

---

## Data Citation

This project uses the Loghub dataset. If you use it in research, please cite:

> Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu.
> "Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics."
> IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023.
> https://github.com/logpai/loghub

---

## License

MIT — use it however you want.
