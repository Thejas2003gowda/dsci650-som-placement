# Decentralized Placement of Deep Learning Training Jobs Using Self-Organizing Maps

DSCI-650 course project, Rochester Institute of Technology.
Author: Thejas Nagesh Gowda (individual project).

## What this project is

Independent scheduling agents, each with a self-organizing map (SOM),
place deep-learning training jobs on a simulated GPU cluster. Agents
share fixed-size map summaries instead of full cluster state. A
trace-driven simulator replays Microsoft's Philly trace to compare this
with centralized baselines (FIFO, shortest-job-first, least-loaded) and
Sparrow-style random sampling. Metrics: job completion time, makespan,
GPU utilization and coordination cost.

## Requirements

- Python 3 (tested with 3.14.2)
- Packages in `requirements.txt` (numpy, pandas, pytest)

## Data

- Microsoft Philly trace: `trace-data.tar.gz` from
  https://github.com/msr-fiddle/philly-traces (license CC-BY-4.0;
  cite Jeon et al., USENIX ATC 2019).
- Extract it to `~/philly-traces/trace-data`. It is not stored in this repo.

## How to build and run

Run all commands from the repository root.

1. Set up once:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the tests:
   ```bash
   python3 -m pytest
   ```
3. Clean the trace into `data/jobs.csv`:
   ```bash
   python3 -m sim.trace ~/philly-traces/trace-data/cluster_job_log data/jobs.csv
   ```
   It prints how many jobs were kept and dropped. Expected: 117,325 jobs
   in the log, 111,856 kept.

## Code layout

| Path | What it is |
|---|---|
| `README.md` | This file |
| `AI_USAGE.md` | How AI was used and how I checked it |
| `requirements.txt` | Python packages |
| `sim/trace.py` | Reads the raw Philly job log, drops jobs that can't be replayed, writes one clean row per job |
| `tests/test_trace.py` | Tests for `sim/trace.py` on small hand-made jobs |
