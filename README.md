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

- Python 3
- `pip install -r requirements.txt` (numpy, pandas)

## Data

- Microsoft Philly trace: `trace-data.tar.gz` from
  https://github.com/msr-fiddle/philly-traces (license CC-BY-4.0;
  cite Jeon et al., USENIX ATC 2019).
- Extract it to `~/philly-traces/trace-data`. It is not stored in this repo.

## How to build and run

No runnable code yet. Steps are added with the first runnable code.

## Code layout

| Path | What it is |
|---|---|
| `README.md` | This file |
| `AI_USAGE.md` | How AI was used and how I checked it |
| `requirements.txt` | Python packages |
