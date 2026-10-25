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
4. Compare the two runtime definitions:
   ```bash
   python3 scripts/compare_runtimes.py data/jobs.csv
   ```
5. Build the three cluster options (`data/machines_list.csv`,
   `data/machines_log.csv`, `data/machines_max.csv`):
   ```bash
   python3 -m sim.machines ~/philly-traces/trace-data data
   ```
   Expected: list 2,490 GPUs, log 2,336 GPUs, max 3,918 GPUs.
6. Run strict FIFO with least-loaded placement on each cluster option:
   ```bash
   python3 -m sim.simulator data/jobs.csv data/machines_list.csv data/results_fifo_list.csv
   python3 -m sim.simulator data/jobs.csv data/machines_log.csv data/results_fifo_log.csv
   python3 -m sim.simulator data/jobs.csv data/machines_max.csv data/results_fifo_max.csv
   ```
   Each prints median and p99 job completion time, mean wait, the share
   of jobs that waited, and makespan.

## Data decisions

| Decision | Choice | Why |
|---|---|---|
| Jobs dropped | 5,236 never started; 233 had no usable attempt (161 missing a time, 72 no GPU). 111,856 kept, including Failed and Killed jobs. | Dropped jobs can't be replayed. Failed and Killed jobs still used GPUs. |
| Runtime | `runtime_gpu`: attempt durations added up. `runtime_span` stays in the CSV for comparison only. | Gaps between retries were the real scheduler's waiting, and the GPUs were free then. The span adds 4.4% GPU-hours and holds GPUs during gaps of up to 27 days. |
| GPU count | Largest GPU count of any usable attempt | Attempts run one after another, so a job never holds more than its largest attempt. |
| Known issue | 4 attempts end before they start, all on Nov 5, 2017, around 1:30 AM | Likely the daylight-saving clock change. Those attempts are ignored. |
| Cluster | Not decided yet. Three options are built (list, log, max) and will be compared in the simulator. | The machine list contradicts the job log for 410 of the 415 machines that ran jobs; on 238 of them, jobs used more GPUs than the list says exist. |
| Placement (least-loaded) | Machines with the most free GPUs first, until the job has enough. A job is split across machines only when no single machine has room. | In the trace, 96.5% of attempts with 8 or fewer GPUs ran on one machine, and about 40% of larger ones were split over partly free machines. Limitation: a split job does not run slower in the simulator. |

## Code layout

| Path | What it is |
|---|---|
| `README.md` | This file |
| `AI_USAGE.md` | How AI was used and how I checked it |
| `requirements.txt` | Python packages |
| `sim/trace.py` | Reads the raw Philly job log, drops jobs that can't be replayed, writes one clean row per job |
| `tests/test_trace.py` | Tests for `sim/trace.py` on small hand-made jobs |
| `scripts/compare_runtimes.py` | Compares the two runtime definitions in `data/jobs.csv` |
| `sim/machines.py` | Builds the three cluster options (which machines, how many GPUs each) |
| `tests/test_machines.py` | Tests for `sim/machines.py` on small made-up machines |
| `sim/cluster.py` | Free GPUs per machine during a simulation; places jobs and releases them |
| `tests/test_cluster.py` | Tests for `sim/cluster.py` on tiny made-up clusters |
| `sim/simulator.py` | Event-driven simulator: replays jobs with a strict FIFO queue and writes per-job results |
| `tests/test_simulator.py` | Tests for `sim/simulator.py` on tiny made-up traces, checked by hand |
