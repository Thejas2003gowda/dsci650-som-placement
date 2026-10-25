"""Event-driven simulator: replay jobs on a cluster with a strict FIFO queue.

Only two things ever change the cluster: a job arrives, or a job finishes.
The simulator jumps from one of these events to the next instead of
ticking every second.

Run from the repository root:
    python3 -m sim.simulator data/jobs.csv data/machines_log.csv data/results_fifo_log.csv
"""

import argparse
import csv
import heapq
from collections import deque
from datetime import datetime

import numpy as np

from sim.cluster import Cluster
from sim.machines import load_machines

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Event kinds. At the same second, finishes (0) come before arrivals (1),
# so GPUs freed at that second can be used by a job arriving then.
FINISH = 0
ARRIVE = 1

RESULT_COLUMNS = ["job_id", "num_gpus", "arrival", "start", "end", "wait", "jct"]


def load_jobs_csv(path):
    """Read data/jobs.csv into a list of jobs.

    Each job: job_id, num_gpus, runtime (seconds, from runtime_gpu) and
    arrival (seconds after the first job in the trace arrived).
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    submits = [datetime.strptime(row["submit_time"], TIME_FORMAT) for row in rows]
    first = min(submits)
    return [
        {
            "job_id": row["job_id"],
            "num_gpus": int(row["num_gpus"]),
            "runtime": int(row["runtime_gpu"]),
            "arrival": int((submit - first).total_seconds()),
        }
        for row, submit in zip(rows, submits)
    ]


def simulate(jobs, machines):
    """Replay jobs on the cluster with strict FIFO and least-loaded placement.

    jobs: list of dicts with job_id, arrival, num_gpus, runtime.
    machines: {machine: GPU count}.
    Returns one result per job: arrival, start, end, wait and jct (seconds).
    """
    cluster = Cluster(machines)
    cluster_size = sum(machines.values())
    for job in jobs:
        # Under strict FIFO, a job that can never fit would block everyone forever.
        assert job["num_gpus"] <= cluster_size, f"{job['job_id']} is bigger than the cluster"

    # The heap always hands back the earliest event. Each event is
    # (time, kind, job index); the index makes every event unique, so
    # jobs arriving at the same second keep their order in the file.
    events = [(job["arrival"], ARRIVE, i) for i, job in enumerate(jobs)]
    heapq.heapify(events)

    queue = deque()    # waiting jobs, first come first served
    running = {}       # job index -> its placement {machine: GPUs}
    start = [None] * len(jobs)
    end = [None] * len(jobs)

    while events:
        now, kind, i = heapq.heappop(events)
        if kind == FINISH:
            cluster.release(running.pop(i))
            end[i] = now
        else:
            queue.append(i)

        # Strict FIFO: start jobs from the front until one doesn't fit.
        while queue:
            placement = cluster.place(jobs[queue[0]]["num_gpus"])
            if placement is None:
                break
            j = queue.popleft()
            running[j] = placement
            start[j] = now
            heapq.heappush(events, (now + jobs[j]["runtime"], FINISH, j))

    return [
        {
            "job_id": job["job_id"],
            "num_gpus": job["num_gpus"],
            "arrival": job["arrival"],
            "start": start[i],
            "end": end[i],
            "wait": start[i] - job["arrival"],
            "jct": end[i] - job["arrival"],
        }
        for i, job in enumerate(jobs)
    ]


def summarize(results):
    """Headline numbers: job completion time (JCT), waiting and makespan."""
    jct = np.array([r["jct"] for r in results]) / 60     # minutes
    wait = np.array([r["wait"] for r in results]) / 60   # minutes
    return {
        "median JCT (min)": float(np.median(jct)),
        "p99 JCT (min)": float(np.percentile(jct, 99)),
        "mean wait (min)": float(wait.mean()),
        "jobs that waited (%)": float((wait > 0).mean() * 100),
        "makespan (days)": max(r["end"] for r in results) / 86400,
    }


def write_results(results, path):
    """Save one row per job."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Run strict FIFO on one cluster option.")
    parser.add_argument("jobs", help="clean jobs, e.g. data/jobs.csv")
    parser.add_argument("machines", help="cluster option, e.g. data/machines_log.csv")
    parser.add_argument("output", help="where to write per-job results")
    args = parser.parse_args()

    results = simulate(load_jobs_csv(args.jobs), load_machines(args.machines))
    write_results(results, args.output)
    for name, value in summarize(results).items():
        print(f"{name}: {value:,.2f}")
    print(f"written to: {args.output}")


if __name__ == "__main__":
    main()
