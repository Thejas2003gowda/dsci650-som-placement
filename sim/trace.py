"""Read the Philly job log and turn it into a clean table of jobs.

The simulator needs three facts about each job: when it arrived, how many
GPUs it needs, and how long it runs. This file pulls those out of the raw
JSON log and drops jobs that are missing any of them.

Run it from the repository root:
    python3 -m sim.trace ~/philly-traces/trace-data/cluster_job_log data/jobs.csv
"""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Columns of the clean table, in order.
COLUMNS = [
    "job_id",
    "vc",            # virtual cluster: the team the job belongs to
    "user",
    "status",        # Pass, Failed or Killed
    "submit_time",
    "num_gpus",      # largest GPU count of any usable attempt
    "runtime_gpu",   # seconds: usable attempts' durations added up
    "runtime_span",  # seconds: first usable start to last usable end
]


def parse_time(text):
    """Turn a time string from the trace into a datetime.

    The trace writes a missing time as the text "None", so we turn that
    into Python's None.
    """
    if text is None or text == "None":
        return None
    return datetime.strptime(text, TIME_FORMAT)


def count_gpus(attempt):
    """Number of GPUs one attempt used, added up over all its machines."""
    return sum(len(machine["gpus"]) for machine in attempt["detail"])


def usable_attempts(raw_job):
    """Return (start, end, gpus) for each attempt we can use.

    An attempt is usable when it has a start time and an end time, the end
    is not before the start, and it ran on at least one GPU.
    """
    usable = []
    for attempt in raw_job["attempts"]:
        start = parse_time(attempt["start_time"])
        end = parse_time(attempt["end_time"])
        gpus = count_gpus(attempt)
        if start is None or end is None:
            continue  # a time is missing: we can't know how long it ran
        if end < start:
            continue  # broken time stamps (4 attempts in the trace)
        if gpus == 0:
            continue  # it never ran on a GPU
        usable.append((start, end, gpus))
    return usable


def clean_job(raw_job):
    """Turn one raw job into a clean row.

    Returns (row, None) if the job is kept,
    or (None, reason) if it is dropped.
    """
    if not raw_job["attempts"]:
        return None, "never started"

    attempts = usable_attempts(raw_job)
    if not attempts:
        return None, "no usable attempt"

    starts = [start for start, _, _ in attempts]
    ends = [end for _, end, _ in attempts]

    runtime_gpu = sum((end - start).total_seconds() for start, end, _ in attempts)
    runtime_span = (max(ends) - min(starts)).total_seconds()
    num_gpus = max(gpus for _, _, gpus in attempts)

    row = {
        "job_id": raw_job["jobid"],
        "vc": raw_job["vc"],
        "user": raw_job["user"],
        "status": raw_job["status"],
        "submit_time": raw_job["submitted_time"],
        "num_gpus": num_gpus,
        "runtime_gpu": int(runtime_gpu),
        "runtime_span": int(runtime_span),
    }
    return row, None


def load_jobs(path):
    """Read the raw job log. Return (rows, dropped).

    rows: clean jobs, sorted by submit time (the order they arrive).
    dropped: how many jobs were dropped, for each reason.
    """
    with open(path) as f:
        raw_jobs = json.load(f)

    rows = []
    dropped = Counter()
    for raw_job in raw_jobs:
        row, reason = clean_job(raw_job)
        if row is None:
            dropped[reason] += 1
        else:
            rows.append(row)

    # The raw file is not in time order. The time format sorts correctly
    # as text; job_id breaks ties so the order is the same on every run.
    rows.sort(key=lambda row: (row["submit_time"], row["job_id"]))
    return rows, dropped


def write_csv(rows, path):
    """Save the clean rows as a CSV file, creating its folder if needed."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Clean the Philly job log.")
    parser.add_argument("job_log", help="path to cluster_job_log")
    parser.add_argument("output", help="where to write the clean CSV, e.g. data/jobs.csv")
    args = parser.parse_args()

    rows, dropped = load_jobs(os.path.expanduser(args.job_log))
    write_csv(rows, args.output)

    print(f"jobs in log: {len(rows) + sum(dropped.values())}")
    for reason, count in sorted(dropped.items()):
        print(f"dropped ({reason}): {count}")
    print(f"kept: {len(rows)}")
    print(f"written to: {args.output}")


if __name__ == "__main__":
    main()
