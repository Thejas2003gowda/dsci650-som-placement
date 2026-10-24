"""Compare the two runtime definitions in data/jobs.csv.

runtime_gpu:  time the job actually spent on GPUs (attempts added up)
runtime_span: first start to last end (also counts the waits between retries)

Run from the repository root:
    python3 scripts/compare_runtimes.py data/jobs.csv
"""

import sys

import pandas as pd


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/jobs.csv"
    jobs = pd.read_csv(path)

    # Attempts never overlap, so the span can never be shorter than the GPU time.
    assert (jobs["runtime_span"] >= jobs["runtime_gpu"]).all(), "span shorter than GPU time"

    differ = jobs[jobs["runtime_span"] > jobs["runtime_gpu"]]
    print(f"jobs: {len(jobs):,}")
    print(f"jobs where the two runtimes differ: {len(differ):,} ({len(differ) / len(jobs):.1%})")

    # GPU-hours = GPUs x hours, added up over all jobs: the total work the cluster must do.
    hours_gpu = (jobs["num_gpus"] * jobs["runtime_gpu"]).sum() / 3600
    hours_span = (jobs["num_gpus"] * jobs["runtime_span"]).sum() / 3600
    print(f"total GPU-hours with runtime_gpu:  {hours_gpu:,.0f}")
    print(f"total GPU-hours with runtime_span: {hours_span:,.0f} "
          f"({hours_span / hours_gpu - 1:+.1%})")

    # For the jobs that differ: how much extra time does the span add?
    extra_hours = (differ["runtime_span"] - differ["runtime_gpu"]) / 3600
    print("extra time added by the span, jobs that differ:")
    print(f"  median: {extra_hours.median():.2f} h")
    print(f"  90% of them: under {extra_hours.quantile(0.9):.2f} h")
    print(f"  largest: {extra_hours.max():.2f} h")

    at_least_double = (differ["runtime_span"] >= 2 * differ["runtime_gpu"]).sum()
    print(f"  span is at least 2x the GPU time: {at_least_double:,} jobs")


if __name__ == "__main__":
    main()
