"""Tests for sim/trace.py, using tiny hand-made jobs."""

from datetime import datetime

from sim.trace import clean_job, parse_time


def attempt(start, end, gpus_per_machine):
    """Build one raw attempt. gpus_per_machine is a list, e.g. [8, 8] = two machines."""
    return {
        "start_time": start,
        "end_time": end,
        "detail": [
            {"ip": f"m{i}", "gpus": [f"gpu{g}" for g in range(n)]}
            for i, n in enumerate(gpus_per_machine)
        ],
    }


def job(attempts):
    """Build one raw job around a list of attempts."""
    return {
        "jobid": "job1",
        "vc": "vc1",
        "user": "user1",
        "status": "Pass",
        "submitted_time": "2017-10-01 09:00:00",
        "attempts": attempts,
    }


def test_text_none_is_read_as_missing():
    assert parse_time("None") is None
    assert parse_time("2017-10-01 09:00:00") == datetime(2017, 10, 1, 9, 0, 0)


def test_simple_job_is_kept():
    row, reason = clean_job(job([attempt("2017-10-01 09:00:00", "2017-10-01 10:00:00", [1])]))
    assert reason is None
    assert row["num_gpus"] == 1
    assert row["runtime_gpu"] == 3600
    assert row["runtime_span"] == 3600


def test_retry_gap_counts_only_in_span():
    # Runs 10 min, waits 50 min, runs 30 min.
    row, _ = clean_job(job([
        attempt("2017-10-01 09:00:00", "2017-10-01 09:10:00", [1]),
        attempt("2017-10-01 10:00:00", "2017-10-01 10:30:00", [1]),
    ]))
    assert row["runtime_gpu"] == 40 * 60   # time on GPUs only
    assert row["runtime_span"] == 90 * 60  # first start to last end


def test_gpus_add_up_across_machines_and_largest_attempt_wins():
    row, _ = clean_job(job([
        attempt("2017-10-01 09:00:00", "2017-10-01 09:10:00", [4]),
        attempt("2017-10-01 10:00:00", "2017-10-01 11:00:00", [8, 8]),
    ]))
    assert row["num_gpus"] == 16


def test_job_with_no_attempts_is_dropped():
    row, reason = clean_job(job([]))
    assert row is None
    assert reason == "never started"


def test_attempt_with_missing_end_is_not_used():
    row, reason = clean_job(job([attempt("2017-10-01 09:00:00", "None", [1])]))
    assert row is None
    assert reason == "no usable attempt"


def test_zero_gpu_attempt_adds_no_runtime():
    row, _ = clean_job(job([
        attempt("2017-10-01 09:00:00", "2017-10-01 09:00:30", []),
        attempt("2017-10-01 09:01:00", "2017-10-01 09:11:00", [2]),
    ]))
    assert row["num_gpus"] == 2
    assert row["runtime_gpu"] == 600


def test_attempt_ending_before_it_starts_is_not_used():
    row, reason = clean_job(job([attempt("2017-11-05 01:31:17", "2017-11-05 01:31:02", [1])]))
    assert row is None
    assert reason == "no usable attempt"
