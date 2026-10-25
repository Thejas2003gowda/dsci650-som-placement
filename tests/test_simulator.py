"""Tests for sim/simulator.py on tiny made-up traces, checked by hand."""

import pytest

from sim.simulator import load_jobs_csv, simulate, summarize


def job(name, arrival, gpus, runtime):
    return {"job_id": name, "arrival": arrival, "num_gpus": gpus, "runtime": runtime}


def by_id(results):
    return {r["job_id"]: r for r in results}


def test_job_on_empty_cluster_starts_at_once():
    r = by_id(simulate([job("j1", 0, 2, 10)], {"a": 4}))["j1"]
    assert (r["start"], r["end"], r["wait"], r["jct"]) == (0, 10, 0, 10)


def test_job_waits_until_gpus_are_free():
    # j1 takes all 4 GPUs until t=10; j2 arrives at t=5 and must wait.
    r = by_id(simulate([job("j1", 0, 4, 10), job("j2", 5, 2, 10)], {"a": 4}))
    assert (r["j2"]["start"], r["j2"]["end"], r["j2"]["wait"]) == (10, 20, 5)


def test_strict_fifo_small_job_waits_behind_big_one():
    # At t=2 one GPU is free and j3 needs only 1, but j2 is ahead of it.
    r = by_id(simulate([
        job("j1", 0, 3, 10),
        job("j2", 1, 4, 10),
        job("j3", 2, 1, 5),
    ], {"a": 4}))
    assert r["j2"]["start"] == 10
    assert r["j3"]["start"] == 20   # only after j2 finishes


def test_finish_is_handled_before_arrival_at_same_second():
    r = by_id(simulate([job("j1", 0, 2, 10), job("j2", 10, 2, 5)], {"a": 2}))
    assert r["j2"]["start"] == 10
    assert r["j2"]["wait"] == 0


def test_job_bigger_than_cluster_is_refused():
    with pytest.raises(AssertionError):
        simulate([job("j1", 0, 16, 10)], {"a": 8})


def test_arrival_is_seconds_after_first_job(tmp_path):
    path = tmp_path / "jobs.csv"
    path.write_text(
        "job_id,vc,user,status,submit_time,num_gpus,runtime_gpu,runtime_span\n"
        "j1,v,u,Pass,2017-10-01 09:00:00,1,30,30\n"
        "j2,v,u,Pass,2017-10-01 09:01:00,2,40,50\n"
    )
    jobs = load_jobs_csv(path)
    assert [j["arrival"] for j in jobs] == [0, 60]
    assert jobs[1]["runtime"] == 40   # runtime_gpu, not runtime_span


def test_summary_numbers():
    results = [
        {"jct": 60, "wait": 0, "end": 60},
        {"jct": 180, "wait": 120, "end": 86400},
    ]
    s = summarize(results)
    assert s["median JCT (min)"] == 2.0
    assert s["mean wait (min)"] == 1.0
    assert s["jobs that waited (%)"] == 50.0
    assert s["makespan (days)"] == 1.0
