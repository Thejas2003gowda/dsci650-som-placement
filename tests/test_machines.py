"""Tests for sim/machines.py, using tiny made-up machines."""

from sim.machines import (
    describe,
    larger_of,
    load_machines,
    machines_from_log,
    read_machine_list,
    write_machines,
)


def attempt(start, end, parts):
    """Build one raw attempt. parts is a list like [("m1", ["gpu0", "gpu7"])]."""
    return {
        "start_time": start,
        "end_time": end,
        "detail": [{"ip": ip, "gpus": gpus} for ip, gpus in parts],
    }


def test_machine_list_is_read_with_spaces_and_header(tmp_path):
    path = tmp_path / "cluster_machine_list"
    path.write_text("machineId, number of GPUs, single GPU mem\nm0,2, 12GB\nm1,8, 24GB\n")
    assert read_machine_list(path) == {"m0": 2, "m1": 8}


def test_log_capacity_is_highest_gpu_number_plus_one():
    jobs = [
        {"attempts": [attempt("2017-10-01 09:00:00", "2017-10-01 10:00:00",
                              [("m1", ["gpu0"]), ("m2", ["gpu1"])])]},
        # Missing end time still shows that m1 has a gpu7.
        {"attempts": [attempt("2017-10-02 09:00:00", "None", [("m1", ["gpu7"])])]},
    ]
    assert machines_from_log(jobs) == {"m1": 8, "m2": 2}


def test_larger_of_keeps_every_machine_and_the_bigger_count():
    listed = {"a": 2, "b": 8, "c": 2}   # c never ran a job
    logged = {"a": 8, "b": 2}
    assert larger_of(listed, logged) == {"a": 8, "b": 8, "c": 2}


def test_write_then_load_gives_back_the_same_machines(tmp_path):
    machines = {"m2": 8, "m1": 2}
    path = tmp_path / "machines.csv"
    write_machines(machines, path)
    assert load_machines(path) == machines


def test_describe_counts_machines_gpus_and_sizes():
    assert describe({"a": 2, "b": 8, "c": 8}) == "3 machines, 18 GPUs (1 x 2, 2 x 8)"
