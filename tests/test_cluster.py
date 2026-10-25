"""Tests for sim/cluster.py, using tiny made-up clusters."""

from sim.cluster import Cluster


def test_new_cluster_is_all_free():
    cluster = Cluster({"a": 2, "b": 8})
    assert cluster.total_free() == 10


def test_small_job_goes_to_machine_with_most_free_gpus():
    cluster = Cluster({"a": 2, "b": 8})
    assert cluster.place(1) == {"b": 1}
    assert cluster.free == {"a": 2, "b": 7}


def test_job_that_fits_on_one_machine_is_not_split():
    cluster = Cluster({"a": 8, "b": 8})
    assert cluster.place(8) == {"a": 8}


def test_big_job_spans_machines_emptiest_first():
    cluster = Cluster({"a": 8, "b": 2, "c": 4})
    assert cluster.place(12) == {"a": 8, "c": 4}
    assert cluster.free == {"a": 0, "b": 2, "c": 0}


def test_not_enough_free_gpus_returns_none_and_changes_nothing():
    cluster = Cluster({"a": 2, "b": 2})
    assert cluster.place(5) is None
    assert cluster.free == {"a": 2, "b": 2}


def test_release_gives_the_gpus_back():
    cluster = Cluster({"a": 8, "b": 8})
    placement = cluster.place(12)
    cluster.release(placement)
    assert cluster.free == {"a": 8, "b": 8}


def test_ties_are_broken_by_machine_name():
    cluster = Cluster({"b": 8, "a": 8})
    assert cluster.place(1) == {"a": 1}
