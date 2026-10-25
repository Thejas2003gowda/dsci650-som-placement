"""Build the three cluster options: which machines exist and how many GPUs each has.

The Philly machine list contradicts the job log for most machines, so we
build three versions and compare them in the simulator:

  list: the machine list as given                       -> machines_list.csv
  log:  machines that ran jobs; GPUs = highest GPU
        number any job used on that machine, plus one    -> machines_log.csv
  max:  every machine; GPUs = the larger of list and log -> machines_max.csv

Run from the repository root:
    python3 -m sim.machines ~/philly-traces/trace-data data
"""

import argparse
import csv
import json
import os
from collections import Counter


def read_machine_list(path):
    """Return {machine: GPU count} from Philly's cluster_machine_list file.

    Its header is "machineId, number of GPUs, single GPU mem".
    """
    machines = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip the header
        for row in reader:
            if row:
                machines[row[0].strip()] = int(row[1])
    return machines


def gpu_number(name):
    """Turn a GPU name like "gpu7" into the number 7."""
    return int(name.replace("gpu", ""))


def machines_from_log(raw_jobs):
    """Return {machine: GPU count} as seen in the job log.

    GPUs are numbered from gpu0, so a machine where some job used gpu7
    has at least 8 GPUs. Every attempt counts, even one with a missing
    time: it still shows which GPUs exist.
    """
    machines = {}
    for job in raw_jobs:
        for attempt in job["attempts"]:
            for part in attempt["detail"]:
                machine = part["ip"]
                for gpu in part["gpus"]:
                    count = gpu_number(gpu) + 1
                    machines[machine] = max(machines.get(machine, 0), count)
    return machines


def larger_of(listed, logged):
    """Every machine in either table, with the larger of its two GPU counts."""
    names = set(listed) | set(logged)
    return {m: max(listed.get(m, 0), logged.get(m, 0)) for m in names}


def write_machines(machines, path):
    """Save {machine: GPU count} as a CSV with columns machine,gpus."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["machine", "gpus"])
        for machine in sorted(machines):
            writer.writerow([machine, machines[machine]])


def load_machines(path):
    """Read a CSV written by write_machines back into {machine: GPU count}."""
    with open(path, newline="") as f:
        return {row["machine"]: int(row["gpus"]) for row in csv.DictReader(f)}


def describe(machines):
    """One line: machine count, GPU count, and how many machines of each size."""
    sizes = Counter(machines.values())
    by_size = ", ".join(f"{n} x {g}" for g, n in sorted(sizes.items()))
    return f"{len(machines)} machines, {sum(machines.values())} GPUs ({by_size})"


def main():
    parser = argparse.ArgumentParser(description="Build the three cluster options.")
    parser.add_argument("trace_dir", help="folder with cluster_machine_list and cluster_job_log")
    parser.add_argument("out_dir", help="where to write the machine tables, e.g. data")
    args = parser.parse_args()

    trace_dir = os.path.expanduser(args.trace_dir)
    listed = read_machine_list(os.path.join(trace_dir, "cluster_machine_list"))
    with open(os.path.join(trace_dir, "cluster_job_log")) as f:
        logged = machines_from_log(json.load(f))
    options = {"list": listed, "log": logged, "max": larger_of(listed, logged)}

    os.makedirs(args.out_dir, exist_ok=True)
    for name, machines in options.items():
        path = os.path.join(args.out_dir, f"machines_{name}.csv")
        write_machines(machines, path)
        print(f"{name}: {describe(machines)} -> {path}")

    too_small = sum(1 for m in logged if logged[m] > listed.get(m, 0))
    print(f"machines where jobs used more GPUs than the list says: {too_small}")


if __name__ == "__main__":
    main()
