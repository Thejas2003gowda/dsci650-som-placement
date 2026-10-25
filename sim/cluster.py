"""The cluster's state while the simulation runs: free GPUs on each machine.

The scheduler asks the cluster two things:
  place(num_gpus)    -> which machines a job gets, or None if it can't start now
  release(placement) -> give a finished job's GPUs back
"""


class Cluster:
    def __init__(self, machines):
        """machines: {machine: GPU count}. Every GPU starts free."""
        self.capacity = dict(machines)
        self.free = dict(machines)

    def total_free(self):
        """Free GPUs in the whole cluster."""
        return sum(self.free.values())

    def place(self, num_gpus):
        """Find GPUs for a job: machines with the most free GPUs first.

        Taking the emptiest machines first is least-loaded placement, and it
        also puts the job on as few machines as possible. A job is split
        across machines only when no single machine has enough free GPUs.

        Returns {machine: GPUs taken}, or None if there aren't enough free
        GPUs right now. Nothing changes when it returns None.
        """
        if num_gpus > self.total_free():
            return None

        placement = {}
        needed = num_gpus
        # Most free GPUs first; the machine name breaks ties so runs repeat exactly.
        for machine in sorted(self.free, key=lambda m: (-self.free[m], m)):
            if needed == 0:
                break
            take = min(self.free[machine], needed)
            placement[machine] = take
            needed -= take

        for machine, take in placement.items():
            self.free[machine] -= take
        return placement

    def release(self, placement):
        """Give back the GPUs of a job that finished."""
        for machine, taken in placement.items():
            self.free[machine] += taken
            assert self.free[machine] <= self.capacity[machine], f"{machine}: more free GPUs than it has"
