# Design

Decisions for the multi-agent scheduler, with reasons, and items to revisit.
Implemented so far (Update I): the single-process simulator with strict FIFO
and least-loaded placement (see README.md).

## Changes from the proposal

| Change | Reason |
|---|---|
| RQ3 refined: "How much coordination traffic do cheaper exchange methods (such as sending only updated cells) save compared with full-map exchange, and how much placement quality do they cost?" | Professor's feedback: full-map all-to-all transfer and averaging is naive; examine more efficient transfer. As first written, H3 was true by definition (a full map always has m x n x d values). |
| 111,856 jobs, not 111,751 | Counted by our own parser (see README, Data decisions). |
| Agents are groups of VCs, not one VC each | Needed for RQ1's 2 to 5 agents and to cover the 10 smaller VCs (decision 3). |

## Architecture decisions (Oct 25, 2026)

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | How do agents run? | Each agent is a separate MPI process (mpi4py). The same agent code also runs in a single-process test mode, where the exchange is a plain copy. | separate processes exchange real messages, so coordination bytes are measured on real traffic. The test mode makes debugging easier and checks that the MPI version gives the same results. |
| 2 | What if two agents pick the same GPUs? | Each agent owns a set of machines and places jobs only there. Ownership is recomputed at every exchange, in proportion to each agent's work. | No conflicts and no central process. Matches the proposal ("node sets ... recomputed during exchanges"). GPUs idle in one agent's share stay unused until the next exchange: a staleness cost RQ2 can measure. |
| 3 | Which agent handles which jobs? | The 15 VCs are grouped into k agents (k = 2 to 5): largest VC first, each to the agent with the fewest jobs so far. | Every job has an agent, and RQ1 can vary 2 to 5 agents. With k = 5, the 5 largest VCs (90.7% of jobs) each anchor one agent. One VC holds 46.5% of jobs, so agents are uneven. |
| 4 | How are maps exchanged? | Every τ seconds of simulated time, all agents stop and exchange with MPI Allgather (everyone sends to everyone). Each agent averages what it receives, machine ownership is recomputed, and all continue. | The same message pattern works for full maps (the baseline) and for sending only changed cells (professor's feedback), so bytes are measured the same way for both. |

## How a multi-agent run works

1. Each agent process loads the jobs of its VCs and its starting machines.
2. For τ simulated seconds, each agent works alone: jobs arrive, its SOM is updated, jobs are placed on its own machines, jobs finish.
3. At the end of the window, all agents exchange maps (Allgather). This is the only point where they wait for each other.
4. Each agent averages the maps, machine ownership is recomputed, and the next window starts.
5. When all jobs are done, per-job results are gathered for the metrics.

## To revisit

| Item | When |
|---|---|
| How VCs are grouped into agents: by job count vs by GPU-hours | Phase 2 |
| How re-balancing handles a machine that still has running jobs, and how the first split is made | Phase 2 |
| Rerun each new scheduler on all three cluster options (list, log, max) | Every new scheduler |
| Almost nothing waits at real cluster size: shrink the cluster (RQ3 already varies cluster size) or add team limits? | Decide by Update II |
| Compare our waiting times with the real Philly cluster's | Phase 2 |
| Check that GPUs in use never exceed a machine's size on the real run | Phase 2 |
| Test the MPI install (mpi4py) on Python 3.14 | Right after Update I |
| Cheaper map exchange, e.g. only changed cells (professor's feedback) | Phase 3 |
| Limitations to state: split jobs don't run slower; times in the daylight-saving hour on Nov 5, 2017 | Paper |
| Expand related work (professor's feedback) | Final paper |
