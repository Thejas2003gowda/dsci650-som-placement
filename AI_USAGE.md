# AI usage

## Tools

- Claude (claude.ai), Oct 14-19, 2026: an early prototype simulator. Discarded on Oct 23; no code from it is in this repo.
- Claude (claude.ai), from Oct 23, 2026: planning, exploring the trace data, writing code in small steps from my decisions, and slide layout.

## How I work with AI in this project

- Claude proposes a plan for each step. I approve or change it before anything is written.
- Claude writes the code one small piece at a time with my inputs and direction.
- I run every step on my own machine, look at the results, look for ways to improvise if and make the commit myself.

## What was AI-generated

| Part | Mostly AI? 
|---|---|
| README.md (skeleton) | NO |
| AI_USAGE.md  | NO |
| .gitignore, requirements.txt | NO | 
| sim/trace.py, sim/machines.py, sim/cluster.py, sim/simulator.py | yes |
| tests/ (all four test files) | yes |
| scripts/compare_runtimes.py | yes |
| DESIGN.md | yes, written from my decisions |
| Update I slides | layout and wording by AI; content from my decisions |

## How I checked it

| Part | How I checked it |
|---|---|
| sim/trace.py | Tests pass; ran it on the real trace: 111,856 jobs kept, matching the counts found when exploring the data first |
| scripts/compare_runtimes.py | Ran it on the real data; used its numbers to choose runtime_gpu |
| sim/machines.py | Tests pass; GPU totals (2,490 / 2,336 / 3,918) match the earlier data check |
| sim/cluster.py | Tests pass; worked through placement examples by hand |
| sim/simulator.py | Tests on tiny traces with hand-worked answers; real runs: makespan 136.9 days matches the trace length, and a bigger cluster gives less waiting |
