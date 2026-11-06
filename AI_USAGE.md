# AI usage

This file records which AI tools were used on peregrin, for what, and on which parts of the
repository. It is updated in the same pull request as the work it describes.

## Tools

| Tool | Model | Period |
|---|---|---|
| Claude (claude.ai, in the browser) | Claude Opus 5.5 | before 2026-11-04 |
| Claude Code (Anthropic's CLI coding agent) | Claude Opus 5.5 (`claude-opus-5-5`) | from 2026-11-04 |

## How it was used

Claude Code ran in the repository with access to a shell. It wrote code and documents,
ran tests and benchmarks, and made the commits.

**Commits.** The commits from 2026-11-04 to 2026-11-05 were made by Claude Code under
Carson's git identity. They originally ended with a `Co-Authored-By: Claude` line, which was
removed from the history on 2026-11-05 at Carson's request, before the repository had any
other users. Since then, Claude Code's part in each change is recorded here and in the pull
request descriptions, not in commit trailers.

**Issue and pull request text.** From #1 on, Claude Code drafts issue and PR descriptions,
and each PR says so.

## By area

The extent is stated as it is. Where Claude Code wrote nearly all of something, this says
so.

| Area | What existed before 2026-11-04 | Claude Code's part |
|---|---|---|
| `crates/core` (Rust simulator) | nothing | Designed and wrote all of it: the batch-first API, the tile engine, permutation relabeling, real-state tracking, fusion, vectorized trig, the JSON format, and the golden and unit tests. |
| `crates/python` (bindings, `peregrin.cpu` device, packaging) | nothing | Designed and wrote all of it, including the device tests and the PennyLane version compatibility. |
| `.github/` (CI, release, templates) | nothing | Wrote all of it. |
| `bench/` harness (`run_bench.py`, `worker.py`, `workload.py`, `backends.py`, `summarize.py`) | the original harness in the first commit, written by Claude in the browser | Extended it: `--circuit`, the template interpreter and its elementwise rewrite in `backends.py`, the peregrin backends, `compare.py`, `sweeps/cpu.sh`, and the move to `bench/`. |
| `bench/circuits.py` | the circuits in Carson's paper code, and the survey `quanvolution_circuits.md` | Wrote the registry. The `owner-mri` gates are transcribed from Carson's original code and checked against a copy of it in `bench/tests/test_circuits.py`. |
| `bench/results/`, `bench/RESULTS.md` | `results/stateprep` | Ran every sweep and wrote `RESULTS.md`, on Carson's laptop. |
| `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff`, `DESIGN.md` | an earlier harness README | Drafted. `CODE_OF_CONDUCT.md` is the unmodified Contributor Covenant 2.1. |

The original benchmark harness, the first commit's `run_bench.py`, `worker.py`,
`workload.py`, `backends.py`, `summarize.py` and `README.md`, was written by Claude in the
browser (claude.ai), before Claude Code was used. Its first results, `results/stateprep`,
came from that harness.

`quanvolution_circuits.md`, the survey of circuits from published quanvolutional papers, was
also compiled by Claude in the browser. Benchmarking circuits from other published papers was
Carson's idea, and the survey was compiled at his request.

## Carson's part

These decisions are Carson's. They were made in conversation with Claude Code, which asked
before acting on them:

- the goal and the target workload: quanvolutional preprocessing of MRI patches, from his
  own paper and code;
- which circuits to benchmark: his paper's circuits taken from his code, not from the
  paper's figure, plus the circuits from his survey;
- the project and package names (`peregrin`, `pennylane-peregrin`), and the device names
  `peregrin.cpu` and, later, `peregrin.gpu`;
- the license, the single repository and workspace, f32 and f64 both supported, and CPU first
  with GPU deferred;
- keeping his original paper code private;
- separating the tool from the benchmark harness;
- the order of the work: device, then CI and wheels, then this record and the PennyLane test
  suite.

The internal design of the simulator and the device was proposed and implemented by Claude
Code, and Carson approved going ahead with it. `DESIGN.md` lists those decisions one by one,
and Carson reviews them there.

Carson reviews every pull request before it is merged.
