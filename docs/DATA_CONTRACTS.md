# Data contracts

This repository stores benchmark harnesses, run artifacts and local binaries.
The authoritative overview is [README.md](../README.md). The contracts below
describe the files that are actually present in this checkout; they are not a
claim about production machine state.

## Tracked JSON formats

| File(s) | Format | Validated invariants |
| --- | --- | --- |
| `bin/k2h_benchmark_k2-horizon_*.json` | `version: "1.0.0"` suite written by [k2h_benchmark.py](../bin/k2h_benchmark.py) | Every `(category, name)` pair equals `K2H_TASKS` in the harness, no duplicates, and all summary counts/formatted averages are recomputed from `results`. |
| `bin/runs/**/<run-id>/results.json` | terminal-bench aggregate | `n_resolved`/`n_unresolved`/`resolved_ids`/`unresolved_ids`/`accuracy` agree with the trial entries; task ids are unique. |
| `bin/runs/**/<run-id>/<task>/<trial>/results.json` | terminal-bench trial | Required fields present, parser statuses are `passed`/`failed`, resolved trials have all tests passing, timestamps parse as ISO 8601. |
| `bin/runs/**/<run-id>/run_metadata.json` | terminal-bench metadata | `task_ids` and `accuracy` match the aggregate; dataset is `terminal-bench-core` `0.1.1`; time range is ordered. |
| `bin/runs/**/<run-id>/pareto.json` | cost/accuracy roll-up | Rows are unique, counts and accuracy match the aggregate, and relative `results_json` paths exist (resolved from `bin/`, the harness cwd; absolute paths are external and not checked). |
| `bin/runs/k2h-*/summary.json` | harness run summary | `base_url == server + "/v1"`, positive `n_tasks`, parseable timestamp, and `results_dir` exists under `bin/`. |
| `bin/k2h_benchmark_results_latest.json` | `version: "2.0.0"` report | Server host/port/gateway agree with [k2h-bench.sh](../bin/k2h-bench.sh), accuracy equals `resolved/total`, and the referenced harness scripts exist in `bin/`. |

Referenced trial artifacts are resolved relative to the parent of the run-id
directory, so a missing `sessions/agent.cast` is a contract violation. JSON
files are read-only inputs to the tests; no harness is executed.

## Optional, gitignored artifacts

`.gitignore` excludes `*.tsv` and `*.tar.gz`, so they are not in this checkout.
If present, [test_artifact_contract.py](../tests/test_artifact_contract.py)
checks that a TSV table has a unique, non-empty header, rectangular rows, no
empty cells and unique node ids in column 1. For archives it reads headers
only (never extracts): members must be files/directories with safe relative
paths and bounded sizes.

## Validation

```sh
cd <repo-root>
python3 -m pytest -q
python3 -m py_compile bin/k2h_benchmark.py bin/k2h_terminal_bench.py
```

The validators live in [contracts.py](../tests/contracts.py) and use only the
Python standard library. They require no network, credentials, model files or
Docker daemon. `py_compile` writes bytecode caches; the test suite does not.
