# fleet-data

A public repository for fleet-related data or tooling. Its authoritative data model, collection process, executable entry point, and reproducibility commands have not been verified in this README.

## Data-handling guidelines

- Document each dataset's schema, source, collection interval, and ownership.
- Prefer synthetic, aggregated, or explicitly approved examples.
- Never publish API tokens, credentials, personal data, or sensitive private-network identifiers.
- Clearly separate observed machine state from desired configuration and from synthetic tests.

## Reproducing results

Before publishing derived metrics, add exact input versions, validation commands, provenance, and date ranges. This README does not claim the data is current or approved for production routing decisions.

## Benchmark harnesses

`bin/k2h-bench.sh`, `bin/k2h_benchmark.py`, and `bin/k2h_terminal_bench.py`
expect a llama-server on port 8888; the terminal harness also needs the `tb`
CLI, Docker, Node.js, and a terminal-bench checkout containing
`build-manifest.mjs`, `hardest-first.mjs`, and `score.py`. Missing prerequisites
now produce actionable preflight errors instead of cryptic failures. Existing
positional arguments and run behavior are unchanged when prerequisites exist.

Validate configuration without starting servers, downloading data, installing
tools, or making network calls:

```sh
bash bin/k2h-bench.sh --check                 # or: --dry-run bench|terminal|full
python3 bin/k2h_benchmark.py --check          # or: --dry-run
python3 bin/k2h_terminal_bench.py --check     # or: --dry-run
```

`--check` reports missing prerequisites and exits 2; `--dry-run` prints the
same checks plus the planned commands and exits 0, even if tools are missing.
Live runs perform the same preflight (including a server health probe) before
doing any work.

Machine-specific defaults are overridable with environment variables rather
than by editing the scripts: `K2H_PORT`, `K2H_HOST`, `K2H_HEALTH_HOST`,
`K2H_MODEL_PATH`, `K2H_SERVER_BIN`, `K2H_SERVER_ROOT`, `K2H_LOG_FILE`,
`K2H_PID_FILE`, `K2H_RUN_DIR`, `K2H_GATEWAY_IP`, `K2H_TB_DIR`, `K2H_TB_BIN`,
`K2H_TB_DATASET`, `K2H_NODE_BIN`, and `K2H_SERVER_URL` for the Python harnesses
(also `--url` / `--server-url`).

Run the offline tests with:

```sh
python3 -m py_compile bin/k2h_benchmark.py bin/k2h_terminal_bench.py tests/test_preflight.py
python3 -m unittest discover -s tests
```

## Repository links

- [Browse the default-branch files](https://github.com/scottjoyner/fleet-data/tree/main)
- [Review repository history](https://github.com/scottjoyner/fleet-data/commits/main)

These links refer to this checkout and do not constitute build or deployment verification.
