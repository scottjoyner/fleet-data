"""Contract tests for terminal-bench run artifacts and the aggregate report."""

import json

from contracts import (
    BIN, RUNS, expect_invalid, load_json, require, validate_latest_report,
    validate_pareto, validate_run_directory, validate_run_results, validate_run_summary,
)


def _aggregate_paths():
    candidates = sorted(RUNS.rglob("results.json"))
    return [path for path in candidates if isinstance(load_json(path).get("results"), list)]


def _first_aggregate_path():
    paths = _aggregate_paths()
    require(bool(paths), "no committed run aggregates found")
    return paths[0]


def test_committed_run_directories_are_self_consistent():
    paths = _aggregate_paths()
    require(len(paths) >= 2, f"expected at least 2 run aggregates, found {len(paths)}")
    for path in paths:
        validate_run_directory(path.parent)


def test_summaries_point_at_existing_run_trees():
    summaries = sorted(RUNS.rglob("summary.json"))
    require(bool(summaries), "no run summary found")
    for path in summaries:
        validate_run_summary(load_json(path))


def test_pareto_rows_match_run_results():
    for path in sorted(RUNS.rglob("pareto.json")):
        run_dir = path.parent
        results = load_json(run_dir / "results.json") if (run_dir / "results.json").is_file() else None
        validate_pareto(load_json(path), results)


def test_invalid_counts_and_ids_are_rejected():
    data = load_json(_first_aggregate_path())
    data["n_resolved"] += 1
    expect_invalid(validate_run_results, data)

    data = load_json(_first_aggregate_path())
    data["resolved_ids"] = data["resolved_ids"] + data["resolved_ids"][:1]
    expect_invalid(validate_run_results, data)

    data = load_json(_first_aggregate_path())
    data["accuracy"] = 0.5
    expect_invalid(validate_run_results, data)

    data = load_json(_first_aggregate_path())
    data["results"][0]["is_resolved"] = not data["results"][0]["is_resolved"]
    expect_invalid(validate_run_results, data)

    data = load_json(_first_aggregate_path())
    data["results"][0]["parser_results"] = {"test_one": "maybe"}
    expect_invalid(validate_run_results, data)


def test_missing_recording_is_detected(tmp_path):
    source = _first_aggregate_path().parent
    for item in source.rglob("*"):
        if item.is_file():
            target = tmp_path / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())
    aggregate = next((tmp_path / p.relative_to(source)) for p in source.rglob("results.json") if isinstance(load_json(p).get("results"), list))
    data = load_json(aggregate)
    data["results"][0]["recording_path"] = "does/not/exist.cast"
    aggregate.write_text(json.dumps(data), encoding="utf-8")
    expect_invalid(validate_run_directory, aggregate.parent)


def test_latest_report_matches_shell_constants_and_scripts():
    data = load_json(BIN / "k2h_benchmark_results_latest.json")
    shell_text = (BIN / "k2h-bench.sh").read_text(encoding="utf-8")
    validate_latest_report(data, shell_text)

    altered = json.loads(json.dumps(data))
    altered["server_config"]["port"] = 9999
    expect_invalid(validate_latest_report, altered, shell_text)

    altered = json.loads(json.dumps(data))
    altered["benchmark_results"]["terminal_bench_v4"]["hello_world"]["0.9b"]["accuracy"] = 0.5
    expect_invalid(validate_latest_report, altered, shell_text)

    altered = json.loads(json.dumps(data))
    altered["infrastructure"]["harness"] = "/somewhere/else.py"
    expect_invalid(validate_latest_report, altered, shell_text)
