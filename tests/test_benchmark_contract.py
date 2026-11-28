"""Contract tests for k2h_benchmark.py result files (version 1.0.0)."""

import json

from contracts import BIN, expect_invalid, harness_tasks, load_json, require, validate_benchmark_suite


def _suite(name="k2h_benchmark_k2-horizon_20260916_113407.json"):
    return load_json(BIN / name)


def test_committed_suites_cover_exactly_the_harness_tasks():
    expected = harness_tasks()
    require(bool(expected), "no tasks parsed from harness")
    paths = sorted(BIN.glob("k2h_benchmark_k2-horizon_*.json"))
    require(bool(paths), "no committed benchmark suites found")
    for path in paths:
        validate_benchmark_suite(load_json(path), expected)


def test_summary_must_agree_with_results(tmp_path):
    data = _suite()
    data["summary"]["passed"] += 1
    expect_invalid(validate_benchmark_suite, data)

    data = _suite()
    data["summary"]["categories"]["reasoning"]["total"] = 2
    expect_invalid(validate_benchmark_suite, data)

    data = _suite()
    data["summary"]["avg_gen_speed"] = "999.0 tok/s"
    expect_invalid(validate_benchmark_suite, data)

    saved = tmp_path / "bad-suite.json"
    saved.write_text(json.dumps(data), encoding="utf-8")
    expect_invalid(validate_benchmark_suite, load_json(saved))


def test_duplicate_and_unknown_tasks_are_rejected():
    data = _suite()
    data["results"].append(dict(data["results"][0]))
    expect_invalid(validate_benchmark_suite, data)

    data = _suite()
    data["results"][0]["name"] = "not_a_harness_task"
    expect_invalid(validate_benchmark_suite, data, harness_tasks())


def test_result_field_types_are_enforced():
    data = _suite()
    data["results"][0]["tokens"] = True
    expect_invalid(validate_benchmark_suite, data)

    data = _suite()
    del data["results"][0]["gen_speed"]
    expect_invalid(validate_benchmark_suite, data)
