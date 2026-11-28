"""Standard-library validators for the fleet-data data contracts.

Nothing here opens the network, imports the harness modules, or extracts model
archives. Validators raise ``ValueError`` with a diagnostic message so they can
be used both by the tests and by future ad-hoc checks.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
RUNS = BIN / "runs"
DOCS = ROOT / "docs"


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_number(value: Any) -> bool:
    return type(value) in (int, float)


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return is_number(actual) and abs(actual - expected) <= tolerance


def expect_invalid(check: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Assert that a validator rejects the given input."""
    try:
        check(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected {check.__name__} to reject the input")


# ---------------------------------------------------------------------------
# k2h_benchmark.py output (version 1.0.0)
# ---------------------------------------------------------------------------

BENCH_RESULT_FIELDS = {
    "name", "category", "success", "tokens", "prompt_tokens",
    "gen_speed", "prompt_speed", "time", "output", "error",
}
BENCH_SUMMARY_FIELDS = {
    "total_tasks", "passed", "success_rate", "avg_gen_speed",
    "avg_prompt_speed", "avg_tokens", "total_time", "categories",
}


def harness_tasks(source: Path | str | None = None) -> set[tuple[str, str]]:
    """Read K2H_TASKS from the harness source without importing/executing it."""
    text = Path(source or BIN / "k2h_benchmark.py").read_text(encoding="utf-8")
    found = None
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "K2H_TASKS"
            for target in node.targets
        ):
            found = node.value
    require(found is not None, "K2H_TASKS not found in k2h_benchmark.py")
    definition = ast.literal_eval(found)
    require(isinstance(definition, dict), "K2H_TASKS must be a mapping")
    pairs: set[tuple[str, str]] = set()
    for category, entries in definition.items():
        require(isinstance(category, str) and isinstance(entries, list), "bad K2H_TASKS shape")
        for entry in entries:
            require(isinstance(entry, dict) and isinstance(entry.get("name"), str), "bad task entry")
            pairs.add((category, entry["name"]))
    return pairs


def validate_benchmark_suite(data: Any, expected_tasks: set[tuple[str, str]] | None = None) -> None:
    require(isinstance(data, dict), "benchmark suite must be an object")
    require(data.get("version") == "1.0.0", "unsupported benchmark suite version")
    require(isinstance(data.get("model"), str) and data["model"], "missing model")
    datetime.fromisoformat(str(data.get("timestamp", "")))

    results = data.get("results")
    require(isinstance(results, list) and results, "results must be a non-empty list")
    for index, result in enumerate(results):
        label = f"results[{index}]"
        require(isinstance(result, dict) and BENCH_RESULT_FIELDS <= set(result), f"{label}: missing fields")
        require(isinstance(result["name"], str) and isinstance(result["category"], str), f"{label}: bad identifiers")
        require(type(result["success"]) is bool, f"{label}: success must be bool")
        for field in ("tokens", "prompt_tokens"):
            require(type(result[field]) is int and result[field] >= 0, f"{label}: bad {field}")
        for field in ("gen_speed", "prompt_speed", "time"):
            require(is_number(result[field]) and result[field] >= 0, f"{label}: bad {field}")
        for field in ("output", "error"):
            require(isinstance(result[field], str), f"{label}: bad {field}")

    pairs = [(result["category"], result["name"]) for result in results]
    require(len(set(pairs)) == len(pairs), "duplicate task in benchmark results")
    if expected_tasks is not None:
        require(set(pairs) == expected_tasks, "benchmark tasks differ from K2H_TASKS in the harness")

    summary = data.get("summary")
    require(isinstance(summary, dict) and BENCH_SUMMARY_FIELDS <= set(summary), "bad summary fields")
    total = len(results)
    passed = sum(result["success"] for result in results)
    averages = {
        "total_tasks": total,
        "passed": passed,
        "success_rate": f"{passed / total * 100:.1f}%",
        "avg_gen_speed": f"{sum(r['gen_speed'] for r in results) / total:.1f} tok/s",
        "avg_prompt_speed": f"{sum(r['prompt_speed'] for r in results) / total:.1f} tok/s",
        "avg_tokens": f"{sum(r['tokens'] for r in results) / total:.0f}",
        "total_time": f"{sum(r['time'] for r in results):.1f}s",
    }
    for key, expected in averages.items():
        require(summary[key] == expected, f"summary.{key} disagrees with results")

    categories = summary["categories"]
    require(isinstance(categories, dict) and set(categories) == {c for c, _ in pairs}, "summary.categories mismatch")
    for category, stats in categories.items():
        subset = [result for result in results if result["category"] == category]
        successes = sum(result["success"] for result in subset)
        expected = {
            "total": len(subset),
            "passed": successes,
            "rate": f"{successes / len(subset) * 100:.1f}%",
            "avg_gen_speed": f"{sum(r['gen_speed'] for r in subset) / len(subset):.1f} tok/s",
        }
        require(stats == expected, f"summary.categories[{category}] disagrees with results")


# ---------------------------------------------------------------------------
# terminal-bench run artifacts
# ---------------------------------------------------------------------------

TRIAL_FIELDS = {
    "id", "trial_name", "task_id", "instruction", "is_resolved", "failure_mode",
    "parser_results", "recording_path", "total_input_tokens", "total_output_tokens",
    "trial_started_at", "trial_ended_at", "agent_started_at", "agent_ended_at",
    "test_started_at", "test_ended_at",
}
RUN_FIELDS = {
    "id", "results", "pass_at_k", "n_resolved", "n_unresolved",
    "resolved_ids", "unresolved_ids", "accuracy",
}
METADATA_FIELDS = {"task_ids", "dataset_name", "dataset_version", "accuracy", "start_time", "end_time"}


def validate_trial(trial: Any) -> None:
    require(isinstance(trial, dict) and TRIAL_FIELDS <= set(trial), "trial: missing fields")
    for field in ("id", "trial_name", "task_id", "instruction", "failure_mode", "recording_path"):
        require(isinstance(trial[field], str) and trial[field], f"trial: bad {field}")
    require(type(trial["is_resolved"]) is bool, "trial: is_resolved must be bool")
    parsers = trial["parser_results"]
    require(isinstance(parsers, dict) and parsers, "trial: parser_results must be a non-empty object")
    for name, status in parsers.items():
        require(isinstance(name, str) and status in {"passed", "failed"}, "trial: bad parser result")
    if trial["is_resolved"]:
        require(all(status == "passed" for status in parsers.values()), "resolved trial has failed tests")
    for field in ("total_input_tokens", "total_output_tokens"):
        require(type(trial[field]) is int and trial[field] >= 0, f"trial: bad {field}")
    for field in ("trial_started_at", "trial_ended_at", "agent_started_at", "agent_ended_at", "test_started_at", "test_ended_at"):
        datetime.fromisoformat(str(trial[field]))


def validate_run_results(data: Any) -> dict:
    require(isinstance(data, dict) and RUN_FIELDS <= set(data), "run results: missing fields")
    trials = data["results"]
    require(isinstance(trials, list) and trials, "run results: no trials")
    for trial in trials:
        validate_trial(trial)

    task_ids = [trial["task_id"] for trial in trials]
    require(len(set(task_ids)) == len(task_ids), "run results: duplicate task_id")
    resolved = {tid for tid, trial in zip(task_ids, trials) if trial["is_resolved"]}
    unresolved = set(task_ids) - resolved
    for key, expected in (("resolved_ids", resolved), ("unresolved_ids", unresolved)):
        ids = data[key]
        require(isinstance(ids, list) and len(ids) == len(set(ids)), f"run results: bad {key}")
        require(set(ids) == expected, f"run results: {key} disagrees with trials")
    require(data["n_resolved"] == len(resolved), "run results: bad n_resolved")
    require(data["n_unresolved"] == len(unresolved), "run results: bad n_unresolved")
    require(close(data["accuracy"], len(resolved) / len(trials)), "run results: bad accuracy")
    require(isinstance(data["pass_at_k"], dict), "run results: bad pass_at_k")
    return data


def validate_pareto(data: Any, run_results: dict | None = None, base: Path | None = BIN) -> None:
    fields = {"run_dir", "results_json", "accuracy_official", "n_tasks", "n_resolved",
              "accuracy", "total_usd", "mean_usd_per_task", "usd_per_resolved", "rows"}
    require(isinstance(data, dict) and fields <= set(data), "pareto: missing fields")
    rows = data["rows"]
    require(isinstance(rows, list) and rows, "pareto: no rows")
    for row in rows:
        require(isinstance(row, dict) and {"task_id", "resolved", "usd", "input_tokens", "output_tokens", "steps"} <= set(row), "pareto: bad row")
        require(isinstance(row["task_id"], str) and type(row["resolved"]) is bool, "pareto: bad row identifiers")
        for field in ("usd", "input_tokens", "output_tokens", "steps"):
            require(is_number(row[field]) and row[field] >= 0, f"pareto: bad row {field}")
    ids = [row["task_id"] for row in rows]
    require(len(set(ids)) == len(ids), "pareto: duplicate task_id")
    successes = sum(row["resolved"] for row in rows)
    require(data["n_tasks"] == len(rows), "pareto: bad n_tasks")
    require(data["n_resolved"] == successes, "pareto: bad n_resolved")
    require(close(data["accuracy"], successes / len(rows)), "pareto: bad accuracy")
    require(close(data["accuracy_official"], data["accuracy"]), "pareto: accuracy_official disagrees")
    require(close(data["mean_usd_per_task"], data["total_usd"] / len(rows)), "pareto: bad mean_usd_per_task")
    require((data["usd_per_resolved"] is None) == (successes == 0), "pareto: bad usd_per_resolved")
    if run_results is not None:
        expected = {trial["task_id"]: trial["is_resolved"] for trial in run_results["results"]}
        require(set(ids) == set(expected), "pareto: task ids disagree with run results")
        require(all(row["resolved"] == expected[row["task_id"]] for row in rows), "pareto: resolution disagrees with run results")
    if base is not None:
        # Relative results_json paths are relative to bin/ (the harness cwd),
        # while absolute paths point outside this checkout by design.
        reference = Path(data["results_json"])
        if not reference.is_absolute():
            require((base / reference).is_file(), f"pareto: missing referenced results {reference}")


def validate_run_directory(path: Path) -> dict:
    """Validate an aggregate run dir, its trial dirs, and any local pareto file."""
    path = Path(path)
    results = validate_run_results(load_json(path / "results.json"))
    metadata = load_json(path / "run_metadata.json")
    require(isinstance(metadata, dict) and METADATA_FIELDS <= set(metadata), "run_metadata: missing fields")
    task_ids = [trial["task_id"] for trial in results["results"]]
    require(isinstance(metadata["task_ids"], list) and set(metadata["task_ids"]) == set(task_ids), "run_metadata: task_ids disagree with results")
    require(metadata["dataset_name"] == "terminal-bench-core", "run_metadata: unexpected dataset")
    require(metadata["dataset_version"] == "0.1.1", "run_metadata: unexpected dataset version")
    require(close(metadata["accuracy"], results["accuracy"]), "run_metadata: accuracy disagrees with results")
    require(datetime.fromisoformat(metadata["start_time"]) <= datetime.fromisoformat(metadata["end_time"]), "run_metadata: inverted time range")

    for trial in results["results"]:
        trial_dir = path / trial["task_id"] / trial["trial_name"]
        require(trial_dir.is_dir(), f"missing trial directory {trial_dir}")
        trial_data = load_json(trial_dir / "results.json")
        require(trial_data.get("id") == trial["id"], f"trial id mismatch in {trial_dir}")
        recording = path.parent / trial["recording_path"]
        require(recording.is_file(), f"missing referenced recording {recording}")

    pareto = path / "pareto.json"
    if pareto.is_file():
        validate_pareto(load_json(pareto), results)
    return results


def validate_run_summary(data: Any, bin_dir: Path = BIN) -> None:
    required = {"model", "server", "base_url", "results_dir", "timestamp", "band", "n_tasks"}
    require(isinstance(data, dict) and required <= set(data), "run summary: missing fields")
    require(data["base_url"] == data["server"].rstrip("/") + "/v1", "run summary: base_url disagrees with server")
    require(type(data["n_tasks"]) is int and data["n_tasks"] > 0, "run summary: bad n_tasks")
    datetime.fromisoformat(str(data["timestamp"]))
    require((Path(bin_dir) / data["results_dir"]).is_dir(), "run summary: results_dir does not exist")


# ---------------------------------------------------------------------------
# Aggregate benchmark report (version 2.0.0)
# ---------------------------------------------------------------------------

def shell_constants(text: str) -> dict[str, str | int]:
    pattern = {"PORT": r"^PORT=(\d+)", "HOST": r'^HOST="([^"]+)"', "GATEWAY_IP": r'^GATEWAY_IP="([^"]+)"'}
    values: dict[str, str | int] = {}
    for key, regex in pattern.items():
        match = re.search(regex, text, re.MULTILINE)
        require(match is not None, f"k2h-bench.sh: missing {key}")
        values[key] = int(match.group(1)) if key == "PORT" else match.group(1)
    return values


def validate_latest_report(data: Any, shell_text: str | None = None, bin_dir: Path = BIN) -> None:
    required = {"version", "timestamp", "model", "server_config", "models_tested", "benchmark_results", "findings", "infrastructure"}
    require(isinstance(data, dict) and required <= set(data), "latest report: missing fields")
    require(data["version"] == "2.0.0" and data["model"] == "k2-horizon", "latest report: bad version/model")
    datetime.fromisoformat(str(data["timestamp"]))

    config = data["server_config"]
    require(isinstance(config, dict) and {"host", "port", "gateway_ip", "container_url", "context_size"} <= set(config), "latest report: bad server_config")
    require(config["container_url"] == f"http://{config['gateway_ip']}:{config['port']}/v1", "latest report: container_url mismatch")
    if shell_text is not None:
        constants = shell_constants(shell_text)
        for key in ("PORT", "HOST", "GATEWAY_IP"):
            require(config[key.lower()] == constants[key], f"latest report: server_config {key.lower()} differs from k2h-bench.sh")

    models = data["models_tested"]
    require(isinstance(models, dict) and set(models) == {"k2-horizon-0.9b", "k2-horizon-3.7b"}, "latest report: unexpected models_tested")
    for model in models.values():
        require(isinstance(model, dict) and isinstance(model.get("model_path"), str) and model["model_path"].endswith(".gguf"), "latest report: bad model_path")
        require(isinstance(model.get("parameters"), str) and model["parameters"], "latest report: bad parameters")

    results = data["benchmark_results"]
    require(isinstance(results, dict) and set(results) == {"terminal_bench_v4", "completion_benchmark"}, "latest report: unexpected benchmark_results")
    tasks = results["terminal_bench_v4"]
    require(isinstance(tasks, dict) and tasks, "latest report: no terminal-bench tasks")
    for by_model in tasks.values():
        require(isinstance(by_model, dict) and set(by_model) == {"0.9b", "3.7b"}, "latest report: task must cover both models")
        for entry in by_model.values():
            require(isinstance(entry, dict) and {"accuracy", "resolved", "total"} <= set(entry), "latest report: bad task entry")
            require(type(entry["resolved"]) is int and type(entry["total"]) is int and 0 <= entry["resolved"] <= entry["total"], "latest report: bad resolved/total")
            require(close(entry["accuracy"], entry["resolved"] / entry["total"]), "latest report: bad accuracy")
    completion = results["completion_benchmark"]
    for key in ("gen_speed_tok_s", "prompt_speed_tok_s"):
        require(is_number(completion.get(key)) and completion[key] > 0, f"latest report: bad {key}")

    findings = data["findings"]
    require(isinstance(findings, dict) and isinstance(findings.get("next_steps"), list), "latest report: bad findings")
    infra = data["infrastructure"]
    require(isinstance(infra, dict), "latest report: bad infrastructure")
    for key in ("harness", "bash_harness"):
        target = bin_dir / Path(infra.get(key, "")).name
        require(target.is_file(), f"latest report: referenced script {key} is missing from bin/")


# ---------------------------------------------------------------------------
# Optional, gitignored artifacts (validated only when present)
# ---------------------------------------------------------------------------

def validate_node_table(path: Path | str, delimiter: str = "\t") -> int:
    """Check a generic matrix/table: header, rectangular rows, unique node ids."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    require(bool(rows), "empty table")
    header = [cell.strip() for cell in rows[0]]
    require(all(header), "empty column name")
    require(len(header) == len(set(header)), "duplicate column name")
    seen: set[str] = set()
    for line, row in enumerate(rows[1:], start=2):
        require(len(row) == len(header), f"line {line}: expected {len(header)} columns, got {len(row)}")
        require(all(cell.strip() for cell in row), f"line {line}: empty cell")
        node = row[0].strip()
        require(node not in seen, f"line {line}: duplicate node id {node}")
        seen.add(node)
    return len(rows) - 1


def validate_tar_headers(path: Path | str, *, max_members: int = 200,
                         max_member_bytes: int = 1 << 30,
                         max_total_bytes: int = 1 << 31) -> int:
    """Read archive headers only; never extract. Reject unsafe member shape."""
    count = 0
    total = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            count += 1
            require(count <= max_members, f"too many members (> {max_members})")
            require(member.isfile() or member.isdir(), f"unsupported member type: {member.name}")
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and ".." not in name.parts, f"unsafe member path: {member.name}")
            if member.isfile():
                require(member.size <= max_member_bytes, f"member too large: {member.name}")
                total += member.size
                require(total <= max_total_bytes, f"archive payload too large: {path}")
    return count


def markdown_links(text: str) -> list[str]:
    return re.findall(r"\]\(([^)]+)\)", text)
