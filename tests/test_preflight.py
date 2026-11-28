"""Fast, offline tests for benchmark preflight and error paths.

No llama-server, Docker, downloads, installs or real network calls are used:
network and filesystem lookups are injected with small fakes.
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bm = load_module("k2h_benchmark", BIN / "k2h_benchmark.py")
tb = load_module("k2h_terminal_bench", BIN / "k2h_terminal_bench.py")


class FakeResponse:
    def __init__(self, body):
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class BenchmarkParseArgsTests(unittest.TestCase):
    def test_legacy_positional_arguments_still_work(self):
        config = bm.parse_args(["8888", "k2-horizon"])
        self.assertEqual(config.port, 8888)
        self.assertEqual(config.model_name, "k2-horizon")
        self.assertFalse(config.check)
        self.assertFalse(config.dry_run)

    def test_defaults_match_historical_behavior(self):
        config = bm.parse_args([])
        self.assertEqual((config.port, config.model_name), (8888, "k2-horizon"))

    def test_flags_and_equals_form(self):
        config = bm.parse_args(["--port", "1234", "--model=other", "--url=http://example:9/"])
        self.assertEqual(config.port, 1234)
        self.assertEqual(config.model_name, "other")
        self.assertEqual(config.url, "http://example:9/")

    def test_bad_arguments_raise_actionable_errors(self):
        for argv, fragment in [
            (["--port", "abc"], "invalid port"),
            (["--port=70000"], "invalid port"),
            (["--model"], "requires a value"),
            (["--bogus"], "unknown option"),
            (["8888", "m", "extra"], "unexpected argument"),
        ]:
            with self.subTest(argv=argv), self.assertRaises(ValueError) as ctx:
                bm.parse_args(argv)
            self.assertIn(fragment, str(ctx.exception))


class BenchmarkPreflightTests(unittest.TestCase):
    def test_url_and_port_helpers(self):
        self.assertIsNone(bm.validate_port(8888))
        self.assertIn("out of range", bm.validate_port(0))
        self.assertIn("must be an integer", bm.validate_port("8888"))
        self.assertEqual(bm.resolve_base_url(1234, env={}), "http://localhost:1234")
        self.assertEqual(
            bm.resolve_base_url(1234, env={"K2H_SERVER_URL": "http://gpu:9999/"}),
            "http://gpu:9999",
        )
        self.assertEqual(bm.resolve_base_url(1234, "http://cli:1",
                                             {"K2H_SERVER_URL": "http://env:2"}),
                         "http://cli:1")
        self.assertIn("http://", bm.validate_base_url("ftp://host") or "")
        self.assertIsNone(bm.validate_base_url("http://localhost:8888"))

    def test_config_preflight_is_all_ok_with_defaults(self):
        checks = bm.preflight_config(bm.CliConfig(), env={})
        self.assertEqual([c.name for c in checks],
                         ["port", "model name", "server URL", "benchmark tasks"])
        self.assertTrue(all(c.ok for c in checks), bm.format_checks(checks))
        self.assertEqual(bm.fatal_checks(checks), [])

    def test_config_preflight_reports_failures(self):
        config = bm.CliConfig(port=-1, model_name="", url="ftp://bad", check=True)
        checks = bm.preflight_config(config, env={})
        failures = bm.fatal_checks(checks)
        self.assertEqual(len(failures), 3)
        self.assertTrue(any("port out of range" in c.message for c in failures))
        self.assertTrue(any("model name is empty" in c.message for c in failures))
        self.assertTrue(any("http://" in c.message for c in failures))
        self.assertIn("[FAIL]", bm.format_checks(checks))

    def test_server_check_messages(self):
        healthy = bm.check_server("http://localhost:8888",
                                  open_url=lambda url, timeout: FakeResponse('{"status":"ok"}'))
        self.assertTrue(healthy.ok)
        self.assertIn("localhost:8888", healthy.message)

        def unreachable(url, timeout):
            raise urllib.error.URLError("connection refused")

        down = bm.check_server("http://localhost:8888", open_url=unreachable)
        self.assertFalse(down.ok)
        self.assertIn("llama-server not reachable at localhost:8888", down.message)
        self.assertIn("bin/k2h-bench.sh start", down.message)

        def missing_health(url, timeout):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

        self.assertTrue(bm.check_server("http://localhost:8888",
                                        open_url=missing_health).ok)
        unexpected = bm.check_server("http://localhost:8888",
                                     open_url=lambda url, timeout: FakeResponse("garbage"))
        self.assertFalse(unexpected.ok)

    def test_dry_run_plan_is_offline_and_describes_urls(self):
        plan = bm.build_dry_run_plan(bm.CliConfig(), "http://localhost:8888")
        self.assertEqual(plan["chat_url"], "http://localhost:8888/v1/chat/completions")
        self.assertEqual(plan["completion_url"], "http://localhost:8888/completion")
        self.assertEqual(plan["task_count"], sum(len(v) for v in bm.K2H_TASKS.values()))
        self.assertFalse(plan["external_services_checked"])


class BenchmarkMainTests(unittest.TestCase):
    def test_check_and_dry_run_return_zero_without_network(self):
        with mock.patch.object(bm.urllib.request, "urlopen",
                               side_effect=AssertionError("network call in check mode")):
            for argv in (["--check"], ["--dry-run"]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(bm.main(argv), 0)
                self.assertIn("preflight", out.getvalue())
                if argv == ["--dry-run"]:
                    self.assertIn("chat_url", out.getvalue())

    def test_unreachable_server_aborts_before_writing_results(self):
        down = bm.CheckResult("llama-server", False,
                              "llama-server not reachable at localhost:8888; start it")
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(bm, "check_server", return_value=down), \
             mock.patch.object(os, "getcwd", return_value=tmpdir):
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(bm.main([]), 2)
            self.assertEqual(list(Path(tmpdir).glob("*.json")), [])

    def test_parse_error_exit_code(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bm.main(["--port", "nope"]), 2)


class TerminalParseArgsTests(unittest.TestCase):
    def test_legacy_positional_arguments_still_work(self):
        config = tb.parse_args(["hard", "3", "my-model"])
        self.assertEqual((config.band, config.n_tasks, config.model_name),
                         ("hard", 3, "my-model"))
        self.assertFalse(config.check)

    def test_flags_and_env_precedence(self):
        config = tb.parse_args(["--server-url=http://gpu:9999", "--tb-bin=/tmp/fake-tb",
                                "--n-tasks=2"])
        resolved = tb.resolve_config(config, {"K2H_SERVER_URL": "http://env:1",
                                              "K2H_TB_BIN": "/env/tb",
                                              "K2H_TB_DIR": "/env/tbdir"})
        self.assertEqual(resolved.server_url, "http://gpu:9999")
        self.assertEqual(resolved.tb_bin, "/tmp/fake-tb")
        self.assertEqual(resolved.n_tasks, 2)
        self.assertEqual(resolved.tb_dir, "/env/tbdir")

    def test_bad_arguments(self):
        for argv, fragment in [
            (["--n-tasks", "zero"], "invalid --n-tasks"),
            (["hard", "three"], "invalid n_tasks"),
            (["a", "1", "m", "extra"], "unexpected argument"),
            (["--unknown"], "unknown option"),
            (["--tb-dir"], "requires a value"),
        ]:
            with self.subTest(argv=argv), self.assertRaises(ValueError) as ctx:
                tb.parse_args(argv)
            self.assertIn(fragment, str(ctx.exception))

    def test_container_url(self):
        self.assertEqual(tb.container_base_url("http://127.0.0.1:8888", "172.17.0.1"),
                         "http://172.17.0.1:8888/v1")
        self.assertEqual(tb.container_base_url("http://localhost", "10.0.0.1"),
                         "http://10.0.0.1:8888/v1")


class TerminalPreflightTests(unittest.TestCase):
    def config(self, **kwargs):
        defaults = dict(tb_dir="/fake/tb", tb_bin="/fake/tb-bin/node/bin/tb",
                        server_url="http://127.0.0.1:8888", n_tasks=6)
        defaults.update(kwargs)
        return tb.TerminalCliConfig(**defaults)

    def test_all_prerequisites_present(self):
        existing = {"/fake/tb", "/fake/tb-bin/node/bin/tb"}
        existing |= {f"/fake/tb/{s}" for s in tb.NODE_SCRIPTS}
        existing |= {str(Path.home() / ".cache" / "terminal-bench" /
                         "terminal-bench-core" / "0.1.1")}
        checks = tb.preflight_config(
            self.config(),
            exists=existing.__contains__,
            find_bin=lambda name: {"node": "/usr/bin/node", "docker": "/usr/bin/docker"}.get(name),
        )
        self.assertEqual(tb.fatal_checks(checks), [], tb.format_checks(checks))
        self.assertTrue(all(c.ok for c in checks))

    def test_missing_tb_cli_has_install_hint(self):
        checks = tb.preflight_config(
            self.config(), exists=lambda path: path != "/fake/tb-bin/node/bin/tb",
            find_bin=lambda name: None if name == "tb" else "/usr/bin/" + name)
        failures = tb.fatal_checks(checks)
        messages = "\n".join(c.message for c in failures)
        self.assertIn("tb CLI not found: /fake/tb-bin/node/bin/tb", messages)
        self.assertIn("uv tool install terminal-bench", messages)

    def test_missing_node_scripts_and_docker_are_reported(self):
        checks = tb.preflight_config(
            self.config(),
            exists=lambda path: False,
            find_bin=lambda name: "/usr/bin/node" if name == "node" else None,
        )
        messages = "\n".join(c.message for c in checks)
        self.assertIn("node script missing: /fake/tb/hardest-first.mjs", messages)
        self.assertIn("docker CLI not found", messages)
        self.assertIn("terminal-bench directory not found", messages)
        self.assertTrue(any("dataset cache missing" in c.message
                            and c.severity == "warning" for c in checks))
        self.assertFalse(any(c.severity == "warning" for c in tb.fatal_checks(checks)))

    def test_invalid_task_count_and_url(self):
        checks = tb.preflight_config(self.config(n_tasks=0, server_url="ftp://x"),
                                     exists=lambda path: True,
                                     find_bin=lambda name: "/bin/" + name)
        messages = "\n".join(c.message for c in tb.fatal_checks(checks))
        self.assertIn("n_tasks must be a positive integer", messages)
        self.assertIn("server URL must start with http://", messages)

    def test_server_check_message(self):
        def unreachable(url, timeout):
            raise OSError("refused")
        result = tb.check_server("http://127.0.0.1:8888", open_url=unreachable)
        self.assertFalse(result.ok)
        self.assertIn("llama-server not reachable at 127.0.0.1:8888", result.message)
        self.assertIn("bin/k2h-bench.sh start", result.message)

    def test_dry_run_plan_lists_commands_without_running_them(self):
        plan = tb.build_dry_run_plan(self.config(), "http://172.17.0.1:8888/v1")
        self.assertFalse(plan["external_services_checked"])
        self.assertTrue(any("hardest-first.mjs" in cmd for cmd in plan["commands"]))
        self.assertTrue(any("datasets download" in cmd for cmd in plan["commands"]))


class TerminalRunnerTests(unittest.TestCase):
    def test_dry_run_does_not_create_dirs_or_spawn_processes(self):
        config = tb.TerminalCliConfig(tb_dir="/does/not/exist", tb_bin="/does/not/exist/tb")
        runner = tb.TerminalBenchRunner(config=config)
        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(runner, "results_dir", Path(tmpdir) / "runs" / "new-run"), \
             mock.patch.object(tb.subprocess, "run",
                               side_effect=AssertionError("subprocess in dry-run")):
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertTrue(runner.run_full_benchmark(dry_run=True))
            finally:
                os.chdir(old_cwd)
            self.assertFalse((Path(tmpdir) / "runs").exists())
        self.assertIn("dry-run", out.getvalue())

    def test_check_dependencies_never_installs(self):
        config = tb.TerminalCliConfig()
        runner = tb.TerminalBenchRunner(config=config)
        failing = [tb.CheckResult("tb CLI", False, "tb CLI not found: /x; uv tool install terminal-bench")]
        with mock.patch.object(tb, "preflight_config", return_value=failing), \
             mock.patch.object(tb.subprocess, "run",
                               side_effect=AssertionError("install attempted")), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(runner.check_dependencies())


class TerminalMainTests(unittest.TestCase):
    def test_check_and_dry_run_return_zero_with_offline_config(self):
        checks = [tb.CheckResult("config", True, "ok")]
        with mock.patch.object(tb, "preflight_config", return_value=checks), \
             mock.patch.object(tb.urllib.request, "urlopen",
                               side_effect=AssertionError("network call in check mode")):
            for argv in (["--check"], ["--dry-run"]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(tb.main(argv), 0)
                self.assertIn("preflight", out.getvalue())
                if argv == ["--dry-run"]:
                    self.assertIn("hardest-first.mjs", out.getvalue())

    def test_failed_preflight_returns_two(self):
        checks = [tb.CheckResult("tb CLI", False,
                                 "tb CLI not found; uv tool install terminal-bench")]
        with mock.patch.object(tb, "preflight_config", return_value=checks), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(tb.main(["--check"]), 2)


class ShellScriptTests(unittest.TestCase):
    SCRIPT = BIN / "k2h-bench.sh"

    def run_script(self, args, **env_overrides):
        env = {**os.environ, **env_overrides}
        return subprocess.run([str(self.SCRIPT), *args], env=env, text=True,
                              capture_output=True, timeout=30)

    def test_shell_syntax(self):
        result = subprocess.run(["bash", "-n", str(self.SCRIPT)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_check_reports_missing_tools_without_side_effects(self):
        result = self.run_script(
            ["--check"],
            K2H_PORT="not-a-port",
            K2H_SERVER_BIN="/definitely/missing/llama-server",
            K2H_MODEL_PATH="/definitely/missing/model.gguf",
            K2H_TB_DIR="/definitely/missing/tb",
            K2H_TB_BIN="/definitely/missing/tb-cli",
            K2H_NODE_BIN="definitely-missing-node-binary",
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        output = result.stdout + result.stderr
        for fragment in ("invalid port", "llama-server binary not found",
                         "model file not found", "terminal-bench directory not found",
                         "node script missing", "tb CLI not found",
                         "uv tool install terminal-bench", "not found"):
            self.assertIn(fragment, output)

    def test_dry_run_never_executes_actions(self):
        result = self.run_script(
            ["--dry-run", "bench"],
            K2H_SERVER_BIN="/definitely/missing/llama-server",
            K2H_MODEL_PATH="/definitely/missing/model.gguf",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[dry-run]", result.stdout)
        self.assertIn("/completion", result.stdout)
        self.assertIn("no files written", result.stdout)

    def test_unknown_action_and_option(self):
        self.assertEqual(self.run_script(["nope"]).returncode, 2)
        self.assertEqual(self.run_script(["--bogus"]).returncode, 2)


class JsonValidationTests(unittest.TestCase):
    def test_repository_json_files_parse(self):
        files = sorted(p for p in ROOT.rglob("*.json") if ".git" not in p.parts)
        self.assertTrue(files)
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                with path.open(encoding="utf-8") as handle:
                    json.load(handle)


if __name__ == "__main__":
    unittest.main()
