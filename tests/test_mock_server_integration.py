"""Offline integration tests backed by a local mock llama-server.

The server is a standard-library ``http.server.ThreadingHTTPServer`` bound to
an ephemeral loopback port in a background thread. It speaks only the small
subset of the llama-server API used by the harnesses: ``GET /health`` and
``POST /v1/chat/completions`` (plus ``POST /completion`` for completeness).

No Docker, ``tb`` CLI, model downloads, or external network access is needed.
"""

import contextlib
import http.server
import importlib.util
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bm = load_module("k2h_benchmark_mock", BIN / "k2h_benchmark.py")
tb = load_module("k2h_terminal_bench_mock", BIN / "k2h_terminal_bench.py")

CANNED_CONTENT = ("The mock model replied with a deterministic answer that is "
                  "long enough to exercise the harness output truncation: " + "x" * 200)
CANNED_USAGE = {"prompt_tokens": 17, "completion_tokens": 23, "total_tokens": 40}
CANNED_CHAT_RESPONSE = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "model": "k2-horizon",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": CANNED_CONTENT},
        "finish_reason": "stop",
    }],
    "usage": CANNED_USAGE,
}


class ServerState:
    """Mutable behavior/capture shared with the request handler."""

    def __init__(self, **overrides):
        self.health_status = 200
        self.health_body = '{"status":"ok"}'
        self.health_delay = 0.0
        self.chat_status = 200
        self.chat_body = None  # None -> canned chat response
        self.chat_delay = 0.0
        self.unknown_status = 404
        self.requests = []
        self._lock = threading.Lock()
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise KeyError(f"unknown mock server option: {key}")
            setattr(self, key, value)

    def record(self, method, path, payload=None):
        with self._lock:
            self.requests.append({"method": method, "path": path, "body": payload})

    @property
    def paths(self):
        return [entry["path"] for entry in self.requests]


def make_handler(state):
    class MockLlamaHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass  # keep test output quiet

        def handle_error(self, request, client_address):
            pass  # late writes to clients that timed out are expected

        def _reply(self, status, payload):
            if isinstance(payload, (dict, list)):
                body = json.dumps(payload).encode("utf-8")
                content_type = "application/json"
            else:
                body = str(payload).encode("utf-8")
                content_type = "text/plain"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            state.record("GET", path)
            if path != "/health":
                self._reply(state.unknown_status, {"error": "not found"})
                return
            if state.health_delay:
                time.sleep(state.health_delay)
            self._reply(state.health_status, state.health_body)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else None
            except ValueError:
                payload = raw.decode("utf-8", errors="replace")
            state.record("POST", path, payload)
            if path not in ("/v1/chat/completions", "/completion"):
                self._reply(state.unknown_status, {"error": "not found"})
                return
            if state.chat_delay:
                time.sleep(state.chat_delay)
            body = CANNED_CHAT_RESPONSE if state.chat_body is None else state.chat_body
            self._reply(state.chat_status, body)

    return MockLlamaHandler


class MockLlamaServer:
    """Context manager for an ephemeral, loopback-only mock llama-server."""

    def __init__(self, **state_overrides):
        self.state = ServerState(**state_overrides)
        self.server = None
        self.thread = None
        self.base_url = None

    def __enter__(self):
        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.state))
        self.server.daemon_threads = True
        host, port = self.server.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="mock-llama-server",
            daemon=True,
        )
        self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        return False


class MockHealthCheckTests(unittest.TestCase):
    def test_benchmark_health_check_succeeds_against_mock(self):
        with MockLlamaServer() as server:
            result = bm.check_server(server.base_url, timeout=2)
        self.assertTrue(result.ok, result.message)
        self.assertIn("reachable at", result.message)
        self.assertIn(server.base_url.split("//")[1], result.message)
        self.assertEqual(server.state.paths, ["/health"])

    def test_terminal_bench_health_check_succeeds_against_mock(self):
        with MockLlamaServer() as server:
            result = tb.check_server(server.base_url, timeout=2)
            port = urllib.parse.urlparse(server.base_url).port
            container_url = tb.container_base_url(server.base_url, "172.17.0.1")
        self.assertTrue(result.ok, result.message)
        self.assertIn("reachable at", result.message)
        self.assertEqual(container_url, f"http://172.17.0.1:{port}/v1")
        self.assertEqual(server.state.paths, ["/health"])

    def test_missing_health_endpoint_is_not_fatal(self):
        with MockLlamaServer(health_status=404) as server:
            result = bm.check_server(server.base_url, timeout=2)
        self.assertTrue(result.ok, result.message)
        self.assertIn("no /health endpoint", result.message)

    def test_unexpected_health_body_is_actionable_failure(self):
        with MockLlamaServer(health_body="not healthy") as server:
            result = bm.check_server(server.base_url, timeout=2)
        self.assertFalse(result.ok)
        self.assertIn("unexpected response", result.message)
        self.assertIn("bin/k2h-bench.sh start", result.message)
        self.assertIn("K2H_SERVER_URL", result.message)

    def test_http_error_from_health_is_actionable_failure(self):
        for module in (bm, tb):
            with self.subTest(module=module.__name__), MockLlamaServer(health_status=503) as server:
                result = module.check_server(server.base_url, timeout=2)
            self.assertFalse(result.ok)
            self.assertIn("HTTP 503", result.message)
            self.assertIn("not reachable at", result.message)
            self.assertIn("bin/k2h-bench.sh start", result.message)
            self.assertIn("K2H_SERVER_URL", result.message)

    def test_health_timeout_is_actionable_failure(self):
        with MockLlamaServer(health_delay=0.3) as server:
            result = bm.check_server(server.base_url, timeout=0.05)
        self.assertFalse(result.ok)
        self.assertIn("not reachable at", result.message)
        self.assertIn("bin/k2h-bench.sh start", result.message)
        self.assertTrue(any(word in result.message.lower()
                            for word in ("timed out", "timeout")), result.message)


class BenchmarkMockRunTests(unittest.TestCase):
    def test_run_task_parses_canned_chat_response(self):
        with MockLlamaServer() as server:
            runner = bm.K2HBenchmarkRunner(0, "k2-horizon", base_url=server.base_url)
            task = {"name": "canned", "prompt": "answer hello", "expect": "text"}
            result = runner.run_task(task, "reasoning")
        self.assertTrue(result.success, result.error)
        self.assertEqual((result.name, result.category), ("canned", "reasoning"))
        self.assertEqual(result.tokens, CANNED_USAGE["completion_tokens"])
        self.assertEqual(result.prompt_tokens, runner._count_tokens(task["prompt"]))
        self.assertEqual(result.output, CANNED_CONTENT[:200])
        self.assertGreater(result.gen_speed, 0)
        self.assertGreater(result.prompt_speed, 0)
        self.assertGreater(result.time, 0)
        self.assertEqual(server.state.paths, ["/v1/chat/completions"])
        request = server.state.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["body"]["model"], "k2-horizon")
        self.assertEqual(request["body"]["messages"][0]["content"], "answer hello")

    def test_chat_http_error_yields_failed_result_with_server_error(self):
        with MockLlamaServer(chat_status=500, chat_body={"error": "overloaded"}) as server:
            runner = bm.K2HBenchmarkRunner(0, "k2-horizon", base_url=server.base_url)
            result = runner.run_task({"name": "canned", "prompt": "hello"}, "reasoning")
        self.assertFalse(result.success)
        self.assertEqual(result.tokens, 0)
        self.assertEqual(result.output, "")
        self.assertIn("500", result.error)

    def test_full_main_run_writes_parseable_results(self):
        with MockLlamaServer() as server, tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            out = io.StringIO()
            try:
                os.chdir(tmpdir)
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                    rc = bm.main(["--url", server.base_url, "--model", "mock-model"])
                files = sorted(Path(tmpdir).glob("k2h_benchmark_mock-model_*.json"))
                self.assertEqual(rc, 0)
                self.assertEqual(len(files), 1, out.getvalue())
                suite = json.loads(files[0].read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)
        expected_tasks = sum(len(tasks) for tasks in bm.K2H_TASKS.values())
        self.assertEqual(suite["model"], "mock-model")
        self.assertEqual(suite["summary"]["total_tasks"], expected_tasks)
        self.assertEqual(suite["summary"]["passed"], expected_tasks)
        self.assertEqual(suite["summary"]["success_rate"], "100.0%")
        self.assertEqual(len(suite["results"]), expected_tasks)
        self.assertEqual(suite["results"][0]["output"], CANNED_CONTENT[:200])
        self.assertIn("K2-HORIZON BENCHMARK RESULTS", out.getvalue())
        self.assertEqual(server.state.paths.count("/health"), 1)
        self.assertEqual(server.state.paths.count("/v1/chat/completions"), expected_tasks)

    def test_main_aborts_on_unhealthy_server_without_chat_requests(self):
        with MockLlamaServer(health_status=503) as server, \
             tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            err = io.StringIO()
            try:
                os.chdir(tmpdir)
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(err):
                    rc = bm.main(["--url", server.base_url])
                self.assertEqual(list(Path(tmpdir).glob("*.json")), [])
            finally:
                os.chdir(old_cwd)
        self.assertEqual(rc, 2)
        self.assertIn("Preflight failed", err.getvalue())
        self.assertIn("HTTP 503", err.getvalue())
        self.assertIn("bin/k2h-bench.sh start", err.getvalue())
        self.assertEqual(server.state.paths, ["/health"])

    def test_check_and_dry_run_do_not_contact_live_server(self):
        with MockLlamaServer() as server:
            for argv in (["--check", "--url", server.base_url],
                         ["--dry-run", "--url", server.base_url]):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(bm.main(argv), 0)
            self.assertEqual(server.state.requests, [])


class TerminalBenchMockRunTests(unittest.TestCase):
    def test_full_run_uses_mock_health_and_writes_summary_without_tb(self):
        with MockLlamaServer() as server, tempfile.TemporaryDirectory() as tmpdir:
            config = tb.TerminalCliConfig(
                band="hard", n_tasks=1, model_name="mock-model",
                server_url=server.base_url, host_ip="172.17.0.1",
                tb_dir=tmpdir, tb_bin=str(Path(tmpdir) / "fake-tb"),
            )
            runner = tb.TerminalBenchRunner(config=config)
            commands = []

            def fake_run(cmd, timeout=120, cwd=None):
                commands.append(cmd)
                return 0, "ok", ""

            results_dir = Path(tmpdir) / "runs" / "mock-run"
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with mock.patch.object(runner, "results_dir", results_dir), \
                     mock.patch.object(runner, "check_dependencies", return_value=True), \
                     mock.patch.object(runner, "_run", side_effect=fake_run), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(runner.run_full_benchmark(band="hard", n_tasks=1))
                summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)
        port = urllib.parse.urlparse(server.base_url).port
        self.assertEqual(summary["model"], "mock-model")
        self.assertEqual(summary["server"], server.base_url)
        self.assertEqual(summary["container_base_url"], f"http://172.17.0.1:{port}/v1")
        self.assertEqual(summary["band"], "hard")
        self.assertTrue(any("hardest-first.mjs" in cmd for cmd in commands if cmd))
        self.assertEqual(server.state.paths, ["/health"])

    def test_unhealthy_server_aborts_before_any_tb_command(self):
        with MockLlamaServer(health_status=500) as server, \
             tempfile.TemporaryDirectory() as tmpdir:
            config = tb.TerminalCliConfig(server_url=server.base_url, tb_dir=tmpdir)
            runner = tb.TerminalBenchRunner(config=config)
            commands = []

            def fake_run(cmd, timeout=120, cwd=None):
                commands.append(cmd)
                return 0, "ok", ""

            old_cwd = os.getcwd()
            err = io.StringIO()
            try:
                os.chdir(tmpdir)
                with mock.patch.object(runner, "results_dir", Path(tmpdir) / "runs" / "mock"), \
                     mock.patch.object(runner, "check_dependencies", return_value=True), \
                     mock.patch.object(runner, "_run", side_effect=fake_run), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(err):
                    self.assertFalse(runner.run_full_benchmark())
            finally:
                os.chdir(old_cwd)
        self.assertEqual(commands, [])
        self.assertIn("HTTP 500", err.getvalue())
        self.assertEqual(server.state.paths, ["/health"])


if __name__ == "__main__":
    unittest.main()
