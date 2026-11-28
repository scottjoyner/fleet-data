#!/usr/bin/env python3
"""
K2-Horizon Benchmark Harness
==============================

Benchmark harness for the K2-Horizon-0.9B model.
Uses the OpenAI-compatible API endpoint at http://127.0.0.1:8888/v1

Usage:
    python3 k2h_benchmark.py [port] [model_name]
    python3 k2h_benchmark.py 8888 k2-horizon
    python3 k2h_benchmark.py --check          # validate config, no services needed
    python3 k2h_benchmark.py --dry-run        # show the plan, no requests/files

Environment:
    K2H_SERVER_URL   Base URL of llama-server (default: http://localhost:<port>)

Exits 2 on failed preflight (with an actionable message) instead of producing
a suite full of connection errors.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

DEFAULT_PORT = 8888
DEFAULT_MODEL_NAME = "k2-horizon"
SERVER_URL_ENV = "K2H_SERVER_URL"

USAGE = (
    "Usage: k2h_benchmark.py [port] [model_name]\n"
    "       k2h_benchmark.py --port N --model NAME [--url URL]\n"
    "       k2h_benchmark.py --check | --dry-run"
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one preflight check (no external services required)."""
    name: str
    ok: bool
    message: str
    severity: str = "error"  # "error" is fatal, "warning" is informational


@dataclass(frozen=True)
class CliConfig:
    port: int = DEFAULT_PORT
    model_name: str = DEFAULT_MODEL_NAME
    url: Optional[str] = None
    check: bool = False
    dry_run: bool = False
    help: bool = False


def format_checks(checks: List[CheckResult]) -> str:
    """Render preflight results consistently (pure)."""
    lines = []
    for check in checks:
        tag = "OK" if check.ok else ("WARN" if check.severity == "warning" else "FAIL")
        lines.append(f"[{tag}] {check.name}: {check.message}")
    return "\n".join(lines)


def fatal_checks(checks: List[CheckResult]) -> List[CheckResult]:
    return [c for c in checks if not c.ok and c.severity != "warning"]


def validate_port(port) -> Optional[str]:
    """Return an error string for an invalid port, else None (pure)."""
    if isinstance(port, bool) or not isinstance(port, int):
        return f"port must be an integer (got {port!r})"
    if not 1 <= port <= 65535:
        return f"port out of range 1-65535: {port}"
    return None


def resolve_base_url(port: int, url: Optional[str] = None,
                     env: Optional[Dict[str, str]] = None) -> str:
    """Choose the server base URL: CLI > env > default (pure)."""
    environ = os.environ if env is None else env
    candidate = url or environ.get(SERVER_URL_ENV) or f"http://localhost:{port}"
    return candidate.rstrip("/")


def validate_base_url(url: str) -> Optional[str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"server URL must start with http:// or https://: {url!r}"
    if not parsed.hostname:
        return f"server URL has no host: {url!r}"
    return None


def parse_args(argv: List[str]) -> CliConfig:
    """Parse CLI arguments without touching the network or filesystem (pure)."""
    port = None
    model_name = None
    url = None
    check = dry_run = show_help = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            show_help = True
        elif arg == "--check":
            check = True
        elif arg == "--dry-run":
            dry_run = True
        elif arg in ("--port", "--model", "--url"):
            if i + 1 >= len(argv):
                raise ValueError(f"{arg} requires a value")
            i += 1
            value = argv[i]
            if arg == "--port":
                try:
                    port = int(value)
                except ValueError:
                    raise ValueError(f"invalid port: {value!r} (expected an integer)")
            elif arg == "--model":
                model_name = value
            else:
                url = value
        elif arg.startswith("--port=") or arg.startswith("--model=") or arg.startswith("--url="):
            key, value = arg.split("=", 1)
            if key == "--port":
                try:
                    port = int(value)
                except ValueError:
                    raise ValueError(f"invalid port: {value!r} (expected an integer)")
            elif key == "--model":
                model_name = value
            else:
                url = value
        elif arg.startswith("-"):
            raise ValueError(f"unknown option: {arg}")
        elif port is None:
            try:
                port = int(arg)
            except ValueError:
                raise ValueError(f"invalid port: {arg!r} (expected an integer)")
        elif model_name is None:
            model_name = arg
        else:
            raise ValueError(f"unexpected argument: {arg!r}")
        i += 1
    effective_port = DEFAULT_PORT if port is None else port
    port_error = validate_port(effective_port)
    if port_error:
        raise ValueError(f"invalid port: {port_error}")
    return CliConfig(
        port=effective_port,
        model_name=model_name or DEFAULT_MODEL_NAME,
        url=url, check=check, dry_run=dry_run, help=show_help,
    )


def preflight_config(config: CliConfig, env: Optional[Dict[str, str]] = None,
                     tasks: Optional[Dict] = None) -> List[CheckResult]:
    """Validate configuration only; safe to call with no external services."""
    checks: List[CheckResult] = []
    port_error = validate_port(config.port)
    checks.append(CheckResult("port", port_error is None,
                              port_error or f"using port {config.port}"))
    checks.append(CheckResult(
        "model name", bool(config.model_name and config.model_name.strip()),
        f"using model {config.model_name!r}" if config.model_name and config.model_name.strip()
        else "model name is empty; pass --model NAME",
    ))
    base_url = resolve_base_url(config.port, config.url, env)
    url_error = validate_base_url(base_url)
    checks.append(CheckResult("server URL", url_error is None,
                              url_error or f"using {base_url} (K2H_SERVER_URL / --url overrides port)"))
    all_tasks = tasks if tasks is not None else K2H_TASKS
    task_count = sum(len(v) for v in all_tasks.values())
    checks.append(CheckResult(
        "benchmark tasks", task_count > 0,
        f"{task_count} tasks in {len(all_tasks)} categories"
        if task_count else "no benchmark tasks configured; check K2H_TASKS",
    ))
    return checks


def check_server(base_url: str, open_url=None, timeout: float = 5.0) -> CheckResult:
    """Probe llama-server /health. `open_url` is injectable for tests."""
    opener = urllib.request.urlopen if open_url is None else open_url
    parsed = urllib.parse.urlparse(base_url)
    address = parsed.netloc or base_url
    health_url = f"{base_url.rstrip('/')}/health"
    hint = (f"llama-server not reachable at {address}; "
            f"start it with: bin/k2h-bench.sh start (or set K2H_SERVER_URL)")
    try:
        with opener(health_url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace").lower()
    except urllib.error.HTTPError as exc:
        # Some OpenAI-compatible servers do not expose /health. HTTP 404 still
        # proves the server is up, so keep the original benchmark behavior.
        exc.close()
        if exc.code == 404:
            return CheckResult("llama-server", True,
                               f"reachable at {address} (no /health endpoint; continuing)")
        return CheckResult("llama-server", False, f"{hint}: HTTP {exc.code}")
    except Exception as exc:
        return CheckResult("llama-server", False, f"{hint}: {exc}")
    if "ok" not in body:
        return CheckResult("llama-server", False,
                           f"{hint}: /health returned an unexpected response: {body[:120]!r}")
    return CheckResult("llama-server", True, f"reachable at {address}")


def build_dry_run_plan(config: CliConfig, base_url: str,
                       tasks: Optional[Dict] = None) -> Dict:
    """Describe what a real run would do; no requests or writes (pure)."""
    all_tasks = tasks if tasks is not None else K2H_TASKS
    return {
        "mode": "dry-run",
        "model": config.model_name,
        "server_base_url": base_url,
        "chat_url": f"{base_url}/v1/chat/completions",
        "completion_url": f"{base_url}/completion",
        "categories": {cat: [task["name"] for task in items]
                       for cat, items in all_tasks.items()},
        "task_count": sum(len(items) for items in all_tasks.values()),
        "output": f"k2h_benchmark_{config.model_name}_<timestamp>.json",
        "external_services_checked": False,
    }


@dataclass
class BenchmarkResult:
    name: str
    category: str
    success: bool
    tokens: int
    prompt_tokens: int
    gen_speed: float
    prompt_speed: float
    time: float
    output: str
    error: str = ""

@dataclass
class BenchmarkSuite:
    version: str
    timestamp: str
    model: str
    results: List[BenchmarkResult]
    summary: Dict

# Test categories and tasks
K2H_TASKS = {
    "tool_calling": [
        {"name": "bash_simple", "prompt": "Run 'echo hello world'", "expect": "function"},
        {"name": "bash_file_ops", "prompt": "Create a file called test.txt with content 'hello' then read it back", "expect": "function"},
        {"name": "bash_search", "prompt": "Find all Python files in the current directory", "expect": "function"},
    ],
    "reasoning": [
        {"name": "explain_variable", "prompt": "Explain what a variable is in programming", "expect": "text"},
        {"name": "explain_gc", "prompt": "How does garbage collection work?", "expect": "text"},
        {"name": "compare_lists", "prompt": "Compare lists and tuples in Python, highlighting their differences", "expect": "text"},
    ],
    "multi_step": [
        {"name": "sequence_ops", "prompt": "First list files in the current directory, then count the Python files", "expect": "function"},
        {"name": "conditional_task", "prompt": "If Python exists, show its version, otherwise install it", "expect": "function"},
        {"name": "loop_task", "prompt": "Show the current directory contents, disk usage, and memory usage", "expect": "function"},
    ],
    "math_logic": [
        {"name": "simple_math", "prompt": "What is the sum of the first 100 prime numbers?", "expect": "text"},
        {"name": "fibonacci", "prompt": "Write a function to compute the nth Fibonacci number and find F(20)", "expect": "text"},
    ],
    "creative_writing": [
        {"name": "story_prompt", "prompt": "Write a short story about a robot discovering it has feelings", "expect": "text"},
        {"name": "poem_prompt", "prompt": "Write a haiku about artificial intelligence", "expect": "text"},
    ],
}

class K2HBenchmarkRunner:
    def __init__(self, port: int, model_name: str, base_url: Optional[str] = None):
        self.port = port
        self.model_name = model_name
        # Default URLs are unchanged; base_url allows env/CLI overrides.
        self.server_base_url = (base_url or f"http://localhost:{port}").rstrip("/")
        self.base_url = f"{self.server_base_url}/v1"
        self.chat_url = f"{self.base_url}/chat/completions"
    
    def _call_api(self, messages: List[Dict], max_tokens: int = 100, temperature: float = 0.1) -> Dict:
        body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            self.chat_url,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            elapsed = time.time() - start
            return result, elapsed
        except Exception as e:
            return {"error": str(e)}, 0
    
    def _call_completion(self, prompt: str, max_tokens: int = 100, temperature: float = 0.1) -> Dict:
        body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(
            f"{self.server_base_url}/completion",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            elapsed = time.time() - start
            return result, elapsed
        except Exception as e:
            return {"error": str(e)}, 0
    
    def _count_tokens(self, text: str) -> int:
        """Rough token count: ~1 token per 4 characters"""
        return max(len(text) // 4, 1)
    
    def run_task(self, task: Dict, category: str) -> BenchmarkResult:
        prompt = task["prompt"]
        name = task["name"]
        
        start_time = time.time()
        prompt_tokens = self._count_tokens(prompt)
        
        try:
            # Try chat completions first
            result, elapsed = self._call_api(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            if "error" in result:
                return BenchmarkResult(
                    name=name, category=category, success=False,
                    tokens=0, prompt_tokens=prompt_tokens, gen_speed=0,
                    prompt_speed=prompt_tokens / elapsed if elapsed > 0 else 0,
                    time=elapsed, output="", error=result["error"]
                )
            
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            gen_speed = completion_tokens / elapsed if elapsed > 0 else 0
            prompt_speed = prompt_tokens / elapsed if elapsed > 0 else 0
            
            return BenchmarkResult(
                name=name, category=category, success=True,
                tokens=completion_tokens, prompt_tokens=prompt_tokens,
                gen_speed=gen_speed, prompt_speed=prompt_speed,
                time=elapsed, output=content[:200]
            )
        except Exception as e:
            return BenchmarkResult(
                name=name, category=category, success=False,
                tokens=0, prompt_tokens=prompt_tokens, gen_speed=0,
                prompt_speed=0, time=time.time() - start_time,
                output="", error=str(e)
            )
    
    def run_category(self, category: str) -> List[BenchmarkResult]:
        tasks = K2H_TASKS.get(category, [])
        return [self.run_task(task, category) for task in tasks]
    
    def run_all(self) -> BenchmarkSuite:
        all_results = []
        for category in K2H_TASKS:
            print(f"Running {category}...")
            results = self.run_category(category)
            all_results.extend(results)
        
        total = len(all_results)
        passed = sum(1 for r in all_results if r.success)
        avg_gen = sum(r.gen_speed for r in all_results) / total if total > 0 else 0
        avg_prompt = sum(r.prompt_speed for r in all_results) / total if total > 0 else 0
        avg_tokens = sum(r.tokens for r in all_results) / total if total > 0 else 0
        total_time = sum(r.time for r in all_results)
        
        summary = {
            "total_tasks": total,
            "passed": passed,
            "success_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%",
            "avg_gen_speed": f"{avg_gen:.1f} tok/s",
            "avg_prompt_speed": f"{avg_prompt:.1f} tok/s",
            "avg_tokens": f"{avg_tokens:.0f}",
            "total_time": f"{total_time:.1f}s",
            "categories": {}
        }
        
        for category in K2H_TASKS:
            cat_results = [r for r in all_results if r.category == category]
            cat_passed = sum(1 for r in cat_results if r.success)
            cat_gen = sum(r.gen_speed for r in cat_results) / len(cat_results) if cat_results else 0
            summary["categories"][category] = {
                "total": len(cat_results),
                "passed": cat_passed,
                "rate": f"{(cat_passed/len(cat_results)*100):.1f}%" if cat_results else "0%",
                "avg_gen_speed": f"{cat_gen:.1f} tok/s"
            }
        
        return BenchmarkSuite(
            version="1.0.0",
            timestamp=datetime.now().isoformat(),
            model=self.model_name,
            results=all_results,
            summary=summary
        )

def print_results(suite: BenchmarkSuite):
    print(f"\n{'='*70}")
    print(f"K2-HORIZON BENCHMARK RESULTS: {suite.model}")
    print(f"{'='*70}")
    print(f"Version: {suite.version}")
    print(f"Timestamp: {suite.timestamp}")
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total tasks: {suite.summary['total_tasks']}")
    print(f"Passed: {suite.summary['passed']}")
    print(f"Success rate: {suite.summary['success_rate']}")
    print(f"Avg generation speed: {suite.summary['avg_gen_speed']}")
    print(f"Avg prompt speed: {suite.summary['avg_prompt_speed']}")
    print(f"Total time: {suite.summary['total_time']}")
    
    print(f"\n{'='*70}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*70}")
    for cat, stats in suite.summary['categories'].items():
        print(f"{cat:<20} {stats['passed']}/{stats['total']} ({stats['rate']}) | {stats['avg_gen_speed']}")
    
    print(f"\n{'='*70}")
    print("DETAILED RESULTS")
    print(f"{'='*70}")
    for result in suite.results:
        status = "PASS" if result.success else "FAIL"
        print(f"{result.category:<15} {result.name:<20} {status} | {result.gen_speed:.1f} tok/s | {result.tokens} tokens | {result.time:.1f}s")
    
    print(f"\n{'='*70}")

def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        config = parse_args(argv)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if config.help:
        print(USAGE)
        return 0

    base_url = resolve_base_url(config.port, config.url)
    checks = preflight_config(config)
    if config.check or config.dry_run:
        print("K2-Horizon benchmark preflight (no external services used)")
        print(format_checks(checks))
        if config.dry_run:
            print(json.dumps(build_dry_run_plan(config, base_url), indent=2))
        return 0 if not fatal_checks(checks) else 2

    failures = fatal_checks(checks)
    if failures:
        print("Preflight failed; benchmark aborted:", file=sys.stderr)
        print(format_checks(failures), file=sys.stderr)
        return 2

    reachable = check_server(base_url)
    print(format_checks([reachable]))
    if not reachable.ok:
        print("Preflight failed; benchmark aborted.", file=sys.stderr)
        return 2

    runner = K2HBenchmarkRunner(config.port, config.model_name, base_url=base_url)
    suite = runner.run_all()
    print_results(suite)

    # Save results
    filename = (f"k2h_benchmark_{config.model_name}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filename, 'w') as f:
        json.dump(asdict(suite), f, indent=2)
    print(f"\nResults saved to {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
