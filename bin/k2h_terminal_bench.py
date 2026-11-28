#!/usr/bin/env python3
"""
K2-Horizon Terminal-Bench v4 Harness
=========================================

Runs terminal-bench v4 tasks against the K2-Horizon model server.
Uses the official terminal-bench (tb) CLI with the Darwin terminal agent.

Docker containers connect to the host at 172.17.0.1 (gateway IP).
The llama-server must be started with --host 0.0.0.0.

Usage:
    python3 k2h_terminal_bench.py [band] [n_tasks] [model_name]
    python3 k2h_terminal_bench.py hard 6
    python3 k2h_terminal_bench.py --check           # validate config, no services needed
    python3 k2h_terminal_bench.py --dry-run          # show planned commands, run nothing

Environment (defaults preserve the original machine setup):
    K2H_SERVER_URL   llama-server URL (default: http://127.0.0.1:8888)
    K2H_HOST_IP      gateway IP seen from Docker containers (default: 172.17.0.1)
    K2H_TB_DIR       terminal-bench checkout (contains the .mjs scripts)
    K2H_TB_BIN       path to the tb CLI (default: /home/scott/.local/bin/tb)
    K2H_TB_DATASET   dataset spec (default: terminal-bench-core==0.1.1)

Preflight failures print an actionable message and exit 2 instead of failing
with a cryptic FileNotFoundError or a wall of failed task results. Nothing is
installed, downloaded, or started by --check / --dry-run.
"""

import json
import shutil
import subprocess
import sys
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_MODEL_NAME = "k2-horizon"
DEFAULT_SERVER_URL = "http://127.0.0.1:8888"
DEFAULT_HOST_IP = "172.17.0.1"
DEFAULT_TB_DIR = ("/home/scott/.local/share/kipnerter/ruview/source/vendor/"
                  "metaharness/packages/darwin-mode/bench/terminal-bench")
DEFAULT_TB_BIN = "/home/scott/.local/bin/tb"
DEFAULT_DATASET = "terminal-bench-core==0.1.1"
NODE_SCRIPTS = ("build-manifest.mjs", "hardest-first.mjs", "score.py")

USAGE = (
    "Usage: k2h_terminal_bench.py [band] [n_tasks] [model_name]\n"
    "       k2h_terminal_bench.py --band hard --n-tasks 6 --model NAME "
    "[--server-url URL] [--tb-dir DIR] [--tb-bin PATH]\n"
    "       k2h_terminal_bench.py --check | --dry-run"
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one preflight check (no external services required)."""
    name: str
    ok: bool
    message: str
    severity: str = "error"  # "error" is fatal, "warning" is informational


@dataclass(frozen=True)
class TerminalCliConfig:
    band: str = ""
    n_tasks: int = 6
    model_name: str = DEFAULT_MODEL_NAME
    server_url: str = DEFAULT_SERVER_URL
    host_ip: str = DEFAULT_HOST_IP
    tb_dir: str = DEFAULT_TB_DIR
    tb_bin: str = DEFAULT_TB_BIN
    dataset: str = DEFAULT_DATASET
    check: bool = False
    dry_run: bool = False
    help: bool = False
    explicit: tuple = ()  # CLI flags that must win over environment variables


def format_checks(checks: List[CheckResult]) -> str:
    lines = []
    for check in checks:
        tag = "OK" if check.ok else ("WARN" if check.severity == "warning" else "FAIL")
        lines.append(f"[{tag}] {check.name}: {check.message}")
    return "\n".join(lines)


def fatal_checks(checks: List[CheckResult]) -> List[CheckResult]:
    return [c for c in checks if not c.ok and c.severity != "warning"]


def env_config(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    return os.environ if env is None else env


def resolve_config(cli: TerminalCliConfig,
                   env: Optional[Dict[str, str]] = None) -> TerminalCliConfig:
    """Layer env vars under explicit CLI values (pure)."""
    environ = env_config(env)
    env_map = {
        "server_url": environ.get("K2H_SERVER_URL"),
        "host_ip": environ.get("K2H_HOST_IP"),
        "tb_dir": environ.get("K2H_TB_DIR"),
        "tb_bin": environ.get("K2H_TB_BIN"),
        "dataset": environ.get("K2H_TB_DATASET"),
    }
    overrides = {key: value for key, value in env_map.items()
                 if value and key not in cli.explicit}
    return TerminalCliConfig(**{**cli.__dict__, **overrides})


def parse_args(argv: List[str]) -> TerminalCliConfig:
    """Parse CLI arguments without touching services or the filesystem (pure)."""
    band = n_tasks = model_name = None
    server_url = host_ip = tb_dir = tb_bin = dataset = None
    check = dry_run = show_help = False
    value_flags = {
        "--band": "band", "--n-tasks": "n_tasks", "--model": "model_name",
        "--server-url": "server_url", "--host-ip": "host_ip",
        "--tb-dir": "tb_dir", "--tb-bin": "tb_bin", "--dataset": "dataset",
    }
    explicit = set()
    positionals = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            show_help = True
        elif arg == "--check":
            check = True
        elif arg == "--dry-run":
            dry_run = True
        elif arg in value_flags or any(arg.startswith(f"{flag}=") for flag in value_flags):
            if "=" in arg:
                flag, value = arg.split("=", 1)
            else:
                flag = arg
                if i + 1 >= len(argv):
                    raise ValueError(f"{flag} requires a value")
                i += 1
                value = argv[i]
            if flag == "--n-tasks":
                try:
                    n_tasks = int(value)
                except ValueError:
                    raise ValueError(f"invalid --n-tasks: {value!r} (expected an integer)")
                explicit.add("n_tasks")
            elif flag == "--band":
                band = value
                explicit.add("band")
            elif flag == "--model":
                model_name = value
                explicit.add("model_name")
            elif flag == "--server-url":
                server_url = value
                explicit.add("server_url")
            elif flag == "--host-ip":
                host_ip = value
                explicit.add("host_ip")
            elif flag == "--tb-dir":
                tb_dir = value
                explicit.add("tb_dir")
            elif flag == "--tb-bin":
                tb_bin = value
                explicit.add("tb_bin")
            elif flag == "--dataset":
                dataset = value
                explicit.add("dataset")
        elif arg.startswith("-"):
            raise ValueError(f"unknown option: {arg}")
        else:
            positionals.append(arg)
        i += 1

    if len(positionals) > 3:
        raise ValueError(f"unexpected argument: {positionals[3]!r}")
    if band is None and positionals:
        band = positionals[0]
        explicit.add("band")
    if n_tasks is None and len(positionals) > 1:
        try:
            n_tasks = int(positionals[1])
        except ValueError:
            raise ValueError(f"invalid n_tasks: {positionals[1]!r} (expected an integer)")
        explicit.add("n_tasks")
    if model_name is None and len(positionals) > 2:
        model_name = positionals[2]
        explicit.add("model_name")

    defaults = TerminalCliConfig()
    return TerminalCliConfig(
        band=band or "",
        n_tasks=defaults.n_tasks if n_tasks is None else n_tasks,
        model_name=model_name or defaults.model_name,
        server_url=server_url or defaults.server_url,
        host_ip=host_ip or defaults.host_ip,
        tb_dir=tb_dir or defaults.tb_dir,
        tb_bin=tb_bin or defaults.tb_bin,
        dataset=dataset or defaults.dataset,
        check=check, dry_run=dry_run, help=show_help,
        explicit=tuple(sorted(explicit)),
    )


def validate_base_url(url: str) -> Optional[str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"server URL must start with http:// or https://: {url!r}"
    if not parsed.hostname:
        return f"server URL has no host: {url!r}"
    return None


def container_base_url(server_url: str, host_ip: str) -> str:
    """URL that Docker containers use to reach the host server (pure)."""
    port = urllib.parse.urlparse(server_url).port or 8888
    return f"http://{host_ip}:{port}/v1"


def check_server(base_url: str, open_url=None, timeout: float = 5.0) -> CheckResult:
    """Probe llama-server /health. `open_url` is injectable for tests."""
    opener = urllib.request.urlopen if open_url is None else open_url
    parsed = urllib.parse.urlparse(base_url)
    address = parsed.netloc or base_url
    hint = (f"llama-server not reachable at {address}; "
            f"start it with: bin/k2h-bench.sh start (or set K2H_SERVER_URL)")
    try:
        with opener(f"{base_url.rstrip('/')}/health", timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace").lower()
    except urllib.error.HTTPError as exc:
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


def preflight_config(config: TerminalCliConfig,
                     exists: Optional[Callable[[str], bool]] = None,
                     find_bin: Optional[Callable[[str], Optional[str]]] = None) -> List[CheckResult]:
    """Validate paths/tools/config only; injectable so tests need no services."""
    path_exists = (lambda p: Path(p).exists()) if exists is None else exists
    which = shutil.which if find_bin is None else find_bin
    checks: List[CheckResult] = []

    checks.append(CheckResult(
        "n_tasks", isinstance(config.n_tasks, int) and config.n_tasks > 0,
        f"using {config.n_tasks} tasks"
        if isinstance(config.n_tasks, int) and config.n_tasks > 0
        else "n_tasks must be a positive integer; pass --n-tasks N",
    ))
    checks.append(CheckResult(
        "model name", bool(config.model_name and config.model_name.strip()),
        f"using model {config.model_name!r}"
        if config.model_name and config.model_name.strip()
        else "model name is empty; pass --model NAME",
    ))
    url_error = validate_base_url(config.server_url)
    checks.append(CheckResult("server URL", url_error is None,
                              url_error or f"using {config.server_url}"))

    checks.append(CheckResult(
        "terminal-bench directory", path_exists(config.tb_dir),
        f"found {config.tb_dir}" if path_exists(config.tb_dir)
        else f"terminal-bench directory not found: {config.tb_dir}; set K2H_TB_DIR "
             f"to the metaharness/darwin-mode/bench/terminal-bench checkout",
    ))

    tb_path = Path(config.tb_bin)
    if path_exists(str(tb_path)):
        checks.append(CheckResult("tb CLI", True, f"found {config.tb_bin}"))
    else:
        on_path = which("tb")
        if on_path:
            checks.append(CheckResult("tb CLI", True, f"found on PATH: {on_path}"))
        else:
            checks.append(CheckResult(
                "tb CLI", False,
                f"tb CLI not found: {config.tb_bin}; install with: "
                f"uv tool install terminal-bench (or set K2H_TB_BIN)",
            ))

    node = which("node")
    checks.append(CheckResult(
        "node runtime", bool(node),
        f"found {node}" if node else "node not found on PATH; install Node.js to run the harness scripts",
    ))
    for script in NODE_SCRIPTS:
        script_path = str(Path(config.tb_dir) / script)
        checks.append(CheckResult(
            f"script {script}", path_exists(script_path),
            f"found {script_path}" if path_exists(script_path)
            else f"node script missing: {script_path}; set K2H_TB_DIR to a "
                 f"terminal-bench checkout that contains {script}",
        ))

    docker = which("docker")
    checks.append(CheckResult(
        "Docker CLI", bool(docker),
        f"found {docker}" if docker else
        "docker CLI not found; terminal-bench tasks run in Docker containers",
    ))

    cache_dir = str(Path.home() / ".cache" / "terminal-bench" /
                    "terminal-bench-core" / "0.1.1")
    checks.append(CheckResult(
        "dataset cache", True if path_exists(cache_dir) else False,
        f"found {cache_dir}" if path_exists(cache_dir) else
        f"dataset cache missing: {cache_dir}; the run will download it "
        f"with: {config.tb_bin} datasets download -d {config.dataset}",
        severity="warning" if not path_exists(cache_dir) else "error",
    ))
    return checks


def build_dry_run_plan(config: TerminalCliConfig, container_url: str) -> Dict:
    """Describe the commands a real run would issue; execute nothing (pure)."""
    return {
        "mode": "dry-run",
        "model": config.model_name,
        "server_url": config.server_url,
        "container_base_url": container_url,
        "tb_dir": config.tb_dir,
        "tb_bin": config.tb_bin,
        "results_dir": f"runs/k2h-<timestamp>/hardest-{config.model_name}",
        "band": config.band or "all",
        "n_tasks": config.n_tasks,
        "commands": [
            f"{config.tb_bin} datasets download -d {config.dataset}",
            "node build-manifest.mjs",
            "node hardest-first.mjs --manifest tbench-manifest.json "
            f"--model {config.model_name} --n {config.n_tasks} --max-steps 30 "
            f"--per-task-cost 1.5 --concurrent 1 --base-url {container_url} "
            + (f"--band {config.band} " if config.band else "")
            + "--out runs/k2h-<timestamp>/hardest-" + config.model_name,
            "python3 score.py <results-dir>",
        ],
        "external_services_checked": False,
    }


class TerminalBenchRunner:
    def __init__(self, model_name: str = "k2-horizon", server_url: str = "http://127.0.0.1:8888",
                 band: str = "", n_tasks: int = 6, config: Optional[TerminalCliConfig] = None):
        if config is None:
            config = resolve_config(TerminalCliConfig(
                band=band, n_tasks=n_tasks, model_name=model_name, server_url=server_url,
                explicit=("model_name", "server_url")))
        self.config = config
        self.model_name = config.model_name
        self.server_url = config.server_url
        # Docker containers use host gateway IP
        self.host_ip = config.host_ip
        self.container_base_url = container_base_url(config.server_url, config.host_ip)
        self.tb_dir = Path(config.tb_dir)
        self.tb_bin = Path(config.tb_bin)
        self.results_dir = Path(f"runs/k2h-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        bin_prefix = str(Path(config.tb_bin).parent)
        self.env = {**os.environ,
                    "PATH": f"{bin_prefix}:/usr/local/bin:/usr/bin:/bin:{os.environ.get('PATH', '')}"}
    
    def _run(self, cmd: list, timeout: int = 120, cwd: str = None) -> tuple[int, str, str]:
        env = {**self.env, "PYTHONPATH": str(self.tb_dir), "DARWIN_BASE_URL": self.container_base_url}
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout expired"
        except Exception as e:
            return -1, "", str(e)
    
    def check_dependencies(self) -> bool:
        """Report missing tools with actionable hints; never auto-install."""
        checks = preflight_config(self.config)
        print("Preflight results (no services are started/installed here):")
        print(format_checks(checks))
        failures = fatal_checks(checks)
        if failures:
            print("\nPreflight failed; benchmark aborted. Fix the FAIL items above.",
                  file=sys.stderr)
            return False
        return True
    
    def download_datasets(self):
        rc, out, err = self._run([str(self.tb_bin), "datasets", "download",
                                  "-d", self.config.dataset], timeout=120)
        if rc != 0:
            print(f"⚠️  Dataset download failed ({self.config.dataset}): {err}")
            print(f"   Install the CLI with: uv tool install terminal-bench")
        return rc
    
    def eval_validate(self, skip: bool = True):
        if skip: return
        val_dir = self.results_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        self._run([str(self.tb_bin), "run", "-d", self.config.dataset, "-t", "hello-world", "--agent", "oracle", "--output-path", str(val_dir / "oracle"), "--cleanup"], timeout=600)
        self._run([str(self.tb_bin), "run", "-d", self.config.dataset, "-t", "hello-world", "--agent", "nop", "--output-path", str(val_dir / "nop"), "--cleanup"], timeout=600)
    
    def build_manifest(self):
        rc, out, err = self._run(["node", "build-manifest.mjs"], timeout=60, cwd=str(self.tb_dir))
        if rc == 0:
            manifest = self.tb_dir / "tbench-manifest.json"
            if manifest.exists():
                m = json.loads(manifest.read_text())
                print(f"✅ Manifest: {len(m['tasks'])} tasks (hard={m['byDifficulty']['hard']}, med={m['byDifficulty']['medium']}, easy={m['byDifficulty']['easy']})")
    
    def run_terminal_bench(self, band: str = "", n_tasks: int = 6):
        os.chdir(self.tb_dir)
        out_dir = self.results_dir / f"hardest-{self.model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "node", "hardest-first.mjs",
            "--manifest", "tbench-manifest.json",
            "--model", self.model_name,
            "--n", str(n_tasks),
            "--max-steps", "30",
            "--per-task-cost", "1.5",
            "--concurrent", "1",
            "--base-url", self.container_base_url,
            "--out", str(out_dir)
        ]
        if band: cmd.extend(["--band", band])
        print(f"\n=== Running Terminal-Bench v4 ===")
        print(f"Model: {self.model_name} | Server: {self.server_url} | Container URL: {self.container_base_url}")
        print(f"Band: {band or 'all'} | Tasks: {n_tasks}")
        rc, out, err = self._run(cmd, timeout=7200, cwd=str(self.tb_dir))
        if rc == 0: print(f"\n✅ Terminal-bench completed.")
        else:
            print(f"\n⚠️  Exit code: {rc}")
            print(f"Stdout: {out[:1000]}")
            print(f"Stderr: {err[:1000]}")
    
    def run_scoring(self):
        out_dir = self.results_dir / f"hardest-{self.model_name}"
        results_jsons = list(out_dir.glob("*/results.json")) + [out_dir / "results.json"]
        for rj in results_jsons:
            if rj.exists():
                self._run(["python3", "score.py", str(rj.parent)], timeout=60, cwd=str(self.tb_dir))
    
    def run_full_benchmark(self, band: str = "", n_tasks: int = 6,
                           skip_eval: bool = True, dry_run: bool = False) -> bool:
        print(f"\n{'='*70}")
        print(f"K2-HORIZON TERMINAL-BENCH v4 BENCHMARK")
        print(f"{'='*70}")
        print(f"Start: {datetime.now().isoformat()}")
        print(f"Server: {self.server_url} | Container URL: {self.container_base_url}")
        print(f"Model: {self.model_name}")
        print(f"{'='*70}")

        if dry_run:
            print("Dry run: no directories, downloads, containers or requests.")
            print(format_checks(preflight_config(self.config)))
            print(json.dumps(build_dry_run_plan(self.config, self.container_base_url), indent=2))
            return True

        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.check_dependencies(): return False
        reachable = check_server(self.server_url)
        if not reachable.ok:
            print(format_checks([reachable]), file=sys.stderr)
            print("Preflight failed; benchmark aborted. Fix the FAIL items above.",
                  file=sys.stderr)
            return False
        print(format_checks([reachable]))
        self.download_datasets()
        self.eval_validate(skip=skip_eval)
        self.build_manifest()
        self.run_terminal_bench(band, n_tasks)
        self.run_scoring()
        summary = {"model": self.model_name, "server": self.server_url, "container_base_url": self.container_base_url, "results_dir": str(self.results_dir), "timestamp": datetime.now().isoformat(), "band": band or "all", "n_tasks": n_tasks}
        with open(self.results_dir / "summary.json", 'w') as f: json.dump(summary, f, indent=2)
        print(f"\n✅ Summary saved to {self.results_dir / 'summary.json'}")
        return True

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    try:
        cli = parse_args(argv)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    if cli.help:
        print(USAGE)
        return 0
    config = resolve_config(cli)
    if cli.check or cli.dry_run:
        checks = preflight_config(config)
        print("K2-Horizon terminal-bench preflight (no external services used)")
        print(format_checks(checks))
        if cli.dry_run:
            print(json.dumps(build_dry_run_plan(
                config, container_base_url(config.server_url, config.host_ip)), indent=2))
        return 0 if not fatal_checks(checks) else 2

    runner = TerminalBenchRunner(config=config)
    ok = runner.run_full_benchmark(band=config.band, n_tasks=config.n_tasks)
    return 0 if ok else 2

if __name__ == "__main__":
    sys.exit(main())
