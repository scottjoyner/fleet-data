#!/usr/bin/env python3
"""
K2-Horizon Terminal-Bench v4 Harness
=========================================

Runs terminal-bench v4 tasks against the K2-Horizon model server.
Uses the official terminal-bench (tb) CLI with the Darwin terminal agent.

Docker containers connect to the host at 172.17.0.1 (gateway IP).
The llama-server must be started with --host 0.0.0.0.

Usage:
    python3 k2h_terminal_bench.py [band] [n_tasks]
    python3 k2h_terminal_bench.py hard 6
    python3 k2h_terminal_bench.py  # default: hard, 6 tasks
"""

import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

class TerminalBenchRunner:
    def __init__(self, model_name: str = "k2-horizon", server_url: str = "http://127.0.0.1:8888"):
        self.model_name = model_name
        self.server_url = server_url
        # Docker containers use host gateway IP
        self.host_ip = "172.17.0.1"
        self.container_base_url = f"http://{self.host_ip}:{self.server_url.split(':')[-1]}/v1"
        self.tb_dir = Path("/home/scott/.local/share/kipnerter/ruview/source/vendor/metaharness/packages/darwin-mode/bench/terminal-bench")
        self.tb_bin = Path("/home/scott/.local/bin/tb")
        self.results_dir = Path(f"runs/k2h-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.env = {**os.environ, "PATH": "/home/scott/.local/bin:/usr/local/bin:/usr/bin:/bin"}
    
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
        rc, out, err = self._run(["which", "tb"])
        if rc != 0:
            rc, out, err = self._run(["uv", "tool", "install", "terminal-bench"], timeout=120)
            if rc != 0: return False
        rc, out, err = self._run(["which", "node"])
        if rc != 0: return False
        cache_dir = Path.home() / ".cache" / "terminal-bench" / "terminal-bench-core" / "0.1.1"
        if not cache_dir.exists(): self.download_datasets()
        return True
    
    def download_datasets(self):
        rc, out, err = self._run([str(self.tb_bin), "datasets", "download", "-d", "terminal-bench-core==0.1.1"], timeout=120)
        if rc != 0: print(f"⚠️  Dataset download: {err}")
    
    def eval_validate(self, skip: bool = True):
        if skip: return
        val_dir = self.results_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        self._run([str(self.tb_bin), "run", "-d", "terminal-bench-core==0.1.1", "-t", "hello-world", "--agent", "oracle", "--output-path", str(val_dir / "oracle"), "--cleanup"], timeout=600)
        self._run([str(self.tb_bin), "run", "-d", "terminal-bench-core==0.1.1", "-t", "hello-world", "--agent", "nop", "--output-path", str(val_dir / "nop"), "--cleanup"], timeout=600)
    
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
    
    def run_full_benchmark(self, band: str = "", n_tasks: int = 6, skip_eval: bool = True):
        print(f"\n{'='*70}")
        print(f"K2-HORIZON TERMINAL-BENCH v4 BENCHMARK")
        print(f"{'='*70}")
        print(f"Start: {datetime.now().isoformat()}")
        print(f"Server: {self.server_url} | Container URL: {self.container_base_url}")
        print(f"Model: {self.model_name}")
        print(f"{'='*70}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.check_dependencies(): sys.exit(1)
        self.download_datasets()
        self.eval_validate(skip=skip_eval)
        self.build_manifest()
        self.run_terminal_bench(band, n_tasks)
        self.run_scoring()
        summary = {"model": self.model_name, "server": self.server_url, "container_base_url": self.container_base_url, "results_dir": str(self.results_dir), "timestamp": datetime.now().isoformat(), "band": band or "all", "n_tasks": n_tasks}
        with open(self.results_dir / "summary.json", 'w') as f: json.dump(summary, f, indent=2)
        print(f"\n✅ Summary saved to {self.results_dir / 'summary.json'}")

def main():
    band = sys.argv[1] if len(sys.argv) > 1 else ""
    n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    model_name = sys.argv[3] if len(sys.argv) > 3 else "k2-horizon"
    TerminalBenchRunner(model_name=model_name).run_full_benchmark(band=band, n_tasks=n_tasks)

if __name__ == "__main__":
    main()
