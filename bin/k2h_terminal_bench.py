#!/usr/bin/env python3
"""
K2-Horizon Terminal-Bench v4 Harness
=========================================

Runs terminal-bench v4 tasks against the K2-Horizon model server at 127.0.0.1:8888.
Uses the official terminal-bench (tb) CLI with the Darwin terminal agent.

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
        self.base_url = f"{server_url}/v1"
        self.tb_dir = Path("/home/scott/.local/share/kipnerter/ruview/source/vendor/metaharness/packages/darwin-mode/bench/terminal-bench")
        self.tb_bin = Path("/home/scott/.local/bin/tb")
        self.results_dir = Path(f"runs/k2h-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.env = {**os.environ, "PATH": "/home/scott/.local/bin:/usr/local/bin:/usr/bin:/bin"}
    
    def _run(self, cmd: list, timeout: int = 120, cwd: str = None) -> tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        env = {**self.env, "PYTHONPATH": str(self.tb_dir), "DARWIN_BASE_URL": self.base_url}
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=cwd, env=env
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout expired"
        except Exception as e:
            return -1, "", str(e)
    
    def check_dependencies(self) -> bool:
        """Check if terminal-bench (tb) and node are available."""
        rc, out, err = self._run(["which", "tb"])
        if rc != 0:
            print("⚠️  terminal-bench (tb) not found. Installing...")
            rc, out, err = self._run(["uv", "tool", "install", "terminal-bench"], timeout=120)
            if rc != 0:
                print(f"❌ ERROR: Failed to install terminal-bench: {err}")
                return False
            print("✅ terminal-bench installed.")
        else:
            print(f"✅ terminal-bench found at {out}")
        
        rc, out, err = self._run(["which", "node"])
        if rc != 0:
            print("❌ ERROR: node not found")
            return False
        print(f"✅ node found at {out}")
        
        # Check the dataset exists
        cache_dir = Path.home() / ".cache" / "terminal-bench" / "terminal-bench-core" / "0.1.1"
        if cache_dir.exists():
            print(f"✅ terminal-bench-core dataset found")
        else:
            print("⚠️  terminal-bench-core dataset not found, downloading...")
            self.download_datasets()
        
        # Check the agent file exists
        agent_file = self.tb_dir / "darwin_terminal_agent.py"
        if agent_file.exists():
            print(f"✅ Agent file found")
        else:
            print(f"❌ ERROR: Agent file not found: {agent_file}")
            return False
        
        return True
    
    def download_datasets(self):
        """Download terminal-bench-core dataset."""
        print("\n=== Downloading Datasets ===")
        cache_dir = Path.home() / ".cache" / "terminal-bench" / "terminal-bench-core" / "0.1.1"
        if cache_dir.exists():
            print("✅ Dataset already present, skipping download.")
            return
        rc, out, err = self._run([str(self.tb_bin), "datasets", "download", "-d", "terminal-bench-core==0.1.1"], timeout=120)
        if rc == 0:
            print("✅ Datasets downloaded.")
        else:
            print(f"⚠️  Dataset download returned code {rc}")
    
    def eval_validate(self, skip: bool = True):
        """Run eval-validation: oracle must PASS, nop must FAIL."""
        if skip:
            print("\n=== Eval Validation (SKIPPED - use --no-skip-eval to enable) ===")
            return
        print("\n=== Eval Validation ===")
        val_dir = self.results_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        
        print("Running oracle (known-good should PASS)...")
        rc, out, err = self._run([
            str(self.tb_bin), "run", "-d", "terminal-bench-core==0.1.1", "-t", "hello-world",
            "--agent", "oracle", "--output-path", str(val_dir / "oracle"), "--cleanup"
        ], timeout=600)
        if rc == 0:
            print("✅ Oracle validation PASSED")
        else:
            print(f"⚠️  Oracle validation returned code {rc}")
        
        print("Running nop (empty agent should FAIL)...")
        rc, out, err = self._run([
            str(self.tb_bin), "run", "-d", "terminal-bench-core==0.1.1", "-t", "hello-world",
            "--agent", "nop", "--output-path", str(val_dir / "nop"), "--cleanup"
        ], timeout=600)
        if rc != 0:
            print("✅ NOP validation PASSED (nop correctly failed)")
        else:
            print("⚠️  NOP unexpectedly passed")
    
    def build_manifest(self):
        """Build hardest-first task manifest."""
        print("\n=== Building Manifest ===")
        rc, out, err = self._run(["node", "build-manifest.mjs"], timeout=60, cwd=str(self.tb_dir))
        if rc == 0:
            print("✅ Manifest built.")
            manifest = self.tb_dir / "tbench-manifest.json"
            if manifest.exists():
                m = json.loads(manifest.read_text())
                print(f"   Tasks: {len(m.get('tasks', []))} (hard={m['byDifficulty']['hard']}, med={m['byDifficulty']['medium']}, easy={m['byDifficulty']['easy']})")
        else:
            print(f"⚠️  Could not build manifest: {err}")
    
    def run_terminal_bench(self, band: str = "", n_tasks: int = 6):
        """Run terminal-bench with the Darwin agent against our local server."""
        os.chdir(self.tb_dir)
        
        out_dir = self.results_dir / f"hardest-{self.model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the node command
        cmd = [
            "node", "hardest-first.mjs",
            "--manifest", "tbench-manifest.json",
            "--model", self.model_name,
            "--n", str(n_tasks),
            "--max-steps", "30",
            "--per-task-cost", "1.5",
            "--concurrent", "1",
            "--base-url", self.base_url,
            "--out", str(out_dir)
        ]
        if band:
            cmd.extend(["--band", band])
        
        print(f"\n=== Running Terminal-Bench v4 ===")
        print(f"Model: {self.model_name}")
        print(f"Server: {self.server_url}")
        print(f"Base URL: {self.base_url}")
        print(f"Band: {band or 'all'}")
        print(f"Tasks: {n_tasks}")
        print(f"Output: {out_dir}")
        
        rc, out, err = self._run(cmd, timeout=7200, cwd=str(self.tb_dir))
        if rc == 0:
            print(f"\n✅ Terminal-bench completed.")
        else:
            print(f"\n⚠️  Terminal-bench exit code: {rc}")
            print(f"Stdout: {out[:2000]}")
            print(f"Stderr: {err[:2000]}")
        
        # Check if results exist
        results_json = out_dir / "results.json"
        if results_json.exists():
            print(f"✅ Results found: {results_json}")
        # Also check subdirs
        subdirs = list(out_dir.glob("*/results.json"))
        if subdirs:
            print(f"✅ Found {len(subdirs)} subdir results")
    
    def run_scoring(self):
        """Run scoring on terminal-bench results."""
        print("\n=== Scoring ===")
        out_dir = self.results_dir / f"hardest-{self.model_name}"
        
        # Find all results.json files
        results_jsons = list(out_dir.glob("*/results.json")) + [out_dir / "results.json"]
        rj_found = [r for r in results_jsons if r.exists()]
        
        if not rj_found:
            print(f"⚠️  No results.json found in {out_dir}")
            return
        
        for rj in rj_found:
            run_dir = rj.parent
            rc, out, err = self._run(["python3", "score.py", str(run_dir)], timeout=60, cwd=str(self.tb_dir))
            if rc == 0:
                print(f"✅ Scoring complete for {run_dir.name}.")
            else:
                print(f"⚠️  Scoring failed for {run_dir.name}: {err}")
    
    def run_full_benchmark(self, band: str = "", n_tasks: int = 6, skip_eval: bool = True):
        """Run the complete terminal-bench pipeline."""
        print(f"\n{'='*70}")
        print(f"K2-HORIZON TERMINAL-BENCH v4 BENCHMARK")
        print(f"{'='*70}")
        print(f"Start: {datetime.now().isoformat()}")
        print(f"Server: {self.server_url}")
        print(f"Model: {self.model_name}")
        print(f"{'='*70}")
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Check dependencies
        if not self.check_dependencies():
            print("❌ FATAL: terminal-bench not available")
            sys.exit(1)
        
        # 2. Download datasets
        self.download_datasets()
        
        # 3. Eval validation (optional, skip by default)
        self.eval_validate(skip=skip_eval)
        
        # 4. Build manifest
        self.build_manifest()
        
        # 5. Run terminal-bench
        self.run_terminal_bench(band, n_tasks)
        
        # 6. Scoring
        self.run_scoring()
        
        print(f"\n{'='*70}")
        print(f"K2-HORIZON TERMINAL-BENCH v4 COMPLETE")
        print(f"Results: {self.results_dir}")
        print(f"{'='*70}")
        
        # Save summary
        summary = {
            "model": self.model_name,
            "server": self.server_url,
            "base_url": self.base_url,
            "results_dir": str(self.results_dir),
            "timestamp": datetime.now().isoformat(),
            "band": band or "all",
            "n_tasks": n_tasks
        }
        with open(self.results_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {self.results_dir / 'summary.json'}")

def main():
    band = sys.argv[1] if len(sys.argv) > 1 else ""
    n_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    model_name = sys.argv[3] if len(sys.argv) > 3 else "k2-horizon"
    
    runner = TerminalBenchRunner(model_name=model_name)
    runner.run_full_benchmark(band=band, n_tasks=n_tasks)

if __name__ == "__main__":
    main()
