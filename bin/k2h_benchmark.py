#!/usr/bin/env python3
"""
K2-Horizon Benchmark Harness
==============================

Benchmark harness for the K2-Horizon-0.9B model.
Uses the OpenAI-compatible API endpoint at http://127.0.0.1:8888/v1

Usage:
    python3 k2h_benchmark.py [port] [model_name]
    python3 k2h_benchmark.py 8888 k2-horizon
"""

import json
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

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
    def __init__(self, port: int, model_name: str):
        self.port = port
        self.model_name = model_name
        self.base_url = f"http://localhost:{port}/v1"
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
            f"http://localhost:{self.port}/completion",
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

if __name__ == "__main__":
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    model_name = sys.argv[2] if len(sys.argv) > 2 else "k2-horizon"
    
    runner = K2HBenchmarkRunner(port, model_name)
    suite = runner.run_all()
    print_results(suite)
    
    # Save results
    filename = f"k2h_benchmark_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(asdict(suite), f, indent=2)
    print(f"\nResults saved to {filename}")
