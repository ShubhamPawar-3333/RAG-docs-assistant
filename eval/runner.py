"""
Promptfoo Evaluation Runner

Python wrapper for running Promptfoo evaluations and processing results.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PromptfooRunner:
    """
    Wrapper for running Promptfoo evaluations.
    
    Provides Python interface for:
    - Running evaluations
    - Parsing results
    - Generating reports
    """
    
    def __init__(self, config_path: str = "eval/promptfooconfig.yaml"):
        """
        Initialize the runner.
        
        Args:
            config_path: Path to promptfoo config file
        """
        self.config_path = Path(config_path)
        self.results_dir = Path("eval/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_eval(
        self,
        output_path: Optional[str] = None,
        no_cache: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run Promptfoo evaluation.
        
        Args:
            output_path: Custom output path for results
            no_cache: Disable caching
            verbose: Show detailed output
            
        Returns:
            Evaluation results dictionary
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.results_dir / f"eval_{timestamp}.json"
        
        cmd = [
            "npx", "promptfoo", "eval",
            "-c", str(self.config_path),
            "-o", str(output_path),
            "--json",
        ]
        
        if no_cache:
            cmd.append("--no-cache")
        
        logger.info(f"Running Promptfoo evaluation: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Promptfoo failed: {result.stderr}")
                return {"error": result.stderr}
            
            # Parse results
            with open(output_path, "r") as f:
                results = json.load(f)
            
            return results
            
        except subprocess.TimeoutExpired:
            logger.error("Promptfoo evaluation timed out")
            return {"error": "Evaluation timed out"}
        except Exception as e:
            logger.error(f"Failed to run Promptfoo: {e}")
            return {"error": str(e)}
    
    def parse_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and summarize evaluation results.
        
        Args:
            results: Raw Promptfoo results
            
        Returns:
            Summary dictionary
        """
        if "error" in results:
            return results
        
        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "avg_latency_ms": 0,
            "total_cost": 0.0,
            "failures": [],
        }
        
        tests = results.get("results", [])
        summary["total_tests"] = len(tests)
        
        latencies = []
        costs = []
        
        for test in tests:
            if test.get("success"):
                summary["passed"] += 1
            else:
                summary["failed"] += 1
                summary["failures"].append({
                    "description": test.get("description", "Unknown"),
                    "error": test.get("error", "Assertion failed"),
                })
            
            if "latencyMs" in test:
                latencies.append(test["latencyMs"])
            if "cost" in test:
                costs.append(test["cost"])
        
        if summary["total_tests"] > 0:
            summary["pass_rate"] = summary["passed"] / summary["total_tests"]
        
        if latencies:
            summary["avg_latency_ms"] = sum(latencies) / len(latencies)
        
        if costs:
            summary["total_cost"] = sum(costs)
        
        return summary
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_format: str = "markdown",
    ) -> str:
        """
        Generate human-readable report.
        
        Args:
            results: Evaluation results
            output_format: Output format (markdown, text)
            
        Returns:
            Formatted report string
        """
        summary = self.parse_results(results)
        
        if output_format == "markdown":
            return self._generate_markdown_report(summary)
        else:
            return self._generate_text_report(summary)
    
    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        report = [
            "# RAG Evaluation Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tests | {summary['total_tests']} |",
            f"| Passed | {summary['passed']} |",
            f"| Failed | {summary['failed']} |",
            f"| Pass Rate | {summary['pass_rate']:.1%} |",
            f"| Avg Latency | {summary['avg_latency_ms']:.0f}ms |",
            f"| Total Cost | ${summary['total_cost']:.4f} |",
            "",
        ]
        
        if summary["failures"]:
            report.extend([
                "## Failures",
                "",
            ])
            for failure in summary["failures"]:
                report.append(f"- **{failure['description']}**: {failure['error']}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_text_report(self, summary: Dict[str, Any]) -> str:
        """Generate plain text report."""
        lines = [
            "=== RAG Evaluation Report ===",
            "",
            f"Total Tests: {summary['total_tests']}",
            f"Passed: {summary['passed']}",
            f"Failed: {summary['failed']}",
            f"Pass Rate: {summary['pass_rate']:.1%}",
            f"Avg Latency: {summary['avg_latency_ms']:.0f}ms",
            f"Total Cost: ${summary['total_cost']:.4f}",
        ]
        
        if summary["failures"]:
            lines.extend(["", "Failures:"])
            for failure in summary["failures"]:
                lines.append(f"  - {failure['description']}: {failure['error']}")
        
        return "\n".join(lines)


def run_evaluation(
    config_path: str = "eval/promptfooconfig.yaml",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run evaluation.
    
    Args:
        config_path: Path to config file
        verbose: Show detailed output
        
    Returns:
        Evaluation summary
    """
    runner = PromptfooRunner(config_path)
    results = runner.run_eval(verbose=verbose)
    return runner.parse_results(results)


if __name__ == "__main__":
    # Run evaluation when executed directly
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    runner = PromptfooRunner()
    results = runner.run_eval()
    
    if "error" in results:
        print(f"Evaluation failed: {results['error']}")
        sys.exit(1)
    
    report = runner.generate_report(results)
    print(report)
