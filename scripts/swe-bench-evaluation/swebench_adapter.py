"""
Adapter to integrate SWE-bench evaluation with rust-benchmark-audit pipeline
"""
import json
from pathlib import Path
from typing import List, Dict
from swebench.harness.run_evaluation import run_instances
from swebench.harness.test_spec import make_test_spec
import docker

class SWEBenchAdapter:
    """Wraps SWE-bench evaluation for rust-benchmark-audit"""
    def __init__(self, work_dir: Path, output_dir: Path):
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.client = docker.from_env()

    def run_swebench_evaluation(
            self, 
            predictions: List[Dict],
            dataset_name: str,
            max_workers: int = 4,
            run_id: str = "rust-audit"
            ) -> Dict:
        """
        Run SWE-bench evaluation on predictions
        
        Args:
            predictions: List of predictions in SWE-bench format
            dataset_name: HuggingFace dataset name
            max_workers: Number of parallel workers
            run_id: Identifier for this evaluation run
            
        Returns:
            Evaluation results dictionary
        """
        # Save predictions to file
        pred_file = self.output_dir / f"predictions_{run_id}.jsonl"
        with open(pred_file, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
        
        # Run SWE-bench evaluation
        from swebench.harness.run_evaluation import main as run_eval_main
        
        # https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md
        results = run_eval_main(
            instance_ids=predictions["instance"],
            dataset_name=predications["dataset_name"],
            predictions_path=str(pred_file),
            max_workers=max_workers,
            run_id=run_id,
            timeout=1800  # 30 minutes per instance
        )
        
        return results
