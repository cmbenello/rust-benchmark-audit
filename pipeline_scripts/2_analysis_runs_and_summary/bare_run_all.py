#!/usr/bin/env python3
"""Run a batch of instances and emit JSONL + summary CSV."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


BENCHMARK_NAMES = {
    "swe-bench_plus-plus": "TuringEnterprises/SWE-Bench-plus-plus",
    "swe-bench_multilingual":"SWE-bench/SWE-bench_Multilingual",
    "multi-swe-bench":"ByteDance-Seed/Multi-SWE-bench"
}
MUTATIONS = ['unsafe', 'unwrap', 'panic!']

# Allow imports from the refactored pipeline directory tree.
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR / "1_patch_mutate_and_eval"))
sys.path.insert(0, str(PIPELINE_DIR / "0_data_construction"))

from policy_checks import count_from_bm_diff
from mutate_patch import mutate_patch_text
from swebench_eval import create_predictions_from_mutated_instances, evaluate_predictions

def load_instances_from_jsonl(filepath: str | Path) -> List[dict]:
    """Load instances from a JSONL file.
    
    Args:
        filepath: Path to the instances JSONL file
        
    Returns:
        List of instance dictionaries
    """
    instances = []
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                instances.append(json.loads(line))
    
    return instances

def instances_policy_checks(instances: List[dict]) -> List[dict]:
    """Run policy checks on a list of instances and return results.
    
    Args:
        instances: List of instance dictionaries
        
    Returns:
        List of dictionaries with source_benchmark, instance_id, and policy_count_results
    """    
    results = []
    for instance in instances:
        instance_id = instance.get("instance_id")
        source_benchmark = instance.get("source_benchmark")
        policy_count_results = count_from_bm_diff(instance.get("fix_patch", ""))
        #TODOD maybe something for tests too?
        results.append({
            "source_benchmark": source_benchmark,
            "instance_id": instance_id,
            "policy_count_results": policy_count_results
        })
    return results

def save_policy_results_to_jsonl(results: List[dict], output_path: str | Path) -> None:
    """Save policy check results to a JSONL file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save the JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

def patch_mutation(instances: List[dict]) -> List[dict]:
    """Apply mutations to the patches of the given instances.
    
    Args:
        instances: List of instance dictionaries
        
    Returns:
        List of mutated instance dictionaries
    """
    mutated_instances = []
    for instance in instances:
        instance['hf_bm'] = BENCHMARK_NAMES.get(instance.get('source_benchmark'))
        #save the gs diff
        diff = instance.get("fix_patch", "")
        #save gs to mutated_instances for evaluation later
        gs_instance = instance.copy()
        gs_instance['mutation'] = 'gs'
        gs_instance['diff'] = diff
        mutated_instances.append(gs_instance)
        for mutation in MUTATIONS:
            #for each of the three types of patch mutations
            mutated_instance = instance.copy()
            mutated_instance['mutation'] = mutation
            mutated_diff = mutate_patch_text(diff, mutation)
            mutated_instance['diff'] = mutated_diff
            mutated_instances.append(mutated_instance)
    return mutated_instances

def mutations_evaluation(mutated_instances: List[dict]) -> List[dict]:
    """Evaluate the mutated instances and return results.
    
    Args:
        mutated_instances: List of mutated instance dictionaries
    """
    prediction_paths = create_predictions_from_mutated_instances(mutated_instances)
    #prediction_paths = {'SWE-bench/SWE-bench_Multilingual': prediction_paths['SWE-bench/SWE-bench_Multilingual']}
    #evaluation_results = evaluate_predictions(prediction_paths)
    #print(evaluation_results)

def main() -> int:
    filtered_instances = load_instances_from_jsonl("data/instances_unified.jsonl")
    #policy_results = instances_policy_checks(filtered_instances)
    #save_policy_results_to_jsonl(policy_results, "results/policy_check_results.jsonl")
    mutated_instances = patch_mutation(filtered_instances)
    create_predictions_from_mutated_instances(mutated_instances)
    return 0

if __name__ == "__main__":
    main()
