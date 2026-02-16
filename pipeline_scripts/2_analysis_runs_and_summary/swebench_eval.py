
import pandas as pd
from swebench.harness import run_evaluation
import json
import sys
import subprocess

def create_predictions_from_dataframe(df, benchmark):
    '''
    output prediction format
    {
    "instance_id": "repo_owner__repo_name-issue_number",
    "model_name_or_path": "your-model-name",
    "model_patch": "the patch content as a string"
    }
    '''
    predictions = []
    for _, row in df.iterrows():
        instance_id = f"{row['instance_id']}"
        model_name_or_path = row['augmentation']
        model_patch = row['patch_diff']
        
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch
        }
        predictions.append(prediction)
    benchmark_path = benchmark.replace('/', '_')
    predictions_path = f"{benchmark_path}_formatted.json"
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    return predictions_path


def evaluate_predictions(predictions_paths):
    results = {}
    for benchmark in predictions_paths.keys():
        print(f"Evaluating benchmark: {benchmark}")
        
        cmd = [
            sys.executable, '-m', 'swebench.harness.run_evaluation',
            '--predictions_path', predictions_paths[benchmark],
            '--dataset_name', benchmark,
            '--max_workers', '4',
            '--run_id', f"{benchmark}_evaluation"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        results[benchmark] = result.stdout
        benchmark_path = benchmark.replace('/', '_')
        with open(f"{benchmark_path}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # testing with swebench-lite gs
    df = pd.read_csv("data/benchmark-sets/test-swebench-lite.csv")
    df['benchmark'] = 'swe-bench/swe-bench_lite'
    df['patch_diff'] = df['patch']
    df['augmentation'] = 'gs'
    
    # Create predictions in the required format for evaluation
    benchmarks = df['benchmark'].unique()
    predictions_paths = {}
    for benchmark in benchmarks:
        benchmark_df = df[df['benchmark'] == benchmark]
        predictions_paths[benchmark] = create_predictions_from_dataframe(benchmark_df, benchmark)
    
    # Evaluate the predictions and save results
    results = evaluate_predictions(predictions_paths)
    print(results)
    
    #cleaning up
    for benchmark, path in predictions_paths.items():
        print(f"Removing temporary predictions file: {benchmark} - {path}")
        os.remove(path)
