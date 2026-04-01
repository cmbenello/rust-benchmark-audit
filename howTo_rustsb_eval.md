### 1. Clone the repository and install the required dependencies:

```
#clone repo 
git clone git@github.com:GhabiX/Rust-SWE-Bench.git

# Navigate into the project directory
cd Rust-bench

# Install the package in editable mode
pip install -e .
```

### 2. Set the environment variable `GITHUB_TOKENS` to a current GitHub Access Token. 

### 3. To build the instance images, use the commands below:

```
cd Rust-bench/swebench/harness
  
python run_evaluation.py \
    --dataset_name user2f86/rustbench\
    --run_id rustbench \
    --max_workers 10\
    --cache_level instance \
    --predictions_path gold \
    --split train \
    --config_path swebench/harness/logs/config.json \
    --build_image_only 1
```

> [!TIP]
> Running evaluations can be resource-intensive. For optimal performance, we recommend the following system specifications:
>
> - **Architecture:** x86_64
> - **Storage:** At least 120GB of free space
> - **RAM:** 16GB or more
> - **CPU:** 8 cores or more
>
> You may need to experiment with the `--max_workers` argument to find the optimal number of workers for your machine, but we recommend using fewer than `min(0.75 * os.cpu_count(), 24)`.

### 4. Run the local, custom evaluation harness

```
python -m swebench.harness.run_evaluation \
    --dataset_name user2f86/rustbench \
    --predictions_path <path_to_your_predictions> \
    --max_workers <num_workers> \
    --run_id <your_run_id> \
    --split train\
    --config_path Rust-bench/swebench/harness/logs/config.json \
    --build_image_only 0
```