This is the mostly-replication package for 'Is it correct to be 'unsafe'? A pilot study of Rust safety in CodeGen SE benchmarks' [A true replication package would subset this directory and delete out any unused data, scripts, or printouts; for posterity and ongoing development, these superfluous files are kept in for now.]

In /data we provide all benchmark data sets and mutations that were used in our pilot analysis. 
    /data/0_benchmark-sets contains the full versions of the three unmutated benchmarks in our study 
    /data/1_manually_sampled_data contains the rows which were manually sampeld from each of those data sets for project coding policies --- sampled by benchmark as well as unified.
    /data/2_frozen_mutated_patches contains the claude-mutated sampled patches (gold standard, panic, unsafe, unwrap)

In /pipeline_scripts we provide all code used in our analysis pipeline 
    /pipeline_scripts/0_data_construction contains relevant scripts for checking the presence of policy-risky language within gold standard patches --- per-project-safety-constructs.txt contains manual annotations to project safety policies 
    /pipeline_scripts/1_analysis_runs_and_summary contains scripts for the analysis pipeline, more information can be found in TODO README.

/results contains the outputs from the evaluation harnesses. 
    /results/20260225_results contains the final policy check results for gold-standard patches 
    /results/20260309_harness_eval_results contains all results from the swe-bench evaluation harness for multilingual and plus-plus