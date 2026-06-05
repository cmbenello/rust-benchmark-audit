import json
import sys
from pathlib import Path
 
def convert(base_dir, model_name, output_path):
    base = Path(base_dir)
    records = []
 
    for patch_file in sorted(base.rglob("patch.diff")):
        instance_id = patch_file.parent.name
        diff_content = patch_file.read_text()
        records.append({
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": diff_content,
        })
 
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
 
    print(f"Wrote {len(records)} records to {output_path}")
 
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: pipeline_scripts/0_data_construction/collate_adv_mutations.py <base_dir> <model_name> <output.jsonl>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
