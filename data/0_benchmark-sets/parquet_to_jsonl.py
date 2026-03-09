import pyarrow.parquet as pq
import json

instance_ids = ["uutils__coreutils-8478", "aptos-labs__aptos-core-16152", "unicode-org__icu4x-6776"]

table = pq.read_table("data/benchmark-sets/20260218_swe-bench_plus-plus.parquet")
df = table.to_pandas()

with open("data/benchmark-sets/filtered_20260225_swe-bench_plus-plus.jsonl", "w") as f:
    for _, row in df.iterrows():
        if row["instance_id"] in instance_ids:
            f.write(json.dumps(row.to_dict()) + "\n")