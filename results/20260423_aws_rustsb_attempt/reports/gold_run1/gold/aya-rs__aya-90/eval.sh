#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/aya/src/obj/mod.rs b/aya/src/obj/mod.rs
--- a/aya/src/obj/mod.rs
+++ b/aya/src/obj/mod.rs
@@ -607,6 +611,7 @@ mod tests {
             address: 0,
             name,
             data,
+            size: data.len() as u64,
             relocations: Vec::new(),
         }
     }

EOF_114329324912
cd "aya"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 3a8e4fe9b91538a0fafd8c91ae96185c1a017651
git clean -fd
