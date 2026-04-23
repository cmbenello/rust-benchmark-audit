#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_buf_mut.rs b/tests/test_buf_mut.rs
--- a/tests/test_buf_mut.rs
+++ b/tests/test_buf_mut.rs
@@ -75,7 +75,7 @@ fn test_mut_slice() {
 fn test_deref_bufmut_forwards() {
     struct Special;
 
-    impl BufMut for Special {
+    unsafe impl BufMut for Special {
         fn remaining_mut(&self) -> usize {
             unreachable!("remaining_mut");
         }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 94c543f74b111e894d16faa43e4ad361b97ee87d
git clean -fd
