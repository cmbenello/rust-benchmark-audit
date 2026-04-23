#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/integration.rs b/tests/integration.rs
--- a/tests/integration.rs
+++ b/tests/integration.rs
@@ -101,6 +101,11 @@ mod integration {
         structured_test(include_str!("structured_tests/total_agg.toml"));
     }
 
+    #[test]
+    fn fields_after_agg_bug() {
+        structured_test(include_str!("structured_tests/fields_after_agg.toml"));
+    }
+
     #[test]
     fn limit() {
         structured_test(include_str!("structured_tests/limit.toml"));
diff --git /dev/null b/tests/structured_tests/fields_after_agg.toml
new file mode 100644
--- /dev/null
+++ b/tests/structured_tests/fields_after_agg.toml
@@ -0,0 +1,10 @@
+query = ' * | parse ":* (" as port | count by port | fields port'
+input = ": 45 ("
+output = """
+port
+------------
+45
+"""
+error = """
+"""
+succeeds = true

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard b264366cb68b0f649486d70b8522b43bb3392865
git clean -fd
