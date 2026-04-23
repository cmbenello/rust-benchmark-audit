#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cli/src/utils/inspect.rs b/crates/cli/src/utils/inspect.rs
--- a/crates/cli/src/utils/inspect.rs
+++ b/crates/cli/src/utils/inspect.rs
@@ -186,7 +212,10 @@ mod test {
       0
     );
     assert!(run_trace.print().is_ok());
-    assert_eq!(ret, "Files scanned: 0, Files skipped: 0\n");
+    assert_eq!(
+      ret,
+      "sg: summary|file: scannedFileCount=0,skippedFileCount=0\n"
+    );
 
     let mut ret = String::new();
     let rule_stats = RuleTrace {
diff --git a/crates/cli/src/utils/inspect.rs b/crates/cli/src/utils/inspect.rs
--- a/crates/cli/src/utils/inspect.rs
+++ b/crates/cli/src/utils/inspect.rs
@@ -208,7 +237,9 @@ mod test {
     assert!(scan_trace.print().is_ok());
     assert_eq!(
       ret,
-      "Files scanned: 0, Files skipped: 0\nEffective rules: 10, Skipped rules: 2\n"
+      r"sg: summary|file: scannedFileCount=0,skippedFileCount=0
+sg: summary|rule: effectiveRuleCount=10,skippedRuleCount=2
+"
     );
   }
 
diff --git a/crates/cli/tests/run_test.rs b/crates/cli/tests/run_test.rs
--- a/crates/cli/tests/run_test.rs
+++ b/crates/cli/tests/run_test.rs
@@ -57,6 +57,6 @@ fn test_inspect() -> Result<()> {
     .assert()
     .success()
     .stdout(contains("alert(1)"))
-    .stderr(contains("Files scanned: 2"));
+    .stderr(contains("scannedFileCount=2"));
   Ok(())
 }

EOF_114329324912
cd "crates/cli"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard ccdc53168d6fbcff91e9ac095a3ef49eb847a410
git clean -fd
