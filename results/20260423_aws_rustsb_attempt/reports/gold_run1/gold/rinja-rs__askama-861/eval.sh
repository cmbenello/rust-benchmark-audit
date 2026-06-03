#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_parser/src/tests.rs b/askama_parser/src/tests.rs
--- a/askama_parser/src/tests.rs
+++ b/askama_parser/src/tests.rs
@@ -788,3 +788,10 @@ fn test_parse_array() {
         )],
     );
 }
+
+#[test]
+fn fuzzed_unicode_slice() {
+    let d = "{eeuuu{b&{!!&{!!11{{
+            0!(!1q҄א!)!!!!!!n!";
+    assert!(Ast::from_str(d, &Syntax::default()).is_err());
+}

EOF_114329324912
cd "askama_parser"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 43e92aa3b6b9cd967a70bd0fd54d1f087d6ed76b
git clean -fd
