#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -169,6 +169,34 @@ fn gnu_compile_assembly() {
     test.cmd(0).must_have("foo.S");
 }
 
+#[test]
+fn gnu_shared() {
+    let test = Test::gnu();
+    test.gcc()
+        .file("foo.c")
+        .shared_flag(true)
+        .static_flag(false)
+        .compile("libfoo.a");
+
+    test.cmd(0)
+        .must_have("-shared")
+        .must_not_have("-static");
+}
+
+#[test]
+fn gnu_static() {
+    let test = Test::gnu();
+    test.gcc()
+        .file("foo.c")
+        .shared_flag(false)
+        .static_flag(true)
+        .compile("libfoo.a");
+
+    test.cmd(0)
+        .must_have("-static")
+        .must_not_have("-shared");
+}
+
 #[test]
 fn msvc_smoke() {
     let test = Test::msvc();

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 0f6c1424aec4994cb6736226b400e88f56fd829e
git clean -fd
