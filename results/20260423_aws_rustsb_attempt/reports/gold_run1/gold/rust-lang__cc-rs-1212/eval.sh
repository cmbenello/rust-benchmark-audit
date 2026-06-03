#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -269,6 +269,20 @@ fn gnu_x86_64_no_plt() {
     test.cmd(0).must_have("-fno-plt");
 }
 
+#[test]
+fn gnu_aarch64_none_no_pic() {
+    for target in &["aarch64-unknown-none-gnu", "aarch64-unknown-none"] {
+        let test = Test::gnu();
+        test.gcc()
+            .target(&target)
+            .host(&target)
+            .file("foo.c")
+            .compile("foo");
+
+        test.cmd(0).must_not_have("-fPIC");
+    }
+}
+
 #[test]
 fn gnu_set_stdlib() {
     reset_env();

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 92a5e28c084aefc101d2f71f236614d9fcaa1f9e
git clean -fd
