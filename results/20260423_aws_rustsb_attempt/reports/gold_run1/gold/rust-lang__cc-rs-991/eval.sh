#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -617,3 +617,33 @@ fn compile_intermediates() {
     assert!(intermediates[1].display().to_string().contains("x86_64"));
     assert!(intermediates[2].display().to_string().contains("x86_64"));
 }
+
+#[test]
+fn clang_android() {
+    let target = "arm-linux-androideabi";
+
+    // On Windows, we don't use the Android NDK shims for Clang, so verify that
+    // we use "clang" and set the target correctly.
+    #[cfg(windows)]
+    {
+        let test = Test::new();
+        test.shim("clang").shim("llvm-ar");
+        test.gcc()
+            .target(target)
+            .host("x86_64-pc-windows-msvc")
+            .file("foo.c")
+            .compile("foo");
+        test.cmd(0).must_have("--target=arm-linux-androideabi");
+    }
+
+    // On non-Windows, we do use the shims, so make sure that we use the shim
+    // and don't set the target.
+    #[cfg(not(windows))]
+    {
+        let test = Test::new();
+        test.shim("arm-linux-androideabi-clang")
+            .shim("arm-linux-androideabi-ar");
+        test.gcc().target(target).file("foo.c").compile("foo");
+        test.cmd(0).must_not_have("--target=arm-linux-androideabi");
+    }
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 915420c16c134e6bb458ce634c663305a0ef2504
git clean -fd
