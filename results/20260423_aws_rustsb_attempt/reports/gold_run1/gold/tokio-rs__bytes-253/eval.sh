#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -258,15 +258,10 @@ fn split_to_oob_mut() {
 }
 
 #[test]
+#[should_panic]
 fn split_to_uninitialized() {
     let mut bytes = BytesMut::with_capacity(1024);
-    let other = bytes.split_to(128);
-
-    assert_eq!(bytes.len(), 0);
-    assert_eq!(bytes.capacity(), 896);
-
-    assert_eq!(other.len(), 0);
-    assert_eq!(other.capacity(), 128);
+    let _other = bytes.split_to(128);
 }
 
 #[test]

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard e0e30f00a1248b1de59405da66cd871ccace4f9f
git clean -fd
