#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -363,6 +363,15 @@ fn extend() {
     assert_eq!(*bytes, LONG[..]);
 }
 
+#[test]
+fn from_static() {
+    let mut a = Bytes::from_static(b"ab");
+    let b = a.split_off(1);
+
+    assert_eq!(a, b"a"[..]);
+    assert_eq!(b, b"b"[..]);
+}
+
 #[test]
 // Only run these tests on little endian systems. CI uses qemu for testing
 // little endian... and qemu doesn't really support threading all that well.

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 627864187c531af38c21aa44315a1b3204f9a175
git clean -fd
