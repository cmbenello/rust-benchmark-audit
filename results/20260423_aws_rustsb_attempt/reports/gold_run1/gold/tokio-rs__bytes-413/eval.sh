#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -929,6 +929,22 @@ fn bytes_buf_mut_advance() {
     }
 }
 
+#[test]
+fn bytes_buf_mut_reuse_when_fully_consumed() {
+    use bytes::{Buf, BytesMut};
+    let mut buf = BytesMut::new();
+    buf.reserve(8192);
+    buf.extend_from_slice(&[0u8; 100][..]);
+
+    let p = &buf[0] as *const u8;
+    buf.advance(100);
+
+    buf.reserve(8192);
+    buf.extend_from_slice(b" ");
+
+    assert_eq!(&buf[0] as *const u8, p);
+}
+
 #[test]
 #[should_panic]
 fn bytes_reserve_overflow() {

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 90e7e650c99b6d231017d9b831a01e95b8c06756
git clean -fd
