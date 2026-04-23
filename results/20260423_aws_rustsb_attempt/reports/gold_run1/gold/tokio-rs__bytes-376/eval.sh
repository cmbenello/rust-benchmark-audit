#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -342,6 +342,72 @@ fn freeze_clone_unique() {
     assert_eq!(c, s);
 }
 
+#[test]
+fn freeze_after_advance() {
+    let s = &b"abcdefgh"[..];
+    let mut b = BytesMut::from(s);
+    b.advance(1);
+    assert_eq!(b, s[1..]);
+    let b = b.freeze();
+    // Verify fix for #352. Previously, freeze would ignore the start offset
+    // for BytesMuts in Vec mode.
+    assert_eq!(b, s[1..]);
+}
+
+#[test]
+fn freeze_after_advance_arc() {
+    let s = &b"abcdefgh"[..];
+    let mut b = BytesMut::from(s);
+    // Make b Arc
+    let _ = b.split_to(0);
+    b.advance(1);
+    assert_eq!(b, s[1..]);
+    let b = b.freeze();
+    assert_eq!(b, s[1..]);
+}
+
+#[test]
+fn freeze_after_split_to() {
+    let s = &b"abcdefgh"[..];
+    let mut b = BytesMut::from(s);
+    let _ = b.split_to(1);
+    assert_eq!(b, s[1..]);
+    let b = b.freeze();
+    assert_eq!(b, s[1..]);
+}
+
+#[test]
+fn freeze_after_truncate() {
+    let s = &b"abcdefgh"[..];
+    let mut b = BytesMut::from(s);
+    b.truncate(7);
+    assert_eq!(b, s[..7]);
+    let b = b.freeze();
+    assert_eq!(b, s[..7]);
+}
+
+#[test]
+fn freeze_after_truncate_arc() {
+    let s = &b"abcdefgh"[..];
+    let mut b = BytesMut::from(s);
+    // Make b Arc
+    let _ = b.split_to(0);
+    b.truncate(7);
+    assert_eq!(b, s[..7]);
+    let b = b.freeze();
+    assert_eq!(b, s[..7]);
+}
+
+#[test]
+fn freeze_after_split_off() {
+    let s = &b"abcdefgh"[..];
+    let mut b = BytesMut::from(s);
+    let _ = b.split_off(7);
+    assert_eq!(b, s[..7]);
+    let b = b.freeze();
+    assert_eq!(b, s[..7]);
+}
+
 #[test]
 fn fns_defined_for_bytes_mut() {
     let mut bytes = BytesMut::from(&b"hello world"[..]);

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard fe6e67386451715c5d609c90a41e98ef80f0e1d1
git clean -fd
