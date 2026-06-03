#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_buf.rs b/tests/test_buf.rs
--- a/tests/test_buf.rs
+++ b/tests/test_buf.rs
@@ -33,15 +33,15 @@ fn test_get_u8() {
 #[test]
 fn test_get_u16() {
     let buf = b"\x21\x54zomg";
-    assert_eq!(0x2154, Cursor::new(buf).get_u16::<byteorder::BigEndian>());
-    assert_eq!(0x5421, Cursor::new(buf).get_u16::<byteorder::LittleEndian>());
+    assert_eq!(0x2154, Cursor::new(buf).get_u16_be());
+    assert_eq!(0x5421, Cursor::new(buf).get_u16_le());
 }
 
 #[test]
 #[should_panic]
 fn test_get_u16_buffer_underflow() {
     let mut buf = Cursor::new(b"\x21");
-    buf.get_u16::<byteorder::BigEndian>();
+    buf.get_u16_be();
 }
 
 #[test]
diff --git a/tests/test_buf_mut.rs b/tests/test_buf_mut.rs
--- a/tests/test_buf_mut.rs
+++ b/tests/test_buf_mut.rs
@@ -41,11 +41,11 @@ fn test_put_u8() {
 #[test]
 fn test_put_u16() {
     let mut buf = Vec::with_capacity(8);
-    buf.put_u16::<byteorder::BigEndian>(8532);
+    buf.put_u16_be(8532);
     assert_eq!(b"\x21\x54", &buf[..]);
 
     buf.clear();
-    buf.put_u16::<byteorder::LittleEndian>(8532);
+    buf.put_u16_le(8532);
     assert_eq!(b"\x54\x21", &buf[..]);
 }
 

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 86c83959dc0f72c94bcb2b6aa57efc178f6a7fa2
git clean -fd
