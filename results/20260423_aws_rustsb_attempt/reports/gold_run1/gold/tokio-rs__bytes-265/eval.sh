#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -98,22 +98,22 @@ fn index() {
 fn slice() {
     let a = Bytes::from(&b"hello world"[..]);
 
-    let b = a.slice(3, 5);
+    let b = a.slice(3..5);
     assert_eq!(b, b"lo"[..]);
 
-    let b = a.slice(0, 0);
+    let b = a.slice(0..0);
     assert_eq!(b, b""[..]);
 
-    let b = a.slice(3, 3);
+    let b = a.slice(3..3);
     assert_eq!(b, b""[..]);
 
-    let b = a.slice(a.len(), a.len());
+    let b = a.slice(a.len()..a.len());
     assert_eq!(b, b""[..]);
 
-    let b = a.slice_to(5);
+    let b = a.slice(..5);
     assert_eq!(b, b"hello"[..]);
 
-    let b = a.slice_from(3);
+    let b = a.slice(3..);
     assert_eq!(b, b"lo world"[..]);
 }
 
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -121,14 +121,14 @@ fn slice() {
 #[should_panic]
 fn slice_oob_1() {
     let a = Bytes::from(&b"hello world"[..]);
-    a.slice(5, inline_cap() + 1);
+    a.slice(5..(inline_cap() + 1));
 }
 
 #[test]
 #[should_panic]
 fn slice_oob_2() {
     let a = Bytes::from(&b"hello world"[..]);
-    a.slice(inline_cap() + 1, inline_cap() + 5);
+    a.slice((inline_cap() + 1)..(inline_cap() + 5));
 }
 
 #[test]
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -689,9 +689,9 @@ fn bytes_unsplit_two_split_offs() {
 fn bytes_unsplit_overlapping_references() {
     let mut buf = Bytes::with_capacity(64);
     buf.extend_from_slice(b"abcdefghijklmnopqrstuvwxyz");
-    let mut buf0010 = buf.slice(0, 10);
-    let buf1020 = buf.slice(10, 20);
-    let buf0515 = buf.slice(5, 15);
+    let mut buf0010 = buf.slice(0..10);
+    let buf1020 = buf.slice(10..20);
+    let buf0515 = buf.slice(5..15);
     buf0010.unsplit(buf1020);
     assert_eq!(b"abcdefghijklmnopqrst", &buf0010[..]);
     assert_eq!(b"fghijklmno", &buf0515[..]);

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard ed7c7b5c5845a4084eb65a364ddf259b1e17ca59
git clean -fd
