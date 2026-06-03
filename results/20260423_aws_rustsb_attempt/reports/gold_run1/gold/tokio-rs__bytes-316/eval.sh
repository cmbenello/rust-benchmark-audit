#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_buf_mut.rs b/tests/test_buf_mut.rs
--- a/tests/test_buf_mut.rs
+++ b/tests/test_buf_mut.rs
@@ -74,7 +74,7 @@ fn test_bufs_vec_mut() {
     // with no capacity
     let mut buf = BytesMut::new();
     assert_eq!(buf.capacity(), 0);
-    assert_eq!(0, buf.bytes_vectored_mut(&mut dst[..]));
+    assert_eq!(1, buf.bytes_vectored_mut(&mut dst[..]));
 
     // with capacity
     let mut buf = BytesMut::with_capacity(64);
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -2,6 +2,8 @@
 
 use bytes::{Bytes, BytesMut, Buf, BufMut};
 
+use std::usize;
+
 const LONG: &'static [u8] = b"mary had a little lamb, little lamb, little lamb";
 const SHORT: &'static [u8] = b"hello world";
 
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -93,8 +95,8 @@ fn fmt_write() {
 
 
     let mut c = BytesMut::with_capacity(64);
-    write!(c, "{}", s).unwrap_err();
-    assert!(c.is_empty());
+    write!(c, "{}", s).unwrap();
+    assert_eq!(c, s[..].as_bytes());
 }
 
 #[test]
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -820,3 +822,34 @@ fn empty_slice_ref_catches_not_an_empty_subset() {
 
     bytes.slice_ref(slice);
 }
+
+#[test]
+fn bytes_buf_mut_advance() {
+    let mut bytes = BytesMut::with_capacity(1024);
+
+    unsafe {
+        let ptr = bytes.bytes_mut().as_ptr();
+        assert_eq!(1024, bytes.bytes_mut().len());
+
+        bytes.advance_mut(10);
+
+        let next = bytes.bytes_mut().as_ptr();
+        assert_eq!(1024 - 10, bytes.bytes_mut().len());
+        assert_eq!(ptr.offset(10), next);
+
+        // advance to the end
+        bytes.advance_mut(1024 - 10);
+
+        // The buffer size is doubled
+        assert_eq!(1024, bytes.bytes_mut().len());
+    }
+}
+
+#[test]
+#[should_panic]
+fn bytes_reserve_overflow() {
+    let mut bytes = BytesMut::with_capacity(1024);
+    bytes.put_slice(b"hello world");
+
+    bytes.reserve(usize::MAX);
+}
diff --git a/tests/test_chain.rs b/tests/test_chain.rs
--- a/tests/test_chain.rs
+++ b/tests/test_chain.rs
@@ -1,7 +1,7 @@
 #![deny(warnings, rust_2018_idioms)]
 
-use bytes::{Buf, BufMut, Bytes, BytesMut};
-use bytes::buf::BufExt;
+use bytes::{Buf, BufMut, Bytes};
+use bytes::buf::{BufExt, BufMutExt};
 use std::io::IoSlice;
 
 #[test]
diff --git a/tests/test_chain.rs b/tests/test_chain.rs
--- a/tests/test_chain.rs
+++ b/tests/test_chain.rs
@@ -15,20 +15,17 @@ fn collect_two_bufs() {
 
 #[test]
 fn writing_chained() {
-    let mut a = BytesMut::with_capacity(64);
-    let mut b = BytesMut::with_capacity(64);
+    let mut a = [0u8; 64];
+    let mut b = [0u8; 64];
 
     {
-        let mut buf = (&mut a).chain(&mut b);
+        let mut buf = (&mut a[..]).chain_mut(&mut b[..]);
 
         for i in 0u8..128 {
             buf.put_u8(i);
         }
     }
 
-    assert_eq!(64, a.len());
-    assert_eq!(64, b.len());
-
     for i in 0..64 {
         let expect = i as u8;
         assert_eq!(expect, a[i]);

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 59aea9e8719d8acff18b586f859011d3c52cfcde
git clean -fd
