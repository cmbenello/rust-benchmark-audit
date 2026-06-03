#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -1174,29 +1174,29 @@ fn shared_is_unique() {
 }
 
 #[test]
-fn test_bytes_make_mut_static() {
+fn test_bytesmut_from_bytes_static() {
     let bs = b"1b23exfcz3r";
 
     // Test STATIC_VTABLE.to_mut
-    let bytes_mut = Bytes::from_static(bs).make_mut();
+    let bytes_mut = BytesMut::from(Bytes::from_static(bs));
     assert_eq!(bytes_mut, bs[..]);
 }
 
 #[test]
-fn test_bytes_make_mut_bytes_mut_vec() {
+fn test_bytesmut_from_bytes_bytes_mut_vec() {
     let bs = b"1b23exfcz3r";
     let bs_long = b"1b23exfcz3r1b23exfcz3r";
 
     // Test case where kind == KIND_VEC
     let mut bytes_mut: BytesMut = bs[..].into();
-    bytes_mut = bytes_mut.freeze().make_mut();
+    bytes_mut = BytesMut::from(bytes_mut.freeze());
     assert_eq!(bytes_mut, bs[..]);
     bytes_mut.extend_from_slice(&bs[..]);
     assert_eq!(bytes_mut, bs_long[..]);
 }
 
 #[test]
-fn test_bytes_make_mut_bytes_mut_shared() {
+fn test_bytesmut_from_bytes_bytes_mut_shared() {
     let bs = b"1b23exfcz3r";
 
     // Set kind to KIND_ARC so that after freeze, Bytes will use bytes_mut.SHARED_VTABLE
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -1207,17 +1207,17 @@ fn test_bytes_make_mut_bytes_mut_shared() {
     let b2 = b1.clone();
 
     // shared.is_unique() = False
-    let mut b1m = b1.make_mut();
+    let mut b1m = BytesMut::from(b1);
     assert_eq!(b1m, bs[..]);
     b1m[0] = b'9';
 
     // shared.is_unique() = True
-    let b2m = b2.make_mut();
+    let b2m = BytesMut::from(b2);
     assert_eq!(b2m, bs[..]);
 }
 
 #[test]
-fn test_bytes_make_mut_bytes_mut_offset() {
+fn test_bytesmut_from_bytes_bytes_mut_offset() {
     let bs = b"1b23exfcz3r";
 
     // Test bytes_mut.SHARED_VTABLE.to_mut impl where offset != 0
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -1227,58 +1227,58 @@ fn test_bytes_make_mut_bytes_mut_offset() {
     let b1 = bytes_mut1.freeze();
     let b2 = bytes_mut2.freeze();
 
-    let b1m = b1.make_mut();
-    let b2m = b2.make_mut();
+    let b1m = BytesMut::from(b1);
+    let b2m = BytesMut::from(b2);
 
     assert_eq!(b2m, bs[9..]);
     assert_eq!(b1m, bs[..9]);
 }
 
 #[test]
-fn test_bytes_make_mut_promotable_even_vec() {
+fn test_bytesmut_from_bytes_promotable_even_vec() {
     let vec = vec![33u8; 1024];
 
     // Test case where kind == KIND_VEC
     let b1 = Bytes::from(vec.clone());
-    let b1m = b1.make_mut();
+    let b1m = BytesMut::from(b1);
     assert_eq!(b1m, vec);
 }
 
 #[test]
-fn test_bytes_make_mut_promotable_even_arc_1() {
+fn test_bytesmut_from_bytes_promotable_even_arc_1() {
     let vec = vec![33u8; 1024];
 
     // Test case where kind == KIND_ARC, ref_cnt == 1
     let b1 = Bytes::from(vec.clone());
     drop(b1.clone());
-    let b1m = b1.make_mut();
+    let b1m = BytesMut::from(b1);
     assert_eq!(b1m, vec);
 }
 
 #[test]
-fn test_bytes_make_mut_promotable_even_arc_2() {
+fn test_bytesmut_from_bytes_promotable_even_arc_2() {
     let vec = vec![33u8; 1024];
 
     // Test case where kind == KIND_ARC, ref_cnt == 2
     let b1 = Bytes::from(vec.clone());
     let b2 = b1.clone();
-    let b1m = b1.make_mut();
+    let b1m = BytesMut::from(b1);
     assert_eq!(b1m, vec);
 
     // Test case where vtable = SHARED_VTABLE, kind == KIND_ARC, ref_cnt == 1
-    let b2m = b2.make_mut();
+    let b2m = BytesMut::from(b2);
     assert_eq!(b2m, vec);
 }
 
 #[test]
-fn test_bytes_make_mut_promotable_even_arc_offset() {
+fn test_bytesmut_from_bytes_promotable_even_arc_offset() {
     let vec = vec![33u8; 1024];
 
     // Test case where offset != 0
     let mut b1 = Bytes::from(vec.clone());
     let b2 = b1.split_off(20);
-    let b1m = b1.make_mut();
-    let b2m = b2.make_mut();
+    let b1m = BytesMut::from(b1);
+    let b2m = BytesMut::from(b2);
 
     assert_eq!(b2m, vec[20..]);
     assert_eq!(b1m, vec[..20]);
diff --git a/tests/test_bytes_odd_alloc.rs b/tests/test_bytes_odd_alloc.rs
--- a/tests/test_bytes_odd_alloc.rs
+++ b/tests/test_bytes_odd_alloc.rs
@@ -6,7 +6,7 @@
 use std::alloc::{GlobalAlloc, Layout, System};
 use std::ptr;
 
-use bytes::Bytes;
+use bytes::{Bytes, BytesMut};
 
 #[global_allocator]
 static ODD: Odd = Odd;
diff --git a/tests/test_bytes_odd_alloc.rs b/tests/test_bytes_odd_alloc.rs
--- a/tests/test_bytes_odd_alloc.rs
+++ b/tests/test_bytes_odd_alloc.rs
@@ -97,50 +97,50 @@ fn test_bytes_into_vec() {
 }
 
 #[test]
-fn test_bytes_make_mut_vec() {
+fn test_bytesmut_from_bytes_vec() {
     let vec = vec![33u8; 1024];
 
     // Test case where kind == KIND_VEC
     let b1 = Bytes::from(vec.clone());
-    let b1m = b1.make_mut();
+    let b1m = BytesMut::from(b1);
     assert_eq!(b1m, vec);
 }
 
 #[test]
-fn test_bytes_make_mut_arc_1() {
+fn test_bytesmut_from_bytes_arc_1() {
     let vec = vec![33u8; 1024];
 
     // Test case where kind == KIND_ARC, ref_cnt == 1
     let b1 = Bytes::from(vec.clone());
     drop(b1.clone());
-    let b1m = b1.make_mut();
+    let b1m = BytesMut::from(b1);
     assert_eq!(b1m, vec);
 }
 
 #[test]
-fn test_bytes_make_mut_arc_2() {
+fn test_bytesmut_from_bytes_arc_2() {
     let vec = vec![33u8; 1024];
 
     // Test case where kind == KIND_ARC, ref_cnt == 2
     let b1 = Bytes::from(vec.clone());
     let b2 = b1.clone();
-    let b1m = b1.make_mut();
+    let b1m = BytesMut::from(b1);
     assert_eq!(b1m, vec);
 
     // Test case where vtable = SHARED_VTABLE, kind == KIND_ARC, ref_cnt == 1
-    let b2m = b2.make_mut();
+    let b2m = BytesMut::from(b2);
     assert_eq!(b2m, vec);
 }
 
 #[test]
-fn test_bytes_make_mut_arc_offset() {
+fn test_bytesmut_from_bytes_arc_offset() {
     let vec = vec![33u8; 1024];
 
     // Test case where offset != 0
     let mut b1 = Bytes::from(vec.clone());
     let b2 = b1.split_off(20);
-    let b1m = b1.make_mut();
-    let b2m = b2.make_mut();
+    let b1m = BytesMut::from(b1);
+    let b2m = BytesMut::from(b2);
 
     assert_eq!(b2m, vec[20..]);
     assert_eq!(b1m, vec[..20]);

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard caf520ac7f2c466d26bd88eca33ddc53c408e17e
git clean -fd
