#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_buf.rs b/tests/test_buf.rs
--- a/tests/test_buf.rs
+++ b/tests/test_buf.rs
@@ -9,17 +9,17 @@ fn test_fresh_cursor_vec() {
     let mut buf = &b"hello"[..];
 
     assert_eq!(buf.remaining(), 5);
-    assert_eq!(buf.bytes(), b"hello");
+    assert_eq!(buf.chunk(), b"hello");
 
     buf.advance(2);
 
     assert_eq!(buf.remaining(), 3);
-    assert_eq!(buf.bytes(), b"llo");
+    assert_eq!(buf.chunk(), b"llo");
 
     buf.advance(3);
 
     assert_eq!(buf.remaining(), 0);
-    assert_eq!(buf.bytes(), b"");
+    assert_eq!(buf.chunk(), b"");
 }
 
 #[test]
diff --git a/tests/test_buf.rs b/tests/test_buf.rs
--- a/tests/test_buf.rs
+++ b/tests/test_buf.rs
@@ -53,7 +53,7 @@ fn test_bufs_vec() {
 
     let mut dst = [IoSlice::new(b1), IoSlice::new(b2)];
 
-    assert_eq!(1, buf.bytes_vectored(&mut dst[..]));
+    assert_eq!(1, buf.chunks_vectored(&mut dst[..]));
 }
 
 #[test]
diff --git a/tests/test_buf.rs b/tests/test_buf.rs
--- a/tests/test_buf.rs
+++ b/tests/test_buf.rs
@@ -63,9 +63,9 @@ fn test_vec_deque() {
     let mut buffer: VecDeque<u8> = VecDeque::new();
     buffer.extend(b"hello world");
     assert_eq!(11, buffer.remaining());
-    assert_eq!(b"hello world", buffer.bytes());
+    assert_eq!(b"hello world", buffer.chunk());
     buffer.advance(6);
-    assert_eq!(b"world", buffer.bytes());
+    assert_eq!(b"world", buffer.chunk());
     buffer.extend(b" piece");
     let mut out = [0; 11];
     buffer.copy_to_slice(&mut out);
diff --git a/tests/test_buf.rs b/tests/test_buf.rs
--- a/tests/test_buf.rs
+++ b/tests/test_buf.rs
@@ -81,8 +81,8 @@ fn test_deref_buf_forwards() {
             unreachable!("remaining");
         }
 
-        fn bytes(&self) -> &[u8] {
-            unreachable!("bytes");
+        fn chunk(&self) -> &[u8] {
+            unreachable!("chunk");
         }
 
         fn advance(&mut self, _: usize) {
diff --git a/tests/test_buf_mut.rs b/tests/test_buf_mut.rs
--- a/tests/test_buf_mut.rs
+++ b/tests/test_buf_mut.rs
@@ -11,7 +11,7 @@ fn test_vec_as_mut_buf() {
 
     assert_eq!(buf.remaining_mut(), usize::MAX);
 
-    assert!(buf.bytes_mut().len() >= 64);
+    assert!(buf.chunk_mut().len() >= 64);
 
     buf.put(&b"zomg"[..]);
 
diff --git a/tests/test_buf_mut.rs b/tests/test_buf_mut.rs
--- a/tests/test_buf_mut.rs
+++ b/tests/test_buf_mut.rs
@@ -81,8 +81,8 @@ fn test_deref_bufmut_forwards() {
             unreachable!("remaining_mut");
         }
 
-        fn bytes_mut(&mut self) -> &mut UninitSlice {
-            unreachable!("bytes_mut");
+        fn chunk_mut(&mut self) -> &mut UninitSlice {
+            unreachable!("chunk_mut");
         }
 
         unsafe fn advance_mut(&mut self, _: usize) {
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -912,20 +912,20 @@ fn bytes_buf_mut_advance() {
     let mut bytes = BytesMut::with_capacity(1024);
 
     unsafe {
-        let ptr = bytes.bytes_mut().as_mut_ptr();
-        assert_eq!(1024, bytes.bytes_mut().len());
+        let ptr = bytes.chunk_mut().as_mut_ptr();
+        assert_eq!(1024, bytes.chunk_mut().len());
 
         bytes.advance_mut(10);
 
-        let next = bytes.bytes_mut().as_mut_ptr();
-        assert_eq!(1024 - 10, bytes.bytes_mut().len());
+        let next = bytes.chunk_mut().as_mut_ptr();
+        assert_eq!(1024 - 10, bytes.chunk_mut().len());
         assert_eq!(ptr.offset(10), next);
 
         // advance to the end
         bytes.advance_mut(1024 - 10);
 
         // The buffer size is doubled
-        assert_eq!(1024, bytes.bytes_mut().len());
+        assert_eq!(1024, bytes.chunk_mut().len());
     }
 }
 
diff --git a/tests/test_chain.rs b/tests/test_chain.rs
--- a/tests/test_chain.rs
+++ b/tests/test_chain.rs
@@ -62,7 +62,7 @@ fn vectored_read() {
             IoSlice::new(b4),
         ];
 
-        assert_eq!(2, buf.bytes_vectored(&mut iovecs));
+        assert_eq!(2, buf.chunks_vectored(&mut iovecs));
         assert_eq!(iovecs[0][..], b"hello"[..]);
         assert_eq!(iovecs[1][..], b"world"[..]);
         assert_eq!(iovecs[2][..], b""[..]);
diff --git a/tests/test_chain.rs b/tests/test_chain.rs
--- a/tests/test_chain.rs
+++ b/tests/test_chain.rs
@@ -83,7 +83,7 @@ fn vectored_read() {
             IoSlice::new(b4),
         ];
 
-        assert_eq!(2, buf.bytes_vectored(&mut iovecs));
+        assert_eq!(2, buf.chunks_vectored(&mut iovecs));
         assert_eq!(iovecs[0][..], b"llo"[..]);
         assert_eq!(iovecs[1][..], b"world"[..]);
         assert_eq!(iovecs[2][..], b""[..]);
diff --git a/tests/test_chain.rs b/tests/test_chain.rs
--- a/tests/test_chain.rs
+++ b/tests/test_chain.rs
@@ -104,7 +104,7 @@ fn vectored_read() {
             IoSlice::new(b4),
         ];
 
-        assert_eq!(1, buf.bytes_vectored(&mut iovecs));
+        assert_eq!(1, buf.chunks_vectored(&mut iovecs));
         assert_eq!(iovecs[0][..], b"world"[..]);
         assert_eq!(iovecs[1][..], b""[..]);
         assert_eq!(iovecs[2][..], b""[..]);
diff --git a/tests/test_chain.rs b/tests/test_chain.rs
--- a/tests/test_chain.rs
+++ b/tests/test_chain.rs
@@ -125,7 +125,7 @@ fn vectored_read() {
             IoSlice::new(b4),
         ];
 
-        assert_eq!(1, buf.bytes_vectored(&mut iovecs));
+        assert_eq!(1, buf.chunks_vectored(&mut iovecs));
         assert_eq!(iovecs[0][..], b"ld"[..]);
         assert_eq!(iovecs[1][..], b""[..]);
         assert_eq!(iovecs[2][..], b""[..]);
diff --git a/tests/test_take.rs b/tests/test_take.rs
--- a/tests/test_take.rs
+++ b/tests/test_take.rs
@@ -8,5 +8,5 @@ fn long_take() {
     // overrun the buffer. Regression test for #138.
     let buf = b"hello world".take(100);
     assert_eq!(11, buf.remaining());
-    assert_eq!(b"hello world", buf.bytes());
+    assert_eq!(b"hello world", buf.chunk());
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 54f5ced6c58c47f721836a9444654de4c8d687a1
git clean -fd
