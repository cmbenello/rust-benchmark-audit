#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -1283,3 +1283,73 @@ fn test_bytes_make_mut_promotable_even_arc_offset() {
     assert_eq!(b2m, vec[20..]);
     assert_eq!(b1m, vec[..20]);
 }
+
+#[test]
+fn try_reclaim_empty() {
+    let mut buf = BytesMut::new();
+    assert_eq!(false, buf.try_reclaim(6));
+    buf.reserve(6);
+    assert_eq!(true, buf.try_reclaim(6));
+    let cap = buf.capacity();
+    assert!(cap >= 6);
+    assert_eq!(false, buf.try_reclaim(cap + 1));
+
+    let mut buf = BytesMut::new();
+    buf.reserve(6);
+    let cap = buf.capacity();
+    assert!(cap >= 6);
+    let mut split = buf.split();
+    drop(buf);
+    assert_eq!(0, split.capacity());
+    assert_eq!(true, split.try_reclaim(6));
+    assert_eq!(false, split.try_reclaim(cap + 1));
+}
+
+#[test]
+fn try_reclaim_vec() {
+    let mut buf = BytesMut::with_capacity(6);
+    buf.put_slice(b"abc");
+    // Reclaiming a ludicrous amount of space should calmly return false
+    assert_eq!(false, buf.try_reclaim(usize::MAX));
+
+    assert_eq!(false, buf.try_reclaim(6));
+    buf.advance(2);
+    assert_eq!(4, buf.capacity());
+    // We can reclaim 5 bytes, because the byte in the buffer can be moved to the front. 6 bytes
+    // cannot be reclaimed because there is already one byte stored
+    assert_eq!(false, buf.try_reclaim(6));
+    assert_eq!(true, buf.try_reclaim(5));
+    buf.advance(1);
+    assert_eq!(true, buf.try_reclaim(6));
+    assert_eq!(6, buf.capacity());
+}
+
+#[test]
+fn try_reclaim_arc() {
+    let mut buf = BytesMut::with_capacity(6);
+    buf.put_slice(b"abc");
+    let x = buf.split().freeze();
+    buf.put_slice(b"def");
+    // Reclaiming a ludicrous amount of space should calmly return false
+    assert_eq!(false, buf.try_reclaim(usize::MAX));
+
+    let y = buf.split().freeze();
+    let z = y.clone();
+    assert_eq!(false, buf.try_reclaim(6));
+    drop(x);
+    drop(z);
+    assert_eq!(false, buf.try_reclaim(6));
+    drop(y);
+    assert_eq!(true, buf.try_reclaim(6));
+    assert_eq!(6, buf.capacity());
+    assert_eq!(0, buf.len());
+    buf.put_slice(b"abc");
+    buf.put_slice(b"def");
+    assert_eq!(6, buf.capacity());
+    assert_eq!(6, buf.len());
+    assert_eq!(false, buf.try_reclaim(6));
+    buf.advance(4);
+    assert_eq!(true, buf.try_reclaim(4));
+    buf.advance(2);
+    assert_eq!(true, buf.try_reclaim(6));
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 7a5154ba8b54970b7bb07c4902bc8a7981f4e57c
git clean -fd
