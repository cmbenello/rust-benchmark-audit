#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -515,6 +515,34 @@ fn reserve_in_arc_unique_doubles() {
     assert_eq!(2000, bytes.capacity());
 }
 
+#[test]
+fn reserve_in_arc_unique_does_not_overallocate_after_split() {
+    let mut bytes = BytesMut::from(LONG);
+    let orig_capacity = bytes.capacity();
+    drop(bytes.split_off(LONG.len() / 2));
+
+    // now bytes is Arc and refcount == 1
+
+    let new_capacity = bytes.capacity();
+    bytes.reserve(orig_capacity - new_capacity);
+    assert_eq!(bytes.capacity(), orig_capacity);
+}
+
+#[test]
+fn reserve_in_arc_unique_does_not_overallocate_after_multiple_splits() {
+    let mut bytes = BytesMut::from(LONG);
+    let orig_capacity = bytes.capacity();
+    for _ in 0..10 {
+        drop(bytes.split_off(LONG.len() / 2));
+
+        // now bytes is Arc and refcount == 1
+
+        let new_capacity = bytes.capacity();
+        bytes.reserve(orig_capacity - new_capacity);
+    }
+    assert_eq!(bytes.capacity(), orig_capacity);
+}
+
 #[test]
 fn reserve_in_arc_nonunique_does_not_overallocate() {
     let mut bytes = BytesMut::with_capacity(1000);

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 38fd42acbaced11ff19f0a4ca2af44a308af5063
git clean -fd
