#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -378,6 +378,21 @@ fn reserve_max_original_capacity_value() {
     assert_eq!(bytes.capacity(), 64 * 1024);
 }
 
+// Without either looking at the internals of the BytesMut or doing weird stuff
+// with the memory allocator, there's no good way to automatically verify from
+// within the program that this actually recycles memory. Instead, just exercise
+// the code path to ensure that the results are correct.
+#[test]
+fn reserve_vec_recycling() {
+    let mut bytes = BytesMut::from(Vec::with_capacity(16));
+    assert_eq!(bytes.capacity(), 16);
+    bytes.put("0123456789012345");
+    bytes.advance(10);
+    assert_eq!(bytes.capacity(), 6);
+    bytes.reserve(8);
+    assert_eq!(bytes.capacity(), 16);
+}
+
 #[test]
 fn reserve_in_arc_unique_does_not_overallocate() {
     let mut bytes = BytesMut::with_capacity(1000);

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard d656d37180db722114f960609c3e6b934b242aee
git clean -fd
