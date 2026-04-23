#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_buf_mut.rs b/tests/test_buf_mut.rs
--- a/tests/test_buf_mut.rs
+++ b/tests/test_buf_mut.rs
@@ -45,13 +45,12 @@ fn test_put_u16() {
 }
 
 #[test]
+#[should_panic(expected = "cannot advance")]
 fn test_vec_advance_mut() {
-    // Regression test for carllerche/bytes#108.
+    // Verify fix for #354
     let mut buf = Vec::with_capacity(8);
     unsafe {
         buf.advance_mut(12);
-        assert_eq!(buf.len(), 12);
-        assert!(buf.capacity() >= 12, "capacity: {}", buf.capacity());
     }
 }
 

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard fe6e67386451715c5d609c90a41e98ef80f0e1d1
git clean -fd
