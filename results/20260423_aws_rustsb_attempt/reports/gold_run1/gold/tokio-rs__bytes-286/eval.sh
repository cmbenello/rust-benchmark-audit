#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -528,7 +528,7 @@ fn stress() {
 
     for i in 0..ITERS {
         let data = [i as u8; 256];
-        let buf = Arc::new(Bytes::from(&data[..]));
+        let buf = Arc::new(Bytes::copy_from_slice(&data[..]));
 
         let barrier = Arc::new(Barrier::new(THREADS));
         let mut joins = Vec::with_capacity(THREADS);
diff --git a/tests/test_bytes.rs b/tests/test_bytes.rs
--- a/tests/test_bytes.rs
+++ b/tests/test_bytes.rs
@@ -888,7 +888,7 @@ fn slice_ref_catches_not_an_empty_subset() {
 #[test]
 #[should_panic]
 fn empty_slice_ref_catches_not_an_empty_subset() {
-    let bytes = Bytes::from(&b""[..]);
+    let bytes = Bytes::copy_from_slice(&b""[..]);
     let slice = &b""[0..0];
 
     bytes.slice_ref(slice);

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard b6cb346adfaae89bce44bfa337652e6d218d38c4
git clean -fd
