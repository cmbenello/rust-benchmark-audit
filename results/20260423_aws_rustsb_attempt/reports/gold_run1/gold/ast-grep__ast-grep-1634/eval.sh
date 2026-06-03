#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/core/src/node.rs b/crates/core/src/node.rs
--- a/crates/core/src/node.rs
+++ b/crates/core/src/node.rs
@@ -742,7 +748,6 @@ if (a) {
   }
 
   #[test]
-  #[ignore = "TODO: fix column to be unicode character"]
   fn test_unicode_pos() {
     let root = Tsx.ast_grep("🦀");
     let root = root.root();
diff --git a/crates/core/src/source.rs b/crates/core/src/source.rs
--- a/crates/core/src/source.rs
+++ b/crates/core/src/source.rs
@@ -189,6 +191,25 @@ impl Content for String {
   fn encode_bytes(bytes: &[Self::Underlying]) -> Cow<str> {
     String::from_utf8_lossy(bytes)
   }
+
+  /// This is an O(n) operation. We assume the col will not be a
+  /// huge number in reality. This may be problematic for special
+  /// files like compressed js
+  fn get_char_column(&self, _col: usize, offset: usize) -> usize {
+    let src = self.as_bytes();
+    let mut col = 0;
+    // TODO: is it possible to use SIMD here???
+    for &b in src[..offset].iter().rev() {
+      if b == b'\n' {
+        break;
+      }
+      // https://en.wikipedia.org/wiki/UTF-8#Description
+      if b & 0b1100_0000 != 0b1000_0000 {
+        col += 1;
+      }
+    }
+    col
+  }
 }
 
 #[cfg(test)]
diff --git a/crates/pyo3/tests/test_range.py b/crates/pyo3/tests/test_range.py
--- a/crates/pyo3/tests/test_range.py
+++ b/crates/pyo3/tests/test_range.py
@@ -48,5 +48,4 @@ def test_unicode():
     assert node is not None
     assert node.range().start.index == 5
     assert node.range().start.line == 0
-    # TODO: Fix this, it should be 5 in character
-    # assert node.range().start.column == 5
\ No newline at end of file
+    assert node.range().start.column == 5
\ No newline at end of file

EOF_114329324912
cd "crates/core"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard b87dad753fb2ce87cae17d488bac4da3fd62a5a7
git clean -fd
