#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/testing/tests/loops.rs b/testing/tests/loops.rs
--- a/testing/tests/loops.rs
+++ b/testing/tests/loops.rs
@@ -1,3 +1,5 @@
+use std::ops::Range;
+
 use askama::Template;
 
 #[derive(Template)]
diff --git a/testing/tests/loops.rs b/testing/tests/loops.rs
--- a/testing/tests/loops.rs
+++ b/testing/tests/loops.rs
@@ -131,3 +133,84 @@ fn test_for_zip_ranges() {
         "0 10 30 1 11 31 2 12 32 3 13 33 4 14 34 5 15 35 6 16 36 7 17 37 8 18 38 9 19 39 "
     );
 }
+
+struct ForVecAttrVec {
+    iterable: Vec<i32>,
+}
+
+#[derive(Template)]
+#[template(
+    source = "{% for x in v %}{% for y in x.iterable %}{{ y }} {% endfor %}{% endfor %}",
+    ext = "txt"
+)]
+struct ForVecAttrVecTemplate {
+    v: Vec<ForVecAttrVec>,
+}
+
+#[test]
+fn test_for_vec_attr_vec() {
+    let t = ForVecAttrVecTemplate {
+        v: vec![
+            ForVecAttrVec {
+                iterable: vec![1, 2],
+            },
+            ForVecAttrVec {
+                iterable: vec![3, 4],
+            },
+            ForVecAttrVec {
+                iterable: vec![5, 6],
+            },
+        ],
+    };
+    assert_eq!(t.render().unwrap(), "1 2 3 4 5 6 ");
+}
+
+struct ForVecAttrSlice {
+    iterable: &'static [i32],
+}
+
+#[derive(Template)]
+#[template(
+    source = "{% for x in v %}{% for y in x.iterable %}{{ y }} {% endfor %}{% endfor %}",
+    ext = "txt"
+)]
+struct ForVecAttrSliceTemplate {
+    v: Vec<ForVecAttrSlice>,
+}
+
+#[test]
+fn test_for_vec_attr_slice() {
+    let t = ForVecAttrSliceTemplate {
+        v: vec![
+            ForVecAttrSlice { iterable: &[1, 2] },
+            ForVecAttrSlice { iterable: &[3, 4] },
+            ForVecAttrSlice { iterable: &[5, 6] },
+        ],
+    };
+    assert_eq!(t.render().unwrap(), "1 2 3 4 5 6 ");
+}
+
+struct ForVecAttrRange {
+    iterable: Range<usize>,
+}
+
+#[derive(Template)]
+#[template(
+    source = "{% for x in v %}{% for y in x.iterable.clone() %}{{ y }} {% endfor %}{% endfor %}",
+    ext = "txt"
+)]
+struct ForVecAttrRangeTemplate {
+    v: Vec<ForVecAttrRange>,
+}
+
+#[test]
+fn test_for_vec_attr_range() {
+    let t = ForVecAttrRangeTemplate {
+        v: vec![
+            ForVecAttrRange { iterable: 1..3 },
+            ForVecAttrRange { iterable: 3..5 },
+            ForVecAttrRange { iterable: 5..7 },
+        ],
+    };
+    assert_eq!(t.render().unwrap(), "1 2 3 4 5 6 ");
+}

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features --test "loops"
cd ../
git reset --hard 49252d2457f280026c020d0df46733578eb959a5
git clean -fd
