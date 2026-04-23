#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/config/src/rule/range.rs b/crates/config/src/rule/range.rs
--- a/crates/config/src/rule/range.rs
+++ b/crates/config/src/rule/range.rs
@@ -102,8 +102,11 @@ mod test {
   #[test]
   fn test_invalid_range() {
     let range = RangeMatcher::<TS>::try_new(
-      SerializablePosition { row: 0, column: 10 },
-      SerializablePosition { row: 0, column: 5 },
+      SerializablePosition {
+        line: 0,
+        column: 10,
+      },
+      SerializablePosition { line: 0, column: 5 },
     );
     assert!(range.is_err());
   }
diff --git a/crates/config/src/rule/range.rs b/crates/config/src/rule/range.rs
--- a/crates/config/src/rule/range.rs
+++ b/crates/config/src/rule/range.rs
@@ -113,8 +116,14 @@ mod test {
     let cand = TS::Tsx.ast_grep("class A { a = 123 }");
     let cand = cand.root();
     let pattern = RangeMatcher::new(
-      SerializablePosition { row: 0, column: 10 },
-      SerializablePosition { row: 0, column: 17 },
+      SerializablePosition {
+        line: 0,
+        column: 10,
+      },
+      SerializablePosition {
+        line: 0,
+        column: 17,
+      },
     );
     assert!(pattern.find_node(cand).is_some());
   }
diff --git a/crates/config/src/rule/range.rs b/crates/config/src/rule/range.rs
--- a/crates/config/src/rule/range.rs
+++ b/crates/config/src/rule/range.rs
@@ -124,8 +133,14 @@ mod test {
     let cand = TS::Tsx.ast_grep("class A { a = 123 }");
     let cand = cand.root();
     let pattern = RangeMatcher::new(
-      SerializablePosition { row: 0, column: 10 },
-      SerializablePosition { row: 0, column: 15 },
+      SerializablePosition {
+        line: 0,
+        column: 10,
+      },
+      SerializablePosition {
+        line: 0,
+        column: 15,
+      },
     );
     assert!(pattern.find_node(cand).is_none(),);
   }
diff --git a/crates/config/src/rule/range.rs b/crates/config/src/rule/range.rs
--- a/crates/config/src/rule/range.rs
+++ b/crates/config/src/rule/range.rs
@@ -136,8 +151,8 @@ mod test {
       .ast_grep("class A { \n b = () => { \n const c = 1 \n const d = 3 \n return c + d \n } }");
     let cand = cand.root();
     let pattern = RangeMatcher::new(
-      SerializablePosition { row: 1, column: 1 },
-      SerializablePosition { row: 5, column: 2 },
+      SerializablePosition { line: 1, column: 1 },
+      SerializablePosition { line: 5, column: 2 },
     );
     assert!(pattern.find_node(cand).is_some());
   }
diff --git a/crates/config/src/rule/range.rs b/crates/config/src/rule/range.rs
--- a/crates/config/src/rule/range.rs
+++ b/crates/config/src/rule/range.rs
@@ -147,8 +162,11 @@ mod test {
     let cand = TS::Tsx.ast_grep("let a = '🦄'");
     let cand = cand.root();
     let pattern = RangeMatcher::new(
-      SerializablePosition { row: 0, column: 8 },
-      SerializablePosition { row: 0, column: 11 },
+      SerializablePosition { line: 0, column: 8 },
+      SerializablePosition {
+        line: 0,
+        column: 11,
+      },
     );
     let node = pattern.find_node(cand);
     assert!(node.is_some());
diff --git a/crates/napi/__test__/index.spec.ts b/crates/napi/__test__/index.spec.ts
--- a/crates/napi/__test__/index.spec.ts
+++ b/crates/napi/__test__/index.spec.ts
@@ -387,8 +387,8 @@ test("find node by range", (t) => {
   const node = sg.root().find({
     rule: {
       range: {
-        start: { row: 0, column: 16 },
-        end: { row: 4, column: 1 },
+        start: { line: 0, column: 16 },
+        end: { line: 4, column: 1 },
       }
     }
   })
diff --git a/crates/pyo3/tests/test_range.py b/crates/pyo3/tests/test_range.py
--- a/crates/pyo3/tests/test_range.py
+++ b/crates/pyo3/tests/test_range.py
@@ -54,7 +54,7 @@ def test_unicode_range_rule():
     source = "ハロ = console.log(世界)".strip()
     sg = SgRoot(source, "javascript")
     root = sg.root()
-    node = root.find(range={"start": {"row": 0, "column": 17}, "end": {"row": 0, "column": 19}})
+    node = root.find(range={"start": {"line": 0, "column": 17}, "end": {"line": 0, "column": 19}})
     assert node
     assert node.range().start.index == 17
     assert node.range().start.line == 0
diff --git a/crates/pyo3/tests/test_rule.py b/crates/pyo3/tests/test_rule.py
--- a/crates/pyo3/tests/test_rule.py
+++ b/crates/pyo3/tests/test_rule.py
@@ -153,14 +153,14 @@ def test_pattern():
 
 def test_range_rule():
     node = root.find(range={
-        "start": {"row": 0, "column": 9},
-        "end": {"row": 0, "column": 13},
+        "start": {"line": 0, "column": 9},
+        "end": {"line": 0, "column": 13},
     })
     assert node
     assert node.text() == "test"
     node = root.find(range={
-        "start": {"row": 0, "column": 9},
-        "end": {"row": 0, "column": 12},
+        "start": {"line": 0, "column": 9},
+        "end": {"line": 0, "column": 12},
     })
     assert not node
 

EOF_114329324912
cd "crates/config"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard bfd4945591b9959ba59309eaa7d2e8f7861f163b
git clean -fd
