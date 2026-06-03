#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1136,6 +1138,25 @@ mod tests {
         );
     }
 
+    #[test]
+    fn test_parse_root_path() {
+        let syntax = Syntax::default();
+        assert_eq!(
+            super::parse("{{ std::string::String::new() }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                Expr::PathCall(vec!["std", "string", "String", "new"], vec![]),
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ ::std::string::String::new() }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                Expr::PathCall(vec!["", "std", "string", "String", "new"], vec![]),
+            )],
+        );
+    }
+
     #[test]
     fn change_delimiters_parse_filter() {
         let syntax = Syntax {
diff --git a/testing/tests/simple.rs b/testing/tests/simple.rs
--- a/testing/tests/simple.rs
+++ b/testing/tests/simple.rs
@@ -290,6 +290,15 @@ fn test_path_func_call() {
     assert_eq!(PathFunctionTemplate.render().unwrap(), "Hello, world4123!");
 }
 
+#[derive(Template)]
+#[template(source = "{{ ::std::string::ToString::to_string(123) }}", ext = "txt")]
+struct RootPathFunctionTemplate;
+
+#[test]
+fn test_root_path_func_call() {
+    assert_eq!(RootPathFunctionTemplate.render().unwrap(), "123");
+}
+
 #[derive(Template)]
 #[template(source = "Hello, {{ Self::world3(self, \"123\", 4) }}!", ext = "txt")]
 struct FunctionTemplate;

EOF_114329324912
cd "askama_shared"
cargo test --no-fail-fast --all-features
cd ../
cd "testing"
cargo test --no-fail-fast --all-features --test "simple"
cd ../
git reset --hard a199defeca2dfc6aa3e972acca82c96db07f99e9
git clean -fd
