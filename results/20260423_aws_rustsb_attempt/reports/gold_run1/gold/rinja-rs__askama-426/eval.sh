#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1126,7 +1126,41 @@ mod tests {
 
     #[test]
     fn test_parse_filter() {
-        super::parse("{{ strvar|e }}", &Syntax::default()).unwrap();
+        use Expr::*;
+        let syntax = Syntax::default();
+        assert_eq!(
+            super::parse("{{ strvar|e }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                Filter("e", vec![Var("strvar")]),
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ 2|abs }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                Filter("abs", vec![NumLit("2")]),
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ -2|abs }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                Filter("abs", vec![Unary("-", NumLit("2").into())]),
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ (1 - 2)|abs }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                Filter(
+                    "abs",
+                    vec![Group(
+                        BinOp("-", NumLit("1").into(), NumLit("2").into()).into()
+                    )]
+                ),
+            )],
+        );
     }
 
     #[test]

EOF_114329324912
cd "askama_shared"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard c29ecd68714bddf5e27a9e347c902faa23b2a545
git clean -fd
