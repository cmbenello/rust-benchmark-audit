#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1078,12 +1074,13 @@ pub fn parse<'a>(src: &'a str, syntax: &'a Syntax<'a>) -> Result<Vec<Node<'a>>,
 
 #[cfg(test)]
 mod tests {
+    use super::{Expr, Node, WS};
     use crate::Syntax;
 
     fn check_ws_split(s: &str, res: &(&str, &str, &str)) {
         let node = super::split_ws_parts(s.as_bytes());
         match node {
-            super::Node::Lit(lws, s, rws) => {
+            Node::Lit(lws, s, rws) => {
                 assert_eq!(lws, res.0);
                 assert_eq!(s, res.1);
                 assert_eq!(rws, res.2);
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1118,12 +1115,9 @@ mod tests {
     fn test_parse_var_call() {
         assert_eq!(
             super::parse("{{ function(\"123\", 3) }}", &Syntax::default()).unwrap(),
-            vec![super::Node::Expr(
-                super::WS(false, false),
-                super::Expr::VarCall(
-                    "function",
-                    vec![super::Expr::StrLit("123"), super::Expr::NumLit("3")]
-                ),
+            vec![Node::Expr(
+                WS(false, false),
+                Expr::VarCall("function", vec![Expr::StrLit("123"), Expr::NumLit("3")]),
             )],
         );
     }
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1132,11 +1126,11 @@ mod tests {
     fn test_parse_path_call() {
         assert_eq!(
             super::parse("{{ self::function(\"123\", 3) }}", &Syntax::default()).unwrap(),
-            vec![super::Node::Expr(
-                super::WS(false, false),
-                super::Expr::PathCall(
+            vec![Node::Expr(
+                WS(false, false),
+                Expr::PathCall(
                     vec!["self", "function"],
-                    vec![super::Expr::StrLit("123"), super::Expr::NumLit("3")],
+                    vec![Expr::StrLit("123"), Expr::NumLit("3")],
                 ),
             )],
         );
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1152,6 +1146,152 @@ mod tests {
 
         super::parse("{~ strvar|e ~}", &syntax).unwrap();
     }
+
+    #[test]
+    fn test_precedence() {
+        use Expr::*;
+        let syntax = Syntax::default();
+        assert_eq!(
+            super::parse("{{ a + b == c }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "==",
+                    BinOp("+", Var("a").into(), Var("b").into()).into(),
+                    Var("c").into(),
+                )
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ a + b * c - d / e }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "-",
+                    BinOp(
+                        "+",
+                        Var("a").into(),
+                        BinOp("*", Var("b").into(), Var("c").into()).into(),
+                    )
+                    .into(),
+                    BinOp("/", Var("d").into(), Var("e").into()).into(),
+                )
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ a * (b + c) / -d }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "/",
+                    BinOp(
+                        "*",
+                        Var("a").into(),
+                        Group(BinOp("+", Var("b").into(), Var("c").into()).into()).into()
+                    )
+                    .into(),
+                    Unary("-", Var("d").into()).into()
+                )
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ a || b && c || d && e }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "||",
+                    BinOp(
+                        "||",
+                        Var("a").into(),
+                        BinOp("&&", Var("b").into(), Var("c").into()).into(),
+                    )
+                    .into(),
+                    BinOp("&&", Var("d").into(), Var("e").into()).into(),
+                )
+            )],
+        );
+    }
+
+    #[test]
+    fn test_associativity() {
+        use Expr::*;
+        let syntax = Syntax::default();
+        assert_eq!(
+            super::parse("{{ a + b + c }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "+",
+                    BinOp("+", Var("a").into(), Var("b").into()).into(),
+                    Var("c").into()
+                )
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ a * b * c }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "*",
+                    BinOp("*", Var("a").into(), Var("b").into()).into(),
+                    Var("c").into()
+                )
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ a && b && c }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "&&",
+                    BinOp("&&", Var("a").into(), Var("b").into()).into(),
+                    Var("c").into()
+                )
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ a + b - c + d }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "+",
+                    BinOp(
+                        "-",
+                        BinOp("+", Var("a").into(), Var("b").into()).into(),
+                        Var("c").into()
+                    )
+                    .into(),
+                    Var("d").into()
+                )
+            )],
+        );
+        assert_eq!(
+            super::parse("{{ a == b != c > d > e == f }}", &syntax).unwrap(),
+            vec![Node::Expr(
+                WS(false, false),
+                BinOp(
+                    "==",
+                    BinOp(
+                        ">",
+                        BinOp(
+                            ">",
+                            BinOp(
+                                "!=",
+                                BinOp("==", Var("a").into(), Var("b").into()).into(),
+                                Var("c").into()
+                            )
+                            .into(),
+                            Var("d").into()
+                        )
+                        .into(),
+                        Var("e").into()
+                    )
+                    .into(),
+                    Var("f").into()
+                )
+            )],
+        );
+    }
 }
 
 type ParserError<'a, T> = Result<(&'a [u8], T), nom::Err<nom::error::Error<&'a [u8]>>>;

EOF_114329324912
cd "askama_shared"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard f4065b09b91f5d00efa5644915cdd83bfb7d393a
git clean -fd
