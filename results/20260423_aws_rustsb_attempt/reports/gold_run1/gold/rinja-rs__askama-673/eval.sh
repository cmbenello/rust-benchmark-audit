#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_shared/src/lib.rs b/askama_shared/src/lib.rs
--- a/askama_shared/src/lib.rs
+++ b/askama_shared/src/lib.rs
@@ -701,5 +705,14 @@ mod tests {
         )
         .unwrap();
         assert_eq!(config.whitespace, WhitespaceHandling::Preserve);
+
+        let config = Config::new(
+            r#"
+            [general]
+            whitespace = "minimize"
+            "#,
+        )
+        .unwrap();
+        assert_eq!(config.whitespace, WhitespaceHandling::Minimize);
     }
 }
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1443,12 +1447,12 @@ mod tests {
     #[test]
     fn change_delimiters_parse_filter() {
         let syntax = Syntax {
-            expr_start: "{~",
-            expr_end: "~}",
+            expr_start: "{=",
+            expr_end: "=}",
             ..Syntax::default()
         };
 
-        super::parse("{~ strvar|e ~}", &syntax).unwrap();
+        super::parse("{= strvar|e =}", &syntax).unwrap();
     }
 
     #[test]
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1665,31 +1669,31 @@ mod tests {
         );
         assert_eq!(
             super::parse("{#- #}", s).unwrap(),
-            vec![Node::Comment(Ws(Some(Whitespace::Trim), None))],
+            vec![Node::Comment(Ws(Some(Whitespace::Suppress), None))],
         );
         assert_eq!(
             super::parse("{# -#}", s).unwrap(),
-            vec![Node::Comment(Ws(None, Some(Whitespace::Trim)))],
+            vec![Node::Comment(Ws(None, Some(Whitespace::Suppress)))],
         );
         assert_eq!(
             super::parse("{#--#}", s).unwrap(),
             vec![Node::Comment(Ws(
-                Some(Whitespace::Trim),
-                Some(Whitespace::Trim)
+                Some(Whitespace::Suppress),
+                Some(Whitespace::Suppress)
             ))],
         );
         assert_eq!(
             super::parse("{#- foo\n bar -#}", s).unwrap(),
             vec![Node::Comment(Ws(
-                Some(Whitespace::Trim),
-                Some(Whitespace::Trim)
+                Some(Whitespace::Suppress),
+                Some(Whitespace::Suppress)
             ))],
         );
         assert_eq!(
             super::parse("{#- foo\n {#- bar\n -#} baz -#}", s).unwrap(),
             vec![Node::Comment(Ws(
-                Some(Whitespace::Trim),
-                Some(Whitespace::Trim)
+                Some(Whitespace::Suppress),
+                Some(Whitespace::Suppress)
             ))],
         );
         assert_eq!(
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1721,6 +1725,35 @@ mod tests {
                 Some(Whitespace::Preserve)
             ))],
         );
+        assert_eq!(
+            super::parse("{#~ #}", s).unwrap(),
+            vec![Node::Comment(Ws(Some(Whitespace::Minimize), None))],
+        );
+        assert_eq!(
+            super::parse("{# ~#}", s).unwrap(),
+            vec![Node::Comment(Ws(None, Some(Whitespace::Minimize)))],
+        );
+        assert_eq!(
+            super::parse("{#~~#}", s).unwrap(),
+            vec![Node::Comment(Ws(
+                Some(Whitespace::Minimize),
+                Some(Whitespace::Minimize)
+            ))],
+        );
+        assert_eq!(
+            super::parse("{#~ foo\n bar ~#}", s).unwrap(),
+            vec![Node::Comment(Ws(
+                Some(Whitespace::Minimize),
+                Some(Whitespace::Minimize)
+            ))],
+        );
+        assert_eq!(
+            super::parse("{#~ foo\n {#~ bar\n ~#} baz -~#}", s).unwrap(),
+            vec![Node::Comment(Ws(
+                Some(Whitespace::Minimize),
+                Some(Whitespace::Minimize)
+            ))],
+        );
 
         assert_eq!(
             super::parse("{# foo {# bar #} {# {# baz #} qux #} #}", s).unwrap(),
diff --git /dev/null b/testing/test_minimize.toml
new file mode 100644
--- /dev/null
+++ b/testing/test_minimize.toml
@@ -0,0 +1,2 @@
+[general]
+whitespace = "minimize"
diff --git a/testing/tests/whitespace.rs b/testing/tests/whitespace.rs
--- a/testing/tests/whitespace.rs
+++ b/testing/tests/whitespace.rs
@@ -41,3 +41,46 @@ fn test_extra_whitespace() {
     template.nested_1.nested_2.hash.insert("key", "value");
     assert_eq!(template.render().unwrap(), "\n0\n0\n0\n0\n\n\n\n0\n0\n0\n0\n0\n\na0\na1\nvalue\n\n\n\n\n\n[\n  &quot;a0&quot;,\n  &quot;a1&quot;,\n  &quot;a2&quot;,\n  &quot;a3&quot;\n]\n[\n  &quot;a0&quot;,\n  &quot;a1&quot;,\n  &quot;a2&quot;,\n  &quot;a3&quot;\n][\n  &quot;a0&quot;,\n  &quot;a1&quot;,\n  &quot;a2&quot;,\n  &quot;a3&quot;\n]\n[\n  &quot;a1&quot;\n][\n  &quot;a1&quot;\n]\n[\n  &quot;a1&quot;,\n  &quot;a2&quot;\n][\n  &quot;a1&quot;,\n  &quot;a2&quot;\n]\n[\n  &quot;a1&quot;\n][\n  &quot;a1&quot;\n]1-1-1\n3333 3\n2222 2\n0000 0\n3333 3\n\ntruefalse\nfalsefalsefalse\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
 }
+
+macro_rules! test_template_minimize {
+    ($source:literal, $rendered:expr) => {{
+        #[derive(Template)]
+        #[template(source = $source, ext = "txt", config = "test_minimize.toml")]
+        struct CondWs;
+
+        assert_eq!(CondWs.render().unwrap(), $rendered);
+    }};
+}
+
+macro_rules! test_template {
+    ($source:literal, $rendered:expr) => {{
+        #[derive(Template)]
+        #[template(source = $source, ext = "txt")]
+        struct CondWs;
+
+        assert_eq!(CondWs.render().unwrap(), $rendered);
+    }};
+}
+
+#[test]
+fn test_minimize_whitespace() {
+    test_template_minimize!(
+        "\n1\r\n{%  if true  %}\n\n2\r\n\r\n{%  endif  %} 3\r\n\r\n\r\n",
+        "\n1\n\n2\n 3"
+    );
+    test_template_minimize!(
+        "\n1\r\n{%+  if true  %}\n\n2\r\n\r\n{%  endif  %} 3\r\n\r\n\r\n",
+        "\n1\r\n\n2\n 3"
+    );
+    test_template_minimize!(
+        "\n1\r\n{%-  if true  %}\n\n2\r\n\r\n{%  endif  %} 3\r\n\r\n\r\n",
+        "\n1\n2\n 3"
+    );
+    test_template_minimize!(" \n1 \n{%  if true  %} 2 {%  endif  %}3 ", " \n1\n 2 3");
+
+    test_template!(
+        "\n1\r\n{%~  if true  ~%}\n\n2\r\n\r\n{%~  endif  ~%} 3\r\n\r\n\r\n",
+        "\n1\n\n2\n 3"
+    );
+    test_template!(" \n1 \n{%~  if true  ~%} 2 {%~  endif  ~%}3 ", " \n1\n 2 3");
+}

EOF_114329324912
cd "askama_shared"
cargo test --no-fail-fast --all-features
cd ../
cd "testing"
cargo test --no-fail-fast --all-features --test "whitespace"
cd ../
git reset --hard 358f7cd07dc42ba4189d2661461ff0b43a27c304
git clean -fd
