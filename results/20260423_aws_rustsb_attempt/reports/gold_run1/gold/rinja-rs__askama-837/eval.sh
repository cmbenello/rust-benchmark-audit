#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_derive/src/parser/tests.rs b/askama_derive/src/parser/tests.rs
--- a/askama_derive/src/parser/tests.rs
+++ b/askama_derive/src/parser/tests.rs
@@ -221,6 +221,51 @@ fn test_parse_root_path() {
     );
 }
 
+#[test]
+fn test_rust_macro() {
+    let syntax = Syntax::default();
+    assert_eq!(
+        super::parse("{{ vec!(1, 2, 3) }}", &syntax).unwrap(),
+        vec![Node::Expr(
+            Ws(None, None),
+            Expr::RustMacro(vec!["vec"], "1, 2, 3",),
+        )],
+    );
+    assert_eq!(
+        super::parse("{{ alloc::vec!(1, 2, 3) }}", &syntax).unwrap(),
+        vec![Node::Expr(
+            Ws(None, None),
+            Expr::RustMacro(vec!["alloc", "vec"], "1, 2, 3",),
+        )],
+    );
+    assert_eq!(
+        super::parse("{{a!()}}", &syntax).unwrap(),
+        [Node::Expr(Ws(None, None), Expr::RustMacro(vec!["a"], ""))],
+    );
+    assert_eq!(
+        super::parse("{{a !()}}", &syntax).unwrap(),
+        [Node::Expr(Ws(None, None), Expr::RustMacro(vec!["a"], ""))],
+    );
+    assert_eq!(
+        super::parse("{{a! ()}}", &syntax).unwrap(),
+        [Node::Expr(Ws(None, None), Expr::RustMacro(vec!["a"], ""))],
+    );
+    assert_eq!(
+        super::parse("{{a ! ()}}", &syntax).unwrap(),
+        [Node::Expr(Ws(None, None), Expr::RustMacro(vec!["a"], ""))],
+    );
+    assert_eq!(
+        super::parse("{{A!()}}", &syntax).unwrap(),
+        [Node::Expr(Ws(None, None), Expr::RustMacro(vec!["A"], ""),)],
+    );
+    assert_eq!(
+        &*super::parse("{{a.b.c!( hello )}}", &syntax)
+            .unwrap_err()
+            .msg,
+        "problems parsing template source at row 1, column 7 near:\n\"!( hello )}}\"",
+    );
+}
+
 #[test]
 fn change_delimiters_parse_filter() {
     let syntax = Syntax {
diff --git /dev/null b/testing/templates/rust-macros-full-path.html
new file mode 100644
--- /dev/null
+++ b/testing/templates/rust-macros-full-path.html
@@ -0,0 +1,1 @@
+Hello, {{ foo::hello2!() }}!
diff --git a/testing/tests/rust_macro.rs b/testing/tests/rust_macro.rs
--- a/testing/tests/rust_macro.rs
+++ b/testing/tests/rust_macro.rs
@@ -11,11 +11,31 @@ macro_rules! hello {
 struct RustMacrosTemplate {}
 
 #[test]
-fn main() {
+fn macro_basic() {
     let template = RustMacrosTemplate {};
     assert_eq!("Hello, world!", template.render().unwrap());
 }
 
+mod foo {
+    macro_rules! hello2 {
+        () => {
+            "world"
+        };
+    }
+
+    pub(crate) use hello2;
+}
+
+#[derive(Template)]
+#[template(path = "rust-macros-full-path.html")]
+struct RustMacrosFullPathTemplate {}
+
+#[test]
+fn macro_full_path() {
+    let template = RustMacrosFullPathTemplate {};
+    assert_eq!("Hello, world!", template.render().unwrap());
+}
+
 macro_rules! call_a_or_b_on_tail {
     ((a: $a:expr, b: $b:expr, c: $c:expr), call a: $($tail:expr),*) => {
         ($a)($($tail),*)
diff --git a/testing/tests/rust_macro.rs b/testing/tests/rust_macro.rs
--- a/testing/tests/rust_macro.rs
+++ b/testing/tests/rust_macro.rs
@@ -47,7 +67,7 @@ fn day(_: u16, _: &str, d: u8) -> u8 {
 struct RustMacrosArgTemplate {}
 
 #[test]
-fn args() {
+fn macro_with_args() {
     let template = RustMacrosArgTemplate {};
     assert_eq!("2021\nJuly\n2", template.render().unwrap());
 }

EOF_114329324912
cd "askama_derive"
cargo test --no-fail-fast --all-features
cd ../
cd "testing"
cargo test --no-fail-fast --all-features --test "rust_macro"
cd ../
git reset --hard 9de9af4a006021a63f705e420be4cdef3eb6af82
git clean -fd
