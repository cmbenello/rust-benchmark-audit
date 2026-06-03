#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_shared/src/filters/json.rs b/askama_shared/src/filters/json.rs
--- a/askama_shared/src/filters/json.rs
+++ b/askama_shared/src/filters/json.rs
@@ -22,6 +22,8 @@ mod tests {
 
     #[test]
     fn test_json() {
+        assert_eq!(json(Html, true).unwrap().to_string(), "true");
+        assert_eq!(json(Html, "foo").unwrap().to_string(), r#""foo""#);
         assert_eq!(json(Html, &true).unwrap().to_string(), "true");
         assert_eq!(json(Html, &"foo").unwrap().to_string(), r#""foo""#);
         assert_eq!(
diff --git a/askama_shared/src/filters/mod.rs b/askama_shared/src/filters/mod.rs
--- a/askama_shared/src/filters/mod.rs
+++ b/askama_shared/src/filters/mod.rs
@@ -480,36 +480,36 @@ mod tests {
 
     #[test]
     fn test_truncate() {
-        assert_eq!(truncate(&"hello", &2).unwrap(), "he...");
+        assert_eq!(truncate(&"hello", 2).unwrap(), "he...");
         let a = String::from("您好");
         assert_eq!(a.len(), 6);
         assert_eq!(String::from("您").len(), 3);
-        assert_eq!(truncate(&"您好", &1).unwrap(), "您...");
-        assert_eq!(truncate(&"您好", &2).unwrap(), "您...");
-        assert_eq!(truncate(&"您好", &3).unwrap(), "您...");
-        assert_eq!(truncate(&"您好", &4).unwrap(), "您好...");
-        assert_eq!(truncate(&"您好", &6).unwrap(), "您好");
-        assert_eq!(truncate(&"您好", &7).unwrap(), "您好");
+        assert_eq!(truncate(&"您好", 1).unwrap(), "您...");
+        assert_eq!(truncate(&"您好", 2).unwrap(), "您...");
+        assert_eq!(truncate(&"您好", 3).unwrap(), "您...");
+        assert_eq!(truncate(&"您好", 4).unwrap(), "您好...");
+        assert_eq!(truncate(&"您好", 6).unwrap(), "您好");
+        assert_eq!(truncate(&"您好", 7).unwrap(), "您好");
         let s = String::from("🤚a🤚");
         assert_eq!(s.len(), 9);
         assert_eq!(String::from("🤚").len(), 4);
-        assert_eq!(truncate(&"🤚a🤚", &1).unwrap(), "🤚...");
-        assert_eq!(truncate(&"🤚a🤚", &2).unwrap(), "🤚...");
-        assert_eq!(truncate(&"🤚a🤚", &3).unwrap(), "🤚...");
-        assert_eq!(truncate(&"🤚a🤚", &4).unwrap(), "🤚...");
-        assert_eq!(truncate(&"🤚a🤚", &5).unwrap(), "🤚a...");
-        assert_eq!(truncate(&"🤚a🤚", &6).unwrap(), "🤚a🤚...");
-        assert_eq!(truncate(&"🤚a🤚", &9).unwrap(), "🤚a🤚");
-        assert_eq!(truncate(&"🤚a🤚", &10).unwrap(), "🤚a🤚");
+        assert_eq!(truncate(&"🤚a🤚", 1).unwrap(), "🤚...");
+        assert_eq!(truncate(&"🤚a🤚", 2).unwrap(), "🤚...");
+        assert_eq!(truncate(&"🤚a🤚", 3).unwrap(), "🤚...");
+        assert_eq!(truncate(&"🤚a🤚", 4).unwrap(), "🤚...");
+        assert_eq!(truncate(&"🤚a🤚", 5).unwrap(), "🤚a...");
+        assert_eq!(truncate(&"🤚a🤚", 6).unwrap(), "🤚a🤚...");
+        assert_eq!(truncate(&"🤚a🤚", 9).unwrap(), "🤚a🤚");
+        assert_eq!(truncate(&"🤚a🤚", 10).unwrap(), "🤚a🤚");
     }
 
     #[test]
     fn test_indent() {
-        assert_eq!(indent(&"hello", &2).unwrap(), "hello");
-        assert_eq!(indent(&"hello\n", &2).unwrap(), "hello\n");
-        assert_eq!(indent(&"hello\nfoo", &2).unwrap(), "hello\n  foo");
+        assert_eq!(indent(&"hello", 2).unwrap(), "hello");
+        assert_eq!(indent(&"hello\n", 2).unwrap(), "hello\n");
+        assert_eq!(indent(&"hello\nfoo", 2).unwrap(), "hello\n  foo");
         assert_eq!(
-            indent(&"hello\nfoo\n bar", &4).unwrap(),
+            indent(&"hello\nfoo\n bar", 4).unwrap(),
             "hello\n    foo\n     bar"
         );
     }
diff --git a/askama_shared/src/filters/mod.rs b/askama_shared/src/filters/mod.rs
--- a/askama_shared/src/filters/mod.rs
+++ b/askama_shared/src/filters/mod.rs
@@ -518,22 +518,22 @@ mod tests {
     #[test]
     #[allow(clippy::float_cmp)]
     fn test_into_f64() {
-        assert_eq!(into_f64(&1).unwrap(), 1.0_f64);
-        assert_eq!(into_f64(&1.9).unwrap(), 1.9_f64);
-        assert_eq!(into_f64(&-1.9).unwrap(), -1.9_f64);
-        assert_eq!(into_f64(&(INFINITY as f32)).unwrap(), INFINITY);
-        assert_eq!(into_f64(&(-INFINITY as f32)).unwrap(), -INFINITY);
+        assert_eq!(into_f64(1).unwrap(), 1.0_f64);
+        assert_eq!(into_f64(1.9).unwrap(), 1.9_f64);
+        assert_eq!(into_f64(-1.9).unwrap(), -1.9_f64);
+        assert_eq!(into_f64(INFINITY as f32).unwrap(), INFINITY);
+        assert_eq!(into_f64(-INFINITY as f32).unwrap(), -INFINITY);
     }
 
     #[cfg(feature = "num-traits")]
     #[test]
     fn test_into_isize() {
-        assert_eq!(into_isize(&1).unwrap(), 1_isize);
-        assert_eq!(into_isize(&1.9).unwrap(), 1_isize);
-        assert_eq!(into_isize(&-1.9).unwrap(), -1_isize);
-        assert_eq!(into_isize(&(1.5_f64)).unwrap(), 1_isize);
-        assert_eq!(into_isize(&(-1.5_f64)).unwrap(), -1_isize);
-        match into_isize(&INFINITY) {
+        assert_eq!(into_isize(1).unwrap(), 1_isize);
+        assert_eq!(into_isize(1.9).unwrap(), 1_isize);
+        assert_eq!(into_isize(-1.9).unwrap(), -1_isize);
+        assert_eq!(into_isize(1.5_f64).unwrap(), 1_isize);
+        assert_eq!(into_isize(-1.5_f64).unwrap(), -1_isize);
+        match into_isize(INFINITY) {
             Err(Fmt(fmt::Error)) => {}
             _ => panic!("Should return error of type Err(Fmt(fmt::Error))"),
         };
diff --git a/askama_shared/src/filters/yaml.rs b/askama_shared/src/filters/yaml.rs
--- a/askama_shared/src/filters/yaml.rs
+++ b/askama_shared/src/filters/yaml.rs
@@ -22,6 +22,8 @@ mod tests {
 
     #[test]
     fn test_yaml() {
+        assert_eq!(yaml(Html, true).unwrap().to_string(), "---\ntrue");
+        assert_eq!(yaml(Html, "foo").unwrap().to_string(), "---\nfoo");
         assert_eq!(yaml(Html, &true).unwrap().to_string(), "---\ntrue");
         assert_eq!(yaml(Html, &"foo").unwrap().to_string(), "---\nfoo");
         assert_eq!(
diff --git a/askama_shared/src/parser.rs b/askama_shared/src/parser.rs
--- a/askama_shared/src/parser.rs
+++ b/askama_shared/src/parser.rs
@@ -1129,6 +1170,19 @@ mod tests {
         super::parse("{{ strvar|e }}", &Syntax::default()).unwrap();
     }
 
+    #[test]
+    fn test_parse_numbers() {
+        let syntax = Syntax::default();
+        assert_eq!(
+            super::parse("{{ 2 }}", &syntax).unwrap(),
+            vec![Node::Expr(WS(false, false), Expr::NumLit("2"),)],
+        );
+        assert_eq!(
+            super::parse("{{ 2.5 }}", &syntax).unwrap(),
+            vec![Node::Expr(WS(false, false), Expr::NumLit("2.5"),)],
+        );
+    }
+
     #[test]
     fn test_parse_var_call() {
         assert_eq!(
diff --git a/testing/tests/loops.rs b/testing/tests/loops.rs
--- a/testing/tests/loops.rs
+++ b/testing/tests/loops.rs
@@ -91,6 +91,18 @@ fn test_for_method_call() {
     assert_eq!(t.render().unwrap(), "123");
 }
 
+#[derive(Template)]
+#[template(
+    source = "{% for i in ::std::iter::repeat(\"a\").take(5) %}{{ i }}{% endfor %}",
+    ext = "txt"
+)]
+struct ForPathCallTemplate;
+
+#[test]
+fn test_for_path_call() {
+    assert_eq!(ForPathCallTemplate.render().unwrap(), "aaaaa");
+}
+
 #[derive(Template)]
 #[template(
     source = "{% for i in [1, 2, 3, 4, 5][3..] %}{{ i }}{% endfor %}",
diff --git a/testing/tests/simple.rs b/testing/tests/simple.rs
--- a/testing/tests/simple.rs
+++ b/testing/tests/simple.rs
@@ -265,7 +265,7 @@ fn test_slice_literal() {
 #[derive(Template)]
 #[template(source = "Hello, {{ world(\"123\", 4) }}!", ext = "txt")]
 struct FunctionRefTemplate {
-    world: fn(s: &str, v: &u8) -> String,
+    world: fn(s: &str, v: u8) -> String,
 }
 
 #[test]
diff --git a/testing/tests/simple.rs b/testing/tests/simple.rs
--- a/testing/tests/simple.rs
+++ b/testing/tests/simple.rs
@@ -277,7 +277,7 @@ fn test_func_ref_call() {
 }
 
 #[allow(clippy::trivially_copy_pass_by_ref)]
-fn world2(s: &str, v: &u8) -> String {
+fn world2(s: &str, v: u8) -> String {
     format!("world{}{}", v, s)
 }
 
diff --git a/testing/tests/simple.rs b/testing/tests/simple.rs
--- a/testing/tests/simple.rs
+++ b/testing/tests/simple.rs
@@ -291,7 +291,7 @@ fn test_path_func_call() {
 }
 
 #[derive(Template)]
-#[template(source = "{{ ::std::string::ToString::to_string(123) }}", ext = "txt")]
+#[template(source = "{{ ::std::string::String::from(\"123\") }}", ext = "txt")]
 struct RootPathFunctionTemplate;
 
 #[test]
diff --git a/testing/tests/simple.rs b/testing/tests/simple.rs
--- a/testing/tests/simple.rs
+++ b/testing/tests/simple.rs
@@ -305,7 +305,7 @@ struct FunctionTemplate;
 
 impl FunctionTemplate {
     #[allow(clippy::trivially_copy_pass_by_ref)]
-    fn world3(&self, s: &str, v: &u8) -> String {
+    fn world3(&self, s: &str, v: u8) -> String {
         format!("world{}{}", s, v)
     }
 }

EOF_114329324912
cd "askama_shared"
cargo test --no-fail-fast --all-features
cd ../
cd "testing"
cargo test --no-fail-fast --all-features --test "loops"
cargo test --no-fail-fast --all-features --test "simple"
cd ../
git reset --hard c29ecd68714bddf5e27a9e347c902faa23b2a545
git clean -fd
