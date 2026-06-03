#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_shared/src/filters/mod.rs b/askama_shared/src/filters/mod.rs
--- a/askama_shared/src/filters/mod.rs
+++ b/askama_shared/src/filters/mod.rs
@@ -460,22 +460,22 @@ mod tests {
     #[test]
     #[allow(clippy::float_cmp)]
     fn test_into_f64() {
-        assert_eq!(into_f64(1).unwrap(), 1.0 as f64);
-        assert_eq!(into_f64(1.9).unwrap(), 1.9 as f64);
-        assert_eq!(into_f64(-1.9).unwrap(), -1.9 as f64);
-        assert_eq!(into_f64(INFINITY as f32).unwrap(), INFINITY);
-        assert_eq!(into_f64(-INFINITY as f32).unwrap(), -INFINITY);
+        assert_eq!(into_f64(&1).unwrap(), 1.0 as f64);
+        assert_eq!(into_f64(&1.9).unwrap(), 1.9 as f64);
+        assert_eq!(into_f64(&-1.9).unwrap(), -1.9 as f64);
+        assert_eq!(into_f64(&(INFINITY as f32)).unwrap(), INFINITY);
+        assert_eq!(into_f64(&(-INFINITY as f32)).unwrap(), -INFINITY);
     }
 
     #[cfg(feature = "num-traits")]
     #[test]
     fn test_into_isize() {
-        assert_eq!(into_isize(1).unwrap(), 1 as isize);
-        assert_eq!(into_isize(1.9).unwrap(), 1 as isize);
-        assert_eq!(into_isize(-1.9).unwrap(), -1 as isize);
-        assert_eq!(into_isize(1.5 as f64).unwrap(), 1 as isize);
-        assert_eq!(into_isize(-1.5 as f64).unwrap(), -1 as isize);
-        match into_isize(INFINITY) {
+        assert_eq!(into_isize(&1).unwrap(), 1 as isize);
+        assert_eq!(into_isize(&1.9).unwrap(), 1 as isize);
+        assert_eq!(into_isize(&-1.9).unwrap(), -1 as isize);
+        assert_eq!(into_isize(&(1.5 as f64)).unwrap(), 1 as isize);
+        assert_eq!(into_isize(&(-1.5 as f64)).unwrap(), -1 as isize);
+        match into_isize(&INFINITY) {
             Err(Fmt(fmt::Error)) => {}
             _ => panic!("Should return error of type Err(Fmt(fmt::Error))"),
         };
diff --git a/testing/tests/filters.rs b/testing/tests/filters.rs
--- a/testing/tests/filters.rs
+++ b/testing/tests/filters.rs
@@ -50,6 +50,20 @@ fn filter_fmt() {
     assert_eq!(t.render().unwrap(), "\"formatted\"");
 }
 
+#[derive(Template)]
+#[template(
+    source = "{{ 1|into_f64 }} {{ 1.9|into_isize }}",
+    ext = "txt",
+    escape = "none"
+)]
+struct IntoNumbersTemplate;
+
+#[test]
+fn into_numbers_fmt() {
+    let t = IntoNumbersTemplate;
+    assert_eq!(t.render().unwrap(), "1 1");
+}
+
 #[derive(Template)]
 #[template(source = "{{ s|myfilter }}", ext = "txt")]
 struct MyFilterTemplate<'a> {

EOF_114329324912
cd "askama_shared"
cargo test --no-fail-fast --all-features
cd ../
cd "testing"
cargo test --no-fail-fast --all-features --test "filters"
cd ../
git reset --hard 17b9d06814cee84bfd57b73e1941b63187ec5f65
git clean -fd
