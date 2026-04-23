#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/gcc-test/build.rs b/gcc-test/build.rs
--- a/gcc-test/build.rs
+++ b/gcc-test/build.rs
@@ -14,7 +14,7 @@ fn main() {
         .flag_if_supported("-Wall")
         .flag_if_supported("-Wfoo-bar-this-flag-does-not-exist")
         .define("FOO", None)
-        .define("BAR", Some("1"))
+        .define("BAR", "1")
         .compile("libfoo.a");
 
     gcc::Config::new()
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -152,7 +152,7 @@ fn gnu_include() {
 fn gnu_define() {
     let test = Test::gnu();
     test.gcc()
-        .define("FOO", Some("bar"))
+        .define("FOO", "bar")
         .define("BAR", None)
         .file("foo.c")
         .compile("libfoo.a");
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -266,7 +266,7 @@ fn msvc_include() {
 fn msvc_define() {
     let test = Test::msvc();
     test.gcc()
-        .define("FOO", Some("bar"))
+        .define("FOO", "bar")
         .define("BAR", None)
         .file("foo.c")
         .compile("libfoo.a");

EOF_114329324912
cd "gcc-test"
cargo test --no-fail-fast --all-features
cd ../
cargo test --no-fail-fast --all-features
git reset --hard 88ac58e25c673adbf30fec17efd67011c105b44a
git clean -fd
