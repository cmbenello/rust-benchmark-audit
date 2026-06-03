#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/gcc-test/build.rs b/gcc-test/build.rs
--- a/gcc-test/build.rs
+++ b/gcc-test/build.rs
@@ -9,7 +9,7 @@ fn main() {
     fs::remove_dir_all(&out).unwrap();
     fs::create_dir(&out).unwrap();
 
-    gcc::Config::new()
+    gcc::Build::new()
         .file("src/foo.c")
         .flag_if_supported("-Wall")
         .flag_if_supported("-Wfoo-bar-this-flag-does-not-exist")
diff --git a/gcc-test/build.rs b/gcc-test/build.rs
--- a/gcc-test/build.rs
+++ b/gcc-test/build.rs
@@ -17,7 +17,7 @@ fn main() {
         .define("BAR", "1")
         .compile("libfoo.a");
 
-    gcc::Config::new()
+    gcc::Build::new()
         .file("src/bar1.c")
         .file("src/bar2.c")
         .include("src/include")
diff --git a/gcc-test/build.rs b/gcc-test/build.rs
--- a/gcc-test/build.rs
+++ b/gcc-test/build.rs
@@ -28,17 +28,17 @@ fn main() {
     let file = format!("src/{}.{}",
                        file,
                        if target.contains("msvc") { "asm" } else { "S" });
-    gcc::Config::new()
+    gcc::Build::new()
         .file(file)
         .compile("libasm.a");
 
-    gcc::Config::new()
+    gcc::Build::new()
         .file("src/baz.cpp")
         .cpp(true)
         .compile("libbaz.a");
 
     if target.contains("windows") {
-        gcc::Config::new()
+        gcc::Build::new()
             .file("src/windows.c")
             .compile("libwindows.a");
     }
diff --git a/gcc-test/build.rs b/gcc-test/build.rs
--- a/gcc-test/build.rs
+++ b/gcc-test/build.rs
@@ -81,12 +81,12 @@ fn main() {
 
     // This tests whether we  can build a library but not link it to the main
     // crate.  The test module will do its own linking.
-    gcc::Config::new()
+    gcc::Build::new()
         .cargo_metadata(false)
         .file("src/opt_linkage.c")
         .compile("libOptLinkage.a");
 
-    let out = gcc::Config::new()
+    let out = gcc::Build::new()
         .file("src/expand.c")
         .expand();
     let out = String::from_utf8(out).unwrap();
diff --git a/tests/support/mod.rs b/tests/support/mod.rs
--- a/tests/support/mod.rs
+++ b/tests/support/mod.rs
@@ -55,8 +55,8 @@ impl Test {
         self
     }
 
-    pub fn gcc(&self) -> gcc::Config {
-        let mut cfg = gcc::Config::new();
+    pub fn gcc(&self) -> gcc::Build {
+        let mut cfg = gcc::Build::new();
         let mut path = env::split_paths(&env::var_os("PATH").unwrap()).collect::<Vec<_>>();
         path.insert(0, self.td.path().to_owned());
         let target = if self.msvc {

EOF_114329324912
cd "gcc-test"
cargo test --no-fail-fast --all-features
cd ../
cargo test --no-fail-fast --all-features
git reset --hard 5824f286a3547badb18962965ac9d1d0a8a1a4d4
git clean -fd
