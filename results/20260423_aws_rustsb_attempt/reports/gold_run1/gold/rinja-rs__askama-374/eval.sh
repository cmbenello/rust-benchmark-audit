#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/testing/Cargo.toml b/testing/Cargo.toml
--- a/testing/Cargo.toml
+++ b/testing/Cargo.toml
@@ -15,6 +15,7 @@ serde_json = { version = "1.0", optional = true }
 
 [dev-dependencies]
 criterion = "0.3"
+trybuild = "1.0"
 
 [[bench]]
 name = "all"
diff --git /dev/null b/testing/tests/ui.rs
new file mode 100644
--- /dev/null
+++ b/testing/tests/ui.rs
@@ -0,0 +1,7 @@
+use trybuild::TestCases;
+
+#[cfg_attr(not(windows), test)]
+fn ui() {
+    let t = TestCases::new();
+    t.compile_fail("tests/ui/*.rs");
+}
diff --git /dev/null b/testing/tests/ui/incorrect_path.rs
new file mode 100644
--- /dev/null
+++ b/testing/tests/ui/incorrect_path.rs
@@ -0,0 +1,8 @@
+use askama::Template;
+
+#[derive(Template)]
+#[template(path = "thisdoesnotexist.html")]
+struct MyTemplate;
+
+fn main() {
+}
diff --git /dev/null b/testing/tests/ui/incorrect_path.stderr
new file mode 100644
--- /dev/null
+++ b/testing/tests/ui/incorrect_path.stderr
@@ -0,0 +1,7 @@
+error: template "thisdoesnotexist.html" not found in directories ["$WORKSPACE/target/tests/askama_testing/templates"]
+ --> $DIR/incorrect_path.rs:3:10
+  |
+3 | #[derive(Template)]
+  |          ^^^^^^^^
+  |
+  = note: this error originates in a derive macro (in Nightly builds, run with -Z macro-backtrace for more info)

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard ac7d9e8031b4df489a0d58dc3459bc931f1ff870
git clean -fd
