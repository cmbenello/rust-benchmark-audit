#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -493,6 +493,47 @@ fn adds_local_source_with_version_flag() {
     assert_eq!(val["version"].as_str(), Some("0.4.3"));
 }
 
+#[test]
+fn adds_local_source_with_version_flag_and_semver_metadata() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.sample");
+
+    // dependency not present beforehand
+    let toml = get_toml(&manifest);
+    assert!(toml["dependencies"].is_none());
+
+    execute_command(
+        &["add", "local", "--vers", "0.4.3+useless-metadata.1.0.0", "--path", "/path/to/pkg"],
+        &manifest,
+    );
+
+    let toml = get_toml(&manifest);
+    let val = &toml["dependencies"]["local"];
+    assert_eq!(val["path"].as_str(), Some("/path/to/pkg"));
+    assert_eq!(val["version"].as_str(), Some("0.4.3"));
+
+    // check this works with other flags (e.g. --dev) as well
+    let toml = get_toml(&manifest);
+    assert!(toml["dev-dependencies"].is_none());
+
+    execute_command(
+        &[
+            "add",
+            "local-dev",
+            "--vers",
+            "0.4.3",
+            "--path",
+            "/path/to/pkg-dev",
+            "--dev",
+        ],
+        &manifest,
+    );
+
+    let toml = get_toml(&manifest);
+    let val = &toml["dev-dependencies"]["local-dev"];
+    assert_eq!(val["path"].as_str(), Some("/path/to/pkg-dev"));
+    assert_eq!(val["version"].as_str(), Some("0.4.3"));
+}
+
 #[test]
 fn adds_local_source_with_inline_version_notation() {
     let (_tmpdir, manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.sample");

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 579f9497aee2757a6132d9fbde6fd6ef2fafa884
git clean -fd
