#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -14,6 +14,19 @@ fn remove_existing_dependency() {
     assert!(toml["dependencies"]["docopt"].is_none());
 }
 
+#[test]
+fn remove_multiple_existing_dependencies() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/rm/Cargo.toml.sample");
+
+    let toml = get_toml(&manifest);
+    assert!(!toml["dependencies"]["docopt"].is_none());
+    assert!(!toml["dependencies"]["semver"].is_none());
+    execute_command(&["rm", "docopt", "semver"], &manifest);
+    let toml = get_toml(&manifest);
+    assert!(toml["dependencies"]["docopt"].is_none());
+    assert!(toml["dependencies"]["semver"].is_none());
+}
+
 #[test]
 fn remove_existing_dependency_from_specific_section() {
     let (_tmpdir, manifest) = clone_out_test("tests/fixtures/rm/Cargo.toml.sample");
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -23,7 +36,7 @@ fn remove_existing_dependency_from_specific_section() {
     assert!(!toml["dev-dependencies"]["regex"].is_none());
     execute_command(&["rm", "--dev", "regex"], &manifest);
     let toml = get_toml(&manifest);
-    assert!(toml["dev-dependencies"].is_none());
+    assert!(toml["dev-dependencies"]["regex"].is_none());
 
     // Test removing build dependency.
     let toml = get_toml(&manifest);
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -33,15 +46,28 @@ fn remove_existing_dependency_from_specific_section() {
     assert!(toml["build-dependencies"].is_none());
 }
 
+#[test]
+fn remove_multiple_existing_dependencies_from_specific_section() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/rm/Cargo.toml.sample");
+
+    // Test removing dev dependency.
+    let toml = get_toml(&manifest);
+    assert!(!toml["dev-dependencies"]["regex"].is_none());
+    assert!(!toml["dev-dependencies"]["serde"].is_none());
+    execute_command(&["rm", "--dev", "regex", "serde"], &manifest);
+    let toml = get_toml(&manifest);
+    assert!(toml["dev-dependencies"].is_none());
+}
+
 #[test]
 fn remove_section_after_removed_last_dependency() {
     let (_tmpdir, manifest) = clone_out_test("tests/fixtures/rm/Cargo.toml.sample");
 
     let toml = get_toml(&manifest);
     assert!(!toml["dev-dependencies"]["regex"].is_none());
-    assert_eq!(toml["dev-dependencies"].as_table().unwrap().len(), 1);
+    assert_eq!(toml["dev-dependencies"].as_table().unwrap().len(), 2);
 
-    execute_command(&["rm", "--dev", "regex"], &manifest);
+    execute_command(&["rm", "--dev", "regex", "serde"], &manifest);
 
     let toml = get_toml(&manifest);
     assert!(toml["dev-dependencies"].is_none());
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -79,13 +105,15 @@ fn invalid_dependency() {
         "rm",
         "invalid_dependency_name",
         &format!("--manifest-path={}", manifest),
-    ]).fails_with(1)
-        .and()
-        .stderr().is(
-            "Command failed due to unhandled error: The dependency `invalid_dependency_name` could \
-             not be found in `dependencies`.",
-        )
-        .unwrap();
+    ])
+    .fails_with(1)
+    .and()
+    .stderr()
+    .contains(
+        "Command failed due to unhandled error: The dependency `invalid_dependency_name` could \
+         not be found in `dependencies`.",
+    )
+    .unwrap();
 }
 
 #[test]
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -99,14 +127,15 @@ fn invalid_section() {
         "semver",
         "--build",
         &format!("--manifest-path={}", manifest),
-    ]).fails_with(1)
-        .and()
-        .stderr()
-        .is(
-            "Command failed due to unhandled error: The table `build-dependencies` could not be \
-             found.",
-        )
-        .unwrap();
+    ])
+    .fails_with(1)
+    .and()
+    .stderr()
+    .contains(
+        "Command failed due to unhandled error: The table `build-dependencies` could not be \
+         found.",
+    )
+    .unwrap();
 }
 
 #[test]
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -117,16 +146,18 @@ fn invalid_dependency_in_section() {
         "target/debug/cargo-rm",
         "rm",
         "semver",
+        "regex",
         "--dev",
         &format!("--manifest-path={}", manifest),
-    ]).fails_with(1)
-        .and()
-        .stderr()
-        .is(
-            "Command failed due to unhandled error: The dependency `semver` could not be found in \
-             `dev-dependencies`.",
-        )
-        .unwrap();
+    ])
+    .fails_with(1)
+    .and()
+    .stderr()
+    .contains(
+        "Command failed due to unhandled error: The dependency `semver` could not be found in \
+         `dev-dependencies`.",
+    )
+    .unwrap();
 }
 
 #[test]
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -139,6 +170,7 @@ fn no_argument() {
 
 Usage:
     cargo rm <crate> [--dev|--build] [options]
+    cargo rm <crates>... [--dev|--build] [options]
     cargo rm (-h|--help)
     cargo rm --version")
         .unwrap();
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -154,6 +186,7 @@ fn unknown_flags() {
 
 Usage:
     cargo rm <crate> [--dev|--build] [options]
+    cargo rm <crates>... [--dev|--build] [options]
     cargo rm (-h|--help)
     cargo rm --version")
         .unwrap();
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -168,9 +201,28 @@ fn rm_prints_message() {
         "rm",
         "semver",
         &format!("--manifest-path={}", manifest),
-    ]).succeeds()
-        .and()
-        .stdout()
-        .is("Removing semver from dependencies")
-        .unwrap();
+    ])
+    .succeeds()
+    .and()
+    .stdout()
+    .is("Removing semver from dependencies")
+    .unwrap();
+}
+
+#[test]
+fn rm_prints_messages_for_multiple() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/rm/Cargo.toml.sample");
+
+    assert_cli::Assert::command(&[
+        "target/debug/cargo-rm",
+        "rm",
+        "semver",
+        "docopt",
+        &format!("--manifest-path={}", manifest),
+    ])
+    .succeeds()
+    .and()
+    .stdout()
+    .is("Removing semver from dependencies\n    Removing docopt from dependencies")
+    .unwrap();
 }
diff --git a/tests/fixtures/rm/Cargo.toml.sample b/tests/fixtures/rm/Cargo.toml.sample
--- a/tests/fixtures/rm/Cargo.toml.sample
+++ b/tests/fixtures/rm/Cargo.toml.sample
@@ -19,3 +19,4 @@ clippy = {git = "https://github.com/Manishearth/rust-clippy.git", optional = tru
 
 [dev-dependencies]
 regex = "0.1.41"
+serde = "1.0.90"

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 26a641e71f1190d9a8764b538211396f71e6c0a0
git clean -fd
