#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/bin/upgrade/main.rs b/src/bin/upgrade/main.rs
--- a/src/bin/upgrade/main.rs
+++ b/src/bin/upgrade/main.rs
@@ -16,8 +16,8 @@ extern crate error_chain;
 
 use crate::errors::*;
 use cargo_edit::{
-    find, get_latest_dependency, registry_url, update_registry_index, CrateName, Dependency,
-    LocalManifest,
+    find, get_latest_dependency, manifest_from_pkgid, registry_url, update_registry_index,
+    CrateName, Dependency, LocalManifest,
 };
 use failure::Fail;
 use std::collections::{HashMap, HashSet};
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -33,4 +34,5 @@ pub use crate::fetch::{
     get_latest_dependency, update_registry_index,
 };
 pub use crate::manifest::{find, LocalManifest, Manifest};
+pub use crate::metadata::manifest_from_pkgid;
 pub use crate::registry::registry_url;
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -4,8 +4,8 @@ extern crate pretty_assertions;
 use std::process;
 mod utils;
 use crate::utils::{
-    clone_out_test, execute_bad_command, execute_command, get_command_path, get_toml,
-    setup_alt_registry_config,
+    clone_out_test, copy_workspace_test, execute_bad_command, execute_command,
+    execute_command_for_pkg, get_command_path, get_toml, setup_alt_registry_config,
 };
 
 /// Some of the tests need to have a crate name that does not exist on crates.io. Hence this rather
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1353,3 +1353,22 @@ toml_edit = "0.1.5"
 "#
     );
 }
+
+#[test]
+fn add_dependency_to_workspace_member() {
+    let (tmpdir, _root_manifest, workspace_manifests) = copy_workspace_test();
+    execute_command_for_pkg(&["add", "toml"], "one", &tmpdir);
+
+    let one = workspace_manifests
+        .iter()
+        .map(|manifest| get_toml(manifest))
+        .find(|manifest| manifest["package"]["name"].as_str() == Some("one"))
+        .expect("Couldn't find workspace member `one'");
+
+    assert_eq!(
+        one["dependencies"]["toml"]
+            .as_str()
+            .expect("toml dependency did not exist"),
+        "toml--CURRENT_VERSION_TEST",
+    );
+}
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -1,5 +1,8 @@
 mod utils;
-use crate::utils::{clone_out_test, execute_command, get_command_path, get_toml};
+use crate::utils::{
+    clone_out_test, copy_workspace_test, execute_command, execute_command_for_pkg,
+    get_command_path, get_toml,
+};
 
 #[test]
 fn remove_existing_dependency() {
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -225,3 +228,17 @@ fn rm_prints_messages_for_multiple() {
     .is("Removing semver from dependencies\n    Removing docopt from dependencies")
     .unwrap();
 }
+
+#[test]
+fn rm_dependency_from_workspace_member() {
+    let (tmpdir, _root_manifest, workspace_manifests) = copy_workspace_test();
+    execute_command_for_pkg(&["rm", "libc"], "one", &tmpdir);
+
+    let one = workspace_manifests
+        .iter()
+        .map(|manifest| get_toml(manifest))
+        .find(|manifest| manifest["package"]["name"].as_str() == Some("one"))
+        .expect("Couldn't find workspace member `one'");
+
+    assert!(one["dependencies"]["libc"].as_str().is_none());
+}
diff --git a/tests/cargo-upgrade.rs b/tests/cargo-upgrade.rs
--- a/tests/cargo-upgrade.rs
+++ b/tests/cargo-upgrade.rs
@@ -5,58 +5,10 @@ use std::fs;
 
 mod utils;
 use crate::utils::{
-    clone_out_test, execute_command, execute_command_in_dir, get_command_path, get_toml,
-    setup_alt_registry_config,
+    clone_out_test, copy_workspace_test, execute_command, execute_command_for_pkg,
+    execute_command_in_dir, get_command_path, get_toml, setup_alt_registry_config,
 };
 
-/// Helper function that copies the workspace test into a temporary directory.
-pub fn copy_workspace_test() -> (tempfile::TempDir, String, Vec<String>) {
-    // Create a temporary directory and copy in the root manifest, the dummy rust file, and
-    // workspace member manifests.
-    let tmpdir = tempfile::tempdir().expect("failed to construct temporary directory");
-
-    let (root_manifest_path, workspace_manifest_paths) = {
-        // Helper to copy in files to the temporary workspace. The standard library doesn't have a
-        // good equivalent of `cp -r`, hence this oddity.
-        let copy_in = |dir, file| {
-            let file_path = tmpdir
-                .path()
-                .join(dir)
-                .join(file)
-                .to_str()
-                .unwrap()
-                .to_string();
-
-            fs::create_dir_all(tmpdir.path().join(dir)).unwrap();
-
-            fs::copy(
-                format!("tests/fixtures/workspace/{}/{}", dir, file),
-                &file_path,
-            )
-            .unwrap_or_else(|err| panic!("could not copy test file: {}", err));
-
-            file_path
-        };
-
-        let root_manifest_path = copy_in(".", "Cargo.toml");
-        copy_in(".", "dummy.rs");
-        copy_in(".", "Cargo.lock");
-
-        let workspace_manifest_paths = ["one", "two", "implicit/three", "explicit/four"]
-            .iter()
-            .map(|member| copy_in(member, "Cargo.toml"))
-            .collect::<Vec<_>>();
-
-        (root_manifest_path, workspace_manifest_paths)
-    };
-
-    (
-        tmpdir,
-        root_manifest_path,
-        workspace_manifest_paths.to_owned(),
-    )
-}
-
 // Verify that an upgraded Cargo.toml matches what we expect.
 #[test]
 fn upgrade_as_expected() {
diff --git a/tests/cargo-upgrade.rs b/tests/cargo-upgrade.rs
--- a/tests/cargo-upgrade.rs
+++ b/tests/cargo-upgrade.rs
@@ -382,6 +334,25 @@ fn upgrade_workspace() {
     }
 }
 
+#[test]
+fn upgrade_dependency_in_workspace_member() {
+    let (tmpdir, _root_manifest, workspace_manifests) = copy_workspace_test();
+    execute_command_for_pkg(&["upgrade", "libc"], "one", &tmpdir);
+
+    let one = workspace_manifests
+        .iter()
+        .map(|manifest| get_toml(manifest))
+        .find(|manifest| manifest["package"]["name"].as_str() == Some("one"))
+        .expect("Couldn't find workspace member `one'");
+
+    assert_eq!(
+        one["dependencies"]["libc"]
+            .as_str()
+            .expect("libc dependency did not exist"),
+        "libc--CURRENT_VERSION_TEST",
+    );
+}
+
 /// Detect if attempting to run against a workspace root and give a helpful warning.
 #[test]
 #[cfg(feature = "test-external-apis")]
diff --git a/tests/utils.rs b/tests/utils.rs
--- a/tests/utils.rs
+++ b/tests/utils.rs
@@ -3,6 +3,54 @@ use std::ffi::{OsStr, OsString};
 use std::io::prelude::*;
 use std::{env, fs, path::Path, path::PathBuf, process};
 
+/// Helper function that copies the workspace test into a temporary directory.
+pub fn copy_workspace_test() -> (tempfile::TempDir, String, Vec<String>) {
+    // Create a temporary directory and copy in the root manifest, the dummy rust file, and
+    // workspace member manifests.
+    let tmpdir = tempfile::tempdir().expect("failed to construct temporary directory");
+
+    let (root_manifest_path, workspace_manifest_paths) = {
+        // Helper to copy in files to the temporary workspace. The standard library doesn't have a
+        // good equivalent of `cp -r`, hence this oddity.
+        let copy_in = |dir, file| {
+            let file_path = tmpdir
+                .path()
+                .join(dir)
+                .join(file)
+                .to_str()
+                .unwrap()
+                .to_string();
+
+            fs::create_dir_all(tmpdir.path().join(dir)).unwrap();
+
+            fs::copy(
+                format!("tests/fixtures/workspace/{}/{}", dir, file),
+                &file_path,
+            )
+            .unwrap_or_else(|err| panic!("could not copy test file: {}", err));
+
+            file_path
+        };
+
+        let root_manifest_path = copy_in(".", "Cargo.toml");
+        copy_in(".", "dummy.rs");
+        copy_in(".", "Cargo.lock");
+
+        let workspace_manifest_paths = ["one", "two", "implicit/three", "explicit/four"]
+            .iter()
+            .map(|member| copy_in(member, "Cargo.toml"))
+            .collect::<Vec<_>>();
+
+        (root_manifest_path, workspace_manifest_paths)
+    };
+
+    (
+        tmpdir,
+        root_manifest_path,
+        workspace_manifest_paths.to_owned(),
+    )
+}
+
 /// Create temporary working directory with Cargo.toml manifest
 pub fn clone_out_test(source: &str) -> (tempfile::TempDir, String) {
     let tmpdir = tempfile::tempdir().expect("failed to construct temporary directory");
diff --git a/tests/utils.rs b/tests/utils.rs
--- a/tests/utils.rs
+++ b/tests/utils.rs
@@ -54,6 +102,35 @@ where
     }
 }
 
+/// Execute local cargo command, includes `--package`
+pub fn execute_command_for_pkg<S, P>(command: &[S], pkgid: &str, cwd: P)
+where
+    S: AsRef<OsStr>,
+    P: AsRef<Path>,
+{
+    let subcommand_name = &command[0].as_ref();
+    let cwd = cwd.as_ref();
+
+    let call = process::Command::new(&get_command_path(subcommand_name))
+        .args(command)
+        .arg("--package")
+        .arg(pkgid)
+        .current_dir(&cwd)
+        .env("CARGO_IS_TEST", "1")
+        .output()
+        .expect("call to test command failed");
+
+    if !call.status.success() {
+        println!("Status code: {:?}", call.status);
+        println!("STDOUT: {}", String::from_utf8_lossy(&call.stdout));
+        println!("STDERR: {}", String::from_utf8_lossy(&call.stderr));
+        panic!(
+            "cargo-{} failed to execute",
+            subcommand_name.to_string_lossy()
+        )
+    }
+}
+
 /// Execute local cargo command, includes `--manifest-path`
 pub fn execute_command<S>(command: &[S], manifest: &str)
 where

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard b8a8474db183f57e77e456e13ebe36d1b6e12e3f
git clean -fd
