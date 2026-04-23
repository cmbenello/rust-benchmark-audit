#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/bin/upgrade/upgrade.rs b/src/bin/upgrade/upgrade.rs
--- a/src/bin/upgrade/upgrade.rs
+++ b/src/bin/upgrade/upgrade.rs
@@ -1,11 +1,11 @@
 use std::collections::BTreeSet;
 use std::io::Write;
-use std::path::{Path, PathBuf};
+use std::path::PathBuf;
 
 use cargo_edit::{
-    colorize_stderr, find, get_latest_dependency, manifest_from_pkgid, registry_url,
-    set_dep_version, shell_status, shell_warn, update_registry_index, CargoResult, Context,
-    CrateSpec, Dependency, LocalManifest,
+    colorize_stderr, find, get_latest_dependency, registry_url, resolve_manifests, set_dep_version,
+    shell_status, shell_warn, update_registry_index, CargoResult, Context, CrateSpec, Dependency,
+    LocalManifest,
 };
 use clap::Args;
 use indexmap::IndexMap;
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -34,7 +34,7 @@ pub use dependency::Source;
 pub use errors::*;
 pub use fetch::{get_latest_dependency, update_registry_index};
 pub use manifest::{find, get_dep_version, set_dep_version, LocalManifest, Manifest};
-pub use metadata::{manifest_from_pkgid, workspace_members};
+pub use metadata::{manifest_from_pkgid, resolve_manifests, workspace_members};
 pub use registry::registry_url;
 pub use util::{colorize_stderr, shell_print, shell_status, shell_warn, Color, ColorChoice};
 pub use version::{upgrade_requirement, VersionExt};
diff --git a/tests/cmd/upgrade/invalid_manifest.toml b/tests/cmd/upgrade/invalid_manifest.toml
--- a/tests/cmd/upgrade/invalid_manifest.toml
+++ b/tests/cmd/upgrade/invalid_manifest.toml
@@ -3,17 +3,21 @@ args = ["upgrade"]
 status.code = 1
 stdout = ""
 stderr = """
-Error: Unable to parse Cargo.toml
+Error: Invalid manifest
 
 Caused by:
-    0: Manifest not valid TOML
-    1: TOML parse error at line 1, column 6
-         |
-       1 | This is clearly not a valid Cargo.toml.
-         |      ^
-       Unexpected `i`
-       Expected `.` or `=`
-       
+    `cargo metadata` exited with an error: error: failed to parse manifest at `[CWD]/Cargo.toml`
+    
+    Caused by:
+      could not parse input as TOML
+    
+    Caused by:
+      TOML parse error at line 1, column 6
+        |
+      1 | This is clearly not a valid Cargo.toml.
+        |      ^
+      Unexpected `i`
+      Expected `.` or `=`
 """
 fs.sandbox = true
 
diff --git a/tests/cmd/upgrade/invalid_virtual_manifest.toml /dev/null
--- a/tests/cmd/upgrade/invalid_virtual_manifest.toml
+++ /dev/null
@@ -1,11 +0,0 @@
-bin.name = "cargo-upgrade"
-args = ["upgrade", "--manifest-path", "Cargo.toml"]
-status.code = 1
-stdout = ""
-stderr = """
-Error: Found virtual manifest, but this command requires running against an actual package in this workspace. Try adding `--workspace`.
-"""
-fs.sandbox = true
-
-[env.add]
-CARGO_IS_TEST="1"
diff --git a/tests/cmd/upgrade/invalid_workspace_root_manifest.toml b/tests/cmd/upgrade/invalid_workspace_root_manifest.toml
--- a/tests/cmd/upgrade/invalid_workspace_root_manifest.toml
+++ b/tests/cmd/upgrade/invalid_workspace_root_manifest.toml
@@ -3,7 +3,7 @@ args = ["upgrade", "--workspace"]
 status.code = 1
 stdout = ""
 stderr = """
-Error: Failed to get workspace metadata
+Error: Invalid manifest
 
 Caused by:
     `cargo metadata` exited with an error: error: failed to parse manifest at `[CWD]/Cargo.toml`
diff --git /dev/null b/tests/cmd/upgrade/virtual_manifest.out/explicit/four/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/upgrade/virtual_manifest.out/explicit/four/Cargo.toml
@@ -0,0 +1,9 @@
+[package]
+name = "four"
+version = "0.1.0"
+
+[lib]
+path = "../../dummy.rs"
+
+[dependencies]
+libc = "99999.0.0"
diff --git /dev/null b/tests/cmd/upgrade/virtual_manifest.out/implicit/three/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/upgrade/virtual_manifest.out/implicit/three/Cargo.toml
@@ -0,0 +1,9 @@
+[package]
+name = "three"
+version = "0.1.0"
+
+[lib]
+path = "../../dummy.rs"
+
+[dependencies]
+libc = "99999.0.0"
diff --git /dev/null b/tests/cmd/upgrade/virtual_manifest.out/one/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/upgrade/virtual_manifest.out/one/Cargo.toml
@@ -0,0 +1,11 @@
+[package]
+name = "one"
+version = "0.1.0"
+
+[lib]
+path = "../dummy.rs"
+
+[dependencies]
+libc = "99999.0.0"
+rand = "99999.0"
+three = { path = "../implicit/three"}
diff --git /dev/null b/tests/cmd/upgrade/virtual_manifest.out/two/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/upgrade/virtual_manifest.out/two/Cargo.toml
@@ -0,0 +1,11 @@
+[package]
+name = "two"
+version = "0.1.0"
+
+[[bin]]
+name = "two"
+path = "../dummy.rs"
+
+[dependencies]
+libc = "99999.0.0"
+rand = "99999.0"
diff --git /dev/null b/tests/cmd/upgrade/virtual_manifest.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/upgrade/virtual_manifest.toml
@@ -0,0 +1,20 @@
+bin.name = "cargo-upgrade"
+args = ["upgrade", "--manifest-path", "Cargo.toml"]
+status = "success"
+stdout = ""
+stderr = """
+    Checking one's dependencies
+   Upgrading libc: v0.2.28 -> v99999.0.0
+   Upgrading rand: v0.3 -> v99999.0
+    Checking three's dependencies
+   Upgrading libc: v0.2.28 -> v99999.0.0
+    Checking two's dependencies
+   Upgrading libc: v0.2.28 -> v99999.0.0
+   Upgrading rand: v0.2 -> v99999.0
+    Checking four's dependencies
+   Upgrading libc: v0.2.28 -> v99999.0.0
+"""
+fs.sandbox = true
+
+[env.add]
+CARGO_IS_TEST="1"
diff --git /dev/null b/tests/cmd/upgrade/workspace_member_manifest_path.in
new file mode 100644
--- /dev/null
+++ b/tests/cmd/upgrade/workspace_member_manifest_path.in
@@ -0,0 +1,1 @@
+upgrade-workspace.in/
\ No newline at end of file
diff --git /dev/null b/tests/cmd/upgrade/workspace_member_manifest_path.out/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/upgrade/workspace_member_manifest_path.out/Cargo.toml
@@ -0,0 +1,6 @@
+[workspace]
+members = [
+    "one",
+    "two",
+    "explicit/*"
+]
\ No newline at end of file
diff --git a/tests/cmd/upgrade/invalid_virtual_manifest.out/one/Cargo.toml b/tests/cmd/upgrade/workspace_member_manifest_path.out/one/Cargo.toml
--- a/tests/cmd/upgrade/invalid_virtual_manifest.out/one/Cargo.toml
+++ b/tests/cmd/upgrade/workspace_member_manifest_path.out/one/Cargo.toml
@@ -6,6 +6,6 @@ version = "0.1.0"
 path = "../dummy.rs"
 
 [dependencies]
-libc = "0.2.28"
+libc = "99999.0.0"
 rand = "0.3"
-three = { path = "../implicit/three"}
\ No newline at end of file
+three = { path = "../implicit/three"}
diff --git /dev/null b/tests/cmd/upgrade/workspace_member_manifest_path.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/upgrade/workspace_member_manifest_path.toml
@@ -0,0 +1,13 @@
+bin.name = "cargo-upgrade"
+args = ["upgrade", "libc", "--manifest-path", "Cargo.toml"]
+status = "success"
+stdout = ""
+stderr = """
+    Checking one's dependencies
+   Upgrading libc: v0.2.28 -> v99999.0.0
+"""
+fs.sandbox = true
+fs.cwd = "workspace_member_cwd.in/one"
+
+[env.add]
+CARGO_IS_TEST="1"

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 07c27ad3314463f97abdfd6690035aadf129d122
git clean -fd
