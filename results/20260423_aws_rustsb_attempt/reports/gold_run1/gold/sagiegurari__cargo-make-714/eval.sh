#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/Makefile.toml b/Makefile.toml
--- a/Makefile.toml
+++ b/Makefile.toml
@@ -28,6 +28,26 @@ echo "    Unstable Format Environment: ${CARGO_MAKE_TEMP_UNSTABLE_FMT_ENV}"
 echo "*************************************"
 '''
 
+[tasks.test-multi-phases-cleanup]
+script = '''
+#!@duckscript
+
+fn <scope> delete_all
+    files = set ${1}
+    for file in ${files}
+        echo Deleting lock file: ${file}
+        rm ${file}
+    end
+
+    release ${files}
+end
+
+handle = glob_array ./src/**/Cargo.lock
+delete_all ${handle}
+handle = glob_array ./examples/**/Cargo.lock
+delete_all ${handle}
+'''
+
 [tasks.test-multi-phases-flow]
 condition = { env_false = ["CARGO_MAKE_TEMP_UNSTABLE_TEST_ENV"] }
 
diff --git /dev/null b/examples/workspace-inherit/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/examples/workspace-inherit/Cargo.toml
@@ -0,0 +1,12 @@
+
+[workspace]
+members = ["member1"]
+
+[workspace.package]
+version = "1.2.3"
+authors = ["test author"]
+description = "test description"
+documentation = "test docs"
+license = "test license"
+homepage = "https://testpage.com"
+repository = "https://repotest.com"
diff --git a/examples/workspace2/Cargo.toml b/examples/workspace2/Cargo.toml
--- a/examples/workspace2/Cargo.toml
+++ b/examples/workspace2/Cargo.toml
@@ -1,10 +1,4 @@
 
 [workspace]
 members = ["member", "member1", "member2"]
-
 exclude = ["member"]
-
-[dependencies]
-test = "1.0.0"
-member1 = { path = "./member1" }
-member3 = { path = "./member/member3" }
diff --git a/src/lib/environment/crateinfo.rs b/src/lib/environment/crateinfo.rs
--- a/src/lib/environment/crateinfo.rs
+++ b/src/lib/environment/crateinfo.rs
@@ -7,15 +7,65 @@
 #[path = "crateinfo_test.rs"]
 mod crateinfo_test;
 
-use crate::types::{CrateDependency, CrateInfo};
+use crate::types::{CrateDependency, CrateInfo, PackageInfo, Workspace};
 use cargo_metadata::camino::Utf8PathBuf;
-use cargo_metadata::MetadataCommand;
+use cargo_metadata::{Metadata, MetadataCommand};
 use fsio;
 use glob::glob;
+use indexmap::IndexMap;
 use std::env;
 use std::ffi::OsStr;
 use std::path::{Path, PathBuf};
 
+#[derive(Debug, Deserialize)]
+struct CargoConfig {
+    build: Option<CargoConfigBuild>,
+}
+
+#[derive(Debug, Deserialize)]
+#[serde(rename_all = "kebab-case")]
+struct CargoConfigBuild {
+    target: Option<RustTarget>,
+}
+
+#[derive(Debug, Deserialize)]
+#[serde(from = "PathBuf")]
+struct RustTarget(PathBuf);
+
+impl RustTarget {
+    fn name(&self) -> &str {
+        self.0.file_stem().unwrap().to_str().unwrap()
+    }
+}
+
+impl From<PathBuf> for RustTarget {
+    fn from(buf: PathBuf) -> Self {
+        Self(buf)
+    }
+}
+
+impl AsRef<OsStr> for RustTarget {
+    fn as_ref(&self) -> &OsStr {
+        self.0.as_ref()
+    }
+}
+
+#[derive(Serialize, Deserialize, Debug, Clone, Default)]
+/// Holds crate information loaded from the Cargo.toml file.
+struct CrateInfoMinimal {
+    /// workspace info
+    workspace: Option<Workspace>,
+    /// crate dependencies
+    dependencies: Option<IndexMap<String, CrateDependency>>,
+}
+
+impl CrateInfoMinimal {
+    /// Creates and returns a new instance.
+    fn new() -> CrateInfoMinimal {
+        Default::default()
+    }
+}
+
 fn expand_glob_members(glob_member: &str) -> Vec<String> {
     let emulation = envmnt::is("CARGO_MAKE_WORKSPACE_EMULATION");
 
diff --git a/src/lib/environment/crateinfo_test.rs b/src/lib/environment/crateinfo_test.rs
--- a/src/lib/environment/crateinfo_test.rs
+++ b/src/lib/environment/crateinfo_test.rs
@@ -1,4 +1,5 @@
 use super::*;
+use crate::test::is_min_rust_version;
 use crate::types::{CrateDependencyInfo, PackageInfo, Workspace};
 use cargo_metadata::camino::Utf8Path;
 use indexmap::IndexMap;
diff --git a/src/lib/environment/crateinfo_test.rs b/src/lib/environment/crateinfo_test.rs
--- a/src/lib/environment/crateinfo_test.rs
+++ b/src/lib/environment/crateinfo_test.rs
@@ -653,3 +654,20 @@ fn get_crate_target_dir() {
 
     env::set_current_dir(old_cwd).unwrap();
 }
+
+#[test]
+fn load_from_inherit_from_workspace_toml() {
+    if is_min_rust_version("1.64.0") {
+        let crate_info =
+            load_from(Path::new("src/lib/test/workspace-inherit/member1/Cargo.toml").to_path_buf());
+
+        let package_info = crate_info.package.unwrap();
+        assert_eq!(package_info.name.unwrap(), "member1");
+        assert_eq!(package_info.version.unwrap(), "1.2.3");
+        assert_eq!(package_info.description.unwrap(), "test description");
+        assert_eq!(package_info.documentation.unwrap(), "test docs");
+        assert_eq!(package_info.license.unwrap(), "test license");
+        assert_eq!(package_info.homepage.unwrap(), "https://testpage.com");
+        assert_eq!(package_info.repository.unwrap(), "https://repotest.com");
+    }
+}
diff --git a/src/lib/test/mod.rs b/src/lib/test/mod.rs
--- a/src/lib/test/mod.rs
+++ b/src/lib/test/mod.rs
@@ -1,3 +1,4 @@
+use crate::installer::crate_version_check::is_min_version_valid_for_versions;
 use crate::logger;
 use crate::logger::LoggerOptions;
 use crate::types::{Config, ConfigSection, CrateInfo, EnvInfo, FlowInfo, ToolchainSpecifier};
diff --git a/src/lib/test/mod.rs b/src/lib/test/mod.rs
--- a/src/lib/test/mod.rs
+++ b/src/lib/test/mod.rs
@@ -8,6 +9,7 @@ use git_info::types::GitInfo;
 use indexmap::IndexMap;
 use rust_info;
 use rust_info::types::{RustChannel, RustInfo};
+use semver::Version;
 use std::env;
 use std::path::PathBuf;
 
diff --git a/src/lib/test/mod.rs b/src/lib/test/mod.rs
--- a/src/lib/test/mod.rs
+++ b/src/lib/test/mod.rs
@@ -45,6 +47,16 @@ pub(crate) fn is_rust_channel(rust_channel: RustChannel) -> bool {
     current_rust_channel == rust_channel
 }
 
+pub(crate) fn is_min_rust_version(version: &str) -> bool {
+    let rustinfo = rust_info::get();
+    let rust_version = rustinfo.version.unwrap();
+
+    let version_struct = Version::parse(version).unwrap();
+    let rust_version_struct = Version::parse(&rust_version).unwrap();
+
+    is_min_version_valid_for_versions(&version_struct, &rust_version_struct)
+}
+
 pub(crate) fn should_test(panic_if_false: bool) -> bool {
     on_test_startup();
 
diff --git /dev/null b/src/lib/test/workspace-inherit/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/src/lib/test/workspace-inherit/Cargo.toml
@@ -0,0 +1,12 @@
+
+[workspace]
+members = ["member1"]
+
+[workspace.package]
+version = "1.2.3"
+authors = ["test author"]
+description = "test description"
+documentation = "test docs"
+license = "test license"
+homepage = "https://testpage.com"
+repository = "https://repotest.com"
diff --git /dev/null b/src/lib/test/workspace-inherit/member1/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/src/lib/test/workspace-inherit/member1/Cargo.toml
@@ -0,0 +1,9 @@
+[package]
+name = "member1"
+version.workspace = true
+authors.workspace = true
+description.workspace = true
+documentation.workspace = true
+license.workspace = true
+homepage.workspace = true
+repository.workspace = true
diff --git /dev/null b/src/lib/test/workspace-inherit/member1/src/main.rs
new file mode 100644
--- /dev/null
+++ b/src/lib/test/workspace-inherit/member1/src/main.rs
@@ -0,0 +1,3 @@
+fn main() {
+    println!("Hello World!");
+}
diff --git a/src/lib/test/workspace1/Cargo.toml b/src/lib/test/workspace1/Cargo.toml
--- a/src/lib/test/workspace1/Cargo.toml
+++ b/src/lib/test/workspace1/Cargo.toml
@@ -1,10 +1,4 @@
 
 [workspace]
 members = ["member1", "member2"]
-
 exclude = ["member1"]
-
-[dependencies]
-test = "1.0.0"
-member1 = { path = "./member1" }
-member3 = { path = "./member/member3" }
diff --git /dev/null b/src/lib/test/workspace1/member1/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/src/lib/test/workspace1/member1/Cargo.toml
@@ -0,0 +1,3 @@
+[package]
+name = "member1"
+version = "1.0.0"
diff --git /dev/null b/src/lib/test/workspace1/member1/src/lib.rs
new file mode 100644
--- /dev/null
+++ b/src/lib/test/workspace1/member1/src/lib.rs
@@ -0,0 +1,7 @@
+#[cfg(test)]
+mod tests {
+    #[test]
+    fn it_works() {
+        assert_eq!(2 + 2, 4);
+    }
+}
diff --git a/src/lib/test/workspace1/member2/Cargo.toml b/src/lib/test/workspace1/member2/Cargo.toml
--- a/src/lib/test/workspace1/member2/Cargo.toml
+++ b/src/lib/test/workspace1/member2/Cargo.toml
@@ -1,7 +1,3 @@
 [package]
 name = "member2"
 version = "5.4.3"
-
-[dependencies]
-test200 = { version = "1.0.0", features = ["abc"] }
-test100 = { path = "../test100" }
diff --git /dev/null b/src/lib/test/workspace1/member2/src/lib.rs
new file mode 100644
--- /dev/null
+++ b/src/lib/test/workspace1/member2/src/lib.rs
@@ -0,0 +1,7 @@
+#[cfg(test)]
+mod tests {
+    #[test]
+    fn it_works() {
+        assert_eq!(2 + 2, 4);
+    }
+}
diff --git /dev/null b/src/lib/test/workspace1/target/.rustc_info.json
new file mode 100644
--- /dev/null
+++ b/src/lib/test/workspace1/target/.rustc_info.json
@@ -0,0 +1,1 @@
+{"rustc_fingerprint":12441936045838259762,"outputs":{"15872395580024362796":{"success":true,"status":"","code":0,"stdout":"rustc 1.65.0-nightly (ce36e8825 2022-08-28)\nbinary: rustc\ncommit-hash: ce36e88256f09078519f8bc6b21e4dc88f88f523\ncommit-date: 2022-08-28\nhost: armv7-unknown-linux-gnueabihf\nrelease: 1.65.0-nightly\nLLVM version: 15.0.0\n","stderr":""},"12791339521227362961":{"success":true,"status":"","code":0,"stdout":"___\nlib___.rlib\nlib___.so\nlib___.so\nlib___.a\nlib___.so\n","stderr":""},"16870772668069704980":{"success":true,"status":"","code":0,"stdout":"___\nlib___.rlib\nlib___.so\nlib___.so\nlib___.a\nlib___.so\n/home/pi/workspace/home/.rustup/toolchains/nightly-armv7-unknown-linux-gnueabihf\ndebug_assertions\npanic=\"unwind\"\nproc_macro\ntarget_abi=\"eabihf\"\ntarget_arch=\"arm\"\ntarget_endian=\"little\"\ntarget_env=\"gnu\"\ntarget_family=\"unix\"\ntarget_feature=\"aclass\"\ntarget_feature=\"dsp\"\ntarget_feature=\"llvm14-builtins-abi\"\ntarget_feature=\"thumb2\"\ntarget_feature=\"v5te\"\ntarget_feature=\"v6\"\ntarget_feature=\"v6k\"\ntarget_feature=\"v6t2\"\ntarget_feature=\"v7\"\ntarget_feature=\"vfp2\"\ntarget_has_atomic=\"16\"\ntarget_has_atomic=\"32\"\ntarget_has_atomic=\"64\"\ntarget_has_atomic=\"8\"\ntarget_has_atomic=\"ptr\"\ntarget_has_atomic_equal_alignment=\"16\"\ntarget_has_atomic_equal_alignment=\"32\"\ntarget_has_atomic_equal_alignment=\"64\"\ntarget_has_atomic_equal_alignment=\"8\"\ntarget_has_atomic_equal_alignment=\"ptr\"\ntarget_has_atomic_load_store=\"16\"\ntarget_has_atomic_load_store=\"32\"\ntarget_has_atomic_load_store=\"64\"\ntarget_has_atomic_load_store=\"8\"\ntarget_has_atomic_load_store=\"ptr\"\ntarget_os=\"linux\"\ntarget_pointer_width=\"32\"\ntarget_thread_local\ntarget_vendor=\"unknown\"\nunix\n","stderr":""}},"successes":{}}
\ No newline at end of file
diff --git a/src/lib/test/workspace2/Cargo.lock /dev/null
--- a/src/lib/test/workspace2/Cargo.lock
+++ /dev/null
@@ -1,15 +0,0 @@
-# This file is automatically @generated by Cargo.
-# It is not intended for manual editing.
-version = 3
-
-[[package]]
-name = "env_target_dir_and_triple"
-version = "0.1.0"
-
-[[package]]
-name = "target_dir"
-version = "0.1.0"
-
-[[package]]
-name = "target_dir_and_triple"
-version = "0.1.0"

EOF_114329324912
cargo test --no-fail-fast --all-features
cd "src/lib/test/workspace1"
cargo test --no-fail-fast --all-features
cd ../../../../
cd "src/lib/test/workspace1/member2"
cargo test --no-fail-fast --all-features
cd ../../../../../
git reset --hard 6f82c9ec41374598a4fda5dde232fda7a557865d
git clean -fd
