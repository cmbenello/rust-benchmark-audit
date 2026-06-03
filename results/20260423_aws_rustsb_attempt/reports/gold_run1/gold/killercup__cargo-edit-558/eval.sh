#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -92,7 +92,8 @@ add = ["cli"]
 rm = ["cli"]
 upgrade = ["cli"]
 set-version = ["cli"]
-cli = ["atty", "clap"]
+cli = ["color", "clap"]
+color = ["concolor-control/auto"]
 test-external-apis = []
 vendored-openssl = ["git2/vendored-openssl"]
 vendored-libgit2 = ["git2/vendored-libgit2"]
diff --git a/src/bin/upgrade/main.rs b/src/bin/upgrade/main.rs
--- a/src/bin/upgrade/main.rs
+++ b/src/bin/upgrade/main.rs
@@ -16,15 +16,15 @@ extern crate error_chain;
 
 use crate::errors::*;
 use cargo_edit::{
-    find, get_latest_dependency, manifest_from_pkgid, registry_url, update_registry_index,
-    CrateName, Dependency, LocalManifest,
+    colorize_stderr, find, get_latest_dependency, manifest_from_pkgid, registry_url,
+    update_registry_index, CrateName, Dependency, LocalManifest,
 };
 use clap::Parser;
 use std::collections::{HashMap, HashSet};
 use std::io::Write;
 use std::path::{Path, PathBuf};
 use std::process;
-use termcolor::{BufferWriter, Color, ColorChoice, ColorSpec, WriteColor};
+use termcolor::{BufferWriter, Color, ColorSpec, WriteColor};
 use url::Url;
 
 mod errors {
diff --git a/src/fetch.rs b/src/fetch.rs
--- a/src/fetch.rs
+++ b/src/fetch.rs
@@ -7,7 +7,7 @@ use std::env;
 use std::io::Write;
 use std::path::Path;
 use std::time::Duration;
-use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
+use termcolor::{Color, ColorSpec, StandardStream, WriteColor};
 use url::Url;
 
 /// Query latest version from a registry index

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 59b5c040bce40f3df1a39de766176d4880110fa8
git clean -fd
