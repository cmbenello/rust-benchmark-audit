#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1441,7 +1441,7 @@ fn adds_dependency_normalized_name() {
             &format!("--manifest-path={}", manifest),
         ])
         .success()
-        .stdout(predicates::str::contains(
+        .stderr(predicates::str::contains(
             "WARN: Added `linked-hash-map` instead of `linked_hash_map`",
         ));
 
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1758,7 +1758,7 @@ fn add_prints_message() {
         .env("CARGO_IS_TEST", "1")
         .assert()
         .success()
-        .stdout(predicates::str::contains(
+        .stderr(predicates::str::contains(
             "Adding docopt v0.6.0 to dependencies",
         ));
 }
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1780,7 +1780,7 @@ fn add_prints_message_with_section() {
         .env("CARGO_IS_TEST", "1")
         .assert()
         .success()
-        .stdout(predicates::str::contains(
+        .stderr(predicates::str::contains(
             "Adding clap v0.1.0 to optional dependencies for target `mytarget`",
         ));
 }
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1802,7 +1802,7 @@ fn add_prints_message_for_dev_deps() {
         .env("CARGO_IS_TEST", "1")
         .assert()
         .success()
-        .stdout(predicates::str::contains(
+        .stderr(predicates::str::contains(
             "Adding docopt v0.8.0 to dev-dependencies",
         ));
 }
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1824,7 +1824,7 @@ fn add_prints_message_for_build_deps() {
         .env("CARGO_IS_TEST", "1")
         .assert()
         .success()
-        .stdout(predicates::str::contains(
+        .stderr(predicates::str::contains(
             "Adding hello-world v0.1.0 to build-dependencies",
         ));
 }
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1956,7 +1956,7 @@ fn add_prints_message_for_features_deps() {
         .env("CARGO_IS_TEST", "1")
         .assert()
         .success()
-        .stdout(predicates::str::contains(
+        .stderr(predicates::str::contains(
             r#"Adding hello-world v0.1.0 to dependencies with features: ["jui"]"#,
         ));
 }
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -244,7 +244,7 @@ fn rm_prints_message() {
         .args(&["rm", "semver", &format!("--manifest-path={}", manifest)])
         .assert()
         .success()
-        .stdout("    Removing semver from dependencies\n");
+        .stderr("    Removing semver from dependencies\n");
 }
 
 #[test]
diff --git a/tests/cargo-rm.rs b/tests/cargo-rm.rs
--- a/tests/cargo-rm.rs
+++ b/tests/cargo-rm.rs
@@ -261,7 +261,7 @@ fn rm_prints_messages_for_multiple() {
         ])
         .assert()
         .success()
-        .stdout("    Removing semver from dependencies\n    Removing docopt from dependencies\n");
+        .stderr("    Removing semver from dependencies\n    Removing docopt from dependencies\n");
 }
 
 #[test]
diff --git a/tests/cargo-upgrade.rs b/tests/cargo-upgrade.rs
--- a/tests/cargo-upgrade.rs
+++ b/tests/cargo-upgrade.rs
@@ -550,5 +550,5 @@ fn upgrade_prints_messages() {
         ])
         .assert()
         .success()
-        .stdout(predicates::str::contains("docopt v0.8 -> v"));
+        .stderr(predicates::str::contains("docopt v0.8 -> v"));
 }

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard 712aeb70a91322507bb785cfd6d25ffd8fca8fd1
git clean -fd
