#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/config.rs b/src/config.rs
--- a/src/config.rs
+++ b/src/config.rs
@@ -183,6 +184,7 @@ mod tests {
                 include: Some(vec!["Cargo.toml".into()]),
                 exclude: None,
                 ignore: None,
+                init: None,
             })
         );
         assert!(config.placeholders.is_some());
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -1,4 +1,5 @@
 use git2::Repository;
+use indoc::indoc;
 use predicates::prelude::*;
 
 use crate::helpers::project::binary;
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -1434,3 +1435,45 @@ fn error_message_for_invalid_repo_or_user() {
                 .from_utf8(),
         );
 }
+
+#[test]
+fn a_template_can_specify_to_be_generated_into_cwd() -> anyhow::Result<()> {
+    let template = tmp_dir()
+        .file(
+            "Cargo.toml",
+            indoc! {r#"
+                [package]
+                name = "{{project-name}}"
+                description = "A wonderful project"
+                version = "0.1.0"
+                "#},
+        )
+        .file(
+            "cargo-generate.toml",
+            indoc! {r#"
+                [template]
+                init = true
+                "#},
+        )
+        .init_git()
+        .build();
+
+    let dir = tmp_dir().build();
+
+    binary()
+        .arg("gen")
+        .arg("--git")
+        .arg(template.path())
+        .arg("-n")
+        .arg("foobar-project")
+        .arg("--branch")
+        .arg("main")
+        .current_dir(&dir.path())
+        .assert()
+        .success()
+        .stdout(predicates::str::contains("Done!").from_utf8());
+
+    assert!(dir.exists("Cargo.toml"));
+    assert!(!dir.path().join(".git").exists());
+    Ok(())
+}
diff --git a/tests/integration/config_file/favorites.rs b/tests/integration/config_file/favorites.rs
--- a/tests/integration/config_file/favorites.rs
+++ b/tests/integration/config_file/favorites.rs
@@ -297,3 +297,49 @@ fn favorites_default_value_can_be_overridden_by_environment() {
         .read("my-project/Cargo.toml")
         .contains(r#"description = "Overridden value""#));
 }
+
+#[test]
+fn favorite_can_specify_to_be_generated_into_cwd() -> anyhow::Result<()> {
+    let template = tmp_dir()
+        .file(
+            "Cargo.toml",
+            indoc! {r#"
+                [package]
+                name = "{{project-name}}"
+                description = "A wonderful project"
+                version = "0.1.0"
+                "#},
+        )
+        .init_git()
+        .build();
+    let config_dir = tmp_dir()
+        .file(
+            "config.toml",
+            &format!(
+                indoc! {r#"
+                [favorites.favorite]
+                git = "{git}"
+                init = true
+                "#},
+                git = template.path().display().to_string().escape_default(),
+            ),
+        )
+        .build();
+
+    let dir = tmp_dir().build();
+    binary()
+        .arg("generate")
+        .arg("--config")
+        .arg(config_dir.path().join("config.toml"))
+        .arg("--name")
+        .arg("my-proj")
+        .arg("favorite")
+        .current_dir(&dir.path())
+        .assert()
+        .success()
+        .stdout(predicates::str::contains("Done!").from_utf8());
+
+    assert!(dir.read("Cargo.toml").contains("my-proj"));
+    assert!(!dir.path().join(".git").exists());
+    Ok(())
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 7692e625993c5e18a8cea35858bccc4b82afbe14
git clean -fd
