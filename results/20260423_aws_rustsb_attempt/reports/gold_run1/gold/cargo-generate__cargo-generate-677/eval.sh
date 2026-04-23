#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/config.rs b/src/config.rs
--- a/src/config.rs
+++ b/src/config.rs
@@ -184,6 +187,7 @@ mod tests {
                 include: Some(vec!["Cargo.toml".into()]),
                 exclude: None,
                 ignore: None,
+                vcs: None,
                 init: None,
             })
         );
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -1277,6 +1277,46 @@ version = "0.1.0"
     assert!(Repository::open(dir.path().join("foobar-project")).is_err());
 }
 
+#[test]
+fn vsc_none_can_be_specified_in_the_template() {
+    // Build and commit on branch named 'main'
+    let template = tmp_dir()
+        .file(
+            "Cargo.toml",
+            r#"[package]
+name = "{{project-name}}"
+description = "A wonderful project"
+version = "0.1.0"
+"#,
+        )
+        .file(
+            "cargo-generate.toml",
+            r#"[template]
+vcs = "None"
+"#,
+        )
+        .init_git()
+        .build();
+
+    let dir = tmp_dir().build();
+
+    binary()
+        .arg("generate")
+        .arg("--git")
+        .arg(template.path())
+        .arg("--name")
+        .arg("foobar-project")
+        .current_dir(&dir.path())
+        .assert()
+        .success()
+        .stdout(predicates::str::contains("Done!").from_utf8());
+
+    assert!(dir
+        .read("foobar-project/Cargo.toml")
+        .contains("foobar-project"));
+    assert!(Repository::open(dir.path().join("foobar-project")).is_err());
+}
+
 #[test]
 fn it_provides_crate_type_lib() {
     // Build and commit on branch named 'main'
diff --git a/tests/integration/config_file/favorites.rs b/tests/integration/config_file/favorites.rs
--- a/tests/integration/config_file/favorites.rs
+++ b/tests/integration/config_file/favorites.rs
@@ -1,3 +1,5 @@
+use cargo_generate::Vcs;
+use git2::Repository;
 use predicates::prelude::*;
 
 use crate::helpers::project::binary;
diff --git a/tests/integration/config_file/favorites.rs b/tests/integration/config_file/favorites.rs
--- a/tests/integration/config_file/favorites.rs
+++ b/tests/integration/config_file/favorites.rs
@@ -7,7 +9,11 @@ use assert_cmd::prelude::*;
 use indoc::indoc;
 use std::path::PathBuf;
 
-fn create_favorite_config(name: &str, template_path: &Project) -> (Project, PathBuf) {
+fn create_favorite_config(
+    name: &str,
+    template_path: &Project,
+    vcs: Option<Vcs>,
+) -> (Project, PathBuf) {
     let project = tmp_dir()
         .file(
             "cargo-generate",
diff --git a/tests/integration/config_file/favorites.rs b/tests/integration/config_file/favorites.rs
--- a/tests/integration/config_file/favorites.rs
+++ b/tests/integration/config_file/favorites.rs
@@ -17,10 +23,16 @@ fn create_favorite_config(name: &str, template_path: &Project) -> (Project, Path
                     description = "Favorite for the {name} template"
                     git = "{git}"
                     branch = "{branch}"
+                    {vcs}
                     "#},
                 name = name,
                 git = template_path.path().display().to_string().escape_default(),
-                branch = "main"
+                branch = "main",
+                vcs = if let Some(vcs) = vcs {
+                    format!(r#"vcs = "{vcs:?}""#)
+                } else {
+                    String::from("")
+                }
             ),
         )
         .build();
diff --git a/tests/integration/config_file/favorites.rs b/tests/integration/config_file/favorites.rs
--- a/tests/integration/config_file/favorites.rs
+++ b/tests/integration/config_file/favorites.rs
@@ -32,7 +44,7 @@ fn create_favorite_config(name: &str, template_path: &Project) -> (Project, Path
 fn favorite_with_git_becomes_subfolder() {
     let favorite_template = create_template("favorite-template");
     let git_template = create_template("git-template");
-    let (_config, config_path) = create_favorite_config("test", &favorite_template);
+    let (_config, config_path) = create_favorite_config("test", &favorite_template, None);
     let working_dir = tmp_dir().build();
 
     binary()
diff --git a/tests/integration/config_file/favorites.rs b/tests/integration/config_file/favorites.rs
--- a/tests/integration/config_file/favorites.rs
+++ b/tests/integration/config_file/favorites.rs
@@ -134,7 +146,7 @@ fn favorite_with_subfolder() -> anyhow::Result<()> {
 #[test]
 fn it_can_use_favorites() {
     let favorite_template = create_template("favorite-template");
-    let (_config, config_path) = create_favorite_config("test", &favorite_template);
+    let (_config, config_path) = create_favorite_config("test", &favorite_template, None);
     let working_dir = tmp_dir().build();
 
     binary()
diff --git a/tests/integration/config_file/favorites.rs b/tests/integration/config_file/favorites.rs
--- a/tests/integration/config_file/favorites.rs
+++ b/tests/integration/config_file/favorites.rs
@@ -149,15 +161,38 @@ fn it_can_use_favorites() {
         .success()
         .stdout(predicates::str::contains("Done!").from_utf8());
 
+    assert!(Repository::open(working_dir.path().join("favorite-project")).is_ok());
     assert!(working_dir
         .read("favorite-project/Cargo.toml")
         .contains(r#"description = "favorite-template""#));
 }
 
+#[test]
+fn a_favorite_can_set_vcs_to_none_by_default() {
+    let favorite_template = create_template("favorite-template");
+    let (_config, config_path) =
+        create_favorite_config("test", &favorite_template, Some(Vcs::None));
+    let working_dir = tmp_dir().build();
+
+    binary()
+        .arg("generate")
+        .arg("--config")
+        .arg(config_path)
+        .arg("--name")
+        .arg("favorite-project")
+        .arg("test")
+        .current_dir(&working_dir.path())
+        .assert()
+        .success()
+        .stdout(predicates::str::contains("Done!").from_utf8());
+
+    assert!(Repository::open(working_dir.path().join("favorite-project")).is_err());
+}
+
 #[test]
 fn favorites_default_to_git_if_not_defined() {
     let favorite_template = create_template("favorite-template");
-    let (_config, config_path) = create_favorite_config("test", &favorite_template);
+    let (_config, config_path) = create_favorite_config("test", &favorite_template, None);
     let working_dir = tmp_dir().build();
 
     binary()
diff --git a/tests/integration/library.rs b/tests/integration/library.rs
--- a/tests/integration/library.rs
+++ b/tests/integration/library.rs
@@ -1,5 +1,5 @@
 use crate::helpers::project_builder::tmp_dir;
-use cargo_generate::{generate, GenerateArgs, TemplatePath, Vcs};
+use cargo_generate::{generate, GenerateArgs, TemplatePath};
 
 #[test]
 fn it_allows_generate_call_with_public_args_and_returns_the_generated_path() {
diff --git a/tests/integration/library.rs b/tests/integration/library.rs
--- a/tests/integration/library.rs
+++ b/tests/integration/library.rs
@@ -19,7 +19,7 @@ fn it_allows_generate_call_with_public_args_and_returns_the_generated_path() {
         },
         name: Some(String::from("foobar_project")),
         force: true,
-        vcs: Vcs::Git,
+        vcs: None,
         verbose: true,
         template_values_file: None,
         silent: false,

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 1098c62c7f86956ba0c09be314533eaabf9822d5
git clean -fd
