#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -73,70 +73,6 @@ version = "0.1.0"
     assert_eq!(0, references);
 }
 
-#[test]
-fn it_removes_git_history() {
-    let template = tmp_dir()
-        .file(
-            "Cargo.toml",
-            r#"[package]
-name = "{{project-name}}"
-description = "A wonderful project"
-version = "0.1.0"
-"#,
-        )
-        .init_git()
-        .build();
-
-    let dir = tmp_dir().build();
-
-    binary()
-        .arg("generate")
-        .arg("--git")
-        .arg(template.path())
-        .arg("--name")
-        .arg("foobar-project")
-        .current_dir(&dir.path())
-        .assert()
-        .success()
-        .stdout(predicates::str::contains("Done!").from_utf8());
-
-    let target_path = dir.target_path("foobar-project");
-    let repo = git2::Repository::open(&target_path).unwrap();
-    assert_eq!(0, repo.references().unwrap().count());
-}
-
-#[test]
-fn it_removes_git_history_also_on_local_templates() {
-    let template = tmp_dir()
-        .file(
-            "Cargo.toml",
-            r#"[package]
-name = "{{project-name}}"
-description = "A wonderful project"
-version = "0.1.0"
-"#,
-        )
-        .init_git()
-        .build();
-
-    let dir = tmp_dir().build();
-
-    binary()
-        .arg("generate")
-        .arg("--path")
-        .arg(template.path())
-        .arg("--name")
-        .arg("xyz")
-        .current_dir(&dir.path())
-        .assert()
-        .success()
-        .stdout(predicates::str::contains("Done!").from_utf8());
-
-    let target_path = dir.target_path("xyz");
-    let repo = git2::Repository::open(&target_path).unwrap();
-    assert_eq!(0, repo.references().unwrap().count());
-}
-
 #[test]
 fn it_substitutes_projectname_in_cargo_toml() {
     let template = tmp_dir()
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -798,42 +734,6 @@ version = "0.1.0"
     .is_file());
 }
 
-#[test]
-fn it_allows_a_git_branch_to_be_specified() {
-    // Build and commit on branch named 'main'
-    let template = tmp_dir()
-        .file(
-            "Cargo.toml",
-            r#"[package]
-name = "{{project-name}}"
-description = "A wonderful project"
-version = "0.1.0"
-"#,
-        )
-        .init_git()
-        .branch("baz")
-        .build();
-
-    let dir = tmp_dir().build();
-
-    binary()
-        .arg("generate")
-        .arg("--branch")
-        .arg("baz")
-        .arg("--git")
-        .arg(template.path())
-        .arg("--name")
-        .arg("foobar-project")
-        .current_dir(&dir.path())
-        .assert()
-        .success()
-        .stdout(predicates::str::contains("Done!").from_utf8());
-
-    assert!(dir
-        .read("foobar-project/Cargo.toml")
-        .contains("foobar-project"));
-}
-
 #[test]
 fn it_loads_a_submodule() {
     let submodule = tmp_dir()
diff --git /dev/null b/tests/integration/git.rs
new file mode 100644
--- /dev/null
+++ b/tests/integration/git.rs
@@ -0,0 +1,93 @@
+use assert_cmd::prelude::*;
+use git2::Repository;
+use predicates::prelude::*;
+
+use crate::helpers::project::binary;
+use crate::helpers::project_builder::tmp_dir;
+
+#[test]
+fn it_allows_a_git_branch_to_be_specified() {
+    let template = tmp_dir().init_default_template().branch("bak").build();
+    let dir = tmp_dir().build();
+
+    binary()
+        .arg("generate")
+        .arg("--branch")
+        .arg("bak")
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
+}
+
+#[test]
+fn it_removes_git_history() {
+    let template = tmp_dir().init_default_template().build();
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
+    let repo = Repository::open(&dir.path().join("foobar-project")).unwrap();
+    let references = repo.references().unwrap().count();
+    assert_eq!(0, references);
+}
+
+#[test]
+fn it_removes_git_history_also_on_local_templates() {
+    let template = tmp_dir().init_default_template().build();
+    let dir = tmp_dir().build();
+
+    binary()
+        .arg("generate")
+        .arg("--path")
+        .arg(template.path())
+        .arg("--name")
+        .arg("xyz")
+        .current_dir(&dir.path())
+        .assert()
+        .success()
+        .stdout(predicates::str::contains("Done!").from_utf8());
+
+    let target_path = dir.target_path("xyz");
+    let repo = git2::Repository::open(&target_path).unwrap();
+    assert_eq!(0, repo.references().unwrap().count());
+}
+
+#[test]
+fn it_should_init_an_empty_git_repo_even_when_starting_from_a_repo_when_forced() {
+    let template = tmp_dir().init_default_template().build();
+    let target_path = template.path();
+
+    binary()
+        .arg("generate")
+        .arg("--force-git-init")
+        .arg("--git")
+        .arg(template.path())
+        .arg("--name")
+        .arg("foo")
+        .current_dir(&target_path)
+        .assert()
+        .success()
+        .stdout(predicates::str::contains("Done!").from_utf8());
+
+    let repo = Repository::open(&target_path.join("foo")).unwrap();
+    let references = repo.references().unwrap().count();
+    assert_eq!(0, references);
+}
diff --git a/tests/integration/helpers/project_builder.rs b/tests/integration/helpers/project_builder.rs
--- a/tests/integration/helpers/project_builder.rs
+++ b/tests/integration/helpers/project_builder.rs
@@ -24,22 +24,37 @@ pub fn tmp_dir() -> ProjectBuilder {
 }
 
 impl ProjectBuilder {
-    pub fn file(mut self, name: &str, contents: &str) -> ProjectBuilder {
+    /// builds a template with
+    /// - one file `Cargo.toml` in it
+    /// - one placeholder `project-name`
+    pub fn init_default_template(self) -> Self {
+        self.file(
+            "Cargo.toml",
+            r#"[package]
+name = "{{project-name}}"
+description = "A wonderful project"
+version = "0.1.0"
+"#,
+        )
+        .init_git()
+    }
+
+    pub fn file(mut self, name: &str, contents: &str) -> Self {
         self.files.push((name.to_string(), contents.to_string()));
         self
     }
 
-    pub fn init_git(mut self) -> ProjectBuilder {
+    pub fn init_git(mut self) -> Self {
         self.git = true;
         self
     }
 
-    pub fn branch(mut self, branch: &str) -> ProjectBuilder {
+    pub fn branch(mut self, branch: &str) -> Self {
         self.branch = Some(branch.to_owned());
         self
     }
 
-    pub fn add_submodule<I: Into<String>>(mut self, destination: I, path: I) -> ProjectBuilder {
+    pub fn add_submodule<I: Into<String>>(mut self, destination: I, path: I) -> Self {
         self.submodules.push((destination.into(), path.into()));
         self
     }
diff --git a/tests/integration/library.rs b/tests/integration/library.rs
--- a/tests/integration/library.rs
+++ b/tests/integration/library.rs
@@ -36,6 +36,7 @@ version = "0.1.0"
         ssh_identity: None,
         define: vec![],
         init: false,
+        force_git_init: false,
     };
     // need to cd to the dir as we aren't running in the cargo shell.
     assert!(std::env::set_current_dir(&dir.root).is_ok());
diff --git a/tests/integration/main.rs b/tests/integration/main.rs
--- a/tests/integration/main.rs
+++ b/tests/integration/main.rs
@@ -4,6 +4,7 @@ mod helpers;
 mod basics;
 mod favorites;
 mod filenames;
+mod git;
 mod hooks;
 mod library;
 mod values;

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 3853b4c68a4440578e243ede86244a431a6ac5f2
git clean -fd
