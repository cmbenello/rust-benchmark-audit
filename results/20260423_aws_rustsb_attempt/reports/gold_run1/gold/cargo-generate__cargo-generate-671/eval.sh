#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/integration/git.rs b/tests/integration/git.rs
--- a/tests/integration/git.rs
+++ b/tests/integration/git.rs
@@ -32,6 +32,29 @@ fn it_allows_a_git_branch_to_be_specified() {
         .contains("foobar-project"));
 }
 
+#[test]
+fn it_allows_a_git_tag_to_be_specified() {
+    let template = tmp_dir().init_default_template().tag("v1.0").build();
+    let dir = tmp_dir().build();
+
+    binary()
+        .arg("generate")
+        .arg("--tag")
+        .arg("v1.0")
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
 #[test]
 fn it_removes_git_history() {
     let template = tmp_dir().init_default_template().build();
diff --git a/tests/integration/helpers/project_builder.rs b/tests/integration/helpers/project_builder.rs
--- a/tests/integration/helpers/project_builder.rs
+++ b/tests/integration/helpers/project_builder.rs
@@ -11,6 +11,7 @@ pub struct ProjectBuilder {
     root: TempDir,
     git: bool,
     branch: Option<String>,
+    tag: Option<String>,
 }
 
 pub fn tmp_dir() -> ProjectBuilder {
diff --git a/tests/integration/helpers/project_builder.rs b/tests/integration/helpers/project_builder.rs
--- a/tests/integration/helpers/project_builder.rs
+++ b/tests/integration/helpers/project_builder.rs
@@ -20,6 +21,7 @@ pub fn tmp_dir() -> ProjectBuilder {
         root: tempdir().unwrap(),
         git: false,
         branch: None,
+        tag: None,
     }
 }
 
diff --git a/tests/integration/helpers/project_builder.rs b/tests/integration/helpers/project_builder.rs
--- a/tests/integration/helpers/project_builder.rs
+++ b/tests/integration/helpers/project_builder.rs
@@ -55,6 +57,11 @@ version = "0.1.0"
         self
     }
 
+    pub fn tag(mut self, tag: &str) -> Self {
+        self.tag = Some(tag.to_owned());
+        self
+    }
+
     pub fn add_submodule<I: Into<String>>(mut self, destination: I, path: I) -> Self {
         self.submodules.push((destination.into(), path.into()));
         self
diff --git a/tests/integration/helpers/project_builder.rs b/tests/integration/helpers/project_builder.rs
--- a/tests/integration/helpers/project_builder.rs
+++ b/tests/integration/helpers/project_builder.rs
@@ -163,6 +170,40 @@ version = "0.1.0"
                 .assert()
                 .success();
 
+            if let Some(ref tag) = self.tag {
+                Command::new("git")
+                    .arg("tag")
+                    .arg("-a")
+                    .arg(tag)
+                    .arg("-m")
+                    .arg(format!("our test tag {tag}"))
+                    .current_dir(&path)
+                    .assert()
+                    .success();
+
+                for &(ref file, _) in self.files.iter() {
+                    let path = path.join(file);
+                    fs::remove_file(&path).unwrap_or_else(|_| {
+                        panic!("couldn't remove file {path:?}, after commiting tag {tag}")
+                    });
+                }
+
+                Command::new("git")
+                    .arg("add")
+                    .arg("--all")
+                    .current_dir(&path)
+                    .assert()
+                    .success();
+
+                Command::new("git")
+                    .arg("commit")
+                    .arg("--message")
+                    .arg("dummy commit after tag")
+                    .current_dir(&path)
+                    .assert()
+                    .success();
+            }
+
             if self.branch.is_some() {
                 Command::new("git")
                     .arg("checkout")
diff --git a/tests/integration/library.rs b/tests/integration/library.rs
--- a/tests/integration/library.rs
+++ b/tests/integration/library.rs
@@ -12,6 +12,7 @@ fn it_allows_generate_call_with_public_args_and_returns_the_generated_path() {
             auto_path: None,
             git: Some(format!("{}", template.path().display())),
             branch: Some(String::from("main")),
+            tag: None,
             path: None,
             favorite: None,
             subfolder: None,

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard d95b96ddbec990289327ba9ba6ee0a0d51685ece
git clean -fd
