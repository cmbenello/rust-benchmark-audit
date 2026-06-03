#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -61,16 +61,6 @@ indoc = "~2.0"
 url = "~2.5"
 bstr = "~1.9"
 
-[dev-dependencies.cargo-husky]
-version = "~1.5"
-default-features = false
-features = [
-    "prepush-hook",
-    "run-cargo-test",
-    "run-cargo-clippy",
-    "run-cargo-fmt",
-]
-
 [features]
 default = ["vendored-libgit2"]
 vendored-libgit2 = ["git2/vendored-libgit2"]
diff --git a/src/template_variables/project_dir.rs b/src/template_variables/project_dir.rs
--- a/src/template_variables/project_dir.rs
+++ b/src/template_variables/project_dir.rs
@@ -77,3 +78,46 @@ impl TryFrom<(&ProjectNameInput, &UserParsedInput)> for ProjectDir {
         Ok(Self(project_dir))
     }
 }
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::template_variables::ProjectNameInput;
+    use crate::user_parsed_input::UserParsedInputBuilder;
+
+    #[test]
+    fn test_snake_case_is_accepted() {
+        let input = ProjectNameInput("lock_firmware".to_string());
+        let args = UserParsedInputBuilder::for_testing().build();
+
+        let project_dir = ProjectDir::try_from((&input, &args)).unwrap();
+        assert!(project_dir.0.as_path().ends_with("lock_firmware"));
+    }
+
+    #[test]
+    fn test_dash_case_is_accepted() {
+        let input = ProjectNameInput("lock-firmware".to_string());
+        let args = UserParsedInputBuilder::for_testing().build();
+
+        let project_dir = ProjectDir::try_from((&input, &args)).unwrap();
+        assert!(project_dir.0.as_path().ends_with("lock-firmware"));
+    }
+
+    #[test]
+    fn test_converted_to_dash_case() {
+        let input = ProjectNameInput("lockFirmware".to_string());
+        let args = UserParsedInputBuilder::for_testing().build();
+
+        let project_dir = ProjectDir::try_from((&input, &args)).unwrap();
+        assert!(project_dir.0.as_path().ends_with("lock-firmware"));
+    }
+
+    #[test]
+    fn test_not_converted_to_dash_case_when_with_force() {
+        let input = ProjectNameInput("lockFirmware".to_string());
+        let args = UserParsedInputBuilder::for_testing().with_force().build();
+
+        let project_dir = ProjectDir::try_from((&input, &args)).unwrap();
+        assert!(project_dir.0.as_path().ends_with("lockFirmware"));
+    }
+}
diff --git a/src/template_variables/project_name.rs b/src/template_variables/project_name.rs
--- a/src/template_variables/project_name.rs
+++ b/src/template_variables/project_name.rs
@@ -33,3 +33,54 @@ impl Display for ProjectName {
         self.0.fmt(f)
     }
 }
+
+pub fn sanitize_project_name(name: &str) -> String {
+    let snake_case_project_name = name.to_snake_case();
+    if snake_case_project_name == name {
+        snake_case_project_name
+    } else {
+        name.to_kebab_case()
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use crate::user_parsed_input::UserParsedInputBuilder;
+
+    #[test]
+    fn test_snake_case_is_accepted() {
+        let input = ProjectNameInput("lock_firmware".to_string());
+        let args = UserParsedInputBuilder::for_testing().build();
+
+        let project_name = ProjectName::from((&input, &args));
+        assert_eq!(project_name, ProjectName("lock_firmware".into()));
+    }
+
+    #[test]
+    fn test_dash_case_is_accepted() {
+        let input = ProjectNameInput("lock-firmware".to_string());
+        let args = UserParsedInputBuilder::for_testing().build();
+
+        let project_name = ProjectName::from((&input, &args));
+        assert_eq!(project_name, ProjectName("lock-firmware".into()));
+    }
+
+    #[test]
+    fn test_converted_to_dash_case() {
+        let input = ProjectNameInput("lockFirmware".to_string());
+        let args = UserParsedInputBuilder::for_testing().build();
+
+        let project_name = ProjectName::from((&input, &args));
+        assert_eq!(project_name, ProjectName("lock-firmware".into()));
+    }
+
+    #[test]
+    fn test_not_converted_to_dash_case_when_with_force() {
+        let input = ProjectNameInput("lockFirmware".to_string());
+        let args = UserParsedInputBuilder::for_testing().with_force().build();
+
+        let project_name = ProjectName::from((&input, &args));
+        assert_eq!(project_name, ProjectName("lockFirmware".into()));
+    }
+}
diff --git a/src/user_parsed_input.rs b/src/user_parsed_input.rs
--- a/src/user_parsed_input.rs
+++ b/src/user_parsed_input.rs
@@ -13,6 +13,42 @@ use regex::Regex;
 use crate::{app_config::AppConfig, template_variables::CrateType, GenerateArgs, Vcs};
 use log::warn;
 
+#[derive(Debug)]
+#[cfg(test)]
+pub struct UserParsedInputBuilder {
+    subject: UserParsedInput,
+}
+
+#[cfg(test)]
+impl UserParsedInputBuilder {
+    #[cfg(test)]
+    pub(crate) fn for_testing() -> Self {
+        use crate::TemplatePath;
+        Self {
+            subject: UserParsedInput::try_from_args_and_config(
+                AppConfig::default(),
+                &GenerateArgs {
+                    destination: Some(Path::new("/tmp/dest/").to_path_buf()),
+                    template_path: TemplatePath {
+                        path: Some("/tmp".to_string()),
+                        ..TemplatePath::default()
+                    },
+                    ..GenerateArgs::default()
+                },
+            ),
+        }
+    }
+
+    pub const fn with_force(mut self) -> Self {
+        self.subject.force = true;
+        self
+    }
+
+    pub fn build(self) -> UserParsedInput {
+        self.subject
+    }
+}
+
 // Contains parsed information from user.
 #[derive(Debug)]
 pub struct UserParsedInput {
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -156,7 +156,7 @@ fn it_substitutes_os_arch() {
 }
 
 #[test]
-fn it_kebabcases_projectname_when_passed_to_flag() {
+fn it_keeps_snake_case_projectname() {
     let template = tempdir()
         .file(
             "Cargo.toml",
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -181,8 +181,8 @@ version = "0.1.0"
         .stdout(predicates::str::contains("Done!").from_utf8());
 
     assert!(dir
-        .read("foobar-project/Cargo.toml")
-        .contains("foobar-project"));
+        .read("foobar_project/Cargo.toml")
+        .contains("foobar_project"));
 }
 
 #[test]
diff --git a/tests/integration/helpers/project_builder.rs b/tests/integration/helpers/project_builder.rs
--- a/tests/integration/helpers/project_builder.rs
+++ b/tests/integration/helpers/project_builder.rs
@@ -140,6 +140,7 @@ impl ProjectBuilder {
 
                 Command::new("git")
                     .arg("commit")
+                    .arg("--no-gpg-sign")
                     .arg("--message")
                     .arg("initial main commit")
                     .current_dir(path)
diff --git a/tests/integration/helpers/project_builder.rs b/tests/integration/helpers/project_builder.rs
--- a/tests/integration/helpers/project_builder.rs
+++ b/tests/integration/helpers/project_builder.rs
@@ -179,6 +180,7 @@ impl ProjectBuilder {
 
             Command::new("git")
                 .arg("commit")
+                .arg("--no-gpg-sign")
                 .arg("--message")
                 .arg("initial commit")
                 .current_dir(path)
diff --git a/tests/integration/helpers/project_builder.rs b/tests/integration/helpers/project_builder.rs
--- a/tests/integration/helpers/project_builder.rs
+++ b/tests/integration/helpers/project_builder.rs
@@ -212,6 +214,7 @@ impl ProjectBuilder {
 
                 Command::new("git")
                     .arg("commit")
+                    .arg("--no-gpg-sign")
                     .arg("--message")
                     .arg("dummy commit after tag")
                     .current_dir(path)
diff --git a/tests/integration/hooks_and_rhai.rs b/tests/integration/hooks_and_rhai.rs
--- a/tests/integration/hooks_and_rhai.rs
+++ b/tests/integration/hooks_and_rhai.rs
@@ -351,15 +351,15 @@ fn init_hook_can_change_project_name_but_keeps_cli_name_for_destination() {
         .file(
             "cargo-generate.toml",
             indoc! {r#"
-            [hooks]
-            init = ["init.rhai"]
+                [hooks]
+                init = ["init.rhai"]
             "#},
         )
         .file(
             "generated.txt",
             indoc! {r#"
-            {{crate_name}}
-        "#},
+                {{crate_name}}
+            "#},
         )
         .init_git()
         .build();

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 26427861bddd32bc2fdca354a76bc7b093871084
git clean -fd
