#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -462,6 +462,44 @@ version = "0.1.0"
     Ok(())
 }
 
+#[test]
+fn it_can_overwrite_files() -> anyhow::Result<()> {
+    let template = tmp_dir()
+        .file(
+            "Cargo.toml",
+            r#"[package]
+name = "{{project-name}}"
+description = "A wonderful project"
+version = "0.1.0"
+"#,
+        )
+        .init_git()
+        .build();
+    let dir = tmp_dir().build();
+    let _ = binary()
+        .arg("generate")
+        .arg("--git")
+        .arg(template.path())
+        .arg("--name")
+        .arg("my-proj")
+        .arg("--init")
+        .current_dir(&dir.path())
+        .status();
+    binary()
+        .arg("generate")
+        .arg("--git")
+        .arg(template.path())
+        .arg("--name")
+        .arg("overwritten-proj")
+        .arg("--init")
+        .arg("--overwrite")
+        .current_dir(&dir.path())
+        .assert()
+        .success();
+    assert!(dir.read("Cargo.toml").contains("overwritten-proj"));
+    Ok(())
+}
+
 #[test]
 fn it_allows_user_defined_projectname_when_passing_force_flag() {
     let template = tmp_dir()
diff --git a/tests/integration/library.rs b/tests/integration/library.rs
--- a/tests/integration/library.rs
+++ b/tests/integration/library.rs
@@ -35,6 +35,7 @@ fn it_allows_generate_call_with_public_args_and_returns_the_generated_path() {
         destination: Some(dir.clone()),
         force_git_init: false,
         allow_commands: false,
+        overwrite: false,
     };
 
     assert_eq!(

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard e9fcff0cdd84481fd8bcb85a46ce7a92abe2d74c
git clean -fd
