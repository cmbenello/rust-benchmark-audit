#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/bin/add/args.rs b/src/bin/add/args.rs
--- a/src/bin/add/args.rs
+++ b/src/bin/add/args.rs
@@ -395,7 +398,7 @@ mod tests {
         };
 
         assert_eq!(
-            args.parse_dependencies().unwrap(),
+            args.parse_dependencies(None).unwrap(),
             vec![Dependency::new("demo").set_version("0.4.2")]
         );
     }
diff --git a/src/bin/add/args.rs b/src/bin/add/args.rs
--- a/src/bin/add/args.rs
+++ b/src/bin/add/args.rs
@@ -409,7 +412,7 @@ mod tests {
             ..Args::default()
         };
         assert_eq!(
-            args_github.parse_dependencies().unwrap(),
+            args_github.parse_dependencies(None).unwrap(),
             vec![Dependency::new("cargo-edit").set_git(github_url, None)]
         );
 
diff --git a/src/bin/add/args.rs b/src/bin/add/args.rs
--- a/src/bin/add/args.rs
+++ b/src/bin/add/args.rs
@@ -419,7 +422,7 @@ mod tests {
             ..Args::default()
         };
         assert_eq!(
-            args_gitlab.parse_dependencies().unwrap(),
+            args_gitlab.parse_dependencies(None).unwrap(),
             vec![Dependency::new("polly").set_git(gitlab_url, None)]
         );
     }
diff --git a/src/bin/add/args.rs b/src/bin/add/args.rs
--- a/src/bin/add/args.rs
+++ b/src/bin/add/args.rs
@@ -433,7 +436,9 @@ mod tests {
             ..Args::default()
         };
         assert_eq!(
-            args_path.parse_dependencies().unwrap()[0].path().unwrap(),
+            args_path.parse_dependencies(None).unwrap()[0]
+                .path()
+                .unwrap(),
             self_path
         );
     }
diff --git a/src/fetch.rs b/src/fetch.rs
--- a/src/fetch.rs
+++ b/src/fetch.rs
@@ -40,7 +40,12 @@ pub fn get_latest_dependency(
         };
 
         let features = if crate_name == "your-face" {
-            vec!["nose".to_string(), "mouth".to_string(), "eyes".to_string()]
+            vec![
+                "nose".to_string(),
+                "mouth".to_string(),
+                "eyes".to_string(),
+                "ears".to_string(),
+            ]
         } else {
             vec![]
         };
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1311,20 +1311,42 @@ fn adds_features_dependency() {
     let toml = get_toml(&manifest);
     assert!(toml["dependencies"].is_none());
 
-    execute_command(
-        &[
-            "add",
-            "https://github.com/killercup/cargo-edit.git",
-            "--features",
-            "jui",
-        ],
-        &manifest,
-    );
+    execute_command(&["add", "your-face", "--features", "eyes"], &manifest);
 
     // dependency present afterwards
     let toml = get_toml(&manifest);
-    let val = toml["dependencies"]["cargo-edit"]["features"][0].as_str();
-    assert_eq!(val, Some("jui"));
+    let val = toml["dependencies"]["your-face"]["features"][0].as_str();
+    assert_eq!(val, Some("eyes"));
+}
+
+#[test]
+fn warns_on_unknown_features_dependency() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.sample");
+
+    // dependency not present beforehand
+    let toml = get_toml(&manifest);
+    assert!(toml["dependencies"].is_none());
+
+    Command::cargo_bin("cargo-add")
+        .expect("can find bin")
+        .args(&["add", "your-face", "--features", "noze"])
+        .arg("--manifest-path")
+        .arg(&manifest)
+        .env("CARGO_IS_TEST", "1")
+        .assert()
+        .success()
+        .stderr(predicates::str::contains(
+            "Unrecognized features: [\"noze\"]",
+        ))
+        .stderr(predicates::str::contains("eyes"))
+        .stderr(predicates::str::contains("nose"))
+        .stderr(predicates::str::contains("mouth"))
+        .stderr(predicates::str::contains("ears"));
+
+    // dependency is present afterwards
+    let toml = get_toml(&manifest);
+    let val = toml["dependencies"]["your-face"]["features"][0].as_str();
+    assert!(val.is_some());
 }
 
 #[test]
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1421,32 +1443,32 @@ your-face = { version = "99999.0.0", features = ["nose"] }
 }
 
 #[test]
-fn handles_specifying_features_option_multiple_times() {
+fn can_be_forced_to_provide_an_empty_features_list() {
     overwrite_dependency_test(
         &["add", "your-face"],
-        &[
-            "add",
-            "your-face",
-            "--features",
-            "nose",
-            "--features",
-            "mouth",
-        ],
+        &["add", "your-face", "--features", ""],
         r#"
 [dependencies]
-your-face = { version = "99999.0.0", features = ["nose", "mouth"] }
+your-face = { version = "99999.0.0", features = [] }
 "#,
     )
 }
 
 #[test]
-fn can_be_forced_to_provide_an_empty_features_list() {
+fn handles_specifying_features_option_multiple_times() {
     overwrite_dependency_test(
         &["add", "your-face"],
-        &["add", "your-face", "--features", ""],
+        &[
+            "add",
+            "your-face",
+            "--features",
+            "nose",
+            "--features",
+            "mouth",
+        ],
         r#"
 [dependencies]
-your-face = { version = "99999.0.0", features = [] }
+your-face = { version = "99999.0.0", features = ["mouth", "nose"] }
 "#,
     )
 }
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -1458,7 +1480,7 @@ fn parses_space_separated_argument_to_features() {
         &["add", "your-face", "--features", "mouth ears"],
         r#"
 [dependencies]
-your-face = { version = "99999.0.0", features = ["mouth", "ears"] }
+your-face = { version = "99999.0.0", features = ["ears", "mouth"] }
 "#,
     )
 }
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -2010,22 +2032,25 @@ fn add_dependency_to_workspace_member() {
 #[test]
 fn add_prints_message_for_features_deps() {
     let (_tmpdir, manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.sample");
+    let (dep_tmpdir, _dep_manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.features");
 
     Command::cargo_bin("cargo-add")
         .unwrap()
         .args(&[
             "add",
-            "hello-world",
+            "your-face",
             "--vers",
             "0.1.0",
             "--features",
-            "jui",
+            "eyes",
             &format!("--manifest-path={}", manifest),
         ])
+        .arg("--path")
+        .arg(&dep_tmpdir.path())
         .env("CARGO_IS_TEST", "1")
         .assert()
         .success()
         .stderr(predicates::str::contains(
-            r#"Adding hello-world v0.1.0 to dependencies with features: ["jui"]"#,
+            r#"Adding your-face v0.1.0 to dependencies with features: ["eyes"]"#,
         ));
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard c85829043979c1d5ea1845d6657186d1c13a623f
git clean -fd
