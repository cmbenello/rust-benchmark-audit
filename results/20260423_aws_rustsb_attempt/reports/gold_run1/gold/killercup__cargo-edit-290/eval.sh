#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/bin/add/args.rs b/src/bin/add/args.rs
--- a/src/bin/add/args.rs
+++ b/src/bin/add/args.rs
@@ -77,7 +79,9 @@ impl Args {
                             krate
                         } else {
                             get_latest_dependency(crate_name, self.flag_allow_prerelease)?
-                        }.set_optional(self.flag_optional),
+                        }
+                        .set_optional(self.flag_optional)
+                        .set_default_features(!self.flag_no_default_features),
                     )
                 })
                 .collect();
diff --git a/tests/cargo-add.rs b/tests/cargo-add.rs
--- a/tests/cargo-add.rs
+++ b/tests/cargo-add.rs
@@ -590,6 +590,54 @@ fn adds_multiple_optional_dependencies() {
         .expect("optional not a bool"));
 }
 
+#[test]
+fn adds_no_default_features_dependency() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.sample");
+
+    // dependency not present beforehand
+    let toml = get_toml(&manifest);
+    assert!(toml["dependencies"].is_none());
+
+    execute_command(
+        &[
+            "add",
+            "versioned-package",
+            "--vers",
+            ">=0.1.1",
+            "--no-default-features",
+        ],
+        &manifest,
+    );
+
+    // dependency present afterwards
+    let toml = get_toml(&manifest);
+    let val = &toml["dependencies"]["versioned-package"]["default-features"];
+    assert_eq!(val.as_bool().expect("default-features not a bool"), false);
+}
+
+#[test]
+fn adds_multiple_no_default_features_dependencies() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.sample");
+
+    // dependencies not present beforehand
+    let toml = get_toml(&manifest);
+    assert!(toml["dependencies"].is_none());
+
+    execute_command(
+        &["add", "--no-default-features", "my-package1", "my-package2"],
+        &manifest,
+    );
+
+    // dependencies present afterwards
+    let toml = get_toml(&manifest);
+    assert!(!&toml["dependencies"]["my-package1"]["default-features"]
+        .as_bool()
+        .expect("default-features not a bool"));
+    assert!(!&toml["dependencies"]["my-package2"]["default-features"]
+        .as_bool()
+        .expect("default-features not a bool"));
+}
+
 #[test]
 fn adds_dependency_with_target_triple() {
     let (_tmpdir, manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.sample");

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 26a641e71f1190d9a8764b538211396f71e6c0a0
git clean -fd
