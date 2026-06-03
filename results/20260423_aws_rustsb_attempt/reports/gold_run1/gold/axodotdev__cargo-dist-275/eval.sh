#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/cargo-dist-schema/cargo-dist-json-schema.json b/cargo-dist-schema/cargo-dist-json-schema.json
--- a/cargo-dist-schema/cargo-dist-json-schema.json
+++ b/cargo-dist-schema/cargo-dist-json-schema.json
@@ -320,6 +331,19 @@
           }
         }
       }
+    },
+    "SystemInfo": {
+      "description": "Info about the system/toolchain used to build this announcement.\n\nNote that this is info from the machine that generated this file, which *ideally* should be similar to the machines that built all the artifacts, but we can't guarantee that.\n\ndist-manifest.json is by default generated at the start of the build process, and typically on a linux machine because that's usually the fastest/cheapest part of CI infra.",
+      "type": "object",
+      "properties": {
+        "cargo_version_line": {
+          "description": "The version of Cargo used (first line of cargo -vV)\n\nNote that this is the version used on the machine that generated this file, which presumably should be the same version used on all the machines that built all the artifacts, but maybe not! It's more likely to be correct if rust-toolchain.toml is used with a specific pinned version.",
+          "type": [
+            "string",
+            "null"
+          ]
+        }
+      }
     }
   }
 }
diff --git a/cargo-dist-schema/src/lib.rs b/cargo-dist-schema/src/lib.rs
--- a/cargo-dist-schema/src/lib.rs
+++ b/cargo-dist-schema/src/lib.rs
@@ -63,6 +67,26 @@ pub struct DistManifest {
     pub artifacts: BTreeMap<ArtifactId, Artifact>,
 }
 
+/// Info about the system/toolchain used to build this announcement.
+///
+/// Note that this is info from the machine that generated this file,
+/// which *ideally* should be similar to the machines that built all the artifacts, but
+/// we can't guarantee that.
+///
+/// dist-manifest.json is by default generated at the start of the build process,
+/// and typically on a linux machine because that's usually the fastest/cheapest
+/// part of CI infra.
+#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
+pub struct SystemInfo {
+    /// The version of Cargo used (first line of cargo -vV)
+    ///
+    /// Note that this is the version used on the machine that generated this file,
+    /// which presumably should be the same version used on all the machines that
+    /// built all the artifacts, but maybe not! It's more likely to be correct
+    /// if rust-toolchain.toml is used with a specific pinned version.
+    pub cargo_version_line: Option<String>,
+}
+
 /// A Release of an Application
 #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
 pub struct Release {
diff --git a/cargo-dist-schema/src/snapshots/cargo_dist_schema__emit.snap b/cargo-dist-schema/src/snapshots/cargo_dist_schema__emit.snap
--- a/cargo-dist-schema/src/snapshots/cargo_dist_schema__emit.snap
+++ b/cargo-dist-schema/src/snapshots/cargo_dist_schema__emit.snap
@@ -324,6 +335,19 @@ expression: json_schema
           }
         }
       }
+    },
+    "SystemInfo": {
+      "description": "Info about the system/toolchain used to build this announcement.\n\nNote that this is info from the machine that generated this file, which *ideally* should be similar to the machines that built all the artifacts, but we can't guarantee that.\n\ndist-manifest.json is by default generated at the start of the build process, and typically on a linux machine because that's usually the fastest/cheapest part of CI infra.",
+      "type": "object",
+      "properties": {
+        "cargo_version_line": {
+          "description": "The version of Cargo used (first line of cargo -vV)\n\nNote that this is the version used on the machine that generated this file, which presumably should be the same version used on all the machines that built all the artifacts, but maybe not! It's more likely to be correct if rust-toolchain.toml is used with a specific pinned version.",
+          "type": [
+            "string",
+            "null"
+          ]
+        }
+      }
     }
   }
 }
diff --git a/cargo-dist/src/init.rs b/cargo-dist/src/init.rs
--- a/cargo-dist/src/init.rs
+++ b/cargo-dist/src/init.rs
@@ -143,9 +143,8 @@ fn get_new_dist_metadata(
         DistMetadata {
             // If they init with this version we're gonna try to stick to it!
             cargo_dist_version: Some(std::env!("CARGO_PKG_VERSION").parse().unwrap()),
-            // latest stable release at this precise moment
-            // maybe there's something more clever we can do here, but, *shrug*
-            rust_toolchain_version: Some("1.67.1".to_owned()),
+            // deprecated, default to not emitting it
+            rust_toolchain_version: None,
             ci: vec![],
             installers: None,
             targets: cfg.targets.is_empty().not().then(|| cfg.targets.clone()),
diff --git a/cargo-dist/tests/cli-tests.rs b/cargo-dist/tests/cli-tests.rs
--- a/cargo-dist/tests/cli-tests.rs
+++ b/cargo-dist/tests/cli-tests.rs
@@ -90,6 +90,7 @@ fn test_manifest() {
         (r#""announcement_changelog": .*"#, r#""announcement_changelog": "CENSORED""#),
         (r#""announcement_github_body": .*"#, r#""announcement_github_body": "CENSORED""#),
         (r#""announcement_is_prerelease": .*"#, r#""announcement_is_prerelease": "CENSORED""#),
+        (r#""cargo_version_line": .*"#, r#""cargo_version_line": "CENSORED""#),
     ]}, {
         insta::assert_snapshot!(format_outputs(&output));
     });
diff --git a/cargo-dist/tests/cli-tests.rs b/cargo-dist/tests/cli-tests.rs
--- a/cargo-dist/tests/cli-tests.rs
+++ b/cargo-dist/tests/cli-tests.rs
@@ -121,6 +122,7 @@ fn test_lib_manifest() {
         (r#""announcement_changelog": .*"#, r#""announcement_changelog": "CENSORED""#),
         (r#""announcement_github_body": .*"#, r#""announcement_github_body": "CENSORED""#),
         (r#""announcement_is_prerelease": .*"#, r#""announcement_is_prerelease": "CENSORED""#),
+        (r#""cargo_version_line": .*"#, r#""cargo_version_line": "CENSORED""#),
     ]}, {
         insta::assert_snapshot!(format_outputs(&output));
     });
diff --git a/cargo-dist/tests/cli-tests.rs b/cargo-dist/tests/cli-tests.rs
--- a/cargo-dist/tests/cli-tests.rs
+++ b/cargo-dist/tests/cli-tests.rs
@@ -150,6 +152,7 @@ fn test_error_manifest() {
         (r#""announcement_changelog": .*"#, r#""announcement_changelog": "CENSORED""#),
         (r#""announcement_github_body": .*"#, r#""announcement_github_body": "CENSORED""#),
         (r#""announcement_is_prerelease": .*"#, r#""announcement_is_prerelease": "CENSORED""#),
+        (r#""cargo_version_line": .*"#, r#""cargo_version_line": "CENSORED""#),
     ]}, {
         insta::assert_snapshot!(format_outputs(&output));
     });
diff --git a/cargo-dist/tests/snapshots/cli_tests__error_manifest.snap b/cargo-dist/tests/snapshots/cli_tests__error_manifest.snap
--- a/cargo-dist/tests/snapshots/cli_tests__error_manifest.snap
+++ b/cargo-dist/tests/snapshots/cli_tests__error_manifest.snap
@@ -1,6 +1,5 @@
 ---
 source: cargo-dist/tests/cli-tests.rs
-assertion_line: 154
 expression: format_outputs(&output)
 ---
 stdout:
diff --git a/cargo-dist/tests/snapshots/cli_tests__error_manifest.snap b/cargo-dist/tests/snapshots/cli_tests__error_manifest.snap
--- a/cargo-dist/tests/snapshots/cli_tests__error_manifest.snap
+++ b/cargo-dist/tests/snapshots/cli_tests__error_manifest.snap
@@ -8,6 +7,7 @@ stdout:
 
 stderr:
 analyzing workspace:
+ WARN rust-toolchain-version is deprecated, use rust-toolchain.toml if you want pinned toolchains
   cargo-dist (didn't match tag v1.0.0-FAKEVERSION)
     [bin] cargo-dist
   cargo-dist-schema (no binaries)
diff --git a/cargo-dist/tests/snapshots/cli_tests__lib_manifest.snap b/cargo-dist/tests/snapshots/cli_tests__lib_manifest.snap
--- a/cargo-dist/tests/snapshots/cli_tests__lib_manifest.snap
+++ b/cargo-dist/tests/snapshots/cli_tests__lib_manifest.snap
@@ -9,10 +9,14 @@ stdout:
   "announcement_is_prerelease": "CENSORED"
   "announcement_title": "CENSORED"
   "announcement_github_body": "CENSORED"
+  "system_info": {
+    "cargo_version_line": "CENSORED"
+  }
 }
 
 stderr:
 analyzing workspace:
+ WARN rust-toolchain-version is deprecated, use rust-toolchain.toml if you want pinned toolchains
   cargo-dist (didn't match tag cargo-dist-schema-v1.0.0-FAKEVERSION)
     [bin] cargo-dist
   cargo-dist-schema (no binaries)
diff --git a/cargo-dist/tests/snapshots/cli_tests__manifest.snap b/cargo-dist/tests/snapshots/cli_tests__manifest.snap
--- a/cargo-dist/tests/snapshots/cli_tests__manifest.snap
+++ b/cargo-dist/tests/snapshots/cli_tests__manifest.snap
@@ -10,6 +10,9 @@ stdout:
   "announcement_title": "CENSORED"
   "announcement_changelog": "CENSORED"
   "announcement_github_body": "CENSORED"
+  "system_info": {
+    "cargo_version_line": "CENSORED"
+  },
   "releases": [
     {
       "app_name": "cargo-dist",
diff --git a/cargo-dist/tests/snapshots/cli_tests__manifest.snap b/cargo-dist/tests/snapshots/cli_tests__manifest.snap
--- a/cargo-dist/tests/snapshots/cli_tests__manifest.snap
+++ b/cargo-dist/tests/snapshots/cli_tests__manifest.snap
@@ -222,6 +225,7 @@ stdout:
 
 stderr:
 analyzing workspace:
+ WARN rust-toolchain-version is deprecated, use rust-toolchain.toml if you want pinned toolchains
   cargo-dist
     [bin] cargo-dist
   cargo-dist-schema (no binaries)

EOF_114329324912
git status
git diff
cd "cargo-dist-schema"
cargo test --no-fail-fast
cd ../
cd "cargo-dist"
cargo test --no-fail-fast
cd ../
git status
git reset --hard b9856c45e81d5996d691af60e96bc5f9bfe3d990
git clean -fd
