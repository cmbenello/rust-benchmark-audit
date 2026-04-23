#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/bin/upgrade/main.rs b/src/bin/upgrade/main.rs
--- a/src/bin/upgrade/main.rs
+++ b/src/bin/upgrade/main.rs
@@ -224,14 +235,14 @@ impl DesiredUpgrades {
     fn get_upgraded(self, allow_prerelease: bool, manifest_path: &Path) -> Result<ActualUpgrades> {
         self.0
             .into_iter()
-            .map(|(name, version)| {
+            .map(|(dep, version)| {
                 if let Some(v) = version {
-                    Ok((name, v))
+                    Ok((dep, v))
                 } else {
-                    get_latest_dependency(&name, allow_prerelease, manifest_path)
+                    get_latest_dependency(&dep.name, allow_prerelease, manifest_path)
                         .map(|new_dep| {
                             (
-                                name,
+                                dep,
                                 new_dep
                                     .version()
                                     .expect("Invalid dependency type")
diff --git a/src/dependency.rs b/src/dependency.rs
--- a/src/dependency.rs
+++ b/src/dependency.rs
@@ -112,46 +137,151 @@ impl Dependency {
     /// (If the dependency is set as `optional` or `default-features` is set to `false`,
     /// an `InlineTable` is returned in any case.)
     pub fn to_toml(&self) -> (String, toml_edit::Item) {
-        let data: toml_edit::Item =
-            match (self.optional, self.default_features, self.source.clone()) {
-                // Extra short when version flag only
-                (
-                    false,
-                    true,
-                    DependencySource::Version {
-                        version: Some(v),
-                        path: None,
-                    },
-                ) => toml_edit::value(v),
-                // Other cases are represented as an inline table
-                (optional, default_features, source) => {
-                    let mut data = toml_edit::InlineTable::default();
-
-                    match source {
-                        DependencySource::Version { version, path } => {
-                            if let Some(v) = version {
-                                data.get_or_insert("version", v);
-                            }
-                            if let Some(p) = path {
-                                data.get_or_insert("path", p);
-                            }
+        let data: toml_edit::Item = match (
+            self.optional,
+            self.default_features,
+            self.source.clone(),
+            self.rename.as_ref(),
+        ) {
+            // Extra short when version flag only
+            (
+                false,
+                true,
+                DependencySource::Version {
+                    version: Some(v),
+                    path: None,
+                },
+                None,
+            ) => toml_edit::value(v),
+            // Other cases are represented as an inline table
+            (optional, default_features, source, rename) => {
+                let mut data = toml_edit::InlineTable::default();
+
+                match source {
+                    DependencySource::Version { version, path } => {
+                        if let Some(v) = version {
+                            data.get_or_insert("version", v);
                         }
-                        DependencySource::Git(v) => {
-                            data.get_or_insert("git", v);
+                        if let Some(p) = path {
+                            data.get_or_insert("path", p);
                         }
                     }
-                    if self.optional {
-                        data.get_or_insert("optional", optional);
+                    DependencySource::Git(v) => {
+                        data.get_or_insert("git", v);
                     }
-                    if !self.default_features {
-                        data.get_or_insert("default-features", default_features);
-                    }
-
-                    data.fmt();
-                    toml_edit::value(toml_edit::Value::InlineTable(data))
                 }
-            };
+                if self.optional {
+                    data.get_or_insert("optional", optional);
+                }
+                if !self.default_features {
+                    data.get_or_insert("default-features", default_features);
+                }
+                if rename.is_some() {
+                    data.get_or_insert("package", self.name.clone());
+                }
+
+                data.fmt();
+                toml_edit::value(toml_edit::Value::InlineTable(data))
+            }
+        };
+
+        (self.name_in_manifest().to_string(), data)
+    }
+}
+
+#[cfg(test)]
+mod tests {
+    use crate::dependency::Dependency;
+
+    #[test]
+    fn to_toml_simple_dep() {
+        let toml = Dependency::new("dep").to_toml();
+
+        assert_eq!(toml.0, "dep".to_owned());
+    }
+
+    #[test]
+    fn to_toml_simple_dep_with_version() {
+        let toml = Dependency::new("dep").set_version("1.0").to_toml();
+
+        assert_eq!(toml.0, "dep".to_owned());
+        assert_eq!(toml.1.as_str(), Some("1.0"));
+    }
+
+    #[test]
+    fn to_toml_optional_dep() {
+        let toml = Dependency::new("dep").set_optional(true).to_toml();
+
+        assert_eq!(toml.0, "dep".to_owned());
+        assert!(toml.1.is_inline_table());
+
+        let dep = toml.1.as_inline_table().unwrap();
+        assert_eq!(dep.get("optional").unwrap().as_bool(), Some(true));
+    }
+
+    #[test]
+    fn to_toml_dep_without_default_features() {
+        let toml = Dependency::new("dep").set_default_features(false).to_toml();
+
+        assert_eq!(toml.0, "dep".to_owned());
+        assert!(toml.1.is_inline_table());
+
+        let dep = toml.1.as_inline_table().unwrap();
+        assert_eq!(dep.get("default-features").unwrap().as_bool(), Some(false));
+    }
+
+    #[test]
+    fn to_toml_dep_with_path_source() {
+        let toml = Dependency::new("dep").set_path("~/foo/bar").to_toml();
+
+        assert_eq!(toml.0, "dep".to_owned());
+        assert!(toml.1.is_inline_table());
+
+        let dep = toml.1.as_inline_table().unwrap();
+        assert_eq!(dep.get("path").unwrap().as_str(), Some("~/foo/bar"));
+    }
+
+    #[test]
+    fn to_toml_dep_with_git_source() {
+        let toml = Dependency::new("dep")
+            .set_git("https://foor/bar.git")
+            .to_toml();
+
+        assert_eq!(toml.0, "dep".to_owned());
+        assert!(toml.1.is_inline_table());
+
+        let dep = toml.1.as_inline_table().unwrap();
+        assert_eq!(
+            dep.get("git").unwrap().as_str(),
+            Some("https://foor/bar.git")
+        );
+    }
+
+    #[test]
+    fn to_toml_renamed_dep() {
+        let toml = Dependency::new("dep").set_rename("d").to_toml();
+
+        assert_eq!(toml.0, "d".to_owned());
+        assert!(toml.1.is_inline_table());
+
+        let dep = toml.1.as_inline_table().unwrap();
+        assert_eq!(dep.get("package").unwrap().as_str(), Some("dep"));
+    }
+
+    #[test]
+    fn to_toml_complex_dep() {
+        let toml = Dependency::new("dep")
+            .set_version("1.0")
+            .set_default_features(false)
+            .set_rename("d")
+            .to_toml();
+
+        assert_eq!(toml.0, "d".to_owned());
+        assert!(toml.1.is_inline_table());
 
-        (self.name.clone(), data)
+        let dep = toml.1.as_inline_table().unwrap();
+        assert_eq!(dep.get("package").unwrap().as_str(), Some("dep"));
+        assert_eq!(dep.get("version").unwrap().as_str(), Some("1.0"));
+        assert_eq!(dep.get("default-features").unwrap().as_bool(), Some(false));
     }
 }
diff --git a/tests/cargo-upgrade.rs b/tests/cargo-upgrade.rs
--- a/tests/cargo-upgrade.rs
+++ b/tests/cargo-upgrade.rs
@@ -189,6 +189,52 @@ fn upgrade_optional_dependency() {
     assert_eq!(val["optional"].as_bool(), Some(true));
 }
 
+#[test]
+fn upgrade_renamed_dependency_all() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/upgrade/Cargo.toml.renamed_dep");
+
+    execute_command(&["upgrade"], &manifest);
+
+    let toml = get_toml(&manifest);
+
+    let dep1 = &toml["dependencies"]["te"];
+    assert_eq!(
+        dep1["version"].as_str(),
+        Some("toml_edit--CURRENT_VERSION_TEST")
+    );
+
+    let dep2 = &toml["dependencies"]["rx"];
+    assert_eq!(
+        dep2["version"].as_str(),
+        Some("regex--CURRENT_VERSION_TEST")
+    );
+}
+
+#[test]
+fn upgrade_renamed_dependency_inline_specified_only() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/upgrade/Cargo.toml.renamed_dep");
+
+    execute_command(&["upgrade", "toml_edit"], &manifest);
+
+    let toml = get_toml(&manifest);
+    let dep = &toml["dependencies"]["te"];
+    assert_eq!(
+        dep["version"].as_str(),
+        Some("toml_edit--CURRENT_VERSION_TEST")
+    );
+}
+
+#[test]
+fn upgrade_renamed_dependency_table_specified_only() {
+    let (_tmpdir, manifest) = clone_out_test("tests/fixtures/upgrade/Cargo.toml.renamed_dep");
+
+    execute_command(&["upgrade", "regex"], &manifest);
+
+    let toml = get_toml(&manifest);
+    let dep = &toml["dependencies"]["rx"];
+    assert_eq!(dep["version"].as_str(), Some("regex--CURRENT_VERSION_TEST"));
+}
+
 #[test]
 fn upgrade_at() {
     let (_tmpdir, manifest) = clone_out_test("tests/fixtures/add/Cargo.toml.sample");
diff --git /dev/null b/tests/fixtures/upgrade/Cargo.toml.renamed_dep
new file mode 100644
--- /dev/null
+++ b/tests/fixtures/upgrade/Cargo.toml.renamed_dep
@@ -0,0 +1,13 @@
+[package]
+name = "cargo-list-test-fixture"
+version = "0.0.0"
+
+[lib]
+path = "dummy.rs"
+
+[dependencies]
+te = { package = "toml_edit", version = "0.1.5" }
+
+[dependencies.rx]
+package = "regex"
+version = "0.2"
diff --git a/tests/fixtures/upgrade/Cargo.toml.source b/tests/fixtures/upgrade/Cargo.toml.source
--- a/tests/fixtures/upgrade/Cargo.toml.source
+++ b/tests/fixtures/upgrade/Cargo.toml.source
@@ -12,11 +12,16 @@ serde_json = "1.0"
 syn = { version = "0.11.10", default-features = false, features = ["parsing"] }
 tar = { version = "0.4", default-features = false }
 ftp = "2.2.1"
+te = { package = "toml_edit", version = "0.1.5" }
 
 [dependencies.semver]
 features = ["serde"]
 version = "0.7"
 
+[dependencies.rn]
+package = "renamed"
+version = "0.1"
+
 [dev-dependencies]
 assert_cli = "0.2.0"
 tempdir = "0.3"
diff --git a/tests/fixtures/upgrade/Cargo.toml.target b/tests/fixtures/upgrade/Cargo.toml.target
--- a/tests/fixtures/upgrade/Cargo.toml.target
+++ b/tests/fixtures/upgrade/Cargo.toml.target
@@ -12,11 +12,16 @@ serde_json = "serde_json--CURRENT_VERSION_TEST"
 syn = { version = "syn--CURRENT_VERSION_TEST", default-features = false, features = ["parsing"] }
 tar = { version = "tar--CURRENT_VERSION_TEST", default-features = false }
 ftp = "ftp--CURRENT_VERSION_TEST"
+te = { package = "toml_edit", version = "toml_edit--CURRENT_VERSION_TEST" }
 
 [dependencies.semver]
 features = ["serde"]
 version = "semver--CURRENT_VERSION_TEST"
 
+[dependencies.rn]
+package = "renamed"
+version = "renamed--CURRENT_VERSION_TEST"
+
 [dev-dependencies]
 assert_cli = "assert_cli--CURRENT_VERSION_TEST"
 tempdir = "tempdir--CURRENT_VERSION_TEST"

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard e8d8eebfa0733da112197a78dc793f68a5ab6441
git clean -fd
