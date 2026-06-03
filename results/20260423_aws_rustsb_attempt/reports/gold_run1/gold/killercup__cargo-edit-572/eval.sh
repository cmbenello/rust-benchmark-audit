#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/dependency.rs b/src/dependency.rs
--- a/src/dependency.rs
+++ b/src/dependency.rs
@@ -346,7 +369,7 @@ mod tests {
     fn to_toml_dep_with_git_source() {
         let crate_root = dunce::canonicalize(Path::new("/")).expect("root exists");
         let toml = Dependency::new("dep")
-            .set_git("https://foor/bar.git", None)
+            .set_git("https://foor/bar.git", None, None, None)
             .to_toml(&crate_root);
 
         assert_eq!(toml.0, "dep".to_owned());
diff --git /dev/null b/tests/cmd/add/git_rev.in
new file mode 100644
--- /dev/null
+++ b/tests/cmd/add/git_rev.in
@@ -0,0 +1,1 @@
+add-basic.in/
\ No newline at end of file
diff --git /dev/null b/tests/cmd/add/git_rev.out/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/add/git_rev.out/Cargo.toml
@@ -0,0 +1,6 @@
+[package]
+name = "cargo-list-test-fixture"
+version = "0.0.0"
+
+[dependencies]
+git-package = { git = "http://localhost/git-package.git", rev = "423a3" }
diff --git /dev/null b/tests/cmd/add/git_rev.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/add/git_rev.toml
@@ -0,0 +1,11 @@
+bin.name = "cargo-add"
+args = ["add", "git-package", "--git", "http://localhost/git-package.git", "--rev", "423a3"]
+status = "success"
+stdout = ""
+stderr = """
+      Adding git-package to dependencies.
+"""
+fs.sandbox = true
+
+[env.add]
+CARGO_IS_TEST="1"
diff --git /dev/null b/tests/cmd/add/git_tag.in
new file mode 100644
--- /dev/null
+++ b/tests/cmd/add/git_tag.in
@@ -0,0 +1,1 @@
+add-basic.in/
\ No newline at end of file
diff --git /dev/null b/tests/cmd/add/git_tag.out/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/add/git_tag.out/Cargo.toml
@@ -0,0 +1,6 @@
+[package]
+name = "cargo-list-test-fixture"
+version = "0.0.0"
+
+[dependencies]
+git-package = { git = "http://localhost/git-package.git", tag = "v1.0.0" }
diff --git /dev/null b/tests/cmd/add/git_tag.toml
new file mode 100644
--- /dev/null
+++ b/tests/cmd/add/git_tag.toml
@@ -0,0 +1,11 @@
+bin.name = "cargo-add"
+args = ["add", "git-package", "--git", "http://localhost/git-package.git", "--tag", "v1.0.0"]
+status = "success"
+stdout = ""
+stderr = """
+      Adding git-package to dependencies.
+"""
+fs.sandbox = true
+
+[env.add]
+CARGO_IS_TEST="1"
diff --git a/tests/cmd/add/overwrite_git_with_path.in/primary/Cargo.toml b/tests/cmd/add/overwrite_git_with_path.in/primary/Cargo.toml
--- a/tests/cmd/add/overwrite_git_with_path.in/primary/Cargo.toml
+++ b/tests/cmd/add/overwrite_git_with_path.in/primary/Cargo.toml
@@ -5,4 +5,4 @@ name = "cargo-list-test-fixture"
 version = "0.0.0"
 
 [dependencies]
-cargo-list-test-fixture-dependency = { git = "git://git.git", optional = true }
+cargo-list-test-fixture-dependency = { git = "git://git.git", branch = "main", optional = true }

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard 95ec0eedfec9bd9d5cbb1cb6076d8646abd29f81
git clean -fd
