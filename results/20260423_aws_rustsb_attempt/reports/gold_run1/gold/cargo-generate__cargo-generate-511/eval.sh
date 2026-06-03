#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -54,6 +54,11 @@ predicates = "2.1"
 assert_cmd = "2.0"
 indoc = "1.0"
 
+[dev-dependencies.cargo-husky]
+version = "1"
+default-features = false
+features = ["prepush-hook", "run-cargo-test", "run-cargo-clippy", "run-cargo-fmt"]
+
 [features]
 vendored-openssl = ['openssl/vendored']
 
diff --git a/src/git.rs b/src/git.rs
--- a/src/git.rs
+++ b/src/git.rs
@@ -325,22 +334,19 @@ mod tests {
 
     #[test]
     fn should_not_fail_for_ssh_remote_urls() {
-        let config = GitConfig::new(REPO_URL_SSH.into(), None, None).unwrap();
+        let config = GitConfig::new(REPO_URL_SSH, None, None).unwrap();
         assert_eq!(config.kind, RepoKind::RemoteSsh);
     }
 
     #[test]
     #[should_panic(expected = "Invalid git remote 'aslkdgjlaskjdglskj'")]
     fn should_fail_for_non_existing_local_path() {
-        GitConfig::new("aslkdgjlaskjdglskj".into(), None, None).unwrap();
+        GitConfig::new("aslkdgjlaskjdglskj", None, None).unwrap();
     }
 
     #[test]
     fn should_support_a_local_relative_path() {
-        let remote: String = GitConfig::new("src".into(), None, None)
-            .unwrap()
-            .remote
-            .into();
+        let remote: String = GitConfig::new("src", None, None).unwrap().remote.into();
         #[cfg(unix)]
         assert!(
             remote.ends_with("/src"),
diff --git a/src/git.rs b/src/git.rs
--- a/src/git.rs
+++ b/src/git.rs
@@ -369,14 +375,11 @@ mod tests {
         // Absolute path.
         // If this fails because you cloned this repository into a non-UTF-8 directory... all
         // I can say is you probably had it comin'.
-        let remote: String = GitConfig::new(
-            current_dir().unwrap().display().to_string().into(),
-            None,
-            None,
-        )
-        .unwrap()
-        .remote
-        .into();
+        let remote: String =
+            GitConfig::new(current_dir().unwrap().display().to_string(), None, None)
+                .unwrap()
+                .remote
+                .into();
         #[cfg(unix)]
         assert!(remote.starts_with('/'), "remote {} starts with /", &remote);
         #[cfg(windows)]
diff --git a/src/git.rs b/src/git.rs
--- a/src/git.rs
+++ b/src/git.rs
@@ -390,7 +393,7 @@ mod tests {
     #[test]
     fn should_test_happy_path() {
         // Remote HTTPS URL.
-        let cfg = GitConfig::new(REPO_URL.into(), Some("main".to_owned()), None).unwrap();
+        let cfg = GitConfig::new(REPO_URL, Some("main".to_owned()), None).unwrap();
 
         assert_eq!(cfg.remote.as_ref(), Url::parse(REPO_URL).unwrap().as_str());
         assert_eq!(cfg.branch, GitReference::Branch("main".to_owned()));
diff --git a/src/git.rs b/src/git.rs
--- a/src/git.rs
+++ b/src/git.rs
@@ -399,11 +402,39 @@ mod tests {
     #[test]
     fn should_support_abbreviated_repository_short_urls_like() {
         assert_eq!(
-            GitConfig::new_abbr("cargo-generate/cargo-generate".into(), None, None)
+            GitConfig::new_abbr("cargo-generate/cargo-generate", None, None)
                 .unwrap()
                 .remote
                 .as_ref(),
             Url::parse(REPO_URL).unwrap().as_str()
         );
     }
+
+    #[test]
+    fn should_support_abbreviated_repository_short_urls_like_for_github() {
+        assert_eq!(
+            GitConfig::new_abbr("gh:cargo-generate/cargo-generate", None, None)
+                .unwrap()
+                .remote
+                .as_ref(),
+            Url::parse(REPO_URL).unwrap().as_str()
+        );
+    }
+
+    #[test]
+    fn should_support_bb_gl_gh_abbreviations() {
+        assert_eq!(
+            &abbreviated_git_url_to_full_remote("gh:foo/bar"),
+            "https://github.com/foo/bar.git"
+        );
+        assert_eq!(
+            &abbreviated_git_url_to_full_remote("bb:foo/bar"),
+            "https://bitbucket.org/foo/bar.git"
+        );
+        assert_eq!(
+            &abbreviated_git_url_to_full_remote("gl:foo/bar"),
+            "https://gitlab.com/foo/bar.git"
+        );
+        assert_eq!(&abbreviated_git_url_to_full_remote("foo/bar"), "foo/bar");
+    }
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard dbc133f2287b4b19f65facd6e7fcbbe94c73f101
git clean -fd
