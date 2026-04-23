#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/.github/workflows/build.yml b/.github/workflows/build.yml
--- a/.github/workflows/build.yml
+++ b/.github/workflows/build.yml
@@ -79,6 +79,12 @@ jobs:
         run: |
           git config --global user.email "you@example.com"
           git config --global user.name "Your Name"
+      - name: Install SSH key
+        uses: shimataro/ssh-key-action@v2
+        with:
+          key: ${{ secrets.CI_SSH_KEY }}
+          known_hosts: ${{ secrets.KNOWN_HOSTS }}
+          if_key_exists: replace # replace / ignore / fail; optional (defaults to fail)
       - name: cargo test
         run: cargo test --all --locked -- -Z unstable-options
 
diff --git a/src/git.rs b/src/git.rs
--- a/src/git.rs
+++ b/src/git.rs
@@ -58,149 +62,294 @@ impl GitConfig {
     /// [hub].
     ///
     /// [hub]: https://github.com/github/hub
-    pub fn new_abbr(git: &str, branch: Option<String>) -> Result<Self> {
-        Self::new(git, branch.clone()).or_else(|e| {
-            Self::new(&format!("https://github.com/{}.git", git), branch).map_err(|_| e)
+    pub fn new_abbr(
+        git: Cow<'a, str>,
+        branch: Option<String>,
+        identity: Option<PathBuf>,
+    ) -> Result<Self> {
+        Self::new(git.clone(), branch.clone(), identity.clone()).or_else(|_| {
+            let full_remote = format!("https://github.com/{}.git", &git);
+            Self::new(full_remote.into(), branch, identity)
         })
     }
 }
 
+fn canonicalize_path(p: &Path) -> Result<PathBuf> {
+    let p = if p.to_str().unwrap().starts_with("~/") {
+        home()?.join(p.strip_prefix("~/").unwrap())
+    } else {
+        p.to_path_buf()
+    };
+
+    p.canonicalize().context("path does not exist")
+}
+
+#[test]
+fn should_canonicalize() {
+    #[cfg(target_os = "macos")]
+    assert!(canonicalize_path(&PathBuf::from("../"))
+        .unwrap()
+        .starts_with("/Users/"));
+    #[cfg(target_os = "linux")]
+    assert!(canonicalize_path(&PathBuf::from("../"))
+        .unwrap()
+        .starts_with("/home/"));
+    #[cfg(windows)]
+    assert!(canonicalize_path(&PathBuf::from("../"))
+        .unwrap()
+        // not a bug, a feature:
+        // https://stackoverflow.com/questions/41233684/why-does-my-canonicalized-path-get-prefixed-with
+        .to_str()
+        .unwrap()
+        .starts_with("\\\\?\\"));
+}
+
+/// takes care of `~/` paths, defaults to `$HOME/.ssh/id_rsa` and resolves symlinks.
+fn get_private_key_path(identity: Option<PathBuf>) -> Result<PathBuf> {
+    let private_key = identity.unwrap_or(home()?.join(".ssh/id_rsa"));
+
+    canonicalize_path(&private_key).context("private key path was not was incorrect")
+}
+
+fn git_ssh_credentials_callback<'a>(identity: Option<PathBuf>) -> Result<RemoteCallbacks<'a>> {
+    let private_key = get_private_key_path(identity)?;
+    println!(
+        "{} {} `{}` {}",
+        emoji::INFO,
+        style("Using private key:").bold(),
+        style(pretty_path(&private_key)?).bold().yellow(),
+        style("for git-ssh checkout").bold()
+    );
+    let mut cb = RemoteCallbacks::new();
+    cb.credentials(
+        move |_url, username_from_url: Option<&str>, _allowed_types| {
+            Cred::ssh_key(username_from_url.unwrap_or("git"), None, &private_key, None)
+        },
+    );
+    Ok(cb)
+}
+
+/// home path wrapper
+fn home() -> Result<PathBuf> {
+    canonicalize_path(&dirs::home_dir().context("$HOME was not set")?)
+}
+
+#[test]
+fn should_pretty_path() {
+    let p = pretty_path(home().unwrap().as_path().join(".cargo").as_path()).unwrap();
+    #[cfg(unix)]
+    assert_eq!(p, "$HOME/.cargo");
+    #[cfg(windows)]
+    assert_eq!(p, "%userprofile%\\.cargo");
+}
+
+/// prevents from long stupid paths, and replace the home path by the literal `$HOME`
+fn pretty_path(a: &Path) -> Result<String> {
+    #[cfg(unix)]
+    let home_var = "$HOME";
+    #[cfg(windows)]
+    let home_var = "%userprofile%";
+    Ok(a.display()
+        .to_string()
+        .replace(&home()?.display().to_string(), home_var))
+}
+
+/// thanks to @extrawurst for pointing this out
+/// https://github.com/extrawurst/gitui/blob/master/asyncgit/src/sync/branch/mod.rs#L38
+fn get_branch_name_repo(repo: &Repository) -> Result<String> {
+    let iter = repo.branches(None)?;
+
+    for b in iter {
+        let b = b?;
+
+        if b.0.is_head() {
+            let name = b.0.name()?.unwrap_or("");
+            return Ok(name.into());
+        }
+    }
+
+    anyhow::bail!("A repo has no Head")
+}
+
+fn init_all_submodules(repo: &Repository) -> Result<()> {
+    for mut sub in repo.submodules().unwrap() {
+        sub.update(true, None)?;
+    }
+
+    Ok(())
+}
+
 pub(crate) fn create(project_dir: &Path, args: GitConfig) -> Result<String> {
-    let temp = Builder::new().prefix(project_dir).tempdir()?;
-    let config = Config::default()?;
-    let remote = GitRemote::new(&args.remote);
-
-    let ((db, rev), branch_name) = match &args.branch {
-        GitReference::Branch(branch_name) => (
-            remote.checkout(&temp.path(), None, &args.branch, None, &config)?,
-            branch_name.clone(),
-        ),
-        GitReference::DefaultBranch => {
-            // Cargo has a specific behavior for now for handling the "default" branch. It forces
-            // it to the branch named "master" even if the actual default branch of the repository
-            // is something else. They intent to change this behavior in the future but they don't
-            // want to break the compatibility.
-            //
-            // See issues:
-            //  - https://github.com/rust-lang/cargo/issues/8364
-            //  - https://github.com/rust-lang/cargo/issues/8468
-            let repo = git2::Repository::init(&temp.path())?;
-            let mut origin = repo.remote_anonymous(remote.url().as_str())?;
+    let mut builder = RepoBuilder::new();
+    if let GitReference::Branch(branch_name) = &args.branch {
+        builder.branch(branch_name.as_str());
+    }
+
+    let mut fo = FetchOptions::new();
+    match args.kind {
+        RepoKind::LocalFolder => {}
+        RepoKind::RemoteHttp | RepoKind::RemoteHttps => {
             let mut proxy = ProxyOptions::new();
             proxy.auto();
-            origin.connect_auth(git2::Direction::Fetch, None, Some(proxy))?;
-            let default_branch = origin.default_branch()?;
-            let branch_name = default_branch
-                .as_str()
-                .unwrap_or("refs/heads/master")
-                .replace("refs/heads/", "");
-            (
-                remote.checkout(
-                    &temp.path(),
-                    None,
-                    &GitReference::Branch(branch_name.clone()),
-                    None,
-                    &config,
-                )?,
-                branch_name,
-            )
+            fo.proxy_options(proxy);
         }
-        _ => unreachable!(),
-    };
+        RepoKind::RemoteSsh => {
+            let callbacks = git_ssh_credentials_callback(args.identity)?;
+            fo.remote_callbacks(callbacks);
+        }
+        RepoKind::Invalid => {
+            unreachable!()
+        }
+    }
+    builder.fetch_options(fo);
+
+    let repo = builder.clone(args.remote.as_ref(), project_dir)?;
+    let branch = get_branch_name_repo(&repo)?;
+    init_all_submodules(&repo)?;
+    remove_history(project_dir)?;
 
-    // This clones the remote and handles all the submodules
-    db.copy_to(rev, project_dir, &config)?;
-    Ok(branch_name)
+    Ok(branch)
 }
 
-pub(crate) fn remove_history(project_dir: &Path) -> Result<()> {
-    remove_dir_all(project_dir.join(".git")).context("Error cleaning up cloned template")?;
+fn remove_history(project_dir: &Path) -> Result<()> {
+    let git_dir = project_dir.join(".git");
+    if git_dir.exists() {
+        remove_dir_all(git_dir).context("Error cleaning up cloned template")?;
+    }
     Ok(())
 }
 
-pub fn init(project_dir: &Path, branch: &str) -> Result<GitRepository> {
-    GitRepository::discover(project_dir).or_else(|_| {
+pub fn init(project_dir: &Path, branch: &str) -> Result<Repository> {
+    Repository::discover(project_dir).or_else(|_| {
         let mut opts = RepositoryInitOptions::new();
         opts.bare(false);
         opts.initial_head(branch);
-        GitRepository::init_opts(project_dir, &opts).context("Couldn't init new repository")
+        Repository::init_opts(project_dir, &opts).context("Couldn't init new repository")
     })
 }
 
+/// determines what kind of repository we got
+fn determine_repo_kind(remote_url: &str) -> RepoKind {
+    if remote_url.starts_with("git@") {
+        RepoKind::RemoteSsh
+    } else if remote_url.starts_with("http://") {
+        RepoKind::RemoteHttp
+    } else if remote_url.starts_with("https://") {
+        RepoKind::RemoteHttps
+    } else if Path::new(remote_url).exists() {
+        RepoKind::LocalFolder
+    } else {
+        RepoKind::Invalid
+    }
+}
+
 #[cfg(test)]
 mod tests {
     use super::*;
+    use std::env::current_dir;
+    use url::Url;
 
     const REPO_URL: &str = "https://github.com/cargo-generate/cargo-generate.git";
+    const REPO_URL_SSH: &str = "git@github.com:cargo-generate/cargo-generate.git";
 
     #[test]
-    #[should_panic(expected = "invalid port number")]
-    fn should_fail_for_ssh_remote_urls() {
-        GitConfig::new(
-            REPO_URL
-                .replace("https://github.com/", "ssh://git@github.com:")
-                .as_str(),
-            None,
-        )
-        .unwrap();
+    fn should_determine_repo_kind() {
+        for (u, k) in &[
+            (REPO_URL, RepoKind::RemoteHttps),
+            (
+                "http://github.com/cargo-generate/cargo-generate.git",
+                RepoKind::RemoteHttp,
+            ),
+            (REPO_URL_SSH, RepoKind::RemoteSsh),
+            ("./", RepoKind::LocalFolder),
+            ("ftp://foobar.bak", RepoKind::Invalid),
+        ] {
+            let kind = determine_repo_kind(u);
+            assert_eq!(&kind, k, "{} is not a {:?}", u, k);
+        }
+    }
+
+    #[test]
+    fn should_not_fail_for_ssh_remote_urls() {
+        let config = GitConfig::new(REPO_URL_SSH.into(), None, None).unwrap();
+        assert_eq!(config.kind, RepoKind::RemoteSsh);
     }
 
     #[test]
-    #[should_panic(expected = "aslkdgjlaskjdglskj\" doesn't exist")]
+    #[should_panic(expected = "Invalid git remote 'aslkdgjlaskjdglskj'")]
     fn should_fail_for_non_existing_local_path() {
-        GitConfig::new("aslkdgjlaskjdglskj", None).unwrap();
+        GitConfig::new("aslkdgjlaskjdglskj".into(), None, None).unwrap();
     }
 
     #[test]
     fn should_support_a_local_relative_path() {
-        let remote: String = GitConfig::new("src", None).unwrap().remote.into();
+        let remote: String = GitConfig::new("src".into(), None, None)
+            .unwrap()
+            .remote
+            .into();
+        #[cfg(unix)]
         assert!(
             remote.ends_with("/src"),
             "remote {} ends with /src",
             &remote
         );
+        #[cfg(windows)]
+        assert!(
+            remote.ends_with("\\src"),
+            "remote {} ends with \\src",
+            &remote
+        );
 
         #[cfg(unix)]
+        assert!(remote.starts_with('/'), "remote {} starts with /", &remote);
+        #[cfg(windows)]
         assert!(
-            remote.starts_with("file:///"),
-            "remote {} starts with file:///",
+            remote.starts_with("\\\\?\\"),
+            "remote {} starts with \\\\?\\",
             &remote
         );
     }
 
     #[test]
-    #[cfg(unix)]
     fn should_support_a_local_absolute_path() {
         // Absolute path.
         // If this fails because you cloned this repository into a non-UTF-8 directory... all
         // I can say is you probably had it comin'.
-        let remote: String = GitConfig::new(current_dir().unwrap().to_str().unwrap(), None)
-            .unwrap()
-            .remote
-            .into();
+        let remote: String = GitConfig::new(
+            current_dir().unwrap().display().to_string().into(),
+            None,
+            None,
+        )
+        .unwrap()
+        .remote
+        .into();
+        #[cfg(unix)]
+        assert!(remote.starts_with('/'), "remote {} starts with /", &remote);
+        #[cfg(windows)]
         assert!(
-            remote.starts_with("file:///"),
-            "remote {} starts with file:///",
-            remote
+            remote.starts_with("\\\\?\\"),
+            "remote {} starts with \\\\?\\ then the drive letter",
+            &remote
         );
     }
 
     #[test]
     fn should_test_happy_path() {
         // Remote HTTPS URL.
-        let cfg = GitConfig::new(REPO_URL, Some("main".to_owned())).unwrap();
+        let cfg = GitConfig::new(REPO_URL.into(), Some("main".to_owned()), None).unwrap();
 
-        assert_eq!(cfg.remote, Url::parse(REPO_URL).unwrap());
+        assert_eq!(cfg.remote.as_ref(), Url::parse(REPO_URL).unwrap().as_str());
         assert_eq!(cfg.branch, GitReference::Branch("main".to_owned()));
     }
 
     #[test]
     fn should_support_abbreviated_repository_short_urls_like() {
         assert_eq!(
-            GitConfig::new_abbr("cargo-generate/cargo-generate", None)
+            GitConfig::new_abbr("cargo-generate/cargo-generate".into(), None, None)
                 .unwrap()
-                .remote,
-            Url::parse(REPO_URL).unwrap()
+                .remote
+                .as_ref(),
+            Url::parse(REPO_URL).unwrap().as_str()
         );
     }
 }
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -563,7 +563,7 @@ version = "0.1.0"
             .path()
             .join("dangerous.todelete.cargogeneratetests")
     )
-    .expect("should exist")
+    .unwrap()
     .is_file());
 }
 
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -1055,7 +1055,12 @@ _This README was generated with [cargo-readme](https://github.com/livioribeiro/c
         .success()
         .stdout(predicates::str::contains("Done!").from_utf8());
 
-    assert!(dir.read("foobar-project/README.tpl").contains(raw_body));
+    let template = dir.read("foobar-project/README.tpl");
+    assert!(template.contains("{{badges}}"));
+    assert!(template.contains("{{crate}}"));
+    assert!(template.contains("{{project-name}}"));
+    assert!(template.contains("{{readme}}"));
+    assert!(template.contains("{{license}}"));
 }
 
 #[test]
diff --git a/tests/integration/basics.rs b/tests/integration/basics.rs
--- a/tests/integration/basics.rs
+++ b/tests/integration/basics.rs
@@ -1158,3 +1163,76 @@ version = "0.1.0"
     let cargo_toml = dir.read("foobar-project/Cargo.toml");
     assert!(cargo_toml.contains("this is a bin"));
 }
+
+#[cfg(test)]
+#[cfg(unix)]
+mod ssh_remote {
+    use super::*;
+
+    #[test]
+    fn it_should_support_a_public_repo() {
+        let dir = tmp_dir().build();
+
+        binary()
+            .arg("generate")
+            .arg("--git")
+            .arg("git@github.com:ashleygwilliams/wasm-pack-template.git")
+            .arg("--name")
+            .arg("foobar-project")
+            .current_dir(&dir.path())
+            .assert()
+            .success()
+            .stdout(predicates::str::contains("Done!").from_utf8());
+
+        let cargo_toml = dir.read("foobar-project/Cargo.toml");
+        assert!(cargo_toml.contains("foobar-project"));
+    }
+
+    #[test]
+    fn it_should_support_a_private_repo() {
+        let dir = tmp_dir().build();
+
+        binary()
+            .arg("generate")
+            .arg("--git")
+            .arg("git@github.com:cargo-generate/wasm-pack-template.git")
+            .arg("--name")
+            .arg("foobar-project")
+            .current_dir(&dir.path())
+            .assert()
+            .success()
+            .stdout(predicates::str::contains("Done!").from_utf8());
+
+        let cargo_toml = dir.read("foobar-project/Cargo.toml");
+        assert!(cargo_toml.contains("foobar-project"));
+    }
+
+    #[test]
+    #[ignore]
+    // for now only locally working
+    fn it_should_support_a_custom_ssh_key() {
+        let dir = tmp_dir().build();
+
+        binary()
+            .arg("generate")
+            .arg("-i")
+            .arg("~/workspaces/rust/cargo-generate-org/.env/id_rsa_ci")
+            .arg("--git")
+            .arg("git@github.com:cargo-generate/wasm-pack-template.git")
+            .arg("--name")
+            .arg("foobar-project")
+            .current_dir(&dir.path())
+            .assert()
+            .success()
+            .stdout(
+                predicates::str::contains("Using private key:")
+                    .and(predicates::str::contains(
+                        "cargo-generate-org/.env/id_rsa_ci",
+                    ))
+                    .from_utf8(),
+            );
+
+        let cargo_toml = dir.read("foobar-project/Cargo.toml");
+        assert!(cargo_toml.contains("foobar-project"));
+    }
+}
diff --git a/tests/integration/library.rs b/tests/integration/library.rs
--- a/tests/integration/library.rs
+++ b/tests/integration/library.rs
@@ -31,6 +31,7 @@ version = "0.1.0"
         favorite: None,
         bin: true,
         lib: false,
+        ssh_identity: None,
     };
     // need to cd to the dir as we aren't running in the cargo shell.
     assert!(std::env::set_current_dir(&dir.root).is_ok());

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 4daf0d95d5238250edbcb4a596b1c85ae25faec7
git clean -fd
