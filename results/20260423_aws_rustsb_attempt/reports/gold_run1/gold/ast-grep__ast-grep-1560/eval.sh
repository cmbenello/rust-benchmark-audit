#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cli/src/config.rs b/crates/cli/src/config.rs
--- a/crates/cli/src/config.rs
+++ b/crates/cli/src/config.rs
@@ -102,7 +102,7 @@ impl ProjectConfig {
     };
     // sg_config will not use rule dirs and test configs anymore
     register_custom_language(&config.project_dir, sg_config)?;
-    Ok(Some(config))
+    Ok(Ok(config))
   }
 }
 
diff --git a/crates/cli/src/lib.rs b/crates/cli/src/lib.rs
--- a/crates/cli/src/lib.rs
+++ b/crates/cli/src/lib.rs
@@ -11,11 +11,13 @@ mod verify;
 
 use anyhow::Result;
 use clap::{Parser, Subcommand};
+use std::path::PathBuf;
 
 use completions::{run_shell_completion, CompletionsArg};
+use config::ProjectConfig;
 use lsp::{run_language_server, LspArg};
 use new::{run_create_new, NewArg};
-use run::{register_custom_language_if_is_run, run_with_pattern, RunArg};
+use run::{run_with_pattern, RunArg};
 use scan::{run_with_config, ScanArg};
 use utils::exit_with_error;
 use verify::{run_test_rule, TestArg};
diff --git a/crates/cli/src/lib.rs b/crates/cli/src/lib.rs
--- a/crates/cli/src/lib.rs
+++ b/crates/cli/src/lib.rs
@@ -79,10 +84,27 @@ fn try_default_run(args: &[String]) -> Result<Option<RunArg>> {
   }
 }
 
+/// finding project and setup custom language configuration
+fn setup_project_is_possible(args: &[String]) -> Result<ProjectConfig> {
+  let mut config = None;
+  for i in 0..args.len() {
+    if args[i] != "-c" && args[i] != "--config" {
+      continue;
+    }
+    if i + 1 >= args.len() || args[i + 1].starts_with('-') {
+      return Err(anyhow::anyhow!("missing config file after -c"));
+    }
+    let config_file = (&args[i + 1]).into();
+    config = Some(config_file);
+  }
+  ProjectConfig::setup(config)?
+}
+
 // this wrapper function is for testing
 pub fn main_with_args(args: impl Iterator<Item = String>) -> Result<()> {
   let args: Vec<_> = args.collect();
-  register_custom_language_if_is_run(&args)?;
+  let project = setup_project_is_possible(&args);
+  // register_custom_language_if_is_run(&args)?;
   if let Some(arg) = try_default_run(&args)? {
     return run_with_pattern(arg);
   }
diff --git a/crates/cli/src/lib.rs b/crates/cli/src/lib.rs
--- a/crates/cli/src/lib.rs
+++ b/crates/cli/src/lib.rs
@@ -90,10 +112,10 @@ pub fn main_with_args(args: impl Iterator<Item = String>) -> Result<()> {
   let app = App::try_parse_from(args)?;
   match app.command {
     Commands::Run(arg) => run_with_pattern(arg),
-    Commands::Scan(arg) => run_with_config(arg),
-    Commands::Test(arg) => run_test_rule(arg),
-    Commands::New(arg) => run_create_new(arg),
-    Commands::Lsp(arg) => run_language_server(arg),
+    Commands::Scan(arg) => run_with_config(arg, project),
+    Commands::Test(arg) => run_test_rule(arg, project),
+    Commands::New(arg) => run_create_new(arg, project),
+    Commands::Lsp(arg) => run_language_server(arg, project),
     Commands::Completions(arg) => run_shell_completion::<App>(arg),
     Commands::Docs => todo!("todo, generate rule docs based on current config"),
   }
diff --git a/crates/cli/src/lib.rs b/crates/cli/src/lib.rs
--- a/crates/cli/src/lib.rs
+++ b/crates/cli/src/lib.rs
@@ -149,7 +171,7 @@ mod test_cli {
   fn test_no_arg_run() {
     let ret = main_with_args(["sg".to_owned()].into_iter());
     let err = ret.unwrap_err();
-    assert!(err.to_string().contains("sg <COMMAND>"));
+    assert!(err.to_string().contains("sg [OPTIONS] <COMMAND>"));
   }
   #[test]
   fn test_default_subcommand() {
diff --git a/crates/cli/src/lib.rs b/crates/cli/src/lib.rs
--- a/crates/cli/src/lib.rs
+++ b/crates/cli/src/lib.rs
@@ -185,11 +207,11 @@ mod test_cli {
     ok("run -p test --globs '*.js' --globs '*.ts'");
     ok("run -p fubuki -j8");
     ok("run -p test --threads 12");
+    ok("run -p test -l rs -c config.yml"); // global config arg
     error("run test");
     error("run --debug-query test"); // missing lang
     error("run -r Test dir");
     error("run -p test -i --json dir"); // conflict
-    error("run -p test -l rs -c always"); // no color shortcut
     error("run -p test -U");
     error("run -p test --update-all");
     error("run -p test --strictness not");
diff --git a/crates/cli/src/lsp.rs b/crates/cli/src/lsp.rs
--- a/crates/cli/src/lsp.rs
+++ b/crates/cli/src/lsp.rs
@@ -37,12 +30,12 @@ async fn run_language_server_impl(arg: LspArg) -> Result<()> {
   Ok(())
 }
 
-pub fn run_language_server(arg: LspArg) -> Result<()> {
+pub fn run_language_server(arg: LspArg, project: Result<ProjectConfig>) -> Result<()> {
   tokio::runtime::Builder::new_multi_thread()
     .enable_all()
     .build()
     .context(EC::StartLanguageServer)?
-    .block_on(async { run_language_server_impl(arg).await })
+    .block_on(async { run_language_server_impl(arg, project).await })
 }
 
 #[cfg(test)]
diff --git a/crates/cli/src/lsp.rs b/crates/cli/src/lsp.rs
--- a/crates/cli/src/lsp.rs
+++ b/crates/cli/src/lsp.rs
@@ -52,7 +45,7 @@ mod test {
   #[test]
   #[ignore = "test lsp later"]
   fn test_lsp_start() {
-    let arg = LspArg { config: None };
-    assert!(run_language_server(arg).is_err())
+    let arg = LspArg {};
+    assert!(run_language_server(arg, Err(anyhow::anyhow!("error"))).is_err())
   }
 }
diff --git a/crates/cli/src/new.rs b/crates/cli/src/new.rs
--- a/crates/cli/src/new.rs
+++ b/crates/cli/src/new.rs
@@ -369,7 +364,6 @@ mod test {
       name: None,
       lang: None,
       yes: true,
-      config: None,
     };
     create_new_project(arg, tempdir)?;
     assert!(tempdir.join("sgconfig.yml").exists());
diff --git a/crates/cli/src/new.rs b/crates/cli/src/new.rs
--- a/crates/cli/src/new.rs
+++ b/crates/cli/src/new.rs
@@ -377,27 +371,27 @@ mod test {
   }
 
   fn create_rule(temp: &Path) -> Result<()> {
+    let project = ProjectConfig::setup(Some(temp.join("sgconfig.yml")))?;
     let arg = NewArg {
       entity: Some(Entity::Rule),
       name: Some("test-rule".into()),
       lang: Some(SupportLang::Rust.into()),
       yes: true,
-      config: Some(temp.join("sgconfig.yml")),
     };
-    run_create_new(arg)?;
+    run_create_new(arg, project)?;
     assert!(temp.join("rules/test-rule.yml").exists());
     Ok(())
   }
 
   fn create_util(temp: &Path) -> Result<()> {
+    let project = ProjectConfig::setup(Some(temp.join("sgconfig.yml")))?;
     let arg = NewArg {
       entity: Some(Entity::Util),
       name: Some("test-utils".into()),
       lang: Some(SupportLang::Rust.into()),
       yes: true,
-      config: Some(temp.join("sgconfig.yml")),
     };
-    run_create_new(arg)?;
+    run_create_new(arg, project)?;
     assert!(temp.join("utils/test-utils.yml").exists());
     Ok(())
   }
diff --git a/crates/cli/src/scan.rs b/crates/cli/src/scan.rs
--- a/crates/cli/src/scan.rs
+++ b/crates/cli/src/scan.rs
@@ -403,11 +402,9 @@ rule:
       .write_all("fn test() { Some(123) }".as_bytes())
       .unwrap();
     file.sync_all().unwrap();
-    let arg = ScanArg {
-      config: Some(dir.path().join("sgconfig.yml")),
-      ..default_scan_arg()
-    };
-    assert!(run_with_config(arg).is_ok());
+    let project_config = ProjectConfig::setup(Some(dir.path().join("sgconfig.yml"))).unwrap();
+    let arg = default_scan_arg();
+    assert!(run_with_config(arg, project_config).is_ok());
   }
 
   #[test]
diff --git a/crates/cli/src/scan.rs b/crates/cli/src/scan.rs
--- a/crates/cli/src/scan.rs
+++ b/crates/cli/src/scan.rs
@@ -417,7 +414,7 @@ rule:
       inline_rules: Some(inline_rules),
       ..default_scan_arg()
     };
-    assert!(run_with_config(arg).is_ok());
+    assert!(run_with_config(arg, Err(anyhow::anyhow!("not found"))).is_ok());
   }
 
   #[test]
diff --git a/crates/cli/src/scan.rs b/crates/cli/src/scan.rs
--- a/crates/cli/src/scan.rs
+++ b/crates/cli/src/scan.rs
@@ -428,7 +425,7 @@ rule:
       inline_rules: Some(inline_rules),
       ..default_scan_arg()
     };
-    assert!(run_with_config(arg).is_ok());
+    assert!(run_with_config(arg, Err(anyhow::anyhow!("not found"))).is_ok());
   }
 
   // baseline test for coverage
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -160,9 +160,6 @@ fn verify_test_case_simple<'a>(
 
 #[derive(Args)]
 pub struct TestArg {
-  /// Path to the root ast-grep config YAML
-  #[clap(short, long)]
-  config: Option<PathBuf>,
   /// the directories to search test YAML files
   #[clap(short, long)]
   test_dir: Option<PathBuf>,
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -186,9 +183,8 @@ pub struct TestArg {
   filter: Option<Regex>,
 }
 
-pub fn run_test_rule(arg: TestArg) -> Result<()> {
-  let project = ProjectConfig::setup(arg.config.clone())?
-    .ok_or_else(|| anyhow!(ErrorContext::ProjectNotExist))?;
+pub fn run_test_rule(arg: TestArg, project: Result<ProjectConfig>) -> Result<()> {
+  let project = project?;
   if arg.interactive {
     let reporter = InteractiveReporter {
       output: std::io::stdout(),
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -308,7 +304,6 @@ rule:
   #[test]
   fn test_run_verify_error() {
     let arg = TestArg {
-      config: None,
       interactive: false,
       skip_snapshot_tests: true,
       snapshot_dir: None,
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -316,7 +311,7 @@ rule:
       update_all: false,
       filter: None,
     };
-    assert!(run_test_rule(arg).is_err());
+    assert!(run_test_rule(arg, Err(anyhow!("error"))).is_err());
   }
   const TRANSFORM_TEXT: &str = "
 transform:

EOF_114329324912
cd "crates/cli"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard d9014f979fd7ee34cd9f449449a32f25296c34f4
git clean -fd
