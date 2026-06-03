#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cli/src/lib.rs b/crates/cli/src/lib.rs
--- a/crates/cli/src/lib.rs
+++ b/crates/cli/src/lib.rs
@@ -88,7 +88,6 @@ pub fn main_with_args(args: impl Iterator<Item = String>) -> Result<()> {
   }
 
   let app = App::try_parse_from(args)?;
-  // TODO: add test for app parse
   match app.command {
     Commands::Run(arg) => run_with_pattern(arg),
     Commands::Scan(arg) => run_with_config(arg),
diff --git a/crates/cli/src/lib.rs b/crates/cli/src/lib.rs
--- a/crates/cli/src/lib.rs
+++ b/crates/cli/src/lib.rs
@@ -242,6 +241,17 @@ mod test_cli {
     ok("test --update-all");
     error("test --update-all --skip-snapshot-tests");
   }
+  #[test]
+  fn test_new() {
+    ok("new");
+    ok("new project");
+    ok("new -c sgconfig.yml rule");
+    ok("new rule -y");
+    ok("new test -y");
+    ok("new util -y");
+    ok("new rule -c sgconfig.yml");
+    error("new --base-dir");
+  }
 
   #[test]
   fn test_shell() {
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -4,7 +4,7 @@ mod reporter;
 mod snapshot;
 mod test_case;
 
-use crate::config::{register_custom_language, ProjectConfig};
+use crate::config::ProjectConfig;
 use crate::lang::SgLang;
 use crate::utils::ErrorContext;
 use anyhow::{anyhow, Result};
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -56,9 +56,12 @@ where
   })
 }
 
-fn run_test_rule_impl<R: Reporter + Send>(arg: TestArg, reporter: R) -> Result<()> {
-  let project_config = ProjectConfig::by_config_path_must(arg.config.clone())?;
-  let collections = &project_config.find_rules(Default::default())?.0;
+fn run_test_rule_impl<R: Reporter + Send>(
+  arg: TestArg,
+  reporter: R,
+  project: ProjectConfig,
+) -> Result<()> {
+  let collections = &project.find_rules(Default::default())?.0;
   let TestHarness {
     test_cases,
     snapshots,
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -67,7 +70,7 @@ fn run_test_rule_impl<R: Reporter + Send>(arg: TestArg, reporter: R) -> Result<(
     let snapshot_dirname = arg.snapshot_dir.as_deref();
     TestHarness::from_dir(&test_dirname, snapshot_dirname, arg.filter.as_ref())?
   } else {
-    TestHarness::from_config(arg.config, arg.filter.as_ref())?
+    TestHarness::from_config(project, arg.filter.as_ref())?
   };
   let snapshots = (!arg.skip_snapshot_tests).then_some(snapshots);
   let reporter = &Arc::new(Mutex::new(reporter));
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -184,20 +187,20 @@ pub struct TestArg {
 }
 
 pub fn run_test_rule(arg: TestArg) -> Result<()> {
-  let project_config = ProjectConfig::by_config_path(arg.config.clone())?;
-  register_custom_language(project_config)?;
+  let project = ProjectConfig::setup(arg.config.clone())?
+    .ok_or_else(|| anyhow!(ErrorContext::ProjectNotExist))?;
   if arg.interactive {
     let reporter = InteractiveReporter {
       output: std::io::stdout(),
       should_accept_all: false,
     };
-    run_test_rule_impl(arg, reporter)
+    run_test_rule_impl(arg, reporter, project)
   } else {
     let reporter = DefaultReporter {
       output: std::io::stdout(),
       update_all: arg.update_all,
     };
-    run_test_rule_impl(arg, reporter)
+    run_test_rule_impl(arg, reporter, project)
   }
 }
 
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -302,13 +305,8 @@ rule:
     assert!(ret.is_none());
   }
 
-  use codespan_reporting::term::termcolor::Buffer;
   #[test]
   fn test_run_verify_error() {
-    let reporter = DefaultReporter {
-      output: Buffer::no_color(),
-      update_all: false,
-    };
     let arg = TestArg {
       config: None,
       interactive: false,
diff --git a/crates/cli/src/verify.rs b/crates/cli/src/verify.rs
--- a/crates/cli/src/verify.rs
+++ b/crates/cli/src/verify.rs
@@ -318,7 +316,7 @@ rule:
       update_all: false,
       filter: None,
     };
-    assert!(run_test_rule_impl(arg, reporter).is_err());
+    assert!(run_test_rule(arg).is_err());
   }
   const TRANSFORM_TEXT: &str = "
 transform:
diff --git a/crates/cli/src/verify/find_file.rs b/crates/cli/src/verify/find_file.rs
--- a/crates/cli/src/verify/find_file.rs
+++ b/crates/cli/src/verify/find_file.rs
@@ -23,8 +23,8 @@ pub struct TestHarness {
 }
 
 impl TestHarness {
-  pub fn from_config(config_path: Option<PathBuf>, regex_filter: Option<&Regex>) -> Result<Self> {
-    find_tests(config_path, regex_filter)
+  pub fn from_config(project_config: ProjectConfig, regex_filter: Option<&Regex>) -> Result<Self> {
+    find_tests(project_config, regex_filter)
   }
 
   pub fn from_dir(
diff --git a/crates/cli/src/verify/find_file.rs b/crates/cli/src/verify/find_file.rs
--- a/crates/cli/src/verify/find_file.rs
+++ b/crates/cli/src/verify/find_file.rs
@@ -87,13 +87,13 @@ impl<'a> HarnessBuilder<'a> {
 }
 
 pub fn find_tests(
-  config_path: Option<PathBuf>,
+  project_config: ProjectConfig,
   regex_filter: Option<&Regex>,
 ) -> Result<TestHarness> {
   let ProjectConfig {
-    project_dir,
     sg_config,
-  } = ProjectConfig::by_config_path_must(config_path)?;
+    project_dir,
+  } = project_config;
   let test_configs = sg_config.test_configs.unwrap_or_default();
   let mut builder = HarnessBuilder {
     base_dir: project_dir,

EOF_114329324912
cd "crates/cli"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard cfe472f63c7011ef5635ca38fea4846c871a1177
git clean -fd
