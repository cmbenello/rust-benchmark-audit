#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/lib/command_test.rs b/src/lib/command_test.rs
--- a/src/lib/command_test.rs
+++ b/src/lib/command_test.rs
@@ -110,7 +110,7 @@ fn run_command_for_toolchain() {
         let mut task = Task::new();
         task.command = Some("echo".to_string());
         task.args = Some(vec!["test".to_string()]);
-        task.toolchain = Some(toolchain.to_string());
+        task.toolchain = Some(toolchain.into());
 
         let step = Step {
             name: "test".to_string(),
diff --git a/src/lib/installer/cargo_plugin_installer.rs b/src/lib/installer/cargo_plugin_installer.rs
--- a/src/lib/installer/cargo_plugin_installer.rs
+++ b/src/lib/installer/cargo_plugin_installer.rs
@@ -10,10 +10,11 @@ mod cargo_plugin_installer_test;
 use crate::command;
 use crate::installer::crate_version_check;
 use crate::toolchain::wrap_command;
+use crate::types::ToolchainSpecifier;
 use envmnt;
 use std::process::Command;
 
-fn is_crate_installed(toolchain: &Option<String>, crate_name: &str) -> bool {
+fn is_crate_installed(toolchain: &Option<ToolchainSpecifier>, crate_name: &str) -> bool {
     debug!("Getting list of installed cargo commands.");
 
     let mut command_struct = match toolchain {
diff --git a/src/lib/installer/rustup_component_installer.rs b/src/lib/installer/rustup_component_installer.rs
--- a/src/lib/installer/rustup_component_installer.rs
+++ b/src/lib/installer/rustup_component_installer.rs
@@ -9,10 +9,14 @@ mod rustup_component_installer_test;
 
 use crate::command;
 use crate::toolchain::wrap_command;
-use crate::types::InstallRustupComponentInfo;
+use crate::types::{InstallRustupComponentInfo, ToolchainSpecifier};
 use std::process::Command;
 
-pub(crate) fn is_installed(toolchain: &Option<String>, binary: &str, test_args: &[String]) -> bool {
+pub(crate) fn is_installed(
+    toolchain: &Option<ToolchainSpecifier>,
+    binary: &str,
+    test_args: &[String],
+) -> bool {
     let mut command_struct = match toolchain {
         Some(ref toolchain_string) => {
             let command_spec = wrap_command(toolchain_string, binary, &None);
diff --git a/src/lib/installer/rustup_component_installer.rs b/src/lib/installer/rustup_component_installer.rs
--- a/src/lib/installer/rustup_component_installer.rs
+++ b/src/lib/installer/rustup_component_installer.rs
@@ -52,7 +56,7 @@ pub(crate) fn is_installed(toolchain: &Option<String>, binary: &str, test_args:
 }
 
 pub(crate) fn invoke_rustup_install(
-    toolchain: &Option<String>,
+    toolchain: &Option<ToolchainSpecifier>,
     info: &InstallRustupComponentInfo,
 ) -> bool {
     let mut command_spec = Command::new("rustup");
diff --git a/src/lib/logger.rs b/src/lib/logger.rs
--- a/src/lib/logger.rs
+++ b/src/lib/logger.rs
@@ -147,7 +147,7 @@ pub(crate) fn init(options: &LoggerOptions) {
 
             if cfg!(test) {
                 if record_level == LevelFilter::Error {
-                    panic!("test error flow");
+                    panic!("test error flow: {}", message);
                 }
             }
 
diff --git a/src/lib/test/mod.rs b/src/lib/test/mod.rs
--- a/src/lib/test/mod.rs
+++ b/src/lib/test/mod.rs
@@ -1,6 +1,6 @@
 use crate::logger;
 use crate::logger::LoggerOptions;
-use crate::types::{Config, ConfigSection, CrateInfo, EnvInfo, FlowInfo};
+use crate::types::{Config, ConfigSection, CrateInfo, EnvInfo, FlowInfo, ToolchainSpecifier};
 use ci_info;
 use ci_info::types::CiInfo;
 use fsio;
diff --git a/src/lib/test/mod.rs b/src/lib/test/mod.rs
--- a/src/lib/test/mod.rs
+++ b/src/lib/test/mod.rs
@@ -99,7 +99,7 @@ pub(crate) fn is_not_rust_stable() -> bool {
     }
 }
 
-pub(crate) fn get_toolchain() -> String {
+pub(crate) fn get_toolchain() -> ToolchainSpecifier {
     on_test_startup();
 
     let rustinfo = rust_info::get();
diff --git a/src/lib/test/mod.rs b/src/lib/test/mod.rs
--- a/src/lib/test/mod.rs
+++ b/src/lib/test/mod.rs
@@ -110,7 +110,7 @@ pub(crate) fn get_toolchain() -> String {
         RustChannel::Nightly => "nightly",
     };
 
-    toolchain.to_string()
+    toolchain.into()
 }
 
 pub(crate) fn create_empty_flow_info() -> FlowInfo {
diff --git a/src/lib/toolchain.rs b/src/lib/toolchain.rs
--- a/src/lib/toolchain.rs
+++ b/src/lib/toolchain.rs
@@ -7,38 +7,22 @@
 #[path = "toolchain_test.rs"]
 mod toolchain_test;
 
-use crate::types::CommandSpec;
-use std::process::{Command, Stdio};
-
-#[cfg(test)]
-fn should_validate_installed_toolchain() -> bool {
-    use crate::test;
+use cargo_metadata::Version;
+use semver::Prerelease;
 
-    return test::is_not_rust_stable();
-}
-
-#[cfg(not(test))]
-fn should_validate_installed_toolchain() -> bool {
-    return true;
-}
+use crate::types::{CommandSpec, ToolchainSpecifier};
+use std::process::{Command, Stdio};
 
 pub(crate) fn wrap_command(
-    toolchain: &str,
+    toolchain: &ToolchainSpecifier,
     command: &str,
     args: &Option<Vec<String>>,
 ) -> CommandSpec {
-    let validate = should_validate_installed_toolchain();
-
-    if validate && !has_toolchain(toolchain) {
-        error!(
-            "Missing toolchain {}! Please install it using rustup.",
-            &toolchain
-        );
-    }
+    check_toolchain(toolchain);
 
     let mut rustup_args = vec![
         "run".to_string(),
-        toolchain.to_string(),
+        toolchain.channel().to_string(),
         command.to_string(),
     ];
 
diff --git a/src/lib/toolchain_test.rs b/src/lib/toolchain_test.rs
--- a/src/lib/toolchain_test.rs
+++ b/src/lib/toolchain_test.rs
@@ -1,50 +1,66 @@
 use super::*;
-use crate::test;
+use crate::types::ToolchainBoundedSpecifier;
 use envmnt;
 
+fn get_test_env_toolchain() -> ToolchainSpecifier {
+    let channel = envmnt::get_or_panic("CARGO_MAKE_RUST_CHANNEL");
+    let version = envmnt::get_or_panic("CARGO_MAKE_RUST_VERSION");
+
+    ToolchainSpecifier::Bounded(ToolchainBoundedSpecifier {
+        channel,
+        min_version: version,
+    })
+}
+
 #[test]
 #[should_panic]
 fn wrap_command_invalid_toolchain() {
-    if test::is_not_rust_stable() {
-        wrap_command("invalid-chain", "true", &None);
-    } else {
-        panic!("test");
-    }
+    wrap_command(&"invalid-chain".into(), "true", &None);
+}
+
+#[test]
+#[should_panic]
+fn wrap_command_unreachable_version() {
+    let toolchain = ToolchainSpecifier::Bounded(ToolchainBoundedSpecifier {
+        channel: envmnt::get_or_panic("CARGO_MAKE_RUST_CHANNEL"),
+        min_version: "9999.9.9".to_string(), // If we ever reach this version, add another 9
+    });
+    wrap_command(&toolchain, "true", &None);
 }
 
 #[test]
 fn wrap_command_none_args() {
-    let channel = envmnt::get_or_panic("CARGO_MAKE_RUST_CHANNEL");
-    let output = wrap_command(&channel, "true", &None);
+    let toolchain = get_test_env_toolchain();
+    let output = wrap_command(&toolchain, "true", &None);
 
     assert_eq!(output.command, "rustup".to_string());
 
     let args = output.args.unwrap();
     assert_eq!(args.len(), 3);
     assert_eq!(args[0], "run".to_string());
-    assert_eq!(args[1], channel);
+    assert_eq!(args[1], toolchain.channel());
     assert_eq!(args[2], "true".to_string());
 }
 
 #[test]
 fn wrap_command_empty_args() {
-    let channel = envmnt::get_or_panic("CARGO_MAKE_RUST_CHANNEL");
-    let output = wrap_command(&channel, "true", &Some(vec![]));
+    let toolchain = get_test_env_toolchain();
+    let output = wrap_command(&toolchain, "true", &Some(vec![]));
 
     assert_eq!(output.command, "rustup".to_string());
 
     let args = output.args.unwrap();
     assert_eq!(args.len(), 3);
     assert_eq!(args[0], "run".to_string());
-    assert_eq!(args[1], channel);
+    assert_eq!(args[1], toolchain.channel());
     assert_eq!(args[2], "true".to_string());
 }
 
 #[test]
 fn wrap_command_with_args() {
-    let channel = envmnt::get_or_panic("CARGO_MAKE_RUST_CHANNEL");
+    let toolchain = get_test_env_toolchain();
     let output = wrap_command(
-        &channel,
+        &toolchain,
         "true",
         &Some(vec!["echo".to_string(), "test".to_string()]),
     );
diff --git a/src/lib/toolchain_test.rs b/src/lib/toolchain_test.rs
--- a/src/lib/toolchain_test.rs
+++ b/src/lib/toolchain_test.rs
@@ -54,7 +70,7 @@ fn wrap_command_with_args() {
     let args = output.args.unwrap();
     assert_eq!(args.len(), 5);
     assert_eq!(args[0], "run".to_string());
-    assert_eq!(args[1], channel);
+    assert_eq!(args[1], toolchain.channel());
     assert_eq!(args[2], "true".to_string());
     assert_eq!(args[3], "echo".to_string());
     assert_eq!(args[4], "test".to_string());
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1238,6 +1238,47 @@ fn env_value_deserialize_unset() {
     }
 }
 
+#[test]
+fn toolchain_specifier_deserialize_string() {
+    #[derive(Deserialize)]
+    struct Value {
+        toolchain: ToolchainSpecifier,
+    }
+
+    let v: Value = toml::from_str(
+        r#"
+        toolchain = "stable"
+        "#,
+    )
+    .unwrap();
+    assert_eq!(
+        v.toolchain,
+        ToolchainSpecifier::Simple("stable".to_string())
+    );
+}
+
+#[test]
+fn toolchain_specifier_deserialize_min_version() {
+    #[derive(Deserialize)]
+    struct Value {
+        toolchain: ToolchainSpecifier,
+    }
+
+    let v: Value = toml::from_str(
+        r#"
+        toolchain = { channel = "beta", min_version = "1.56" }
+        "#,
+    )
+    .unwrap();
+    assert_eq!(
+        v.toolchain,
+        ToolchainSpecifier::Bounded(ToolchainBoundedSpecifier {
+            channel: "beta".to_string(),
+            min_version: "1.56".to_string(),
+        })
+    );
+}
+
 #[test]
 fn task_new() {
     let task = Task::new();
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1532,7 +1573,7 @@ fn task_extend_extended_have_all_fields() {
         script_extension: Some("ext2".to_string()),
         run_task: Some(RunTaskInfo::Name("task2".to_string())),
         dependencies: Some(vec!["A".into()]),
-        toolchain: Some("toolchain".to_string()),
+        toolchain: Some("toolchain".into()),
         linux: Some(PlatformOverrideTask {
             clear: Some(true),
             install_crate: Some(InstallCrate::Value("my crate2".to_string())),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1576,7 +1617,7 @@ fn task_extend_extended_have_all_fields() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
         windows: Some(PlatformOverrideTask {
             clear: Some(false),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1621,7 +1662,7 @@ fn task_extend_extended_have_all_fields() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
         mac: Some(PlatformOverrideTask {
             clear: None,
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1666,7 +1707,7 @@ fn task_extend_extended_have_all_fields() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
     };
 
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1744,7 +1785,7 @@ fn task_extend_extended_have_all_fields() {
     };
     assert_eq!(run_task_name, "task2".to_string());
     assert_eq!(base.dependencies.unwrap().len(), 1);
-    assert_eq!(base.toolchain.unwrap(), "toolchain");
+    assert_eq!(base.toolchain.unwrap(), "toolchain".into());
     assert!(base.linux.unwrap().clear.unwrap());
     assert!(!base.windows.unwrap().clear.unwrap());
     assert!(base.mac.unwrap().clear.is_none());
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1807,7 +1848,7 @@ fn task_extend_clear_with_no_data() {
         script_extension: Some("ext2".to_string()),
         run_task: Some(RunTaskInfo::Name("task2".to_string())),
         dependencies: Some(vec!["A".into()]),
-        toolchain: Some("toolchain".to_string()),
+        toolchain: Some("toolchain".into()),
         linux: Some(PlatformOverrideTask {
             clear: Some(true),
             install_crate: Some(InstallCrate::Value("my crate2".to_string())),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1851,7 +1892,7 @@ fn task_extend_clear_with_no_data() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
         windows: Some(PlatformOverrideTask {
             clear: Some(false),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1896,7 +1937,7 @@ fn task_extend_clear_with_no_data() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
         mac: Some(PlatformOverrideTask {
             clear: None,
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -1941,7 +1982,7 @@ fn task_extend_clear_with_no_data() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
     };
 
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2042,7 +2083,7 @@ fn task_extend_clear_with_all_data() {
         script_extension: Some("ext2".to_string()),
         run_task: Some(RunTaskInfo::Name("task2".to_string())),
         dependencies: Some(vec!["A".into()]),
-        toolchain: Some("toolchain".to_string()),
+        toolchain: Some("toolchain".into()),
         linux: Some(PlatformOverrideTask {
             clear: Some(true),
             install_crate: Some(InstallCrate::Value("my crate2".to_string())),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2086,7 +2127,7 @@ fn task_extend_clear_with_all_data() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
         windows: Some(PlatformOverrideTask {
             clear: Some(false),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2131,7 +2172,7 @@ fn task_extend_clear_with_all_data() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
         mac: Some(PlatformOverrideTask {
             clear: None,
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2176,7 +2217,7 @@ fn task_extend_clear_with_all_data() {
             script_extension: Some("ext3".to_string()),
             run_task: Some(RunTaskInfo::Name("task3".to_string())),
             dependencies: Some(vec!["A".into()]),
-            toolchain: Some("toolchain".to_string()),
+            toolchain: Some("toolchain".into()),
         }),
     };
 
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2289,7 +2330,7 @@ fn task_get_normalized_task_undefined() {
         script_extension: Some("ext1".to_string()),
         run_task: Some(RunTaskInfo::Name("task1".to_string())),
         dependencies: Some(vec!["1".into()]),
-        toolchain: Some("toolchain2".to_string()),
+        toolchain: Some("toolchain2".into()),
         description: Some("description".to_string()),
         category: Some("category".to_string()),
         workspace: Some(false),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2369,7 +2410,7 @@ fn task_get_normalized_task_undefined() {
     };
     assert_eq!(run_task_name, "task1".to_string());
     assert_eq!(normalized_task.dependencies.unwrap().len(), 1);
-    assert_eq!(normalized_task.toolchain.unwrap(), "toolchain2");
+    assert_eq!(normalized_task.toolchain.unwrap(), "toolchain2".into());
 }
 
 #[test]
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2428,7 +2469,7 @@ fn task_get_normalized_task_with_override_no_clear() {
         script_extension: Some("ext1".to_string()),
         run_task: Some(RunTaskInfo::Name("task1".to_string())),
         dependencies: Some(vec!["1".into()]),
-        toolchain: Some("toolchain1".to_string()),
+        toolchain: Some("toolchain1".into()),
         linux: Some(PlatformOverrideTask {
             clear: None,
             install_crate: Some(InstallCrate::Value("linux_crate".to_string())),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2481,7 +2522,7 @@ fn task_get_normalized_task_with_override_no_clear() {
             script_extension: Some("ext2".to_string()),
             run_task: Some(RunTaskInfo::Name("task2".to_string())),
             dependencies: Some(vec!["1".into(), "2".into()]),
-            toolchain: Some("toolchain2".to_string()),
+            toolchain: Some("toolchain2".into()),
         }),
         windows: None,
         mac: None,
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2563,7 +2604,7 @@ fn task_get_normalized_task_with_override_no_clear() {
     };
     assert_eq!(run_task_name, "task2".to_string());
     assert_eq!(normalized_task.dependencies.unwrap().len(), 2);
-    assert_eq!(normalized_task.toolchain.unwrap(), "toolchain2");
+    assert_eq!(normalized_task.toolchain.unwrap(), "toolchain2".into());
 
     let condition = normalized_task.condition.unwrap();
     assert_eq!(condition.platforms.unwrap().len(), 2);
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2626,7 +2667,7 @@ fn task_get_normalized_task_with_override_clear_false() {
         script_extension: Some("ext1".to_string()),
         run_task: Some(RunTaskInfo::Name("task1".to_string())),
         dependencies: Some(vec!["1".into()]),
-        toolchain: Some("toolchain1".to_string()),
+        toolchain: Some("toolchain1".into()),
         linux: Some(PlatformOverrideTask {
             clear: Some(false),
             install_crate: Some(InstallCrate::Value("linux_crate".to_string())),
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2683,7 +2724,7 @@ fn task_get_normalized_task_with_override_clear_false() {
             script_extension: Some("ext2".to_string()),
             run_task: Some(RunTaskInfo::Name("task2".to_string())),
             dependencies: Some(vec!["1".into(), "2".into()]),
-            toolchain: Some("toolchain2".to_string()),
+            toolchain: Some("toolchain2".into()),
         }),
         windows: None,
         mac: None,
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2765,7 +2806,7 @@ fn task_get_normalized_task_with_override_clear_false() {
     };
     assert_eq!(run_task_name, "task2".to_string());
     assert_eq!(normalized_task.dependencies.unwrap().len(), 2);
-    assert_eq!(normalized_task.toolchain.unwrap(), "toolchain2");
+    assert_eq!(normalized_task.toolchain.unwrap(), "toolchain2".into());
 
     let condition = normalized_task.condition.unwrap();
     assert_eq!(condition.platforms.unwrap().len(), 1);
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2822,7 +2863,7 @@ fn task_get_normalized_task_with_override_clear_false_partial_override() {
         script_extension: Some("ext1".to_string()),
         run_task: Some(RunTaskInfo::Name("task1".to_string())),
         dependencies: Some(vec!["1".into()]),
-        toolchain: Some("toolchain1".to_string()),
+        toolchain: Some("toolchain1".into()),
         description: None,
         category: None,
         workspace: None,
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2929,7 +2970,7 @@ fn task_get_normalized_task_with_override_clear_false_partial_override() {
     };
     assert_eq!(run_task_name, "task1".to_string());
     assert_eq!(normalized_task.dependencies.unwrap().len(), 1);
-    assert_eq!(normalized_task.toolchain.unwrap(), "toolchain1");
+    assert_eq!(normalized_task.toolchain.unwrap(), "toolchain1".into());
 }
 
 #[test]
diff --git a/src/lib/types_test.rs b/src/lib/types_test.rs
--- a/src/lib/types_test.rs
+++ b/src/lib/types_test.rs
@@ -2982,7 +3023,7 @@ fn task_get_normalized_task_with_override_clear_true() {
         script_extension: Some("ext1".to_string()),
         run_task: Some(RunTaskInfo::Name("task1".to_string())),
         dependencies: Some(vec!["1".into()]),
-        toolchain: Some("toolchain1".to_string()),
+        toolchain: Some("toolchain1".into()),
         description: Some("description".to_string()),
         category: Some("category".to_string()),
         workspace: Some(false),

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard ee4e8b40319532079750d26c7d2415c0c6fbc306
git clean -fd
