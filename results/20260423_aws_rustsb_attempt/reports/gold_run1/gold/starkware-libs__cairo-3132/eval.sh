#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cairo-lang-plugins/src/lib.rs b/crates/cairo-lang-plugins/src/lib.rs
--- a/crates/cairo-lang-plugins/src/lib.rs
+++ b/crates/cairo-lang-plugins/src/lib.rs
@@ -13,6 +15,7 @@ mod test;
 /// Gets the list of default plugins to load into the Cairo compiler.
 pub fn get_default_plugins() -> Vec<Arc<dyn SemanticPlugin>> {
     vec![
+        Arc::new(ConstevalIntMacroPlugin::default()),
         Arc::new(DerivePlugin::default()),
         Arc::new(GenerateTraitPlugin::default()),
         Arc::new(PanicablePlugin::default()),
diff --git a/crates/cairo-lang-plugins/src/test.rs b/crates/cairo-lang-plugins/src/test.rs
--- a/crates/cairo-lang-plugins/src/test.rs
+++ b/crates/cairo-lang-plugins/src/test.rs
@@ -15,6 +15,7 @@ cairo_lang_test_utils::test_file_test!(
     expand_plugin,
     "src/test_data",
     {
+        consteval_int: "consteval_int",
         config: "config",
         derive: "derive",
         generate_trait: "generate_trait",
diff --git /dev/null b/crates/cairo-lang-plugins/src/test_data/consteval_int
new file mode 100644
--- /dev/null
+++ b/crates/cairo-lang-plugins/src/test_data/consteval_int
@@ -0,0 +1,107 @@
+//! > Test consteval_int! macro
+
+//! > test_runner_name
+test_expand_plugin
+
+//! > cairo_code
+const a: felt252 = 0;
+
+const b: felt252 = consteval_int!(4 + 5);
+
+const c: felt252 = 4 + 5;
+
+const d: felt252 = consteval_int!(23 + 4 * 5 + (4 + 5) / 2);
+
+const e: u8 = consteval_int!(255 + 1 - 1);
+
+//! > generated_cairo_code
+const a: felt252 = 0;
+
+const b: felt252 = 9;
+
+const c: felt252 = 4 + 5;
+
+const d: felt252 = 47;
+const e: u8 = 255;
+
+//! > expected_diagnostics
+
+//! > ==========================================================================
+
+//! > Test bad consteval_int! macros
+
+//! > test_runner_name
+test_expand_plugin
+
+//! > cairo_code
+const a: felt252 = consteval_int!(func_call(24));
+
+const b: felt252 = consteval_int!('some string');
+
+const c: felt252 = consteval_int!(*24);
+
+const d: felt252 = consteval_int!(~24);
+
+const e: felt252 = consteval_int!(234 < 5);
+
+//! > generated_cairo_code
+const a: felt252 = consteval_int!(func_call(24));
+
+
+const b: felt252 = consteval_int!('some string');
+
+
+const c: felt252 = consteval_int!(*24);
+
+
+const d: felt252 = consteval_int!(~24);
+
+
+const e: felt252 = consteval_int!(234 < 5);
+
+//! > expected_diagnostics
+error: Unsupported expression in consteval_int macro
+ --> dummy_file.cairo:1:35
+const a: felt252 = consteval_int!(func_call(24));
+                                  ^***********^
+
+error: Unsupported expression in consteval_int macro
+ --> dummy_file.cairo:3:35
+const b: felt252 = consteval_int!('some string');
+                                  ^***********^
+
+error: Unsupported unary operator in consteval_int macro
+ --> dummy_file.cairo:5:35
+const c: felt252 = consteval_int!(*24);
+                                  ^*^
+
+error: Unsupported unary operator in consteval_int macro
+ --> dummy_file.cairo:7:35
+const d: felt252 = consteval_int!(~24);
+                                  ^*^
+
+error: Unsupported binary operator in consteval_int macro
+ --> dummy_file.cairo:9:35
+const e: felt252 = consteval_int!(234 < 5);
+                                  ^*****^
+
+//! > ==========================================================================
+
+//! > Test consteval_int! inside functions (currently does nothing)
+
+//! > test_runner_name
+test_expand_plugin
+
+//! > cairo_code
+fn some_func()
+{
+    return consteval_int!(4 + 5);
+}
+
+//! > generated_cairo_code
+fn some_func()
+{
+    return consteval_int!(4 + 5);
+}
+
+//! > expected_diagnostics
diff --git /dev/null b/tests/bug_samples/issue3130.cairo
new file mode 100644
--- /dev/null
+++ b/tests/bug_samples/issue3130.cairo
@@ -0,0 +1,7 @@
+const a: felt252 = consteval_int!((4 + 2 * 3) * 256);
+const b: felt252 = consteval_int!(0xff & (24 + 5 * 2));
+const c: felt252 = consteval_int!(-0xff & (24 + 5 * 2));
+const d: felt252 = consteval_int!(0xff | (24 + 5 * 2));
+
+#[test]
+fn main() {}
diff --git a/tests/bug_samples/lib.cairo b/tests/bug_samples/lib.cairo
--- a/tests/bug_samples/lib.cairo
+++ b/tests/bug_samples/lib.cairo
@@ -16,6 +16,7 @@ mod issue2939;
 mod issue2961;
 mod issue2964;
 mod issue2995;
+mod issue3130;
 mod issue3153;
 mod issue3192;
 mod issue3211;

EOF_114329324912
cd "crates/cairo-lang-plugins"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard 5c98cf17854f2bec359077daa610a72453f04b7f
git clean -fd
