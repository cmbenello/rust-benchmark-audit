#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -114,6 +114,7 @@ fn run_test(
         generate,
         generate_pods,
         None,
+        &[],
     )
     .unwrap()
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -126,6 +127,7 @@ fn run_test_ex(
     generate: &[&str],
     generate_pods: &[&str],
     extra_directives: Option<TokenStream>,
+    defines: &[&str],
 ) {
     do_run_test(
         cxx_code,
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -134,6 +136,7 @@ fn run_test_ex(
         generate,
         generate_pods,
         extra_directives,
+        defines,
     )
     .unwrap()
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -152,6 +155,7 @@ fn run_test_expect_fail(
         generate,
         generate_pods,
         None,
+        &[],
     )
     .expect_err("Unexpected success");
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -171,6 +175,7 @@ fn do_run_test(
     generate: &[&str],
     generate_pods: &[&str],
     extra_directives: Option<TokenStream>,
+    definitions: &[&str],
 ) -> Result<(), TestError> {
     // Step 1: Write the C++ header snippet to a temp file
     let tdir = tempdir().unwrap();
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -226,6 +231,7 @@ fn do_run_test(
     let build_results = crate::builder::build_to_custom_directory(
         &rs_path,
         &[tdir.path()],
+        &definitions,
         Some(target_dir.clone()),
         None,
     )
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -243,6 +249,13 @@ fn do_run_test(
         b.file(cxx_path);
     }
 
+    for d in definitions {
+        let mut it = d.split("=");
+        let k = it.next().unwrap();
+        let v = it.next();
+        b.define(k, v);
+    }
+
     b.out_dir(&target_dir)
         .host(&target)
         .target(&target)
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1975,6 +1988,7 @@ fn test_multiple_classes_with_methods() {
         ],
         &["TrivialStruct", "TrivialClass"],
         None,
+        &[],
     );
 }
 
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -3951,6 +3965,7 @@ fn test_issues_217_222() {
         &["GURL"],
         &[],
         Some(quote! { block!("StringPiece") block!("Replacements") }),
+        &[],
     );
 }
 
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -4112,8 +4127,7 @@ fn test_issue_264() {
         };
         } // namespace operations_research
     "};
-    let rs = quote! {
-    };
+    let rs = quote! {};
     run_test("", hdr, rs, &["operations_research::Solver"], &[]);
 }
 
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -4133,8 +4147,7 @@ fn test_vector_of_pointers() {
         class a {};
         } // namespace operations_research
     "};
-    let rs = quote! {
-    };
+    let rs = quote! {};
     run_test("", hdr, rs, &["operations_research::Solver"], &[]);
 }
 
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -4151,9 +4164,22 @@ fn test_pointer_to_pointer() {
         class a {};
         } // namespace operations_research
     "};
+    let rs = quote! {};
+    run_test("", hdr, rs, &["operations_research::Solver"], &[]);
+}
+
+#[test]
+fn test_defines_effective() {
+    let hdr = indoc! {"
+        #include <cstdint>
+        #ifdef FOO
+        inline uint32_t a() { return 4; }
+        #endif
+    "};
     let rs = quote! {
+        ffi::a();
     };
-    run_test("", hdr, rs, &["operations_research::Solver"], &[]);
+    run_test_ex("", hdr, rs, &["a"], &[], None, &["FOO"]);
 }
 
 // Yet to test:

EOF_114329324912
cd "engine"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 5922890ca02a55bf5a594fe939c8328a8bcf6d83
git clean -fd
