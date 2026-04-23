#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -113,6 +113,29 @@ fn run_test(
         generate,
         generate_pods,
         TestMethod::BeQuick,
+        None,
+    )
+    .unwrap()
+}
+
+/// A positive test, we expect to pass.
+fn run_test_ex(
+    cxx_code: &str,
+    header_code: &str,
+    rust_code: TokenStream,
+    generate: &[&str],
+    generate_pods: &[&str],
+    method: TestMethod,
+    extra_directives: Option<TokenStream>,
+) {
+    do_run_test(
+        cxx_code,
+        header_code,
+        rust_code,
+        generate,
+        generate_pods,
+        method,
+        extra_directives,
     )
     .unwrap()
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -131,6 +154,7 @@ fn run_test_expect_fail(
         generate,
         generate_pods,
         TestMethod::BeQuick,
+        None,
     )
     .expect_err("Unexpected success");
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -151,6 +175,7 @@ fn do_run_test(
     generate: &[&str],
     generate_pods: &[&str],
     method: TestMethod,
+    extra_directives: Option<TokenStream>,
 ) -> Result<(), TestError> {
     // Step 1: Write the C++ header snippet to a temp file
     let tdir = tempdir().unwrap();
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -199,6 +224,7 @@ fn do_run_test(
             #hexathorpe include "input.h"
             #(#generate)*
             #(#generate_pods)*
+            #extra_directives
         );
 
         fn main() {
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -283,25 +309,6 @@ fn do_run_test(
     Ok(())
 }
 
-/// This function runs a test with the full pipeline of build.rs support etc.
-fn run_test_with_full_pipeline(
-    cxx_code: &str,
-    header_code: &str,
-    rust_code: TokenStream,
-    generate: &[&str],
-    generate_pods: &[&str],
-) {
-    do_run_test(
-        cxx_code,
-        header_code,
-        rust_code,
-        generate,
-        generate_pods,
-        TestMethod::UseFullPipeline,
-    )
-    .unwrap();
-}
-
 #[test]
 fn test_return_void() {
     let cxx = indoc! {"
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1755,7 +1762,15 @@ fn test_cycle_string_full_pipeline() {
         assert_eq!(ffi::take_str(s), 3);
     };
     let generate = &["give_str", "take_str"];
-    run_test_with_full_pipeline(cxx, hdr, rs, generate, &[]);
+    run_test_ex(
+        cxx,
+        hdr,
+        rs,
+        generate,
+        &[],
+        TestMethod::UseFullPipeline,
+        None,
+    );
 }
 
 #[test]
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1775,7 +1790,15 @@ fn test_inline_full_pipeline() {
         assert_eq!(ffi::take_str(s), 3);
     };
     let generate = &["give_str", "take_str"];
-    run_test_with_full_pipeline("", hdr, rs, generate, &[]);
+    run_test_ex(
+        "",
+        hdr,
+        rs,
+        generate,
+        &[],
+        TestMethod::UseFullPipeline,
+        None,
+    );
 }
 
 #[test]
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1862,7 +1885,7 @@ fn test_multiple_classes_with_methods() {
         assert_eq!(oc.inc(), 4);
         assert_eq!(oc.inc(), 5);
     };
-    run_test_with_full_pipeline(
+    run_test_ex(
         cxx,
         hdr,
         rs,
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1873,6 +1896,8 @@ fn test_multiple_classes_with_methods() {
             "make_opaque_class",
         ],
         &["TrivialStruct", "TrivialClass"],
+        TestMethod::UseFullPipeline,
+        None,
     );
 }
 
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -3257,8 +3282,7 @@ fn test_root_ns_meth_ret_nonpod() {
 }
 
 #[test]
-#[ignore] // https://github.com/google/autocxx/issues/115
-fn test_nested_struct() {
+fn test_nested_struct_pod() {
     let hdr = indoc! {"
         #include <cstdint>
         struct A {
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -3267,13 +3291,53 @@ fn test_nested_struct() {
                 uint32_t b;
             };
         };
-        void daft(A::B a);
+        inline void daft(A::B a) {};
     "};
     let rs = quote! {
         let b = ffi::B { b: 12 };
         ffi::daft(b);
     };
-    run_test("", hdr, rs, &["daft"], &["B"]);
+    run_test_ex(
+        "",
+        hdr,
+        rs,
+        &["daft"],
+        &["B"],
+        TestMethod::BeQuick,
+        Some(quote! {
+            nested_type!("B", "A::B")
+        }),
+    );
+}
+
+#[test]
+fn test_nested_struct_nonpod() {
+    let hdr = indoc! {"
+        #include <cstdint>
+        struct A {
+            uint32_t a;
+            struct B {
+                B() {}
+                uint32_t b;
+            };
+        };
+        inline void daft(A::B) {}
+    "};
+    let rs = quote! {
+        let b = ffi::B::make_unique();
+        ffi::daft(b);
+    };
+    run_test_ex(
+        "",
+        hdr,
+        rs,
+        &["daft", "B"],
+        &[],
+        TestMethod::BeQuick,
+        Some(quote! {
+            nested_type!("B", "A::B")
+        }),
+    );
 }
 
 // Yet to test:
diff --git a/engine/src/lib.rs b/engine/src/lib.rs
--- a/engine/src/lib.rs
+++ b/engine/src/lib.rs
@@ -37,6 +38,7 @@ mod integration_tests;
 
 use proc_macro2::TokenStream as TokenStream2;
 use std::{fmt::Display, path::PathBuf};
+use type_database::TypeDatabase;
 
 use quote::ToTokens;
 use syn::parse::{Parse, ParseStream, Result as ParseResult};
diff --git a/engine/src/types.rs b/engine/src/types.rs
--- a/engine/src/types.rs
+++ b/engine/src/types.rs
@@ -219,53 +219,6 @@ impl Display for TypeName {
     }
 }
 
-pub(crate) fn type_to_cpp(ty: &Type) -> String {
-    match ty {
-        Type::Path(typ) => {
-            // If this is a std::unique_ptr we do need to pass
-            // its argument through.
-            let root = TypeName::from_type_path(typ).to_cpp_name();
-            let suffix = match &typ.path.segments.last().unwrap().arguments {
-                syn::PathArguments::AngleBracketed(ab) => Some(
-                    ab.args
-                        .iter()
-                        .map(|x| match x {
-                            syn::GenericArgument::Type(gat) => type_to_cpp(gat),
-                            _ => "".to_string(),
-                        })
-                        .join(", "),
-                ),
-                syn::PathArguments::None | syn::PathArguments::Parenthesized(_) => None,
-            };
-            match suffix {
-                None => root,
-                Some(suffix) => format!("{}<{}>", root, suffix),
-            }
-        }
-        Type::Reference(typr) => {
-            let const_bit = match typr.mutability {
-                None => "const ",
-                Some(_) => "",
-            };
-            format!("{}{}&", const_bit, type_to_cpp(typr.elem.as_ref()))
-        }
-        Type::Array(_)
-        | Type::BareFn(_)
-        | Type::Group(_)
-        | Type::ImplTrait(_)
-        | Type::Infer(_)
-        | Type::Macro(_)
-        | Type::Never(_)
-        | Type::Paren(_)
-        | Type::Ptr(_)
-        | Type::Slice(_)
-        | Type::TraitObject(_)
-        | Type::Tuple(_)
-        | Type::Verbatim(_) => panic!("Unsupported type"),
-        _ => panic!("Unknown type"),
-    }
-}
-
 #[cfg(test)]
 mod tests {
     use crate::TypeName;

EOF_114329324912
cd "engine"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 1ce47156f3fe4bbfc013c37616efa77e9577d85f
git clean -fd
