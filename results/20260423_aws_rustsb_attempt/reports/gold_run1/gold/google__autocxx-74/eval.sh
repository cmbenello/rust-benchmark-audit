#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -17,12 +17,12 @@ use log::info;
 use proc_macro2::{Span, TokenStream};
 use quote::quote;
 use quote::ToTokens;
-use syn::Token;
 use std::fs::File;
 use std::io::Write;
 use std::panic::RefUnwindSafe;
 use std::path::{Path, PathBuf};
 use std::sync::Mutex;
+use syn::Token;
 use tempfile::{tempdir, TempDir};
 use test_env_log::test;
 
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -313,7 +313,7 @@ fn test_return_void() {
         void do_nothing();
     "};
     let rs = quote! {
-        ffi::cxxbridge::do_nothing();
+        ffi::do_nothing();
     };
     run_test(cxx, hdr, rs, &["do_nothing"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -331,8 +331,8 @@ fn test_two_funcs() {
         void do_nothing2();
     "};
     let rs = quote! {
-        ffi::cxxbridge::do_nothing1();
-        ffi::cxxbridge::do_nothing2();
+        ffi::do_nothing1();
+        ffi::do_nothing2();
     };
     run_test(cxx, hdr, rs, &["do_nothing1", "do_nothing2"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -354,8 +354,8 @@ fn test_two_funcs_with_definition() {
         void do_nothing2();
     "};
     let rs = quote! {
-        ffi::cxxbridge::do_nothing1();
-        ffi::cxxbridge::do_nothing2();
+        ffi::do_nothing1();
+        ffi::do_nothing2();
     };
     println!("Here");
 
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -375,7 +375,7 @@ fn test_return_i32() {
         uint32_t give_int();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_int(), 5);
+        assert_eq!(ffi::give_int(), 5);
     };
     run_test(cxx, hdr, rs, &["give_int"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -392,7 +392,7 @@ fn test_take_i32() {
         uint32_t take_int(uint32_t a);
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::take_int(3), 6);
+        assert_eq!(ffi::take_int(3), 6);
     };
     run_test(cxx, hdr, rs, &["take_int"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -411,7 +411,7 @@ fn test_give_up_int() {
         std::unique_ptr<uint32_t> give_up();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_up().as_ref().unwrap(), 12);
+        assert_eq!(ffi::give_up().as_ref().unwrap(), 12);
     };
     run_test(cxx, hdr, rs, &["give_up"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -429,7 +429,7 @@ fn test_give_string_up() {
         std::unique_ptr<std::string> give_str_up();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_str_up().as_ref().unwrap().to_str().unwrap(), "Bob");
+        assert_eq!(ffi::give_str_up().as_ref().unwrap().to_str().unwrap(), "Bob");
     };
     run_test(cxx, hdr, rs, &["give_str_up"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -446,7 +446,7 @@ fn test_give_string_plain() {
         std::string give_str();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_str().as_ref().unwrap(), "Bob");
+        assert_eq!(ffi::give_str().as_ref().unwrap(), "Bob");
     };
     run_test(cxx, hdr, rs, &["give_str"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -469,8 +469,8 @@ fn test_cycle_string_up() {
         uint32_t take_str_up(std::unique_ptr<std::string> a);
     "};
     let rs = quote! {
-        let s = ffi::cxxbridge::give_str_up();
-        assert_eq!(ffi::cxxbridge::take_str_up(s), 3);
+        let s = ffi::give_str_up();
+        assert_eq!(ffi::take_str_up(s), 3);
     };
     run_test(cxx, hdr, rs, &["give_str_up", "take_str_up"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -492,8 +492,8 @@ fn test_cycle_string() {
         uint32_t take_str(std::string a);
     "};
     let rs = quote! {
-        let s = ffi::cxxbridge::give_str();
-        assert_eq!(ffi::cxxbridge::take_str(s), 3);
+        let s = ffi::give_str();
+        assert_eq!(ffi::take_str(s), 3);
     };
     let allowed_funcs = &["give_str", "take_str"];
     run_test(cxx, hdr, rs, allowed_funcs, &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -517,8 +517,8 @@ fn test_cycle_string_by_ref() {
         uint32_t take_str(const std::string& a);
     "};
     let rs = quote! {
-        let s = ffi::cxxbridge::give_str();
-        assert_eq!(ffi::cxxbridge::take_str(s.as_ref().unwrap()), 3);
+        let s = ffi::give_str();
+        assert_eq!(ffi::take_str(s.as_ref().unwrap()), 3);
     };
     let allowed_funcs = &["give_str", "take_str"];
     run_test(cxx, hdr, rs, allowed_funcs, &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -542,8 +542,8 @@ fn test_cycle_string_by_mut_ref() {
         uint32_t take_str(std::string& a);
     "};
     let rs = quote! {
-        let mut s = ffi::cxxbridge::give_str();
-        assert_eq!(ffi::cxxbridge::take_str(s.as_mut().unwrap()), 3);
+        let mut s = ffi::give_str();
+        assert_eq!(ffi::take_str(s.as_mut().unwrap()), 3);
     };
     let allowed_funcs = &["give_str", "take_str"];
     run_test(cxx, hdr, rs, allowed_funcs, &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -568,7 +568,7 @@ fn test_give_pod_by_value() {
         Bob give_bob();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_bob().b, 4);
+        assert_eq!(ffi::give_bob().b, 4);
     };
     run_test(cxx, hdr, rs, &["give_bob"], &["Bob"]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -593,7 +593,7 @@ fn test_give_pod_class_by_value() {
         Bob give_bob();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_bob().b, 4);
+        assert_eq!(ffi::give_bob().b, 4);
     };
     run_test(cxx, hdr, rs, &["give_bob"], &["Bob"]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -618,7 +618,7 @@ fn test_give_pod_by_up() {
         std::unique_ptr<Bob> give_bob();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_bob().as_ref().unwrap().b, 4);
+        assert_eq!(ffi::give_bob().as_ref().unwrap().b, 4);
     };
     run_test(cxx, hdr, rs, &["give_bob"], &["Bob"]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -639,8 +639,8 @@ fn test_take_pod_by_value() {
         uint32_t take_bob(Bob a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob { a: 12, b: 13 };
-        assert_eq!(ffi::cxxbridge::take_bob(a), 12);
+        let a = ffi::Bob { a: 12, b: 13 };
+        assert_eq!(ffi::take_bob(a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob"]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -661,8 +661,8 @@ fn test_take_pod_by_ref() {
         uint32_t take_bob(const Bob& a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob { a: 12, b: 13 };
-        assert_eq!(ffi::cxxbridge::take_bob(&a), 12);
+        let a = ffi::Bob { a: 12, b: 13 };
+        assert_eq!(ffi::take_bob(&a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob"]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -684,8 +684,8 @@ fn test_take_pod_by_mut_ref() {
         uint32_t take_bob(Bob& a);
     "};
     let rs = quote! {
-        let mut a = ffi::cxxbridge::Bob { a: 12, b: 13 };
-        assert_eq!(ffi::cxxbridge::take_bob(&mut a), 12);
+        let mut a = ffi::Bob { a: 12, b: 13 };
+        assert_eq!(ffi::take_bob(&mut a), 12);
         assert_eq!(a.b, 14);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -711,8 +711,8 @@ fn test_take_nested_pod_by_value() {
         uint32_t take_bob(Bob a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob { a: 12, b: 13, c: ffi::cxxbridge::Phil { d: 4 } };
-        assert_eq!(ffi::cxxbridge::take_bob(a), 12);
+        let a = ffi::Bob { a: 12, b: 13, c: ffi::Phil { d: 4 } };
+        assert_eq!(ffi::take_bob(a), 12);
     };
     // Should be no need to allowlist Phil below
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -739,8 +739,8 @@ fn test_take_nonpod_by_value() {
         uint32_t take_bob(Bob a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob_make_unique(12, 13);
-        assert_eq!(ffi::cxxbridge::take_bob(a), 12);
+        let a = ffi::Bob_make_unique(12, 13);
+        assert_eq!(ffi::take_bob(a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob", "Bob"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -767,8 +767,8 @@ fn test_take_nonpod_by_ref() {
         uint32_t take_bob(const Bob& a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::make_bob(12);
-        assert_eq!(ffi::cxxbridge::take_bob(&a), 12);
+        let a = ffi::make_bob(12);
+        assert_eq!(ffi::take_bob(&a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob", "Bob", "make_bob"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -795,8 +795,8 @@ fn test_take_nonpod_by_mut_ref() {
         uint32_t take_bob(Bob& a);
     "};
     let rs = quote! {
-        let mut a = ffi::cxxbridge::make_bob(12);
-        assert_eq!(ffi::cxxbridge::take_bob(&mut a), 12);
+        let mut a = ffi::make_bob(12);
+        assert_eq!(ffi::take_bob(&mut a), 12);
     };
     // TODO confirm that the object really was mutated by C++ in this
     // and similar tests.
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -828,8 +828,8 @@ fn test_return_nonpod_by_value() {
         uint32_t take_bob(std::unique_ptr<Bob> a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::give_bob(13);
-        assert_eq!(ffi::cxxbridge::take_bob(a), 13);
+        let a = ffi::give_bob(13);
+        assert_eq!(ffi::take_bob(a), 13);
     };
     run_test(cxx, hdr, rs, &["take_bob", "give_bob", "Bob"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -847,7 +847,7 @@ fn test_get_str_by_up() {
         std::unique_ptr<std::string> get_str();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::get_str().as_ref().unwrap(), "hello");
+        assert_eq!(ffi::get_str().as_ref().unwrap(), "hello");
     };
     run_test(cxx, hdr, rs, &["get_str"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -864,7 +864,7 @@ fn test_get_str_by_value() {
         std::string get_str();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::get_str().as_ref().unwrap(), "hello");
+        assert_eq!(ffi::get_str().as_ref().unwrap(), "hello");
     };
     run_test(cxx, hdr, rs, &["get_str"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -894,8 +894,8 @@ fn test_cycle_nonpod_with_str_by_ref() {
         std::unique_ptr<Bob> make_bob();
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::make_bob();
-        assert_eq!(ffi::cxxbridge::take_bob(a.as_ref().unwrap()), 32);
+        let a = ffi::make_bob();
+        assert_eq!(ffi::take_bob(a.as_ref().unwrap()), 32);
     };
     run_test(cxx, hdr, rs, &["take_bob", "Bob", "make_bob"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -919,8 +919,8 @@ fn test_make_up() {
         uint32_t take_bob(const Bob& a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob::make_unique(); // TODO test with all sorts of arguments.
-        assert_eq!(ffi::cxxbridge::take_bob(a.as_ref().unwrap()), 3);
+        let a = ffi::Bob::make_unique(); // TODO test with all sorts of arguments.
+        assert_eq!(ffi::take_bob(a.as_ref().unwrap()), 3);
     };
     run_test(cxx, hdr, rs, &["Bob", "take_bob"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -944,8 +944,8 @@ fn test_make_up_with_args() {
         uint32_t take_bob(const Bob& a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob_make_unique(12, 13);
-        assert_eq!(ffi::cxxbridge::take_bob(a.as_ref().unwrap()), 12);
+        let a = ffi::Bob_make_unique(12, 13);
+        assert_eq!(ffi::take_bob(a.as_ref().unwrap()), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob", "Bob"], &[]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -966,7 +966,7 @@ fn test_make_up_int() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob::make_unique(3);
+        let a = ffi::Bob::make_unique(3);
         assert_eq!(a.as_ref().unwrap().b, 3);
     };
     run_test(cxx, hdr, rs, &["Bob"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -988,8 +988,8 @@ fn test_enum_with_funcs() {
         Bob give_bob();
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob::BOB_VALUE_2;
-        let b = ffi::cxxbridge::give_bob();
+        let a = ffi::Bob::BOB_VALUE_2;
+        let b = ffi::give_bob();
         assert!(a == b);
     };
     run_test(cxx, hdr, rs, &["Bob", "give_bob"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1006,8 +1006,8 @@ fn test_enum_no_funcs() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob::BOB_VALUE_1;
-        let b = ffi::cxxbridge::Bob::BOB_VALUE_2;
+        let a = ffi::Bob::BOB_VALUE_1;
+        let b = ffi::Bob::BOB_VALUE_2;
         assert!(a != b);
     };
     run_test(cxx, hdr, rs, &["Bob"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1030,8 +1030,8 @@ fn test_take_pod_class_by_value() {
         uint32_t take_bob(Bob a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob { a: 12, b: 13 };
-        assert_eq!(ffi::cxxbridge::take_bob(a), 12);
+        let a = ffi::Bob { a: 12, b: 13 };
+        assert_eq!(ffi::take_bob(a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob"]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1053,7 +1053,7 @@ fn test_pod_method() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let a = ffi::Bob { a: 12, b: 13 };
         assert_eq!(a.get_bob(), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1076,7 +1076,7 @@ fn test_pod_mut_method() {
         };
     "};
     let rs = quote! {
-        let mut a = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let mut a = ffi::Bob { a: 12, b: 13 };
         assert_eq!(a.get_bob(), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1090,9 +1090,9 @@ fn test_define_int() {
         #define BOB 3
     "};
     let rs = quote! {
-        assert_eq!(ffi::defs::BOB, 3);
+        assert_eq!(ffi::BOB, 3);
     };
-    run_test(cxx, hdr, rs, &[], &[]);
+    run_test(cxx, hdr, rs, &["BOB"], &[]);
 }
 
 #[test]
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1103,9 +1103,9 @@ fn test_define_str() {
         #define BOB \"foo\"
     "};
     let rs = quote! {
-        assert_eq!(ffi::defs::BOB, "foo");
+        assert_eq!(std::str::from_utf8(ffi::BOB).unwrap().trim_end_matches(char::from(0)), "foo");
     };
-    run_test(cxx, hdr, rs, &[], &[]);
+    run_test(cxx, hdr, rs, &["BOB"], &[]);
 }
 
 #[test]
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1174,13 +1174,13 @@ fn test_negative_make_nonpod() {
         uint32_t take_bob(const Bob& a);
     "};
     let rs = quote! {
-        ffi::cxxbridge::Bob {};
+        ffi::Bob {};
     };
     let rs2 = quote! {
-        ffi::cxxbridge::Bob { a: 12 };
+        ffi::Bob { a: 12 };
     };
     let rs3 = quote! {
-        ffi::cxxbridge::Bob { do_not_attempt_to_allocate_nonpod_types: [] };
+        ffi::Bob { do_not_attempt_to_allocate_nonpod_types: [] };
     };
     run_test_expect_fail(cxx, hdr, rs, &["take_bob", "Bob", "make_bob"], &[]);
     run_test_expect_fail(cxx, hdr, rs2, &["take_bob", "Bob", "make_bob"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1207,8 +1207,8 @@ fn test_method_pass_pod_by_value() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Anna { a: 14 };
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let a = ffi::Anna { a: 14 };
+        let b = ffi::Bob { a: 12, b: 13 };
         assert_eq!(b.get_bob(a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob", "Anna"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1231,8 +1231,8 @@ fn test_inline_method() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Anna { a: 14 };
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let a = ffi::Anna { a: 14 };
+        let b = ffi::Bob { a: 12, b: 13 };
         assert_eq!(b.get_bob(a), 12);
     };
     run_test("", hdr, rs, &["take_bob"], &["Bob", "Anna"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1258,8 +1258,8 @@ fn test_method_pass_pod_by_reference() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Anna { a: 14 };
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let a = ffi::Anna { a: 14 };
+        let b = ffi::Bob { a: 12, b: 13 };
         assert_eq!(b.get_bob(&a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob", "Anna"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1285,8 +1285,8 @@ fn test_method_pass_pod_by_mut_reference() {
         };
     "};
     let rs = quote! {
-        let mut a = ffi::cxxbridge::Anna { a: 14 };
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let mut a = ffi::Anna { a: 14 };
+        let b = ffi::Bob { a: 12, b: 13 };
         assert_eq!(b.get_bob(&mut a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob", "Anna"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1313,8 +1313,8 @@ fn test_method_pass_pod_by_up() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Anna { a: 14 };
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let a = ffi::Anna { a: 14 };
+        let b = ffi::Bob { a: 12, b: 13 };
         assert_eq!(b.get_bob(cxx::UniquePtr::new(a)), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["Bob", "Anna"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1348,9 +1348,9 @@ fn test_method_pass_nonpod_by_value() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::give_anna();
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
-        assert_eq!(ffi::cxxbridge::get_bob(&b, a), 12);
+        let a = ffi::give_anna();
+        let b = ffi::Bob { a: 12, b: 13 };
+        assert_eq!(ffi::get_bob(&b, a), 12);
         // assert_eq!(b.get_bob(a), 12); // eventual goal
     };
     run_test(
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1390,8 +1390,8 @@ fn test_method_pass_nonpod_by_reference() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::give_anna();
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let a = ffi::give_anna();
+        let b = ffi::Bob { a: 12, b: 13 };
         assert_eq!(b.get_bob(a.as_ref().unwrap()), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob", "Anna", "give_anna"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1425,8 +1425,8 @@ fn test_method_pass_nonpod_by_mut_reference() {
         };
     "};
     let rs = quote! {
-        let mut a = ffi::cxxbridge::give_anna();
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let mut a = ffi::give_anna();
+        let b = ffi::Bob { a: 12, b: 13 };
         assert_eq!(b.get_bob(a.as_mut().unwrap()), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob", "Anna", "give_anna"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1461,8 +1461,8 @@ fn test_method_pass_nonpod_by_up() {
         };
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::give_anna();
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let a = ffi::give_anna();
+        let b = ffi::Bob { a: 12, b: 13 };
         assert_eq!(b.get_bob(a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob", "give_anna"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1492,9 +1492,9 @@ fn test_method_return_nonpod_by_value() {
         };
     "};
     let rs = quote! {
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let b = ffi::Bob { a: 12, b: 13 };
         // let a = b.get_bob(); // eventual goal
-        let a = ffi::cxxbridge::get_anna(&b);
+        let a = ffi::get_anna(&b);
         assert!(!a.is_null());
     };
     run_test(cxx, hdr, rs, &["take_bob", "Anna", "get_anna"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1518,8 +1518,8 @@ fn test_pass_string_by_value() {
         std::unique_ptr<std::string> get_msg();
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::get_msg();
-        let c = ffi::cxxbridge::measure_string(a);
+        let a = ffi::get_msg();
+        let c = ffi::measure_string(a);
         assert_eq!(c, 5);
     };
     run_test(cxx, hdr, rs, &["measure_string", "get_msg"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1537,7 +1537,7 @@ fn test_return_string_by_value() {
         std::string get_msg();
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::get_msg();
+        let a = ffi::get_msg();
         assert!(a.as_ref().unwrap() == "hello");
     };
     run_test(cxx, hdr, rs, &["get_msg"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1566,9 +1566,9 @@ fn test_method_pass_string_by_value() {
         std::unique_ptr<std::string> get_msg();
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::get_msg();
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
-        let c = ffi::cxxbridge::measure_string(&b, a);
+        let a = ffi::get_msg();
+        let b = ffi::Bob { a: 12, b: 13 };
+        let c = ffi::measure_string(&b, a);
         // let c = b.measure_string(a); // eventual goal
         assert_eq!(c, 5);
     };
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1599,9 +1599,9 @@ fn test_method_return_string_by_value() {
         };
     "};
     let rs = quote! {
-        let b = ffi::cxxbridge::Bob { a: 12, b: 13 };
+        let b = ffi::Bob { a: 12, b: 13 };
         // let a = b.get_msg(); // eventual goal
-        let a = ffi::cxxbridge::get_msg(&b);
+        let a = ffi::get_msg(&b);
         assert!(a.as_ref().unwrap() == "hello");
     };
     run_test(cxx, hdr, rs, &["take_bob", "get_msg"], &["Bob"]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1620,7 +1620,7 @@ fn test_pass_rust_string_by_ref() {
         uint32_t measure_string(const rust::String& z);
     "};
     let rs = quote! {
-        let c = ffi::cxxbridge::measure_string(&"hello".to_string());
+        let c = ffi::measure_string(&"hello".to_string());
         assert_eq!(c, 5);
     };
     run_test(cxx, hdr, rs, &["measure_string"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1639,7 +1639,7 @@ fn test_pass_rust_string_by_value() {
         uint32_t measure_string(rust::String z);
     "};
     let rs = quote! {
-        let c = ffi::cxxbridge::measure_string("hello".into());
+        let c = ffi::measure_string("hello".into());
         assert_eq!(c, 5);
     };
     run_test(cxx, hdr, rs, &["measure_string"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1659,7 +1659,7 @@ fn test_pass_rust_str() {
         uint32_t measure_string(rust::Str z);
     "};
     let rs = quote! {
-        let c = ffi::cxxbridge::measure_string("hello");
+        let c = ffi::measure_string("hello");
         assert_eq!(c, 5);
     };
     run_test(cxx, hdr, rs, &["measure_string"], &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1682,8 +1682,8 @@ fn test_cycle_string_full_pipeline() {
         uint32_t take_str(std::string a);
     "};
     let rs = quote! {
-        let s = ffi::cxxbridge::give_str();
-        assert_eq!(ffi::cxxbridge::take_str(s), 3);
+        let s = ffi::give_str();
+        assert_eq!(ffi::take_str(s), 3);
     };
     let allowed_funcs = &["give_str", "take_str"];
     run_test_with_full_pipeline(cxx, hdr, rs, allowed_funcs, &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1702,8 +1702,8 @@ fn test_inline_full_pipeline() {
         }
     "};
     let rs = quote! {
-        let s = ffi::cxxbridge::give_str();
-        assert_eq!(ffi::cxxbridge::take_str(s), 3);
+        let s = ffi::give_str();
+        assert_eq!(ffi::take_str(s), 3);
     };
     let allowed_funcs = &["give_str", "take_str"];
     run_test_with_full_pipeline("", hdr, rs, allowed_funcs, &[]);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1769,7 +1769,7 @@ fn test_multiple_classes_with_methods() {
         uint32_t OpaqueClass::inc() { return ++val_; }
     "};
     let rs = quote! {
-        use ffi::cxxbridge::*;
+        use ffi::*;
 
         let mut ts: TrivialStruct = make_trivial_struct();
         assert_eq!(ts.get(), 0);
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1830,7 +1830,7 @@ fn test_ns_return_struct() {
         A::B::Bob give_bob();
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_bob().b, 4);
+        assert_eq!(ffi::give_bob().b, 4);
     };
     run_test(cxx, hdr, rs, &["give_bob"], &["A::B::Bob"]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1855,8 +1855,8 @@ fn test_ns_take_struct() {
         uint32_t take_bob(A::B::Bob a);
     "};
     let rs = quote! {
-        let a = ffi::cxxbridge::Bob { a: 12, b: 13 };
-        assert_eq!(ffi::cxxbridge::take_bob(a), 12);
+        let a = ffi::Bob { a: 12, b: 13 };
+        assert_eq!(ffi::take_bob(a), 12);
     };
     run_test(cxx, hdr, rs, &["take_bob"], &["A::B::Bob"]);
 }
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -1887,7 +1887,7 @@ fn test_ns_func() {
         }
     "};
     let rs = quote! {
-        assert_eq!(ffi::cxxbridge::give_bob().b, 4);
+        assert_eq!(ffi::give_bob().b, 4);
     };
     run_test(cxx, hdr, rs, &["C::give_bob"], &["A::B::Bob"]);
 }
diff --git a/engine/src/lib.rs b/engine/src/lib.rs
--- a/engine/src/lib.rs
+++ b/engine/src/lib.rs
@@ -38,9 +37,6 @@ use syn::{parse_quote, ItemMod, Macro};
 use additional_cpp_generator::AdditionalCppGenerator;
 use itertools::join;
 use log::{info, warn};
-use preprocessor_parse_callbacks::{PreprocessorDefinitions, PreprocessorParseCallbacks};
-use std::rc::Rc;
-use std::sync::Mutex;
 use types::TypeName;
 
 #[cfg(any(test, feature = "build"))]

EOF_114329324912
cd "engine"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard e93e85e61116ce5f0c1e18c69020ab1eca5704fd
git clean -fd
