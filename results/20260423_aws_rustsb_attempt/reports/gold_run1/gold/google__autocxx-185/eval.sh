#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -3371,7 +3371,6 @@ fn test_cycle_vector() {
 }
 
 #[test]
-#[ignore] // https://github.com/google/autocxx/issues/183
 fn test_typedef_to_std() {
     let hdr = indoc! {"
         #include <string>
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -3385,7 +3384,93 @@ fn test_typedef_to_std() {
         assert_eq!(ffi::take_str("hello".to_cpp()), 5);
     };
     run_test("", hdr, rs, &["take_str"], &[]);
+}
+
+#[test]
+fn test_typedef_to_up_in_fn_call() {
+    let hdr = indoc! {"
+        #include <string>
+        #include <memory>
+        typedef std::unique_ptr<std::string> my_string;
+        inline uint32_t take_str(my_string a) {
+            return a->size();
+        }
+    "};
+    let rs = quote! {
+        use ffi::ToCppString;
+        assert_eq!(ffi::take_str("hello".to_cpp()), 5);
+    };
+    run_test("", hdr, rs, &["take_str"], &[]);
+}
+
+#[test]
+fn test_typedef_in_pod_struct() {
+    let hdr = indoc! {"
+        #include <string>
+        typedef uint32_t my_int;
+        struct A {
+            my_int a;
+        };
+        inline uint32_t take_a(A a) {
+            return a.a;
+        }
+    "};
+    let rs = quote! {
+        let a = ffi::A {
+            a: 32,
+        };
+        assert_eq!(ffi::take_a(a), 32);
+    };
+    run_test("", hdr, rs, &["take_a"], &["A"]);
+}
 
+#[test]
+fn test_typedef_to_std_in_struct() {
+    let hdr = indoc! {"
+        #include <string>
+        typedef std::string my_string;
+        struct A {
+            my_string a;
+        };
+        inline A make_a(std::string b) {
+            A bob;
+            bob.a = b;
+            return bob;
+        }
+        inline uint32_t take_a(A a) {
+            return a.a.size();
+        }
+    "};
+    let rs = quote! {
+        use ffi::ToCppString;
+        assert_eq!(ffi::take_a(ffi::make_a("hello".to_cpp())), 5);
+    };
+    run_test("", hdr, rs, &["make_a", "take_a"], &[]);
+}
+
+#[test]
+fn test_typedef_to_up_in_struct() {
+    let hdr = indoc! {"
+        #include <string>
+        #include <memory>
+        typedef std::unique_ptr<std::string> my_string;
+        struct A {
+            my_string a;
+        };
+        inline A make_a(std::string b) {
+            A bob;
+            bob.a = std::make_unique<std::string>(b);
+            return bob;
+        }
+        inline uint32_t take_a(A a) {
+            return a.a->size();
+        }
+    "};
+    let rs = quote! {
+        use ffi::ToCppString;
+        assert_eq!(ffi::take_a(ffi::make_a("hello".to_cpp())), 5);
+    };
+    run_test("", hdr, rs, &["make_a", "take_a"], &[]);
 }
 
 // Yet to test:
diff --git a/engine/src/lib.rs b/engine/src/lib.rs
--- a/engine/src/lib.rs
+++ b/engine/src/lib.rs
@@ -22,7 +22,6 @@ mod parse;
 mod parse_callbacks;
 mod rust_pretty_printer;
 mod type_to_cpp;
-mod typedef_analyzer;
 mod types;
 
 #[cfg(any(test, feature = "build"))]

EOF_114329324912
cd "engine"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 75c76ab9ab0075cf5c7d01eaaf800213a98a80ec
git clean -fd
