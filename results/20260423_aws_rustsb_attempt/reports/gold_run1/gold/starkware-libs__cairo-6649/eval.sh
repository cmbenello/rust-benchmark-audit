#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cairo-lang-language-server/tests/e2e/hover.rs b/crates/cairo-lang-language-server/tests/e2e/hover.rs
--- a/crates/cairo-lang-language-server/tests/e2e/hover.rs
+++ b/crates/cairo-lang-language-server/tests/e2e/hover.rs
@@ -17,7 +17,8 @@ cairo_lang_test_utils::test_file_test!(
         partial: "partial.txt",
         starknet: "starknet.txt",
         literals: "literals.txt",
-        structs: "structs.txt"
+        structs: "structs.txt",
+        paths: "paths.txt",
     },
     test_hover
 );
diff --git a/crates/cairo-lang-language-server/tests/test_data/hover/basic.txt b/crates/cairo-lang-language-server/tests/test_data/hover/basic.txt
--- a/crates/cairo-lang-language-server/tests/test_data/hover/basic.txt
+++ b/crates/cairo-lang-language-server/tests/test_data/hover/basic.txt
@@ -170,21 +170,31 @@ fn add_two(x: u32) -> u32
 // = source context
     front<caret>_of_house::hosting::add_to_waitlist();
 // = highlight
-No highlight information.
+    <sel>front_of_house</sel>::hosting::add_to_waitlist();
 // = popover
 ```cairo
-fn add_to_waitlist() -> ()
+hello
 ```
+```cairo
+mod front_of_house
+```
+---
+Front of house module.
 
 //! > hover #7
 // = source context
     front_of_house::ho<caret>sting::add_to_waitlist();
 // = highlight
-No highlight information.
+    front_of_house::<sel>hosting</sel>::add_to_waitlist();
 // = popover
 ```cairo
-fn add_to_waitlist() -> ()
+hello::front_of_house
+```
+```cairo
+mod hosting
 ```
+---
+Hosting module.
 
 //! > hover #8
 // = source context
diff --git /dev/null b/crates/cairo-lang-language-server/tests/test_data/hover/paths.txt
new file mode 100644
--- /dev/null
+++ b/crates/cairo-lang-language-server/tests/test_data/hover/paths.txt
@@ -0,0 +1,373 @@
+//! > Hover
+
+//! > test_runner_name
+test_hover
+
+//! > cairo_project.toml
+[crate_roots]
+hello = "src"
+
+[config.global]
+edition = "2023_11"
+
+//! > cairo_code
+/// some_module docstring.
+mod some_<caret>module {
+    /// internal_module docstring.
+    pub mod internal_module<caret> {
+        pub trait MyTrait<T> {
+            fn foo(t: T);
+        }
+
+        pub impl MyTraitImpl of MyTrait<u32> {
+            fn foo(t: u32) {}
+        }
+
+        pub mod nested_internal_module {
+            pub struct PublicStruct {}
+            struct PrivateStruct {}
+
+            pub fn foo() {}
+        }
+    }
+
+    /// internal_module2 docstring.
+    pub mod internal_module2 {
+        use super::internal_module<caret>;
+
+        /// private_submodule docstring.
+        mod <caret>private_submodule {
+            struct PrivateStruct2 {}
+        }
+    }
+}
+
+mod happy_cases {
+    use super::some_<caret>module;
+    use super::some_module::internal_module;
+    use super::some_module::internal_<caret>module::MyTrait;
+    use super::some_module::internal_module::nested_inte<caret>rnal_module;
+
+    impl TraitImpl of super::some_module::internal<caret>_module::MyTrait<felt252> {
+        fn foo(t: felt252) {}
+    }
+
+    impl TraitImpl2 of internal_<caret>module::MyTrait<u8> {
+        fn foo(t: u8) {}
+    }
+
+    fn function_with_path() {
+        super::some_module::internal_module::nested_internal_module::foo();
+        <caret>super::some_module::internal_module::nested_internal_module::foo();
+        super::some_module::inte<caret>rnal_module::nested_internal_module::foo();
+        super::some_module::internal_module::nested_internal<caret>_module::foo();
+    }
+
+    fn function_with_partial_path() {
+        nested_in<caret>ternal_module::foo();
+        internal_<caret>module::MyTraitImpl::foo(0_u32);
+    }
+
+    fn constructor_with_path() {
+        let _ = super::some_module::internal_module::nested_internal_module::PublicStruct {};
+        let _ = <caret>super::some_module::internal_module::nested_internal_module::PublicStruct {};
+        let _ = super::some_module::inte<caret>rnal_module::nested_internal_module::PublicStruct {};
+        let _ = super::some_module::internal_module::nested_internal<caret>_module::PublicStruct {};
+    }
+}
+
+// Although the code itself is semantically invalid because of items' visibility, paths should be shown correctly.
+mod unhappy_cases {
+    use super::some_module::internal_module;
+    use super::some_module::internal_module2::private<caret>_submodule;
+
+    fn private_item {
+        let _ = super::some_module::internal<caret>_module::PrivateStruct {};
+        let _ = private_sub<caret>module::PrivateStruct2 {};
+    }
+}
+
+//! > hover #0
+// = source context
+mod some_<caret>module {
+// = highlight
+mod <sel>some_module</sel> {
+// = popover
+```cairo
+hello
+```
+```cairo
+mod some_module
+```
+---
+some_module docstring.
+
+//! > hover #1
+// = source context
+    pub mod internal_module<caret> {
+// = highlight
+    pub mod <sel>internal_module</sel> {
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #2
+// = source context
+        use super::internal_module<caret>;
+// = highlight
+        use super::<sel>internal_module</sel>;
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #3
+// = source context
+        mod <caret>private_submodule {
+// = highlight
+        mod <sel>private_submodule</sel> {
+// = popover
+```cairo
+hello::some_module::internal_module2
+```
+```cairo
+mod private_submodule
+```
+---
+private_submodule docstring.
+
+//! > hover #4
+// = source context
+    use super::some_<caret>module;
+// = highlight
+    use super::<sel>some_module</sel>;
+// = popover
+```cairo
+hello
+```
+```cairo
+mod some_module
+```
+---
+some_module docstring.
+
+//! > hover #5
+// = source context
+    use super::some_module::internal_<caret>module::MyTrait;
+// = highlight
+    use super::some_module::<sel>internal_module</sel>::MyTrait;
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #6
+// = source context
+    use super::some_module::internal_module::nested_inte<caret>rnal_module;
+// = highlight
+    use super::some_module::internal_module::<sel>nested_internal_module</sel>;
+// = popover
+```cairo
+hello::some_module::internal_module
+```
+```cairo
+mod nested_internal_module
+```
+
+//! > hover #7
+// = source context
+    impl TraitImpl of super::some_module::internal<caret>_module::MyTrait<felt252> {
+// = highlight
+    impl TraitImpl of super::some_module::<sel>internal_module</sel>::MyTrait<felt252> {
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #8
+// = source context
+    impl TraitImpl2 of internal_<caret>module::MyTrait<u8> {
+// = highlight
+    impl TraitImpl2 of <sel>internal_module</sel>::MyTrait<u8> {
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #9
+// = source context
+        <caret>super::some_module::internal_module::nested_internal_module::foo();
+// = highlight
+        <sel>super</sel>::some_module::internal_module::nested_internal_module::foo();
+// = popover
+```cairo
+hello::happy_cases
+```
+```cairo
+fn function_with_path()
+```
+
+//! > hover #10
+// = source context
+        super::some_module::inte<caret>rnal_module::nested_internal_module::foo();
+// = highlight
+        super::some_module::<sel>internal_module</sel>::nested_internal_module::foo();
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #11
+// = source context
+        super::some_module::internal_module::nested_internal<caret>_module::foo();
+// = highlight
+        super::some_module::internal_module::<sel>nested_internal_module</sel>::foo();
+// = popover
+```cairo
+hello::some_module::internal_module
+```
+```cairo
+mod nested_internal_module
+```
+
+//! > hover #12
+// = source context
+        nested_in<caret>ternal_module::foo();
+// = highlight
+        <sel>nested_internal_module</sel>::foo();
+// = popover
+```cairo
+hello::some_module::internal_module
+```
+```cairo
+mod nested_internal_module
+```
+
+//! > hover #13
+// = source context
+        internal_<caret>module::MyTraitImpl::foo(0_u32);
+// = highlight
+        <sel>internal_module</sel>::MyTraitImpl::foo(0_u32);
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #14
+// = source context
+        let _ = <caret>super::some_module::internal_module::nested_internal_module::PublicStruct {};
+// = highlight
+No highlight information.
+// = popover
+```cairo
+hello::some_module::internal_module::nested_internal_module::PublicStruct
+```
+
+//! > hover #15
+// = source context
+        let _ = super::some_module::inte<caret>rnal_module::nested_internal_module::PublicStruct {};
+// = highlight
+        let _ = super::some_module::<sel>internal_module</sel>::nested_internal_module::PublicStruct {};
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #16
+// = source context
+        let _ = super::some_module::internal_module::nested_internal<caret>_module::PublicStruct {};
+// = highlight
+        let _ = super::some_module::internal_module::<sel>nested_internal_module</sel>::PublicStruct {};
+// = popover
+```cairo
+hello::some_module::internal_module
+```
+```cairo
+mod nested_internal_module
+```
+
+//! > hover #17
+// = source context
+    use super::some_module::internal_module2::private<caret>_submodule;
+// = highlight
+    use super::some_module::internal_module2::<sel>private_submodule</sel>;
+// = popover
+```cairo
+hello::some_module::internal_module2
+```
+```cairo
+mod private_submodule
+```
+---
+private_submodule docstring.
+
+//! > hover #18
+// = source context
+        let _ = super::some_module::internal<caret>_module::PrivateStruct {};
+// = highlight
+        let _ = super::some_module::<sel>internal_module</sel>::PrivateStruct {};
+// = popover
+```cairo
+hello::some_module
+```
+```cairo
+mod internal_module
+```
+---
+internal_module docstring.
+
+//! > hover #19
+// = source context
+        let _ = private_sub<caret>module::PrivateStruct2 {};
+// = highlight
+        let _ = <sel>private_submodule</sel>::PrivateStruct2 {};
+// = popover
+```cairo
+hello::some_module::internal_module2
+```
+```cairo
+mod private_submodule
+```
+---
+private_submodule docstring.

EOF_114329324912
cd "crates/cairo-lang-language-server"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard 47fe5bf462ab3a388f2c03535846a8c3a7873337
git clean -fd
