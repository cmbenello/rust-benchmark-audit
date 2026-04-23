#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/docs.md b/docs.md
--- a/docs.md
+++ b/docs.md
@@ -235,6 +235,19 @@ An annotation may be a bool, string (no quotes), or list of strings. If just the
 
 Most annotations are just local overrides for identical settings in the cbindgen.toml, but a few are unique because they don't make sense in a global context. The set of supported annotation are as follows:
 
+### Ignore annotation
+
+cbindgen will automatically ignore any `#[test]` or `#[cfg(test)]` item it
+finds. You can manually ignore other stuff with the `ignore` annotation
+attribute:
+
+```rust
+pub mod my_interesting_mod;
+
+/// cbindgen:ignore
+pub mod my_uninteresting_mod; // This won't be scanned by cbindgen.
+```
+
 ### Struct Annotations
 
 * field-names=\[field1, field2, ...\] -- sets the names of all the fields in the output struct. These names will be output verbatim, and are not eligible for renaming.
diff --git a/src/bindgen/parser.rs b/src/bindgen/parser.rs
--- a/src/bindgen/parser.rs
+++ b/src/bindgen/parser.rs
@@ -202,41 +203,7 @@ impl<'a> Parser<'a> {
             self.cache_expanded_crate.get(&pkg.name).unwrap().clone()
         };
 
-        self.process_expanded_mod(pkg, &mod_parsed)
-    }
-
-    fn process_expanded_mod(&mut self, pkg: &PackageRef, items: &[syn::Item]) -> Result<(), Error> {
-        self.out.load_syn_crate_mod(
-            &self.config,
-            &self.binding_crate_name,
-            &pkg.name,
-            Cfg::join(&self.cfg_stack).as_ref(),
-            items,
-        );
-
-        for item in items {
-            if item.has_test_attr() {
-                continue;
-            }
-            if let syn::Item::Mod(ref item) = *item {
-                let cfg = Cfg::load(&item.attrs);
-                if let Some(ref cfg) = cfg {
-                    self.cfg_stack.push(cfg.clone());
-                }
-
-                if let Some((_, ref inline_items)) = item.content {
-                    self.process_expanded_mod(pkg, inline_items)?;
-                } else {
-                    unreachable!();
-                }
-
-                if cfg.is_some() {
-                    self.cfg_stack.pop();
-                }
-            }
-        }
-
-        Ok(())
+        self.process_mod(pkg, None, &mod_items, 0)
     }
 
     fn parse_mod(
diff --git a/src/bindgen/parser.rs b/src/bindgen/parser.rs
--- a/src/bindgen/parser.rs
+++ b/src/bindgen/parser.rs
@@ -303,36 +271,37 @@ impl<'a> Parser<'a> {
             items,
         );
 
-        for item in items {
-            if item.has_test_attr() {
-                continue;
-            }
-            if let syn::Item::Mod(ref item) = *item {
-                let next_mod_name = item.ident.to_string();
+        for item in nested_modules {
+            let next_mod_name = item.ident.to_string();
 
-                let cfg = Cfg::load(&item.attrs);
-                if let Some(ref cfg) = cfg {
-                    self.cfg_stack.push(cfg.clone());
-                }
+            let cfg = Cfg::load(&item.attrs);
+            if let Some(ref cfg) = cfg {
+                self.cfg_stack.push(cfg.clone());
+            }
 
-                if let Some((_, ref inline_items)) = item.content {
-                    self.process_mod(pkg, &mod_dir.join(&next_mod_name), inline_items, depth)?;
+            if let Some((_, ref inline_items)) = item.content {
+                let next_mod_dir = mod_dir.map(|dir| dir.join(&next_mod_name));
+                self.process_mod(
+                    pkg,
+                    next_mod_dir.as_ref().map(|p| &**p),
+                    inline_items,
+                    depth,
+                )?;
+            } else if let Some(mod_dir) = mod_dir {
+                let next_mod_path1 = mod_dir.join(next_mod_name.clone() + ".rs");
+                let next_mod_path2 = mod_dir.join(next_mod_name.clone()).join("mod.rs");
+
+                if next_mod_path1.exists() {
+                    self.parse_mod(pkg, next_mod_path1.as_path(), depth + 1)?;
+                } else if next_mod_path2.exists() {
+                    self.parse_mod(pkg, next_mod_path2.as_path(), depth + 1)?;
                 } else {
-                    let next_mod_path1 = mod_dir.join(next_mod_name.clone() + ".rs");
-                    let next_mod_path2 = mod_dir.join(next_mod_name.clone()).join("mod.rs");
-
-                    if next_mod_path1.exists() {
-                        self.parse_mod(pkg, next_mod_path1.as_path(), depth + 1)?;
-                    } else if next_mod_path2.exists() {
-                        self.parse_mod(pkg, next_mod_path2.as_path(), depth + 1)?;
-                    } else {
-                        // Last chance to find a module path
-                        let mut path_attr_found = false;
-                        for attr in &item.attrs {
-                            match attr.parse_meta() {
-                                Ok(syn::Meta::NameValue(syn::MetaNameValue {
-                                    path, lit, ..
-                                })) => match lit {
+                    // Last chance to find a module path
+                    let mut path_attr_found = false;
+                    for attr in &item.attrs {
+                        match attr.parse_meta() {
+                            Ok(syn::Meta::NameValue(syn::MetaNameValue { path, lit, .. })) => {
+                                match lit {
                                     syn::Lit::Str(ref path_lit) if path.is_ident("path") => {
                                         path_attr_found = true;
                                         self.parse_mod(
diff --git a/src/bindgen/parser.rs b/src/bindgen/parser.rs
--- a/src/bindgen/parser.rs
+++ b/src/bindgen/parser.rs
@@ -436,18 +410,19 @@ impl Parse {
         self.functions.extend_from_slice(&other.functions);
     }
 
-    pub fn load_syn_crate_mod(
+    fn load_syn_crate_mod<'a>(
         &mut self,
         config: &Config,
         binding_crate_name: &str,
         crate_name: &str,
         mod_cfg: Option<&Cfg>,
-        items: &[syn::Item],
-    ) {
+        items: &'a [syn::Item],
+    ) -> Vec<&'a syn::ItemMod> {
         let mut impls_with_assoc_consts = Vec::new();
+        let mut nested_modules = Vec::new();
 
         for item in items {
-            if item.has_test_attr() {
+            if item.should_skip_parsing() {
                 continue;
             }
             match item {
diff --git a/src/bindgen/utilities.rs b/src/bindgen/utilities.rs
--- a/src/bindgen/utilities.rs
+++ b/src/bindgen/utilities.rs
@@ -71,77 +71,162 @@ impl SynItemFnHelpers for syn::ImplItemMethod {
     }
 }
 
-pub trait SynItemHelpers {
+/// Returns whether this attribute causes us to skip at item. This basically
+/// checks for `#[cfg(test)]`, `#[test]`, `/// cbindgen::ignore` and
+/// variations thereof.
+fn is_skip_item_attr(attr: &syn::Meta) -> bool {
+    match *attr {
+        syn::Meta::Path(ref path) => {
+            // TODO(emilio): It'd be great if rustc allowed us to use a syntax
+            // like `#[cbindgen::ignore]` or such.
+            path.is_ident("test")
+        }
+        syn::Meta::List(ref list) => {
+            if !list.path.is_ident("cfg") {
+                return false;
+            }
+            list.nested.iter().any(|nested| match *nested {
+                syn::NestedMeta::Meta(ref meta) => {
+                    return is_skip_item_attr(meta);
+                }
+                syn::NestedMeta::Lit(..) => false,
+            })
+        }
+        syn::Meta::NameValue(ref name_value) => {
+            if name_value.path.is_ident("doc") {
+                if let syn::Lit::Str(ref content) = name_value.lit {
+                    // FIXME(emilio): Maybe should use the general annotation
+                    // mechanism, but it seems overkill for this.
+                    if content.value().trim() == "cbindgen:ignore" {
+                        return true;
+                    }
+                }
+            }
+            false
+        }
+    }
+}
+
+pub trait SynAttributeHelpers {
+    /// Returns the list of attributes for an item.
+    fn attrs(&self) -> &[syn::Attribute];
+
     /// Searches for attributes like `#[test]`.
     /// Example:
     /// - `item.has_attr_word("test")` => `#[test]`
-    fn has_attr_word(&self, name: &str) -> bool;
-
-    /// Searches for attributes like `#[cfg(test)]`.
-    /// Example:
-    /// - `item.has_attr_list("cfg", &["test"])` => `#[cfg(test)]`
-    fn has_attr_list(&self, name: &str, args: &[&str]) -> bool;
+    fn has_attr_word(&self, name: &str) -> bool {
+        self.attrs()
+            .iter()
+            .filter_map(|x| x.parse_meta().ok())
+            .any(|attr| {
+                if let syn::Meta::Path(ref path) = attr {
+                    path.is_ident(name)
+                } else {
+                    false
+                }
+            })
+    }
 
     fn is_no_mangle(&self) -> bool {
         self.has_attr_word("no_mangle")
     }
 
-    /// Searches for attributes `#[test]` and/or `#[cfg(test)]`.
-    fn has_test_attr(&self) -> bool {
-        self.has_attr_list("cfg", &["test"]) || self.has_attr_word("test")
+    /// Sees whether we should skip parsing a given item.
+    fn should_skip_parsing(&self) -> bool {
+        for attr in self.attrs() {
+            let meta = match attr.parse_meta() {
+                Ok(attr) => attr,
+                Err(..) => return false,
+            };
+            if is_skip_item_attr(&meta) {
+                return true;
+            }
+        }
+
+        false
+    }
+
+    fn attr_name_value_lookup(&self, name: &str) -> Option<String> {
+        self.attrs()
+            .iter()
+            .filter_map(|attr| {
+                let attr = attr.parse_meta().ok()?;
+                if let syn::Meta::NameValue(syn::MetaNameValue {
+                    path,
+                    lit: syn::Lit::Str(lit),
+                    ..
+                }) = attr
+                {
+                    if path.is_ident(name) {
+                        return Some(lit.value());
+                    }
+                }
+                None
+            })
+            .next()
+    }
+
+    fn get_comment_lines(&self) -> Vec<String> {
+        let mut comment = Vec::new();
+
+        for attr in self.attrs() {
+            if attr.style == syn::AttrStyle::Outer {
+                if let Ok(syn::Meta::NameValue(syn::MetaNameValue {
+                    path,
+                    lit: syn::Lit::Str(content),
+                    ..
+                })) = attr.parse_meta()
+                {
+                    if path.is_ident("doc") {
+                        comment.extend(split_doc_attr(&content.value()));
+                    }
+                }
+            }
+        }
+
+        comment
     }
 }
 
 macro_rules! syn_item_match_helper {
     ($s:ident => has_attrs: |$i:ident| $a:block, otherwise: || $b:block) => {
         match *$s {
-            syn::Item::Const(ref item) => (|$i: &syn::ItemConst| $a)(item),
-            syn::Item::Enum(ref item) => (|$i: &syn::ItemEnum| $a)(item),
-            syn::Item::ExternCrate(ref item) => (|$i: &syn::ItemExternCrate| $a)(item),
-            syn::Item::Fn(ref item) => (|$i: &syn::ItemFn| $a)(item),
-            syn::Item::ForeignMod(ref item) => (|$i: &syn::ItemForeignMod| $a)(item),
-            syn::Item::Impl(ref item) => (|$i: &syn::ItemImpl| $a)(item),
-            syn::Item::Macro(ref item) => (|$i: &syn::ItemMacro| $a)(item),
-            syn::Item::Macro2(ref item) => (|$i: &syn::ItemMacro2| $a)(item),
-            syn::Item::Mod(ref item) => (|$i: &syn::ItemMod| $a)(item),
-            syn::Item::Static(ref item) => (|$i: &syn::ItemStatic| $a)(item),
-            syn::Item::Struct(ref item) => (|$i: &syn::ItemStruct| $a)(item),
-            syn::Item::Trait(ref item) => (|$i: &syn::ItemTrait| $a)(item),
-            syn::Item::Type(ref item) => (|$i: &syn::ItemType| $a)(item),
-            syn::Item::Union(ref item) => (|$i: &syn::ItemUnion| $a)(item),
-            syn::Item::Use(ref item) => (|$i: &syn::ItemUse| $a)(item),
-            syn::Item::TraitAlias(ref item) => (|$i: &syn::ItemTraitAlias| $a)(item),
-            syn::Item::Verbatim(_) => (|| $b)(),
+            syn::Item::Const(ref $i) => $a,
+            syn::Item::Enum(ref $i) => $a,
+            syn::Item::ExternCrate(ref $i) => $a,
+            syn::Item::Fn(ref $i) => $a,
+            syn::Item::ForeignMod(ref $i) => $a,
+            syn::Item::Impl(ref $i) => $a,
+            syn::Item::Macro(ref $i) => $a,
+            syn::Item::Macro2(ref $i) => $a,
+            syn::Item::Mod(ref $i) => $a,
+            syn::Item::Static(ref $i) => $a,
+            syn::Item::Struct(ref $i) => $a,
+            syn::Item::Trait(ref $i) => $a,
+            syn::Item::Type(ref $i) => $a,
+            syn::Item::Union(ref $i) => $a,
+            syn::Item::Use(ref $i) => $a,
+            syn::Item::TraitAlias(ref $i) => $a,
+            syn::Item::Verbatim(_) => $b,
             _ => panic!("Unhandled syn::Item:  {:?}", $s),
         }
     };
 }
 
-impl SynItemHelpers for syn::Item {
-    fn has_attr_word(&self, name: &str) -> bool {
+impl SynAttributeHelpers for syn::Item {
+    fn attrs(&self) -> &[syn::Attribute] {
         syn_item_match_helper!(self =>
-            has_attrs: |item| { item.has_attr_word(name) },
-            otherwise: || { false }
-        )
-    }
-
-    fn has_attr_list(&self, name: &str, args: &[&str]) -> bool {
-        syn_item_match_helper!(self =>
-            has_attrs: |item| { item.has_attr_list(name, args) },
-            otherwise: || { false }
+            has_attrs: |item| { &item.attrs },
+            otherwise: || { &[] }
         )
     }
 }
 
 macro_rules! impl_syn_item_helper {
     ($t:ty) => {
-        impl SynItemHelpers for $t {
-            fn has_attr_word(&self, name: &str) -> bool {
-                self.attrs.has_attr_word(name)
-            }
-
-            fn has_attr_list(&self, name: &str, args: &[&str]) -> bool {
-                self.attrs.has_attr_list(name, args)
+        impl SynAttributeHelpers for $t {
+            fn attrs(&self) -> &[syn::Attribute] {
+                &self.attrs
             }
         }
     };
diff --git /dev/null b/tests/expectations/both/ignore.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/ignore.c
@@ -0,0 +1,6 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void no_ignore_root(void);
diff --git /dev/null b/tests/expectations/both/ignore.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/ignore.compat.c
@@ -0,0 +1,14 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void no_ignore_root(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/ignore.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/ignore.c
@@ -0,0 +1,6 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void no_ignore_root(void);
diff --git /dev/null b/tests/expectations/ignore.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/ignore.compat.c
@@ -0,0 +1,14 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void no_ignore_root(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/ignore.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/ignore.cpp
@@ -0,0 +1,10 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+extern "C" {
+
+void no_ignore_root();
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/tag/ignore.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/ignore.c
@@ -0,0 +1,6 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void no_ignore_root(void);
diff --git /dev/null b/tests/expectations/tag/ignore.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/ignore.compat.c
@@ -0,0 +1,14 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void no_ignore_root(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/ignore.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/ignore.rs
@@ -0,0 +1,12 @@
+/// cbindgen:ignore
+#[no_mangle]
+pub extern "C" fn root() {}
+
+/// cbindgen:ignore
+///
+/// Something else.
+#[no_mangle]
+pub extern "C" fn another_root() {}
+
+#[no_mangle]
+pub extern "C" fn no_ignore_root() {}

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard b6b88f8c3024288287368b377e4d928ddcd2b9e2
git clean -fd
