#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/bindgen/cargo/cargo_metadata.rs b/src/bindgen/cargo/cargo_metadata.rs
--- a/src/bindgen/cargo/cargo_metadata.rs
+++ b/src/bindgen/cargo/cargo_metadata.rs
@@ -24,23 +26,28 @@ use serde_json;
 /// Starting point for metadata returned by `cargo metadata`
 pub struct Metadata {
     /// A list of all crates referenced by this crate (and the crate itself)
-    pub packages: Vec<Package>,
+    pub packages: HashSet<Package>,
     version: usize,
     /// path to the workspace containing the `Cargo.lock`
     pub workspace_root: String,
 }
 
+/// A reference to a package including it's name and the specific version.
+#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
+pub struct PackageRef {
+    pub name: String,
+    pub version: String,
+}
+
 #[derive(Clone, Deserialize, Debug)]
 /// A crate
 pub struct Package {
-    /// Name as given in the `Cargo.toml`
-    pub name: String,
-    /// Version given in the `Cargo.toml`
-    pub version: String,
+    #[serde(flatten)]
+    pub name_and_version: PackageRef,
     id: String,
     source: Option<String>,
     /// List of dependencies of this particular package
-    pub dependencies: Vec<Dependency>,
+    pub dependencies: HashSet<Dependency>,
     /// Targets provided by the crate (lib, bin, example, test, ...)
     pub targets: Vec<Target>,
     features: HashMap<String, Vec<String>>,
diff --git a/tests/expectations/both/rename-crate.c b/tests/expectations/both/rename-crate.c
--- a/tests/expectations/both/rename-crate.c
+++ b/tests/expectations/both/rename-crate.c
@@ -3,6 +3,24 @@
 #include <stdint.h>
 #include <stdlib.h>
 
+#if !defined(DEFINE_FREEBSD)
+typedef struct NoExternTy {
+  uint8_t field;
+} NoExternTy;
+#endif
+
+#if !defined(DEFINE_FREEBSD)
+typedef struct ContainsNoExternTy {
+  NoExternTy field;
+} ContainsNoExternTy;
+#endif
+
+#if defined(DEFINE_FREEBSD)
+typedef struct ContainsNoExternTy {
+  uint64_t field;
+} ContainsNoExternTy;
+#endif
+
 typedef struct RenamedTy {
   uint64_t y;
 } RenamedTy;
diff --git a/tests/expectations/both/rename-crate.c b/tests/expectations/both/rename-crate.c
--- a/tests/expectations/both/rename-crate.c
+++ b/tests/expectations/both/rename-crate.c
@@ -11,6 +29,8 @@ typedef struct Foo {
   int32_t x;
 } Foo;
 
+void no_extern_func(ContainsNoExternTy a);
+
 void renamed_func(RenamedTy a);
 
 void root(Foo a);
diff --git a/tests/expectations/rename-crate.c b/tests/expectations/rename-crate.c
--- a/tests/expectations/rename-crate.c
+++ b/tests/expectations/rename-crate.c
@@ -3,6 +3,24 @@
 #include <stdint.h>
 #include <stdlib.h>
 
+#if !defined(DEFINE_FREEBSD)
+typedef struct {
+  uint8_t field;
+} NoExternTy;
+#endif
+
+#if !defined(DEFINE_FREEBSD)
+typedef struct {
+  NoExternTy field;
+} ContainsNoExternTy;
+#endif
+
+#if defined(DEFINE_FREEBSD)
+typedef struct {
+  uint64_t field;
+} ContainsNoExternTy;
+#endif
+
 typedef struct {
   uint64_t y;
 } RenamedTy;
diff --git a/tests/expectations/rename-crate.c b/tests/expectations/rename-crate.c
--- a/tests/expectations/rename-crate.c
+++ b/tests/expectations/rename-crate.c
@@ -11,6 +29,8 @@ typedef struct {
   int32_t x;
 } Foo;
 
+void no_extern_func(ContainsNoExternTy a);
+
 void renamed_func(RenamedTy a);
 
 void root(Foo a);
diff --git a/tests/expectations/rename-crate.cpp b/tests/expectations/rename-crate.cpp
--- a/tests/expectations/rename-crate.cpp
+++ b/tests/expectations/rename-crate.cpp
@@ -2,6 +2,24 @@
 #include <cstdint>
 #include <cstdlib>
 
+#if !defined(DEFINE_FREEBSD)
+struct NoExternTy {
+  uint8_t field;
+};
+#endif
+
+#if !defined(DEFINE_FREEBSD)
+struct ContainsNoExternTy {
+  NoExternTy field;
+};
+#endif
+
+#if defined(DEFINE_FREEBSD)
+struct ContainsNoExternTy {
+  uint64_t field;
+};
+#endif
+
 struct RenamedTy {
   uint64_t y;
 };
diff --git a/tests/expectations/rename-crate.cpp b/tests/expectations/rename-crate.cpp
--- a/tests/expectations/rename-crate.cpp
+++ b/tests/expectations/rename-crate.cpp
@@ -12,6 +30,8 @@ struct Foo {
 
 extern "C" {
 
+void no_extern_func(ContainsNoExternTy a);
+
 void renamed_func(RenamedTy a);
 
 void root(Foo a);
diff --git a/tests/expectations/tag/rename-crate.c b/tests/expectations/tag/rename-crate.c
--- a/tests/expectations/tag/rename-crate.c
+++ b/tests/expectations/tag/rename-crate.c
@@ -3,6 +3,24 @@
 #include <stdint.h>
 #include <stdlib.h>
 
+#if !defined(DEFINE_FREEBSD)
+struct NoExternTy {
+  uint8_t field;
+};
+#endif
+
+#if !defined(DEFINE_FREEBSD)
+struct ContainsNoExternTy {
+  struct NoExternTy field;
+};
+#endif
+
+#if defined(DEFINE_FREEBSD)
+struct ContainsNoExternTy {
+  uint64_t field;
+};
+#endif
+
 struct RenamedTy {
   uint64_t y;
 };
diff --git a/tests/expectations/tag/rename-crate.c b/tests/expectations/tag/rename-crate.c
--- a/tests/expectations/tag/rename-crate.c
+++ b/tests/expectations/tag/rename-crate.c
@@ -11,6 +29,8 @@ struct Foo {
   int32_t x;
 };
 
+void no_extern_func(struct ContainsNoExternTy a);
+
 void renamed_func(struct RenamedTy a);
 
 void root(struct Foo a);
diff --git a/tests/rust/rename-crate/Cargo.lock b/tests/rust/rename-crate/Cargo.lock
--- a/tests/rust/rename-crate/Cargo.lock
+++ b/tests/rust/rename-crate/Cargo.lock
@@ -4,9 +4,16 @@
 name = "dependency"
 version = "0.1.0"
 
+[[package]]
+name = "no-extern"
+version = "0.1.0"
+
 [[package]]
 name = "old-dep-name"
 version = "0.1.0"
+dependencies = [
+ "no-extern 0.1.0",
+]
 
 [[package]]
 name = "rename-crate"
diff --git a/tests/rust/rename-crate/cbindgen.toml b/tests/rust/rename-crate/cbindgen.toml
--- a/tests/rust/rename-crate/cbindgen.toml
+++ b/tests/rust/rename-crate/cbindgen.toml
@@ -1,2 +1,4 @@
 [parse]
 parse_deps = true
+[defines]
+"target_os = freebsd" = "DEFINE_FREEBSD"
diff --git /dev/null b/tests/rust/rename-crate/no-extern/Cargo.lock
new file mode 100644
--- /dev/null
+++ b/tests/rust/rename-crate/no-extern/Cargo.lock
@@ -0,0 +1,6 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
+[[package]]
+name = "no-extern"
+version = "0.1.0"
+
diff --git /dev/null b/tests/rust/rename-crate/no-extern/Cargo.toml
new file mode 100644
--- /dev/null
+++ b/tests/rust/rename-crate/no-extern/Cargo.toml
@@ -0,0 +1,7 @@
+[package]
+name = "no-extern"
+version = "0.1.0"
+authors = ["cbindgen"]
+edition = "2018"
+
+[dependencies]
diff --git /dev/null b/tests/rust/rename-crate/no-extern/src/lib.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/rename-crate/no-extern/src/lib.rs
@@ -0,0 +1,4 @@
+#[repr(C)]
+pub struct NoExternTy {
+    field: u8,
+}
diff --git a/tests/rust/rename-crate/old-dep/Cargo.lock b/tests/rust/rename-crate/old-dep/Cargo.lock
--- a/tests/rust/rename-crate/old-dep/Cargo.lock
+++ b/tests/rust/rename-crate/old-dep/Cargo.lock
@@ -1,6 +1,13 @@
 # This file is automatically @generated by Cargo.
 # It is not intended for manual editing.
 [[package]]
-name = "old-dep"
+name = "no-extern"
 version = "0.1.0"
 
+[[package]]
+name = "old-dep-name"
+version = "0.1.0"
+dependencies = [
+ "no-extern 0.1.0",
+]
+
diff --git a/tests/rust/rename-crate/old-dep/Cargo.toml b/tests/rust/rename-crate/old-dep/Cargo.toml
--- a/tests/rust/rename-crate/old-dep/Cargo.toml
+++ b/tests/rust/rename-crate/old-dep/Cargo.toml
@@ -4,4 +4,5 @@ version = "0.1.0"
 authors = ["cbindgen"]
 edition = "2018"
 
-[dependencies]
+[target.'cfg(not(target_os = "freebsd"))'.dependencies]
+no-extern = { path = "../no-extern/" }
diff --git a/tests/rust/rename-crate/old-dep/src/lib.rs b/tests/rust/rename-crate/old-dep/src/lib.rs
--- a/tests/rust/rename-crate/old-dep/src/lib.rs
+++ b/tests/rust/rename-crate/old-dep/src/lib.rs
@@ -2,3 +2,15 @@
 pub struct RenamedTy {
     y: u64,
 }
+
+#[cfg(not(target_os = "freebsd"))]
+#[repr(C)]
+pub struct ContainsNoExternTy {
+    pub field: no_extern::NoExternTy,
+}
+
+#[cfg(target_os = "freebsd")]
+#[repr(C)]
+pub struct ContainsNoExternTy {
+    pub field: u64,
+}
diff --git a/tests/rust/rename-crate/src/lib.rs b/tests/rust/rename-crate/src/lib.rs
--- a/tests/rust/rename-crate/src/lib.rs
+++ b/tests/rust/rename-crate/src/lib.rs
@@ -1,3 +1,5 @@
+#![allow(unused_variables)]
+
 extern crate dependency as internal_name;
 extern crate renamed_dep;
 
diff --git a/tests/rust/rename-crate/src/lib.rs b/tests/rust/rename-crate/src/lib.rs
--- a/tests/rust/rename-crate/src/lib.rs
+++ b/tests/rust/rename-crate/src/lib.rs
@@ -11,3 +13,8 @@ pub extern "C" fn root(a: Foo) {
 #[no_mangle]
 pub extern "C" fn renamed_func(a: RenamedTy) {
 }
+
+
+#[no_mangle]
+pub extern "C" fn no_extern_func(a: ContainsNoExternTy) {
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
cd "tests/rust/rename-crate/old-dep"
cargo test --no-fail-fast --all-features
cd ../../../../
cd "tests/rust/rename-crate"
cargo test --no-fail-fast --all-features
cd ../../../
git reset --hard 46aed0802ae6b3e766dfb3f36680221c29f9d2fc
git clean -fd
