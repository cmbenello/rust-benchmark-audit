#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/.travis.yml b/.travis.yml
--- a/.travis.yml
+++ b/.travis.yml
@@ -9,11 +9,11 @@ addons:
       sources:
         - ubuntu-toolchain-r-test
       packages:
-        - gcc-4.9
-        - g++-4.9
-env:
-  - MATRIX_EVAL="CC=gcc-4.9 && CXX=g++-4.9"
+        - gcc-7
+        - g++-7
 script:
+  - export CC=gcc-7
+  - export CXX=g++-7
   - cargo fmt --all -- --check
   - cargo build --verbose
   - cargo test --verbose
diff --git a/src/bindgen/ir/constant.rs b/src/bindgen/ir/constant.rs
--- a/src/bindgen/ir/constant.rs
+++ b/src/bindgen/ir/constant.rs
@@ -285,20 +297,92 @@ impl Item for Constant {
     }
 }
 
-impl Source for Constant {
-    fn write<F: Write>(&self, config: &Config, out: &mut SourceWriter<F>) {
+impl Constant {
+    pub fn write_declaration<F: Write>(
+        &self,
+        config: &Config,
+        out: &mut SourceWriter<F>,
+        associated_to_struct: &Struct,
+    ) {
+        debug_assert!(self.associated_to.is_some());
+        debug_assert!(config.language == Language::Cxx);
+        debug_assert!(!associated_to_struct.is_transparent);
+        debug_assert!(config.structure.associated_constants_in_body);
+        debug_assert!(config.constant.allow_static_const);
+
+        if let Type::ConstPtr(..) = self.ty {
+            out.write("static ");
+        } else {
+            out.write("static const ");
+        }
+        self.ty.write(config, out);
+        write!(out, " {};", self.export_name())
+    }
+
+    pub fn write<F: Write>(
+        &self,
+        config: &Config,
+        out: &mut SourceWriter<F>,
+        associated_to_struct: Option<&Struct>,
+    ) {
+        if let Some(assoc) = associated_to_struct {
+            if assoc.is_generic() {
+                return; // Not tested / implemented yet, so bail out.
+            }
+        }
+
+        let associated_to_transparent = associated_to_struct.map_or(false, |s| s.is_transparent);
+
+        let in_body = associated_to_struct.is_some()
+            && config.language == Language::Cxx
+            && config.structure.associated_constants_in_body
+            && config.constant.allow_static_const
+            && !associated_to_transparent;
+
         let condition = (&self.cfg).to_condition(config);
         condition.write_before(config, out);
+
+        let name = if in_body {
+            Cow::Owned(format!(
+                "{}::{}",
+                associated_to_struct.unwrap().export_name(),
+                self.export_name(),
+            ))
+        } else if self.associated_to.is_none() {
+            Cow::Borrowed(self.export_name())
+        } else {
+            let associated_name = match associated_to_struct {
+                Some(s) => Cow::Borrowed(s.export_name()),
+                None => {
+                    let mut name = self.associated_to.as_ref().unwrap().name().to_owned();
+                    config.export.rename(&mut name);
+                    Cow::Owned(name)
+                }
+            };
+
+            Cow::Owned(format!("{}_{}", associated_name, self.export_name()))
+        };
+
+        let value = match self.value {
+            Literal::Struct {
+                ref fields,
+                ref path,
+                ..
+            } if out.bindings().struct_is_transparent(path) => &fields[0].1,
+            _ => &self.value,
+        };
+
         if config.constant.allow_static_const && config.language == Language::Cxx {
+            out.write(if in_body { "inline " } else { "static " });
             if let Type::ConstPtr(..) = self.ty {
-                out.write("static ");
+                // Nothing.
             } else {
-                out.write("static const ");
+                out.write("const ");
             }
             self.ty.write(config, out);
-            write!(out, " {} = {};", self.export_name(), self.value)
+            write!(out, " {} = {};", name, value)
         } else {
-            write!(out, "#define {} {}", self.export_name(), self.value)
+            write!(out, "#define {} {}", name, value)
         }
         condition.write_after(config, out);
     }
diff --git a/src/bindgen/parser.rs b/src/bindgen/parser.rs
--- a/src/bindgen/parser.rs
+++ b/src/bindgen/parser.rs
@@ -210,8 +211,8 @@ impl Parser {
             if item.has_test_attr() {
                 continue;
             }
-            match item {
-                &syn::Item::Mod(ref item) => {
+            match *item {
+                syn::Item::Mod(ref item) => {
                     let cfg = Cfg::load(&item.attrs);
                     if let &Some(ref cfg) = &cfg {
                         self.cfg_stack.push(cfg.clone());
diff --git a/src/bindgen/parser.rs b/src/bindgen/parser.rs
--- a/src/bindgen/parser.rs
+++ b/src/bindgen/parser.rs
@@ -319,8 +320,8 @@ impl Parser {
             if item.has_test_attr() {
                 continue;
             }
-            match item {
-                &syn::Item::Mod(ref item) => {
+            match *item {
+                syn::Item::Mod(ref item) => {
                     let next_mod_name = item.ident.to_string();
 
                     let cfg = Cfg::load(&item.attrs);
diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -48,7 +48,7 @@ def gxx(src):
     if gxx_bin == None:
         gxx_bin = 'g++'
 
-    subprocess.check_output([gxx_bin, "-D", "DEFINED", "-std=c++11", "-c", src, "-o", "tests/expectations/tmp.o"])
+    subprocess.check_output([gxx_bin, "-D", "DEFINED", "-std=c++17", "-c", src, "-o", "tests/expectations/tmp.o"])
     os.remove("tests/expectations/tmp.o")
 
 def run_compile_test(rust_src, verify, c, style=""):
diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -129,3 +129,5 @@ def run_compile_test(rust_src, verify, c, style=""):
         print("Fail - %s" % test)
 
 print("Tests complete. %i passed, %i failed." % (num_pass, num_fail))
+if num_fail > 0:
+    sys.exit(1)
diff --git a/tests/expectations/assoc_constant.c b/tests/expectations/assoc_constant.c
--- a/tests/expectations/assoc_constant.c
+++ b/tests/expectations/assoc_constant.c
@@ -3,12 +3,10 @@
 #include <stdint.h>
 #include <stdlib.h>
 
-#define Foo_GA 10
-
-#define Foo_ZO 3.14
-
 typedef struct {
 
 } Foo;
+#define Foo_GA 10
+#define Foo_ZO 3.14
 
 void root(Foo x);
diff --git a/tests/expectations/assoc_constant.cpp b/tests/expectations/assoc_constant.cpp
--- a/tests/expectations/assoc_constant.cpp
+++ b/tests/expectations/assoc_constant.cpp
@@ -2,13 +2,11 @@
 #include <cstdint>
 #include <cstdlib>
 
-static const int32_t Foo_GA = 10;
-
-static const float Foo_ZO = 3.14;
-
 struct Foo {
 
 };
+static const int32_t Foo_GA = 10;
+static const float Foo_ZO = 3.14;
 
 extern "C" {
 
diff --git /dev/null b/tests/expectations/associated_in_body.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/associated_in_body.c
@@ -0,0 +1,19 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ * Constants shared by multiple CSS Box Alignment properties
+ * These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+ */
+typedef struct {
+  uint8_t bits;
+} StyleAlignFlags;
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = 1 << 1 }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = 1 << 2 }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = 1 << 3 }
+
+void root(StyleAlignFlags flags);
diff --git /dev/null b/tests/expectations/associated_in_body.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/associated_in_body.cpp
@@ -0,0 +1,25 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+
+/// Constants shared by multiple CSS Box Alignment properties
+/// These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+struct StyleAlignFlags {
+  uint8_t bits;
+  static const StyleAlignFlags AUTO;
+  static const StyleAlignFlags NORMAL;
+  static const StyleAlignFlags START;
+  static const StyleAlignFlags END;
+  static const StyleAlignFlags FLEX_START;
+};
+inline const StyleAlignFlags StyleAlignFlags::AUTO = (StyleAlignFlags){ .bits = 0 };
+inline const StyleAlignFlags StyleAlignFlags::NORMAL = (StyleAlignFlags){ .bits = 1 };
+inline const StyleAlignFlags StyleAlignFlags::START = (StyleAlignFlags){ .bits = 1 << 1 };
+inline const StyleAlignFlags StyleAlignFlags::END = (StyleAlignFlags){ .bits = 1 << 2 };
+inline const StyleAlignFlags StyleAlignFlags::FLEX_START = (StyleAlignFlags){ .bits = 1 << 3 };
+
+extern "C" {
+
+void root(StyleAlignFlags flags);
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/bitflags.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/bitflags.c
@@ -0,0 +1,19 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ * Constants shared by multiple CSS Box Alignment properties
+ * These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+ */
+typedef struct {
+  uint8_t bits;
+} AlignFlags;
+#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_START (AlignFlags){ .bits = 1 << 1 }
+#define AlignFlags_END (AlignFlags){ .bits = 1 << 2 }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = 1 << 3 }
+
+void root(AlignFlags flags);
diff --git /dev/null b/tests/expectations/bitflags.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/bitflags.cpp
@@ -0,0 +1,20 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+
+/// Constants shared by multiple CSS Box Alignment properties
+/// These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+struct AlignFlags {
+  uint8_t bits;
+};
+static const AlignFlags AlignFlags_AUTO = (AlignFlags){ .bits = 0 };
+static const AlignFlags AlignFlags_NORMAL = (AlignFlags){ .bits = 1 };
+static const AlignFlags AlignFlags_START = (AlignFlags){ .bits = 1 << 1 };
+static const AlignFlags AlignFlags_END = (AlignFlags){ .bits = 1 << 2 };
+static const AlignFlags AlignFlags_FLEX_START = (AlignFlags){ .bits = 1 << 3 };
+
+extern "C" {
+
+void root(AlignFlags flags);
+
+} // extern "C"
diff --git a/tests/expectations/both/assoc_constant.c b/tests/expectations/both/assoc_constant.c
--- a/tests/expectations/both/assoc_constant.c
+++ b/tests/expectations/both/assoc_constant.c
@@ -3,12 +3,10 @@
 #include <stdint.h>
 #include <stdlib.h>
 
-#define Foo_GA 10
-
-#define Foo_ZO 3.14
-
 typedef struct Foo {
 
 } Foo;
+#define Foo_GA 10
+#define Foo_ZO 3.14
 
 void root(Foo x);
diff --git /dev/null b/tests/expectations/both/associated_in_body.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/associated_in_body.c
@@ -0,0 +1,19 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ * Constants shared by multiple CSS Box Alignment properties
+ * These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+ */
+typedef struct StyleAlignFlags {
+  uint8_t bits;
+} StyleAlignFlags;
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = 1 << 1 }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = 1 << 2 }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = 1 << 3 }
+
+void root(StyleAlignFlags flags);
diff --git /dev/null b/tests/expectations/both/bitflags.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/bitflags.c
@@ -0,0 +1,19 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ * Constants shared by multiple CSS Box Alignment properties
+ * These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+ */
+typedef struct AlignFlags {
+  uint8_t bits;
+} AlignFlags;
+#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_START (AlignFlags){ .bits = 1 << 1 }
+#define AlignFlags_END (AlignFlags){ .bits = 1 << 2 }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = 1 << 3 }
+
+void root(AlignFlags flags);
diff --git /dev/null b/tests/expectations/both/const_transparent.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/const_transparent.c
@@ -0,0 +1,8 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef uint8_t Transparent;
+
+#define FOO 0
diff --git a/tests/expectations/both/prefixed_struct_literal.c b/tests/expectations/both/prefixed_struct_literal.c
--- a/tests/expectations/both/prefixed_struct_literal.c
+++ b/tests/expectations/both/prefixed_struct_literal.c
@@ -7,9 +7,8 @@ typedef struct PREFIXFoo {
   int32_t a;
   uint32_t b;
 } PREFIXFoo;
+#define PREFIXFoo_FOO (PREFIXFoo){ .a = 42, .b = 47 }
 
 #define PREFIXBAR (PREFIXFoo){ .a = 42, .b = 1337 }
 
-#define PREFIXFoo_FOO (PREFIXFoo){ .a = 42, .b = 47 }
-
 void root(PREFIXFoo x);
diff --git a/tests/expectations/both/struct_literal.c b/tests/expectations/both/struct_literal.c
--- a/tests/expectations/both/struct_literal.c
+++ b/tests/expectations/both/struct_literal.c
@@ -7,9 +7,8 @@ typedef struct Foo {
   int32_t a;
   uint32_t b;
 } Foo;
+#define Foo_FOO (Foo){ .a = 42, .b = 47 }
 
 #define BAR (Foo){ .a = 42, .b = 1337 }
 
-#define Foo_FOO (Foo){ .a = 42, .b = 47 }
-
 void root(Foo x);
diff --git a/tests/expectations/both/transparent.c b/tests/expectations/both/transparent.c
--- a/tests/expectations/both/transparent.c
+++ b/tests/expectations/both/transparent.c
@@ -20,12 +20,10 @@ typedef DummyStruct TransparentComplexWrapper_i32;
 typedef uint32_t TransparentPrimitiveWrapper_i32;
 
 typedef uint32_t TransparentPrimitiveWithAssociatedConstants;
-
-#define EnumWithAssociatedConstantInImpl_TEN 10
-
+#define TransparentPrimitiveWithAssociatedConstants_ZERO 0
 #define TransparentPrimitiveWithAssociatedConstants_ONE 1
 
-#define TransparentPrimitiveWithAssociatedConstants_ZERO 0
+#define EnumWithAssociatedConstantInImpl_TEN 10
 
 void root(TransparentComplexWrappingStructTuple a,
           TransparentPrimitiveWrappingStructTuple b,
diff --git /dev/null b/tests/expectations/const_transparent.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/const_transparent.c
@@ -0,0 +1,8 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef uint8_t Transparent;
+
+#define FOO 0
diff --git /dev/null b/tests/expectations/const_transparent.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/const_transparent.cpp
@@ -0,0 +1,7 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+
+using Transparent = uint8_t;
+
+static const Transparent FOO = 0;
diff --git a/tests/expectations/prefixed_struct_literal.c b/tests/expectations/prefixed_struct_literal.c
--- a/tests/expectations/prefixed_struct_literal.c
+++ b/tests/expectations/prefixed_struct_literal.c
@@ -7,9 +7,8 @@ typedef struct {
   int32_t a;
   uint32_t b;
 } PREFIXFoo;
+#define PREFIXFoo_FOO (PREFIXFoo){ .a = 42, .b = 47 }
 
 #define PREFIXBAR (PREFIXFoo){ .a = 42, .b = 1337 }
 
-#define PREFIXFoo_FOO (PREFIXFoo){ .a = 42, .b = 47 }
-
 void root(PREFIXFoo x);
diff --git a/tests/expectations/prefixed_struct_literal.cpp b/tests/expectations/prefixed_struct_literal.cpp
--- a/tests/expectations/prefixed_struct_literal.cpp
+++ b/tests/expectations/prefixed_struct_literal.cpp
@@ -6,11 +6,10 @@ struct PREFIXFoo {
   int32_t a;
   uint32_t b;
 };
+static const PREFIXFoo PREFIXFoo_FOO = (PREFIXFoo){ .a = 42, .b = 47 };
 
 static const PREFIXFoo PREFIXBAR = (PREFIXFoo){ .a = 42, .b = 1337 };
 
-static const PREFIXFoo PREFIXFoo_FOO = (PREFIXFoo){ .a = 42, .b = 47 };
-
 extern "C" {
 
 void root(PREFIXFoo x);
diff --git a/tests/expectations/struct_literal.c b/tests/expectations/struct_literal.c
--- a/tests/expectations/struct_literal.c
+++ b/tests/expectations/struct_literal.c
@@ -7,9 +7,8 @@ typedef struct {
   int32_t a;
   uint32_t b;
 } Foo;
+#define Foo_FOO (Foo){ .a = 42, .b = 47 }
 
 #define BAR (Foo){ .a = 42, .b = 1337 }
 
-#define Foo_FOO (Foo){ .a = 42, .b = 47 }
-
 void root(Foo x);
diff --git a/tests/expectations/struct_literal.cpp b/tests/expectations/struct_literal.cpp
--- a/tests/expectations/struct_literal.cpp
+++ b/tests/expectations/struct_literal.cpp
@@ -6,11 +6,10 @@ struct Foo {
   int32_t a;
   uint32_t b;
 };
+static const Foo Foo_FOO = (Foo){ .a = 42, .b = 47 };
 
 static const Foo BAR = (Foo){ .a = 42, .b = 1337 };
 
-static const Foo Foo_FOO = (Foo){ .a = 42, .b = 47 };
-
 extern "C" {
 
 void root(Foo x);
diff --git a/tests/expectations/tag/assoc_constant.c b/tests/expectations/tag/assoc_constant.c
--- a/tests/expectations/tag/assoc_constant.c
+++ b/tests/expectations/tag/assoc_constant.c
@@ -3,12 +3,10 @@
 #include <stdint.h>
 #include <stdlib.h>
 
-#define Foo_GA 10
-
-#define Foo_ZO 3.14
-
 struct Foo {
 
 };
+#define Foo_GA 10
+#define Foo_ZO 3.14
 
 void root(struct Foo x);
diff --git /dev/null b/tests/expectations/tag/associated_in_body.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/associated_in_body.c
@@ -0,0 +1,19 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ * Constants shared by multiple CSS Box Alignment properties
+ * These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+ */
+struct StyleAlignFlags {
+  uint8_t bits;
+};
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = 1 << 1 }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = 1 << 2 }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = 1 << 3 }
+
+void root(struct StyleAlignFlags flags);
diff --git /dev/null b/tests/expectations/tag/bitflags.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/bitflags.c
@@ -0,0 +1,19 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ * Constants shared by multiple CSS Box Alignment properties
+ * These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+ */
+struct AlignFlags {
+  uint8_t bits;
+};
+#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_START (AlignFlags){ .bits = 1 << 1 }
+#define AlignFlags_END (AlignFlags){ .bits = 1 << 2 }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = 1 << 3 }
+
+void root(struct AlignFlags flags);
diff --git /dev/null b/tests/expectations/tag/const_transparent.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/const_transparent.c
@@ -0,0 +1,8 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef uint8_t Transparent;
+
+#define FOO 0
diff --git a/tests/expectations/tag/prefixed_struct_literal.c b/tests/expectations/tag/prefixed_struct_literal.c
--- a/tests/expectations/tag/prefixed_struct_literal.c
+++ b/tests/expectations/tag/prefixed_struct_literal.c
@@ -7,9 +7,8 @@ struct PREFIXFoo {
   int32_t a;
   uint32_t b;
 };
+#define PREFIXFoo_FOO (PREFIXFoo){ .a = 42, .b = 47 }
 
 #define PREFIXBAR (PREFIXFoo){ .a = 42, .b = 1337 }
 
-#define PREFIXFoo_FOO (PREFIXFoo){ .a = 42, .b = 47 }
-
 void root(struct PREFIXFoo x);
diff --git a/tests/expectations/tag/struct_literal.c b/tests/expectations/tag/struct_literal.c
--- a/tests/expectations/tag/struct_literal.c
+++ b/tests/expectations/tag/struct_literal.c
@@ -7,9 +7,8 @@ struct Foo {
   int32_t a;
   uint32_t b;
 };
+#define Foo_FOO (Foo){ .a = 42, .b = 47 }
 
 #define BAR (Foo){ .a = 42, .b = 1337 }
 
-#define Foo_FOO (Foo){ .a = 42, .b = 47 }
-
 void root(struct Foo x);
diff --git a/tests/expectations/tag/transparent.c b/tests/expectations/tag/transparent.c
--- a/tests/expectations/tag/transparent.c
+++ b/tests/expectations/tag/transparent.c
@@ -20,12 +20,10 @@ typedef struct DummyStruct TransparentComplexWrapper_i32;
 typedef uint32_t TransparentPrimitiveWrapper_i32;
 
 typedef uint32_t TransparentPrimitiveWithAssociatedConstants;
-
-#define EnumWithAssociatedConstantInImpl_TEN 10
-
+#define TransparentPrimitiveWithAssociatedConstants_ZERO 0
 #define TransparentPrimitiveWithAssociatedConstants_ONE 1
 
-#define TransparentPrimitiveWithAssociatedConstants_ZERO 0
+#define EnumWithAssociatedConstantInImpl_TEN 10
 
 void root(TransparentComplexWrappingStructTuple a,
           TransparentPrimitiveWrappingStructTuple b,
diff --git a/tests/expectations/transparent.c b/tests/expectations/transparent.c
--- a/tests/expectations/transparent.c
+++ b/tests/expectations/transparent.c
@@ -20,12 +20,10 @@ typedef DummyStruct TransparentComplexWrapper_i32;
 typedef uint32_t TransparentPrimitiveWrapper_i32;
 
 typedef uint32_t TransparentPrimitiveWithAssociatedConstants;
-
-#define EnumWithAssociatedConstantInImpl_TEN 10
-
+#define TransparentPrimitiveWithAssociatedConstants_ZERO 0
 #define TransparentPrimitiveWithAssociatedConstants_ONE 1
 
-#define TransparentPrimitiveWithAssociatedConstants_ZERO 0
+#define EnumWithAssociatedConstantInImpl_TEN 10
 
 void root(TransparentComplexWrappingStructTuple a,
           TransparentPrimitiveWrappingStructTuple b,
diff --git a/tests/expectations/transparent.cpp b/tests/expectations/transparent.cpp
--- a/tests/expectations/transparent.cpp
+++ b/tests/expectations/transparent.cpp
@@ -21,12 +21,10 @@ template<typename T>
 using TransparentPrimitiveWrapper = uint32_t;
 
 using TransparentPrimitiveWithAssociatedConstants = uint32_t;
-
-static const TransparentPrimitiveWrappingStructure EnumWithAssociatedConstantInImpl_TEN = 10;
-
+static const TransparentPrimitiveWithAssociatedConstants TransparentPrimitiveWithAssociatedConstants_ZERO = 0;
 static const TransparentPrimitiveWithAssociatedConstants TransparentPrimitiveWithAssociatedConstants_ONE = 1;
 
-static const TransparentPrimitiveWithAssociatedConstants TransparentPrimitiveWithAssociatedConstants_ZERO = 0;
+static const TransparentPrimitiveWrappingStructure EnumWithAssociatedConstantInImpl_TEN = 10;
 
 extern "C" {
 
diff --git /dev/null b/tests/rust/associated_in_body.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/associated_in_body.rs
@@ -0,0 +1,22 @@
+bitflags! {
+    /// Constants shared by multiple CSS Box Alignment properties
+    ///
+    /// These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+    #[derive(MallocSizeOf, ToComputedValue)]
+    #[repr(C)]
+    pub struct AlignFlags: u8 {
+        /// 'auto'
+        const AUTO = 0;
+        /// 'normal'
+        const NORMAL = 1;
+        /// 'start'
+        const START = 1 << 1;
+        /// 'end'
+        const END = 1 << 2;
+        /// 'flex-start'
+        const FLEX_START = 1 << 3;
+    }
+}
+
+#[no_mangle]
+pub extern "C" fn root(flags: AlignFlags) {}
diff --git /dev/null b/tests/rust/associated_in_body.toml
new file mode 100644
--- /dev/null
+++ b/tests/rust/associated_in_body.toml
@@ -0,0 +1,5 @@
+[struct]
+associated_constants_in_body = true
+
+[export]
+prefix = "Style" # Just ensuring they play well together :)
diff --git /dev/null b/tests/rust/bitflags.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/bitflags.rs
@@ -0,0 +1,22 @@
+bitflags! {
+    /// Constants shared by multiple CSS Box Alignment properties
+    ///
+    /// These constants match Gecko's `NS_STYLE_ALIGN_*` constants.
+    #[derive(MallocSizeOf, ToComputedValue)]
+    #[repr(C)]
+    pub struct AlignFlags: u8 {
+        /// 'auto'
+        const AUTO = 0;
+        /// 'normal'
+        const NORMAL = 1;
+        /// 'start'
+        const START = 1 << 1;
+        /// 'end'
+        const END = 1 << 2;
+        /// 'flex-start'
+        const FLEX_START = 1 << 3;
+    }
+}
+
+#[no_mangle]
+pub extern "C" fn root(flags: AlignFlags) {}
diff --git /dev/null b/tests/rust/const_transparent.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/const_transparent.rs
@@ -0,0 +1,4 @@
+#[repr(transparent)]
+struct Transparent { field: u8 }
+
+const FOO: Transparent = Transparent { field: 0 };
diff --git a/tests/rust/derive-eq/Cargo.lock b/tests/rust/derive-eq/Cargo.lock
--- a/tests/rust/derive-eq/Cargo.lock
+++ b/tests/rust/derive-eq/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "derive-eq"
 version = "0.1.0"
diff --git a/tests/rust/expand/Cargo.lock b/tests/rust/expand/Cargo.lock
--- a/tests/rust/expand/Cargo.lock
+++ b/tests/rust/expand/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "expand"
 version = "0.1.0"
diff --git a/tests/rust/expand_default_features/Cargo.lock b/tests/rust/expand_default_features/Cargo.lock
--- a/tests/rust/expand_default_features/Cargo.lock
+++ b/tests/rust/expand_default_features/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "expand"
 version = "0.1.0"
diff --git a/tests/rust/expand_features/Cargo.lock b/tests/rust/expand_features/Cargo.lock
--- a/tests/rust/expand_features/Cargo.lock
+++ b/tests/rust/expand_features/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "expand"
 version = "0.1.0"
diff --git a/tests/rust/expand_no_default_features/Cargo.lock b/tests/rust/expand_no_default_features/Cargo.lock
--- a/tests/rust/expand_no_default_features/Cargo.lock
+++ b/tests/rust/expand_no_default_features/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "expand"
 version = "0.1.0"
diff --git a/tests/rust/mod_attr/Cargo.lock b/tests/rust/mod_attr/Cargo.lock
--- a/tests/rust/mod_attr/Cargo.lock
+++ b/tests/rust/mod_attr/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "mod_attr"
 version = "0.1.0"
diff --git a/tests/rust/mod_path/Cargo.lock b/tests/rust/mod_path/Cargo.lock
--- a/tests/rust/mod_path/Cargo.lock
+++ b/tests/rust/mod_path/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "mod_path"
 version = "0.1.0"
diff --git a/tests/rust/rename-crate/Cargo.lock b/tests/rust/rename-crate/Cargo.lock
--- a/tests/rust/rename-crate/Cargo.lock
+++ b/tests/rust/rename-crate/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "dependency"
 version = "0.1.0"
diff --git a/tests/rust/workspace/Cargo.lock b/tests/rust/workspace/Cargo.lock
--- a/tests/rust/workspace/Cargo.lock
+++ b/tests/rust/workspace/Cargo.lock
@@ -1,3 +1,5 @@
+# This file is automatically @generated by Cargo.
+# It is not intended for manual editing.
 [[package]]
 name = "child"
 version = "0.1.0"

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard e712cc42c759ace3e47a6ee9fdbff8c4f337cec1
git clean -fd
