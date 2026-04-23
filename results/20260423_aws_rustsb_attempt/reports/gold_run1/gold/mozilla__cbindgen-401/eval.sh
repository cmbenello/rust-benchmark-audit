#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/expectations/associated_in_body.cpp b/tests/expectations/associated_in_body.cpp
--- a/tests/expectations/associated_in_body.cpp
+++ b/tests/expectations/associated_in_body.cpp
@@ -32,11 +32,11 @@ struct StyleAlignFlags {
   static const StyleAlignFlags END;
   static const StyleAlignFlags FLEX_START;
 };
-inline const StyleAlignFlags StyleAlignFlags::AUTO = (StyleAlignFlags){ .bits = 0 };
-inline const StyleAlignFlags StyleAlignFlags::NORMAL = (StyleAlignFlags){ .bits = 1 };
-inline const StyleAlignFlags StyleAlignFlags::START = (StyleAlignFlags){ .bits = 1 << 1 };
-inline const StyleAlignFlags StyleAlignFlags::END = (StyleAlignFlags){ .bits = 1 << 2 };
-inline const StyleAlignFlags StyleAlignFlags::FLEX_START = (StyleAlignFlags){ .bits = 1 << 3 };
+inline const StyleAlignFlags StyleAlignFlags::AUTO = StyleAlignFlags{ /* .bits = */ 0 };
+inline const StyleAlignFlags StyleAlignFlags::NORMAL = StyleAlignFlags{ /* .bits = */ 1 };
+inline const StyleAlignFlags StyleAlignFlags::START = StyleAlignFlags{ /* .bits = */ 1 << 1 };
+inline const StyleAlignFlags StyleAlignFlags::END = StyleAlignFlags{ /* .bits = */ 1 << 2 };
+inline const StyleAlignFlags StyleAlignFlags::FLEX_START = StyleAlignFlags{ /* .bits = */ 1 << 3 };
 
 extern "C" {
 
diff --git a/tests/expectations/bitflags.cpp b/tests/expectations/bitflags.cpp
--- a/tests/expectations/bitflags.cpp
+++ b/tests/expectations/bitflags.cpp
@@ -27,11 +27,11 @@ struct AlignFlags {
     return *this;
   }
 };
-static const AlignFlags AlignFlags_AUTO = (AlignFlags){ .bits = 0 };
-static const AlignFlags AlignFlags_NORMAL = (AlignFlags){ .bits = 1 };
-static const AlignFlags AlignFlags_START = (AlignFlags){ .bits = 1 << 1 };
-static const AlignFlags AlignFlags_END = (AlignFlags){ .bits = 1 << 2 };
-static const AlignFlags AlignFlags_FLEX_START = (AlignFlags){ .bits = 1 << 3 };
+static const AlignFlags AlignFlags_AUTO = AlignFlags{ /* .bits = */ 0 };
+static const AlignFlags AlignFlags_NORMAL = AlignFlags{ /* .bits = */ 1 };
+static const AlignFlags AlignFlags_START = AlignFlags{ /* .bits = */ 1 << 1 };
+static const AlignFlags AlignFlags_END = AlignFlags{ /* .bits = */ 1 << 2 };
+static const AlignFlags AlignFlags_FLEX_START = AlignFlags{ /* .bits = */ 1 << 3 };
 
 extern "C" {
 
diff --git /dev/null b/tests/expectations/both/struct_literal_order.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/struct_literal_order.c
@@ -0,0 +1,24 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct ABC {
+  float a;
+  uint32_t b;
+  uint32_t c;
+} ABC;
+#define ABC_abc (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_bac (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_cba (ABC){ .a = 1.0, .b = 2, .c = 3 }
+
+typedef struct BAC {
+  uint32_t b;
+  float a;
+  int32_t c;
+} BAC;
+#define BAC_abc (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_bac (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_cba (BAC){ .b = 1, .a = 2.0, .c = 3 }
+
+void root(ABC a1, BAC a2);
diff --git /dev/null b/tests/expectations/both/struct_literal_order.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/struct_literal_order.compat.c
@@ -0,0 +1,32 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct ABC {
+  float a;
+  uint32_t b;
+  uint32_t c;
+} ABC;
+#define ABC_abc (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_bac (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_cba (ABC){ .a = 1.0, .b = 2, .c = 3 }
+
+typedef struct BAC {
+  uint32_t b;
+  float a;
+  int32_t c;
+} BAC;
+#define BAC_abc (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_bac (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_cba (BAC){ .b = 1, .a = 2.0, .c = 3 }
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(ABC a1, BAC a2);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git a/tests/expectations/prefixed_struct_literal.cpp b/tests/expectations/prefixed_struct_literal.cpp
--- a/tests/expectations/prefixed_struct_literal.cpp
+++ b/tests/expectations/prefixed_struct_literal.cpp
@@ -7,9 +7,9 @@ struct PREFIXFoo {
   int32_t a;
   uint32_t b;
 };
-static const PREFIXFoo PREFIXFoo_FOO = (PREFIXFoo){ .a = 42, .b = 47 };
+static const PREFIXFoo PREFIXFoo_FOO = PREFIXFoo{ /* .a = */ 42, /* .b = */ 47 };
 
-static const PREFIXFoo PREFIXBAR = (PREFIXFoo){ .a = 42, .b = 1337 };
+static const PREFIXFoo PREFIXBAR = PREFIXFoo{ /* .a = */ 42, /* .b = */ 1337 };
 
 extern "C" {
 
diff --git a/tests/expectations/prefixed_struct_literal_deep.cpp b/tests/expectations/prefixed_struct_literal_deep.cpp
--- a/tests/expectations/prefixed_struct_literal_deep.cpp
+++ b/tests/expectations/prefixed_struct_literal_deep.cpp
@@ -13,7 +13,7 @@ struct PREFIXFoo {
   PREFIXBar bar;
 };
 
-static const PREFIXFoo PREFIXVAL = (PREFIXFoo){ .a = 42, .b = 1337, .bar = (PREFIXBar){ .a = 323 } };
+static const PREFIXFoo PREFIXVAL = PREFIXFoo{ /* .a = */ 42, /* .b = */ 1337, /* .bar = */ PREFIXBar{ /* .a = */ 323 } };
 
 extern "C" {
 
diff --git a/tests/expectations/struct_literal.cpp b/tests/expectations/struct_literal.cpp
--- a/tests/expectations/struct_literal.cpp
+++ b/tests/expectations/struct_literal.cpp
@@ -9,12 +9,12 @@ struct Foo {
   int32_t a;
   uint32_t b;
 };
-static const Foo Foo_FOO = (Foo){ .a = 42, .b = 47 };
-static const Foo Foo_FOO2 = (Foo){ .a = 42, .b = 47 };
-static const Foo Foo_FOO3 = (Foo){ .a = 42, .b = 47 };
+static const Foo Foo_FOO = Foo{ /* .a = */ 42, /* .b = */ 47 };
+static const Foo Foo_FOO2 = Foo{ /* .a = */ 42, /* .b = */ 47 };
+static const Foo Foo_FOO3 = Foo{ /* .a = */ 42, /* .b = */ 47 };
 
 
-static const Foo BAR = (Foo){ .a = 42, .b = 1337 };
+static const Foo BAR = Foo{ /* .a = */ 42, /* .b = */ 1337 };
 
 
 
diff --git /dev/null b/tests/expectations/struct_literal_order.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/struct_literal_order.c
@@ -0,0 +1,24 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct {
+  float a;
+  uint32_t b;
+  uint32_t c;
+} ABC;
+#define ABC_abc (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_bac (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_cba (ABC){ .a = 1.0, .b = 2, .c = 3 }
+
+typedef struct {
+  uint32_t b;
+  float a;
+  int32_t c;
+} BAC;
+#define BAC_abc (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_bac (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_cba (BAC){ .b = 1, .a = 2.0, .c = 3 }
+
+void root(ABC a1, BAC a2);
diff --git /dev/null b/tests/expectations/struct_literal_order.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/struct_literal_order.compat.c
@@ -0,0 +1,32 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct {
+  float a;
+  uint32_t b;
+  uint32_t c;
+} ABC;
+#define ABC_abc (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_bac (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_cba (ABC){ .a = 1.0, .b = 2, .c = 3 }
+
+typedef struct {
+  uint32_t b;
+  float a;
+  int32_t c;
+} BAC;
+#define BAC_abc (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_bac (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_cba (BAC){ .b = 1, .a = 2.0, .c = 3 }
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(ABC a1, BAC a2);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/struct_literal_order.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/struct_literal_order.cpp
@@ -0,0 +1,28 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+struct ABC {
+  float a;
+  uint32_t b;
+  uint32_t c;
+};
+static const ABC ABC_abc = ABC{ /* .a = */ 1.0, /* .b = */ 2, /* .c = */ 3 };
+static const ABC ABC_bac = ABC{ /* .a = */ 1.0, /* .b = */ 2, /* .c = */ 3 };
+static const ABC ABC_cba = ABC{ /* .a = */ 1.0, /* .b = */ 2, /* .c = */ 3 };
+
+struct BAC {
+  uint32_t b;
+  float a;
+  int32_t c;
+};
+static const BAC BAC_abc = BAC{ /* .b = */ 1, /* .a = */ 2.0, /* .c = */ 3 };
+static const BAC BAC_bac = BAC{ /* .b = */ 1, /* .a = */ 2.0, /* .c = */ 3 };
+static const BAC BAC_cba = BAC{ /* .b = */ 1, /* .a = */ 2.0, /* .c = */ 3 };
+
+extern "C" {
+
+void root(ABC a1, BAC a2);
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/tag/struct_literal_order.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/struct_literal_order.c
@@ -0,0 +1,24 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+struct ABC {
+  float a;
+  uint32_t b;
+  uint32_t c;
+};
+#define ABC_abc (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_bac (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_cba (ABC){ .a = 1.0, .b = 2, .c = 3 }
+
+struct BAC {
+  uint32_t b;
+  float a;
+  int32_t c;
+};
+#define BAC_abc (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_bac (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_cba (BAC){ .b = 1, .a = 2.0, .c = 3 }
+
+void root(struct ABC a1, struct BAC a2);
diff --git /dev/null b/tests/expectations/tag/struct_literal_order.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/struct_literal_order.compat.c
@@ -0,0 +1,32 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+struct ABC {
+  float a;
+  uint32_t b;
+  uint32_t c;
+};
+#define ABC_abc (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_bac (ABC){ .a = 1.0, .b = 2, .c = 3 }
+#define ABC_cba (ABC){ .a = 1.0, .b = 2, .c = 3 }
+
+struct BAC {
+  uint32_t b;
+  float a;
+  int32_t c;
+};
+#define BAC_abc (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_bac (BAC){ .b = 1, .a = 2.0, .c = 3 }
+#define BAC_cba (BAC){ .b = 1, .a = 2.0, .c = 3 }
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(struct ABC a1, struct BAC a2);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/struct_literal_order.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/struct_literal_order.rs
@@ -0,0 +1,28 @@
+#[repr(C)]
+struct ABC {
+    pub a: f32,
+    pub b: u32,
+    pub c: u32,
+}
+
+#[repr(C)]
+struct BAC {
+    pub b: u32,
+    pub a: f32,
+    pub c: i32,
+}
+
+impl ABC {
+    pub const abc: ABC = ABC { a: 1.0, b: 2, c: 3 };
+    pub const bac: ABC = ABC { b: 2, a: 1.0, c: 3 };
+    pub const cba: ABC = ABC { c: 3, b: 2, a: 1.0 };
+}
+
+impl BAC {
+    pub const abc: BAC = BAC { a: 2.0, b: 1, c: 3 };
+    pub const bac: BAC = BAC { b: 1, a: 2.0, c: 3 };
+    pub const cba: BAC = BAC { c: 3, b: 1, a: 2.0 };
+}
+
+#[no_mangle]
+pub extern "C" fn root(a1: ABC, a2: BAC) {}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 5b4cda0d95690f00a1088f6b43726a197d03dad0
git clean -fd
