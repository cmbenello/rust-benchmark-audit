#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/expectations/both/cell.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/cell.c
@@ -0,0 +1,14 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct NotReprC_RefCell_i32 NotReprC_RefCell_i32;
+
+typedef NotReprC_RefCell_i32 Foo;
+
+typedef struct MyStruct {
+  int32_t number;
+} MyStruct;
+
+void root(const Foo *a, const MyStruct *with_cell);
diff --git /dev/null b/tests/expectations/both/cell.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/cell.compat.c
@@ -0,0 +1,22 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct NotReprC_RefCell_i32 NotReprC_RefCell_i32;
+
+typedef NotReprC_RefCell_i32 Foo;
+
+typedef struct MyStruct {
+  int32_t number;
+} MyStruct;
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(const Foo *a, const MyStruct *with_cell);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/cell.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/cell.c
@@ -0,0 +1,14 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct NotReprC_RefCell_i32 NotReprC_RefCell_i32;
+
+typedef NotReprC_RefCell_i32 Foo;
+
+typedef struct {
+  int32_t number;
+} MyStruct;
+
+void root(const Foo *a, const MyStruct *with_cell);
diff --git /dev/null b/tests/expectations/cell.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/cell.compat.c
@@ -0,0 +1,22 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct NotReprC_RefCell_i32 NotReprC_RefCell_i32;
+
+typedef NotReprC_RefCell_i32 Foo;
+
+typedef struct {
+  int32_t number;
+} MyStruct;
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(const Foo *a, const MyStruct *with_cell);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/cell.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/cell.cpp
@@ -0,0 +1,22 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+template<typename T>
+struct NotReprC;
+
+template<typename T>
+struct RefCell;
+
+using Foo = NotReprC<RefCell<int32_t>>;
+
+struct MyStruct {
+  int32_t number;
+};
+
+extern "C" {
+
+void root(const Foo *a, const MyStruct *with_cell);
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/tag/cell.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/cell.c
@@ -0,0 +1,14 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+struct NotReprC_RefCell_i32;
+
+typedef struct NotReprC_RefCell_i32 Foo;
+
+struct MyStruct {
+  int32_t number;
+};
+
+void root(const Foo *a, const struct MyStruct *with_cell);
diff --git /dev/null b/tests/expectations/tag/cell.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/cell.compat.c
@@ -0,0 +1,22 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+struct NotReprC_RefCell_i32;
+
+typedef struct NotReprC_RefCell_i32 Foo;
+
+struct MyStruct {
+  int32_t number;
+};
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(const Foo *a, const struct MyStruct *with_cell);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/cell.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/cell.rs
@@ -0,0 +1,11 @@
+#[repr(C)]
+pub struct MyStruct {
+    number: std::cell::Cell<i32>,
+}
+
+pub struct NotReprC<T> { inner: T }
+
+pub type Foo = NotReprC<std::cell::RefCell<i32>>;
+
+#[no_mangle]
+pub extern "C" fn root(a: &Foo, with_cell: &MyStruct) {}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 6654f99251769e9e037824d471f9f39e8d536b90
git clean -fd
