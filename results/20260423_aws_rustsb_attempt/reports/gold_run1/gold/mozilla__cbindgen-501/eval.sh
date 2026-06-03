#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/expectations/both/exclude_generic_monomorph.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/exclude_generic_monomorph.c
@@ -0,0 +1,15 @@
+#include <stdint.h>
+
+typedef uint64_t Option_Foo;
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct Bar {
+  Option_Foo foo;
+} Bar;
+
+void root(Bar f);
diff --git /dev/null b/tests/expectations/both/exclude_generic_monomorph.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/exclude_generic_monomorph.compat.c
@@ -0,0 +1,23 @@
+#include <stdint.h>
+
+typedef uint64_t Option_Foo;
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct Bar {
+  Option_Foo foo;
+} Bar;
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(Bar f);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/exclude_generic_monomorph.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/exclude_generic_monomorph.c
@@ -0,0 +1,15 @@
+#include <stdint.h>
+
+typedef uint64_t Option_Foo;
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct {
+  Option_Foo foo;
+} Bar;
+
+void root(Bar f);
diff --git /dev/null b/tests/expectations/exclude_generic_monomorph.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/exclude_generic_monomorph.compat.c
@@ -0,0 +1,23 @@
+#include <stdint.h>
+
+typedef uint64_t Option_Foo;
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct {
+  Option_Foo foo;
+} Bar;
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(Bar f);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/exclude_generic_monomorph.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/exclude_generic_monomorph.cpp
@@ -0,0 +1,15 @@
+#include <stdint.h>
+
+typedef uint64_t Option_Foo;
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct {
+  Option_Foo foo;
+} Bar;
+
+void root(Bar f);
diff --git /dev/null b/tests/expectations/tag/exclude_generic_monomorph.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/exclude_generic_monomorph.c
@@ -0,0 +1,15 @@
+#include <stdint.h>
+
+typedef uint64_t Option_Foo;
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+struct Bar {
+  Option_Foo foo;
+};
+
+void root(struct Bar f);
diff --git /dev/null b/tests/expectations/tag/exclude_generic_monomorph.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/exclude_generic_monomorph.compat.c
@@ -0,0 +1,23 @@
+#include <stdint.h>
+
+typedef uint64_t Option_Foo;
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+struct Bar {
+  Option_Foo foo;
+};
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(struct Bar f);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/exclude_generic_monomorph.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/exclude_generic_monomorph.rs
@@ -0,0 +1,10 @@
+#[repr(transparent)]
+pub struct Foo(NonZeroU64);
+
+#[repr(C)]
+pub struct Bar {
+  foo: Option<Foo>,
+}
+
+#[no_mangle]
+pub extern "C" fn root(f: Bar) {}
diff --git /dev/null b/tests/rust/exclude_generic_monomorph.toml
new file mode 100644
--- /dev/null
+++ b/tests/rust/exclude_generic_monomorph.toml
@@ -0,0 +1,11 @@
+language = "C"
+header = """
+#include <stdint.h>
+
+typedef uint64_t Option_Foo;
+"""
+
+[export]
+exclude = [
+  "Option_Foo",
+]

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 6fd245096dcd5c50c1065b4bd6ce62a09df0b39b
git clean -fd
