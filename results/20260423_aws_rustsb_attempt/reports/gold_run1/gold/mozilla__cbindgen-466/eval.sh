#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/expectations/both/function_sort_name.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/function_sort_name.c
@@ -0,0 +1,12 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void A(void);
+
+void B(void);
+
+void C(void);
+
+void D(void);
diff --git /dev/null b/tests/expectations/both/function_sort_name.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/function_sort_name.compat.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void A(void);
+
+void B(void);
+
+void C(void);
+
+void D(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/both/function_sort_none.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/function_sort_none.c
@@ -0,0 +1,12 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void C(void);
+
+void B(void);
+
+void D(void);
+
+void A(void);
diff --git /dev/null b/tests/expectations/both/function_sort_none.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/function_sort_none.compat.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void C(void);
+
+void B(void);
+
+void D(void);
+
+void A(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/function_sort_name.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/function_sort_name.c
@@ -0,0 +1,12 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void A(void);
+
+void B(void);
+
+void C(void);
+
+void D(void);
diff --git /dev/null b/tests/expectations/function_sort_name.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/function_sort_name.compat.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void A(void);
+
+void B(void);
+
+void C(void);
+
+void D(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/function_sort_name.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/function_sort_name.cpp
@@ -0,0 +1,16 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+extern "C" {
+
+void A();
+
+void B();
+
+void C();
+
+void D();
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/function_sort_none.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/function_sort_none.c
@@ -0,0 +1,12 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void C(void);
+
+void B(void);
+
+void D(void);
+
+void A(void);
diff --git /dev/null b/tests/expectations/function_sort_none.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/function_sort_none.compat.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void C(void);
+
+void B(void);
+
+void D(void);
+
+void A(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/function_sort_none.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/function_sort_none.cpp
@@ -0,0 +1,16 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+extern "C" {
+
+void C();
+
+void B();
+
+void D();
+
+void A();
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/tag/function_sort_name.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/function_sort_name.c
@@ -0,0 +1,12 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void A(void);
+
+void B(void);
+
+void C(void);
+
+void D(void);
diff --git /dev/null b/tests/expectations/tag/function_sort_name.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/function_sort_name.compat.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void A(void);
+
+void B(void);
+
+void C(void);
+
+void D(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/tag/function_sort_none.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/function_sort_none.c
@@ -0,0 +1,12 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void C(void);
+
+void B(void);
+
+void D(void);
+
+void A(void);
diff --git /dev/null b/tests/expectations/tag/function_sort_none.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/function_sort_none.compat.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void C(void);
+
+void B(void);
+
+void D(void);
+
+void A(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/function_sort_name.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/function_sort_name.rs
@@ -0,0 +1,15 @@
+#[no_mangle]
+pub extern "C" fn C()
+{ }
+
+#[no_mangle]
+pub extern "C" fn B()
+{ }
+
+#[no_mangle]
+pub extern "C" fn D()
+{ }
+
+#[no_mangle]
+pub extern "C" fn A()
+{ }
diff --git /dev/null b/tests/rust/function_sort_none.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/function_sort_none.rs
@@ -0,0 +1,15 @@
+#[no_mangle]
+pub extern "C" fn C()
+{ }
+
+#[no_mangle]
+pub extern "C" fn B()
+{ }
+
+#[no_mangle]
+pub extern "C" fn D()
+{ }
+
+#[no_mangle]
+pub extern "C" fn A()
+{ }
diff --git /dev/null b/tests/rust/function_sort_none.toml
new file mode 100644
--- /dev/null
+++ b/tests/rust/function_sort_none.toml
@@ -0,0 +1,2 @@
+[fn]
+sort_by = "None"

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 4cb762ec8f24f8ef3e12fcd716326a1207a88018
git clean -fd
