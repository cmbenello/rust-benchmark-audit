#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/expectations/both/global_variable.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/global_variable.c
@@ -0,0 +1,8 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+extern const char CONST_GLOBAL_ARRAY[128];
+
+extern char MUT_GLOBAL_ARRAY[128];
diff --git /dev/null b/tests/expectations/both/global_variable.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/global_variable.compat.c
@@ -0,0 +1,16 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+extern const char CONST_GLOBAL_ARRAY[128];
+
+extern char MUT_GLOBAL_ARRAY[128];
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/global_variable.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/global_variable.c
@@ -0,0 +1,8 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+extern const char CONST_GLOBAL_ARRAY[128];
+
+extern char MUT_GLOBAL_ARRAY[128];
diff --git /dev/null b/tests/expectations/global_variable.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/global_variable.compat.c
@@ -0,0 +1,16 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+extern const char CONST_GLOBAL_ARRAY[128];
+
+extern char MUT_GLOBAL_ARRAY[128];
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/global_variable.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/global_variable.cpp
@@ -0,0 +1,12 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+extern "C" {
+
+extern const char CONST_GLOBAL_ARRAY[128];
+
+extern char MUT_GLOBAL_ARRAY[128];
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/tag/global_variable.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/global_variable.c
@@ -0,0 +1,8 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+extern const char CONST_GLOBAL_ARRAY[128];
+
+extern char MUT_GLOBAL_ARRAY[128];
diff --git /dev/null b/tests/expectations/tag/global_variable.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/global_variable.compat.c
@@ -0,0 +1,16 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+extern const char CONST_GLOBAL_ARRAY[128];
+
+extern char MUT_GLOBAL_ARRAY[128];
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/global_variable.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/global_variable.rs
@@ -0,0 +1,5 @@
+#[no_mangle]
+pub static mut MUT_GLOBAL_ARRAY: [c_char; 128] = [0; 128];
+
+#[no_mangle]
+pub static CONST_GLOBAL_ARRAY: [c_char; 128] = [0; 128];

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard dfa6e5f9824aac416d267420f8730225011d5c8f
git clean -fd
