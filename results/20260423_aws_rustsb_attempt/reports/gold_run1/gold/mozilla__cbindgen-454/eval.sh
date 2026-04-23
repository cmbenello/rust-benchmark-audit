#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/expectations/both/documentation_attr.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/documentation_attr.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ *With doc attr, each attr contribute to one line of document
+ *like this one with a new line character at its end
+ *and this one as well. So they are in the same paragraph
+ *
+ *Line ends with one new line should not break
+ *
+ *Line ends with two spaces and a new line
+ *should break to next line
+ *
+ *Line ends with two new lines
+ *
+ *Should break to next paragraph
+ */
+void root(void);
diff --git /dev/null b/tests/expectations/both/documentation_attr.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/documentation_attr.compat.c
@@ -0,0 +1,28 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+/**
+ *With doc attr, each attr contribute to one line of document
+ *like this one with a new line character at its end
+ *and this one as well. So they are in the same paragraph
+ *
+ *Line ends with one new line should not break
+ *
+ *Line ends with two spaces and a new line
+ *should break to next line
+ *
+ *Line ends with two new lines
+ *
+ *Should break to next paragraph
+ */
+void root(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/documentation_attr.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/documentation_attr.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ *With doc attr, each attr contribute to one line of document
+ *like this one with a new line character at its end
+ *and this one as well. So they are in the same paragraph
+ *
+ *Line ends with one new line should not break
+ *
+ *Line ends with two spaces and a new line
+ *should break to next line
+ *
+ *Line ends with two new lines
+ *
+ *Should break to next paragraph
+ */
+void root(void);
diff --git /dev/null b/tests/expectations/documentation_attr.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/documentation_attr.compat.c
@@ -0,0 +1,28 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+/**
+ *With doc attr, each attr contribute to one line of document
+ *like this one with a new line character at its end
+ *and this one as well. So they are in the same paragraph
+ *
+ *Line ends with one new line should not break
+ *
+ *Line ends with two spaces and a new line
+ *should break to next line
+ *
+ *Line ends with two new lines
+ *
+ *Should break to next paragraph
+ */
+void root(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/documentation_attr.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/documentation_attr.cpp
@@ -0,0 +1,22 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+extern "C" {
+
+///With doc attr, each attr contribute to one line of document
+///like this one with a new line character at its end
+///and this one as well. So they are in the same paragraph
+///
+///Line ends with one new line should not break
+///
+///Line ends with two spaces and a new line
+///should break to next line
+///
+///Line ends with two new lines
+///
+///Should break to next paragraph
+void root();
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/tag/documentation_attr.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/documentation_attr.c
@@ -0,0 +1,20 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+/**
+ *With doc attr, each attr contribute to one line of document
+ *like this one with a new line character at its end
+ *and this one as well. So they are in the same paragraph
+ *
+ *Line ends with one new line should not break
+ *
+ *Line ends with two spaces and a new line
+ *should break to next line
+ *
+ *Line ends with two new lines
+ *
+ *Should break to next paragraph
+ */
+void root(void);
diff --git /dev/null b/tests/expectations/tag/documentation_attr.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/documentation_attr.compat.c
@@ -0,0 +1,28 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+/**
+ *With doc attr, each attr contribute to one line of document
+ *like this one with a new line character at its end
+ *and this one as well. So they are in the same paragraph
+ *
+ *Line ends with one new line should not break
+ *
+ *Line ends with two spaces and a new line
+ *should break to next line
+ *
+ *Line ends with two new lines
+ *
+ *Should break to next paragraph
+ */
+void root(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/documentation_attr.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/documentation_attr.rs
@@ -0,0 +1,12 @@
+#[doc="With doc attr, each attr contribute to one line of document"]
+#[doc="like this one with a new line character at its end"]
+#[doc="and this one as well. So they are in the same paragraph"]
+#[doc=""]
+#[doc="Line ends with one new line\nshould not break"]
+#[doc=""]
+#[doc="Line ends with two spaces and a new line  \nshould break to next line"]
+#[doc=""]
+#[doc="Line ends with two new lines\n\nShould break to next paragraph"]
+#[no_mangle]
+pub extern "C" fn root() {
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard ff8e5d591dc8bf91a7309c54f0deb67899eeea87
git clean -fd
