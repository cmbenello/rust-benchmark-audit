#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/bindgen/utilities.rs b/src/bindgen/utilities.rs
--- a/src/bindgen/utilities.rs
+++ b/src/bindgen/utilities.rs
@@ -39,6 +39,24 @@ pub fn find_first_some<T>(slice: &[Option<T>]) -> Option<&T> {
     None
 }
 
+pub trait SynItemFnHelpers: SynItemHelpers {
+    fn exported_name(&self) -> Option<String>;
+}
+
+impl SynItemFnHelpers for syn::ItemFn {
+    fn exported_name(&self) -> Option<String> {
+        self.attrs
+            .attr_name_value_lookup("export_name")
+            .or_else(|| {
+                if self.is_no_mangle() {
+                    Some(self.sig.ident.to_string())
+                } else {
+                    None
+                }
+            })
+    }
+}
+
 pub trait SynItemHelpers {
     /// Searches for attributes like `#[test]`.
     /// Example:
diff --git /dev/null b/tests/expectations/both/export_name.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/export_name.c
@@ -0,0 +1,6 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void do_the_thing_with_export_name(void);
diff --git /dev/null b/tests/expectations/both/export_name.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/export_name.compat.c
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
+void do_the_thing_with_export_name(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/export_name.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/export_name.c
@@ -0,0 +1,6 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void do_the_thing_with_export_name(void);
diff --git /dev/null b/tests/expectations/export_name.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/export_name.compat.c
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
+void do_the_thing_with_export_name(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/export_name.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/export_name.cpp
@@ -0,0 +1,10 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+extern "C" {
+
+void do_the_thing_with_export_name();
+
+} // extern "C"
diff --git /dev/null b/tests/expectations/tag/export_name.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/export_name.c
@@ -0,0 +1,6 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+void do_the_thing_with_export_name(void);
diff --git /dev/null b/tests/expectations/tag/export_name.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/export_name.compat.c
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
+void do_the_thing_with_export_name(void);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/export_name.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/export_name.rs
@@ -0,0 +1,4 @@
+#[export_name = "do_the_thing_with_export_name"]
+pub extern "C" fn do_the_thing() {
+  println!("doing the thing!");
+}
\ No newline at end of file

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard f5d76c44c466b47d1c776acd9974df838f30d431
git clean -fd
