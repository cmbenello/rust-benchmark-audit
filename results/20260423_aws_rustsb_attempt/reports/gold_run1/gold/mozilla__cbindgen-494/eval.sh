#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/expectations/associated_constant_panic.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/associated_constant_panic.c
@@ -0,0 +1,4 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
diff --git /dev/null b/tests/expectations/associated_constant_panic.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/associated_constant_panic.compat.c
@@ -0,0 +1,4 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
diff --git /dev/null b/tests/expectations/associated_constant_panic.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/associated_constant_panic.cpp
@@ -0,0 +1,4 @@
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
diff --git /dev/null b/tests/expectations/both/associated_constant_panic.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/associated_constant_panic.c
@@ -0,0 +1,4 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
diff --git /dev/null b/tests/expectations/both/associated_constant_panic.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/associated_constant_panic.compat.c
@@ -0,0 +1,4 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
diff --git /dev/null b/tests/expectations/tag/associated_constant_panic.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/associated_constant_panic.c
@@ -0,0 +1,4 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
diff --git /dev/null b/tests/expectations/tag/associated_constant_panic.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/associated_constant_panic.compat.c
@@ -0,0 +1,4 @@
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
diff --git /dev/null b/tests/rust/associated_constant_panic.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/associated_constant_panic.rs
@@ -0,0 +1,7 @@
+pub trait F {
+    const B: u8;
+}
+
+impl F for u16 {
+    const B: u8 = 3;
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 17d7aad7d07dce8aa665aedbc75c39953afe1600
git clean -fd
