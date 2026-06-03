#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/cflags_shell_escaped.rs
new file mode 100644
--- /dev/null
+++ b/tests/cflags_shell_escaped.rs
@@ -0,0 +1,24 @@
+mod support;
+
+use crate::support::Test;
+use std::env;
+
+/// This test is in its own module because it modifies the environment and would affect other tests
+/// when run in parallel with them.
+#[test]
+fn gnu_test_parse_shell_escaped_flags() {
+    env::set_var("CFLAGS", "foo \"bar baz\"");
+    env::set_var("CC_SHELL_ESCAPED_FLAGS", "1");
+    let test = Test::gnu();
+    test.gcc().file("foo.c").compile("foo");
+
+    test.cmd(0).must_have("foo").must_have("bar baz");
+
+    env::remove_var("CC_SHELL_ESCAPED_FLAGS");
+    let test = Test::gnu();
+    test.gcc().file("foo.c").compile("foo");
+
+    test.cmd(0)
+        .must_have("foo")
+        .must_have_in_order("\"bar", "baz\"");
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard eb3390615747fde57f7098797b2acb1aec80f539
git clean -fd
