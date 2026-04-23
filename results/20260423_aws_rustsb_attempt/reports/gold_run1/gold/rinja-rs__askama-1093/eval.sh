#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/testing/templates/fragment-include.html
new file mode 100644
--- /dev/null
+++ b/testing/templates/fragment-include.html
@@ -0,0 +1,9 @@
+{% extends "fragment-base.html" %}
+
+{% block body %}
+{% include "included.html" %}
+{% endblock %}
+
+{% block other_body %}
+<p>Don't render me.</p>
+{% endblock %}
\ No newline at end of file
diff --git a/testing/tests/block_fragments.rs b/testing/tests/block_fragments.rs
--- a/testing/tests/block_fragments.rs
+++ b/testing/tests/block_fragments.rs
@@ -103,3 +103,15 @@ fn test_specific_block() {
     let t = RenderInPlace { s1 };
     assert_eq!(t.render().unwrap(), "\nSection: [abc]\n");
 }
+
+#[derive(Template)]
+#[template(path = "fragment-include.html", block = "body")]
+struct FragmentInclude<'a> {
+    s: &'a str,
+}
+
+#[test]
+fn test_fragment_include() {
+    let fragment_include = FragmentInclude { s: "world" };
+    assert_eq!(fragment_include.render().unwrap(), "\nINCLUDED: world\n");
+}

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features --test "block_fragments"
cd ../
git reset --hard 668bd6f2c1f60dc25143360976a5c775901af889
git clean -fd
