#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/testing/tests/loops.rs b/testing/tests/loops.rs
--- a/testing/tests/loops.rs
+++ b/testing/tests/loops.rs
@@ -83,6 +83,16 @@ fn test_for_array() {
     assert_eq!(t.render().unwrap(), "123");
 }
 
+#[derive(Template)]
+#[template(source = "{% for i in [1, 2, 3, ] %}{{ i }}{% endfor %}", ext = "txt")]
+struct ForArrayTailingCommaTemplate;
+
+#[test]
+fn test_for_array_trailing_comma() {
+    let t = ForArrayTailingCommaTemplate;
+    assert_eq!(t.render().unwrap(), "123");
+}
+
 #[derive(Template)]
 #[template(
     source = "{% for i in [1, 2, 3].iter() %}{{ i }}{% endfor %}",

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features --test "loops"
cd ../
git reset --hard 53b4b518f9a230665029560df038c318b2e55458
git clean -fd
