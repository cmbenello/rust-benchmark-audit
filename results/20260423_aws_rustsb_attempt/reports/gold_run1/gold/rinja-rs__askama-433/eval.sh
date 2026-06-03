#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/testing/tests/simple.rs b/testing/tests/simple.rs
--- a/testing/tests/simple.rs
+++ b/testing/tests/simple.rs
@@ -72,6 +72,36 @@ fn test_variables_no_escape() {
     );
 }
 
+#[derive(Template)]
+#[template(
+    source = "{{ foo }} {{ foo_bar }} {{ FOO }} {{ FOO_BAR }} {{ self::FOO }} {{ self::FOO_BAR }} {{ Self::BAR }} {{ Self::BAR_BAZ }}",
+    ext = "txt"
+)]
+struct ConstTemplate {
+    foo: &'static str,
+    foo_bar: &'static str,
+}
+
+impl ConstTemplate {
+    const BAR: &'static str = "BAR";
+    const BAR_BAZ: &'static str = "BAR BAZ";
+}
+
+#[test]
+fn test_constants() {
+    let t = ConstTemplate {
+        foo: "foo",
+        foo_bar: "foo bar",
+    };
+    assert_eq!(
+        t.render().unwrap(),
+        "foo foo bar FOO FOO BAR FOO FOO BAR BAR BAR BAZ"
+    );
+}
+
+const FOO: &str = "FOO";
+const FOO_BAR: &str = "FOO BAR";
+
 #[derive(Template)]
 #[template(path = "if.html")]
 struct IfTemplate {

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features --test "simple"
cd ../
git reset --hard 560d219c269bbf291a4f78e8ef3ffeb0d02ffdef
git clean -fd
