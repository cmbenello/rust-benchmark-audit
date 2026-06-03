#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/testing/templates/nested-macro-args.html
new file mode 100644
--- /dev/null
+++ b/testing/templates/nested-macro-args.html
@@ -0,0 +1,9 @@
+{%- macro outer(first) -%}
+{%- call inner(first, "second") -%}
+{%- endmacro -%}
+
+{%- macro inner(first, second) -%}
+{{ first }} {{ second }}
+{%- endmacro -%}
+
+{%- call outer("first") -%}
diff --git a/testing/tests/macro.rs b/testing/tests/macro.rs
--- a/testing/tests/macro.rs
+++ b/testing/tests/macro.rs
@@ -53,3 +53,13 @@ fn test_short_circuit() {
     let t = ShortCircuitTemplate {};
     assert_eq!(t.render().unwrap(), "truetruetruefalsetruetrue");
 }
+
+#[derive(Template)]
+#[template(path = "nested-macro-args.html")]
+struct NestedMacroArgsTemplate {}
+
+#[test]
+fn test_nested_macro_with_args() {
+    let t = NestedMacroArgsTemplate {};
+    assert_eq!(t.render().unwrap(), "first second");
+}

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features --test "macro"
cd ../
git reset --hard b318d7cbcded2c6dfc66bbe19687f1246a9a9eab
git clean -fd
