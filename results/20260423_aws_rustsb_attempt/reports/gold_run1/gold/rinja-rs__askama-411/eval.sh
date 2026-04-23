#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/testing/templates/let-shadow.html
new file mode 100644
--- /dev/null
+++ b/testing/templates/let-shadow.html
@@ -0,0 +1,22 @@
+{%- let a = 1 -%}
+{%- let b -%}
+
+{%- if cond -%}
+    {%- let b = 22 -%}
+    {{ b }}-
+
+    {%- let b = 33 -%}
+    {{ a }}-{{ b }}-
+{%- else -%}
+    {%- let b = 222 -%}
+    {{ b }}-
+
+    {%- let b = 333 -%}
+    {{ a }}-{{ b }}-
+
+    {%- let (a, b) = Self::tuple() -%}
+    {{ a }}-{{ b }}-
+{%- endif -%}
+
+{%- let a = 11 -%}
+{{ a }}-{{ b }}
diff --git a/testing/tests/vars.rs b/testing/tests/vars.rs
--- a/testing/tests/vars.rs
+++ b/testing/tests/vars.rs
@@ -46,6 +46,27 @@ fn test_let_decl() {
     assert_eq!(t.render().unwrap(), "bar");
 }
 
+#[derive(Template)]
+#[template(path = "let-shadow.html")]
+struct LetShadowTemplate {
+    cond: bool,
+}
+
+impl LetShadowTemplate {
+    fn tuple() -> (i32, i32) {
+        (4, 5)
+    }
+}
+
+#[test]
+fn test_let_shadow() {
+    let t = LetShadowTemplate { cond: true };
+    assert_eq!(t.render().unwrap(), "22-1-33-11-22");
+
+    let t = LetShadowTemplate { cond: false };
+    assert_eq!(t.render().unwrap(), "222-1-333-4-5-11-222");
+}
+
 #[derive(Template)]
 #[template(source = "{% for v in self.0 %}{{ v }}{% endfor %}", ext = "txt")]
 struct SelfIterTemplate(Vec<usize>);

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features --test "vars"
cd ../
git reset --hard c7697cbd406dce0962618e99a6b78116b14fdb1c
git clean -fd
