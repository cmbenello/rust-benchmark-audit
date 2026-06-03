#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/testing/tests/macro.rs b/testing/tests/macro.rs
--- a/testing/tests/macro.rs
+++ b/testing/tests/macro.rs
@@ -95,3 +95,31 @@ fn test_macro_self_arg() {
     let t = MacroSelfArgTemplate { s: "foo" };
     assert_eq!(t.render().unwrap(), "foo");
 }
+
+#[derive(Template)]
+#[template(
+    source = "{%- macro thrice(param1, param2) -%}
+{{ param1 }} {{ param2 }}
+{% endmacro -%}
+
+{%- call thrice(param1=2, param2=3) -%}
+{%- call thrice(param2=3, param1=2) -%}
+{%- call thrice(3, param2=2) -%}
+",
+    ext = "html"
+)]
+struct MacroNamedArg;
+
+#[test]
+// We check that it's always the correct values passed to the
+// expected argument.
+fn test_named_argument() {
+    assert_eq!(
+        MacroNamedArg.render().unwrap(),
+        "\
+2 3
+2 3
+3 2
+"
+    );
+}
diff --git /dev/null b/testing/tests/ui/macro_named_argument.rs
new file mode 100644
--- /dev/null
+++ b/testing/tests/ui/macro_named_argument.rs
@@ -0,0 +1,45 @@
+use askama::Template;
+
+#[derive(Template)]
+#[template(source = "{%- macro thrice(param1, param2) -%}
+{{ param1 }} {{ param2 }}
+{%- endmacro -%}
+
+{%- call thrice(param1=2, param3=3) -%}", ext = "html")]
+struct InvalidNamedArg;
+
+#[derive(Template)]
+#[template(source = "{%- macro thrice(param1, param2) -%}
+{{ param1 }} {{ param2 }}
+{%- endmacro -%}
+
+{%- call thrice(param1=2, param1=3) -%}", ext = "html")]
+struct InvalidNamedArg2;
+
+// Ensures that filters can't have named arguments.
+#[derive(Template)]
+#[template(source = "{%- macro thrice(param1, param2) -%}
+{{ param1 }} {{ param2 }}
+{%- endmacro -%}
+
+{%- call thrice(3, param1=2) | filter(param1=12) -%}", ext = "html")]
+struct InvalidNamedArg3;
+
+// Ensures that named arguments can only be passed last.
+#[derive(Template)]
+#[template(source = "{%- macro thrice(param1, param2) -%}
+{{ param1 }} {{ param2 }}
+{%- endmacro -%}
+{%- call thrice(param1=2, 3) -%}", ext = "html")]
+struct InvalidNamedArg4;
+
+// Ensures that named arguments can't be used for arguments before them.
+#[derive(Template)]
+#[template(source = "{%- macro thrice(param1, param2) -%}
+{{ param1 }} {{ param2 }}
+{%- endmacro -%}
+{%- call thrice(3, param1=2) -%}", ext = "html")]
+struct InvalidNamedArg5;
+
+fn main() {
+}
diff --git /dev/null b/testing/tests/ui/macro_named_argument.stderr
new file mode 100644
--- /dev/null
+++ b/testing/tests/ui/macro_named_argument.stderr
@@ -0,0 +1,44 @@
+error: no argument named `param3` in macro "thrice"
+ --> tests/ui/macro_named_argument.rs:3:10
+  |
+3 | #[derive(Template)]
+  |          ^^^^^^^^
+  |
+  = note: this error originates in the derive macro `Template` (in Nightly builds, run with -Z macro-backtrace for more info)
+
+error: named argument `param1` was passed more than once
+       problems parsing template source at row 5, column 15 near:
+       "(param1=2, param1=3) -%}"
+  --> tests/ui/macro_named_argument.rs:11:10
+   |
+11 | #[derive(Template)]
+   |          ^^^^^^^^
+   |
+   = note: this error originates in the derive macro `Template` (in Nightly builds, run with -Z macro-backtrace for more info)
+
+error: problems parsing template source at row 5, column 29 near:
+       "| filter(param1=12) -%}"
+  --> tests/ui/macro_named_argument.rs:20:10
+   |
+20 | #[derive(Template)]
+   |          ^^^^^^^^
+   |
+   = note: this error originates in the derive macro `Template` (in Nightly builds, run with -Z macro-backtrace for more info)
+
+error: named arguments must always be passed last
+       problems parsing template source at row 4, column 15 near:
+       "(param1=2, 3) -%}"
+  --> tests/ui/macro_named_argument.rs:29:10
+   |
+29 | #[derive(Template)]
+   |          ^^^^^^^^
+   |
+   = note: this error originates in the derive macro `Template` (in Nightly builds, run with -Z macro-backtrace for more info)
+
+error: cannot have unnamed argument (`param2`) after named argument in macro "thrice"
+  --> tests/ui/macro_named_argument.rs:37:10
+   |
+37 | #[derive(Template)]
+   |          ^^^^^^^^
+   |
+   = note: this error originates in the derive macro `Template` (in Nightly builds, run with -Z macro-backtrace for more info)

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 80238d7f48fd86ef939e74df9fdc9678ee78a208
git clean -fd
