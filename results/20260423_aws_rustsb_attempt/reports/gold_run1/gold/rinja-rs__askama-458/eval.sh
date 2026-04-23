#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_shared/src/input.rs b/askama_shared/src/input.rs
--- a/askama_shared/src/input.rs
+++ b/askama_shared/src/input.rs
@@ -227,3 +242,51 @@ impl FromStr for Print {
         })
     }
 }
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn test_ext() {
+        assert_eq!(extension(Path::new("foo-bar.txt")), Some("txt"));
+        assert_eq!(extension(Path::new("foo-bar.html")), Some("html"));
+        assert_eq!(extension(Path::new("foo-bar.unknown")), Some("unknown"));
+
+        assert_eq!(extension(Path::new("foo/bar/baz.txt")), Some("txt"));
+        assert_eq!(extension(Path::new("foo/bar/baz.html")), Some("html"));
+        assert_eq!(extension(Path::new("foo/bar/baz.unknown")), Some("unknown"));
+    }
+
+    #[test]
+    fn test_double_ext() {
+        assert_eq!(extension(Path::new("foo-bar.html.txt")), Some("txt"));
+        assert_eq!(extension(Path::new("foo-bar.txt.html")), Some("html"));
+        assert_eq!(extension(Path::new("foo-bar.txt.unknown")), Some("unknown"));
+
+        assert_eq!(extension(Path::new("foo/bar/baz.html.txt")), Some("txt"));
+        assert_eq!(extension(Path::new("foo/bar/baz.txt.html")), Some("html"));
+        assert_eq!(
+            extension(Path::new("foo/bar/baz.txt.unknown")),
+            Some("unknown")
+        );
+    }
+
+    #[test]
+    fn test_skip_jinja_ext() {
+        assert_eq!(extension(Path::new("foo-bar.html.j2")), Some("html"));
+        assert_eq!(extension(Path::new("foo-bar.html.jinja")), Some("html"));
+        assert_eq!(extension(Path::new("foo-bar.html.jinja2")), Some("html"));
+
+        assert_eq!(extension(Path::new("foo/bar/baz.txt.j2")), Some("txt"));
+        assert_eq!(extension(Path::new("foo/bar/baz.txt.jinja")), Some("txt"));
+        assert_eq!(extension(Path::new("foo/bar/baz.txt.jinja2")), Some("txt"));
+    }
+
+    #[test]
+    fn test_only_jinja_ext() {
+        assert_eq!(extension(Path::new("foo-bar.j2")), Some("j2"));
+        assert_eq!(extension(Path::new("foo-bar.jinja")), Some("jinja"));
+        assert_eq!(extension(Path::new("foo-bar.jinja2")), Some("jinja2"));
+    }
+}
diff --git /dev/null b/testing/templates/foo.html
new file mode 100644
--- /dev/null
+++ b/testing/templates/foo.html
@@ -0,0 +1,1 @@
+foo.html
\ No newline at end of file
diff --git /dev/null b/testing/templates/foo.html.jinja
new file mode 100644
--- /dev/null
+++ b/testing/templates/foo.html.jinja
@@ -0,0 +1,1 @@
+foo.html.jinja
\ No newline at end of file
diff --git /dev/null b/testing/templates/foo.jinja
new file mode 100644
--- /dev/null
+++ b/testing/templates/foo.jinja
@@ -0,0 +1,1 @@
+foo.jinja
\ No newline at end of file
diff --git /dev/null b/testing/tests/ext.rs
new file mode 100644
--- /dev/null
+++ b/testing/tests/ext.rs
@@ -0,0 +1,67 @@
+use askama::Template;
+
+#[derive(Template)]
+#[template(path = "foo.html")]
+struct PathHtml;
+
+#[test]
+fn test_path_ext_html() {
+    let t = PathHtml;
+    assert_eq!(t.render().unwrap(), "foo.html");
+    assert_eq!(t.extension(), Some("html"));
+}
+
+#[derive(Template)]
+#[template(path = "foo.jinja")]
+struct PathJinja;
+
+#[test]
+fn test_path_ext_jinja() {
+    let t = PathJinja;
+    assert_eq!(t.render().unwrap(), "foo.jinja");
+    assert_eq!(t.extension(), Some("jinja"));
+}
+
+#[derive(Template)]
+#[template(path = "foo.html.jinja")]
+struct PathHtmlJinja;
+
+#[test]
+fn test_path_ext_html_jinja() {
+    let t = PathHtmlJinja;
+    assert_eq!(t.render().unwrap(), "foo.html.jinja");
+    assert_eq!(t.extension(), Some("html"));
+}
+
+#[derive(Template)]
+#[template(path = "foo.html", ext = "txt")]
+struct PathHtmlAndExtTxt;
+
+#[test]
+fn test_path_ext_html_and_ext_txt() {
+    let t = PathHtmlAndExtTxt;
+    assert_eq!(t.render().unwrap(), "foo.html");
+    assert_eq!(t.extension(), Some("txt"));
+}
+
+#[derive(Template)]
+#[template(path = "foo.jinja", ext = "txt")]
+struct PathJinjaAndExtTxt;
+
+#[test]
+fn test_path_ext_jinja_and_ext_txt() {
+    let t = PathJinjaAndExtTxt;
+    assert_eq!(t.render().unwrap(), "foo.jinja");
+    assert_eq!(t.extension(), Some("txt"));
+}
+
+#[derive(Template)]
+#[template(path = "foo.html.jinja", ext = "txt")]
+struct PathHtmlJinjaAndExtTxt;
+
+#[test]
+fn test_path_ext_html_jinja_and_ext_txt() {
+    let t = PathHtmlJinjaAndExtTxt;
+    assert_eq!(t.render().unwrap(), "foo.html.jinja");
+    assert_eq!(t.extension(), Some("txt"));
+}

EOF_114329324912
cd "askama_shared"
cargo test --no-fail-fast --all-features
cd ../
cd "testing"
cargo test --no-fail-fast --all-features --test "ext"
cd ../
git reset --hard 96a4328d642191e4925a8f822c9d2cf311f2f9ec
git clean -fd
