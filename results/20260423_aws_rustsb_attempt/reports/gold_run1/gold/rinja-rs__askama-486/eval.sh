#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/testing/tests/filters.rs b/testing/tests/filters.rs
--- a/testing/tests/filters.rs
+++ b/testing/tests/filters.rs
@@ -21,7 +21,7 @@ fn filter_escape() {
     };
     assert_eq!(
         s.render().unwrap(),
-        "&#x2f;&#x2f; my &lt;html&gt; is &quot;unsafe&quot; &amp; \
+        "// my &lt;html&gt; is &quot;unsafe&quot; &amp; \
          should be &#x27;escaped&#x27;"
     );
 }
diff --git a/testing/tests/simple.rs b/testing/tests/simple.rs
--- a/testing/tests/simple.rs
+++ b/testing/tests/simple.rs
@@ -40,12 +40,9 @@ struct EscapeTemplate<'a> {
 
 #[test]
 fn test_escape() {
-    let s = EscapeTemplate { name: "<>&\"'/" };
+    let s = EscapeTemplate { name: "<>&\"'" };
 
-    assert_eq!(
-        s.render().unwrap(),
-        "Hello, &lt;&gt;&amp;&quot;&#x27;&#x2f;!"
-    );
+    assert_eq!(s.render().unwrap(), "Hello, &lt;&gt;&amp;&quot;&#x27;!");
 }
 
 #[derive(Template)]

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features --test "filters"
cargo test --no-fail-fast --all-features --test "simple"
cd ../
git reset --hard 92df4d1fe49e8fde5ca13f13b8236102bc16b969
git clean -fd
