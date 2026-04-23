#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -1321,3 +1321,17 @@ pub mod issue154 {
         }
     }
 }
+
+// https://github.com/dtolnay/async-trait/issues/158
+pub mod issue158 {
+    use async_trait::async_trait;
+
+    fn f() {}
+
+    #[async_trait]
+    pub trait Trait {
+        async fn f(&self) {
+            self::f()
+        }
+    }
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 6bff4e0c5935b30b4d5af42bb06d9972866c752d
git clean -fd
