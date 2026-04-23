#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -1258,3 +1258,23 @@ pub mod issue147 {
         }
     }
 }
+
+// https://github.com/dtolnay/async-trait/issues/149
+pub mod issue149 {
+    use async_trait::async_trait;
+
+    pub struct Thing;
+    pub trait Ret {}
+    impl Ret for Thing {}
+
+    pub async fn ok() -> &'static dyn Ret {
+        return &Thing;
+    }
+
+    #[async_trait]
+    pub trait Trait {
+        async fn fail() -> &'static dyn Ret {
+            return &Thing;
+        }
+    }
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 2c4cde7ce826c77fe08a8f67d6fdc47b023adf0f
git clean -fd
