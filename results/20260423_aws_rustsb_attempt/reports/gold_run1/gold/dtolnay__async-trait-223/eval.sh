#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -1450,3 +1450,14 @@ pub mod issue204 {
         async fn g(arg: *const impl Trait);
     }
 }
+
+// https://github.com/dtolnay/async-trait/issues/210
+pub mod issue210 {
+    use async_trait::async_trait;
+    use std::sync::Arc;
+
+    #[async_trait]
+    pub trait Trait {
+        async fn f(self: Arc<Self>) {}
+    }
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 9ed6489ee0cedf31fef8a06a1dcd1f18b4afdd39
git clean -fd
