#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -1489,3 +1489,37 @@ pub mod issue226 {
         }
     }
 }
+
+// https://github.com/dtolnay/async-trait/issues/232
+pub mod issue232 {
+    use async_trait::async_trait;
+
+    #[async_trait]
+    pub trait Generic<T> {
+        async fn take_ref(&self, thing: &T);
+    }
+
+    pub struct One;
+
+    #[async_trait]
+    impl<T> Generic<T> for One {
+        async fn take_ref(&self, _: &T) {}
+    }
+
+    pub struct Two;
+
+    #[async_trait]
+    impl<T: Sync> Generic<(T, T)> for Two {
+        async fn take_ref(&self, (a, b): &(T, T)) {
+            let _ = a;
+            let _ = b;
+        }
+    }
+
+    pub struct Three;
+
+    #[async_trait]
+    impl<T> Generic<(T, T, T)> for Three {
+        async fn take_ref(&self, (_a, _b, _c): &(T, T, T)) {}
+    }
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard d71c74de7b445060462c40f223f10cbe270c5250
git clean -fd
