#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -905,3 +905,35 @@ mod issue104 {
 
     impl_t1!(Foo, 1);
 }
+
+// https://github.com/dtolnay/async-trait/issues/106
+mod issue106 {
+    use async_trait::async_trait;
+    use std::future::Future;
+
+    #[async_trait]
+    pub trait ProcessPool: Send + Sync {
+        type ThreadPool;
+
+        async fn spawn<F, Fut, T>(&self, work: F) -> T
+        where
+            F: FnOnce(&Self::ThreadPool) -> Fut + Send,
+            Fut: Future<Output = T> + 'static;
+    }
+
+    #[async_trait]
+    impl<P: ?Sized> ProcessPool for &P
+    where
+        P: ProcessPool,
+    {
+        type ThreadPool = P::ThreadPool;
+
+        async fn spawn<F, Fut, T>(&self, work: F) -> T
+        where
+            F: FnOnce(&Self::ThreadPool) -> Fut + Send,
+            Fut: Future<Output = T> + 'static,
+        {
+            (*self).spawn(work).await
+        }
+    }
+}

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard 7c746b6ee601c51c557c9f4c32230f5c9a9934be
git clean -fd
