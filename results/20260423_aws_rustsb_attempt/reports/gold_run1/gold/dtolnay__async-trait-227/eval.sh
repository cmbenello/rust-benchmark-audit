#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -54,6 +54,10 @@ trait Trait {
     async fn calls_mut(&mut self) {
         self.selfmut().await;
     }
+
+    async fn cfg_param(&self, param: u8);
+    async fn cfg_param_wildcard(&self, _: u8);
+    async fn cfg_param_tuple(&self, (left, right): (u8, u8));
 }
 
 struct Struct;
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -87,6 +91,17 @@ impl Trait for Struct {
     async fn calls_mut(&mut self) {
         self.selfmut().await;
     }
+
+    async fn cfg_param(&self, #[cfg(any())] param: u8, #[cfg(all())] _unused: u8) {}
+
+    async fn cfg_param_wildcard(&self, #[cfg(any())] _: u8, #[cfg(all())] _: u8) {}
+
+    async fn cfg_param_tuple(
+        &self,
+        #[cfg(any())] (left, right): (u8, u8),
+        #[cfg(all())] (_left, _right): (u8, u8),
+    ) {
+    }
 }
 
 pub async fn test() {

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 6050c94ca7be287be1657fcefd299165e39c7ef2
git clean -fd
