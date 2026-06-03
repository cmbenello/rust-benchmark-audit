#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -622,3 +622,30 @@ pub mod issue87 {
         }
     }
 }
+
+// https://github.com/dtolnay/async-trait/issues/89
+pub mod issue89 {
+    #![allow(bare_trait_objects)]
+
+    use async_trait::async_trait;
+
+    #[async_trait]
+    trait Trait {
+        async fn f(&self);
+    }
+
+    #[async_trait]
+    impl Trait for Send + Sync {
+        async fn f(&self) {}
+    }
+
+    #[async_trait]
+    impl Trait for dyn Fn(i8) + Send + Sync {
+        async fn f(&self) {}
+    }
+
+    #[async_trait]
+    impl Trait for (dyn Fn(u8) + Send + Sync) {
+        async fn f(&self) {}
+    }
+}
diff --git /dev/null b/tests/ui/bare-trait-object.rs
new file mode 100644
--- /dev/null
+++ b/tests/ui/bare-trait-object.rs
@@ -0,0 +1,15 @@
+#![deny(bare_trait_objects)]
+
+use async_trait::async_trait;
+
+#[async_trait]
+trait Trait {
+    async fn f(&self);
+}
+
+#[async_trait]
+impl Trait for Send + Sync {
+    async fn f(&self) {}
+}
+
+fn main() {}
diff --git /dev/null b/tests/ui/bare-trait-object.stderr
new file mode 100644
--- /dev/null
+++ b/tests/ui/bare-trait-object.stderr
@@ -0,0 +1,11 @@
+error: trait objects without an explicit `dyn` are deprecated
+  --> $DIR/bare-trait-object.rs:11:16
+   |
+11 | impl Trait for Send + Sync {
+   |                ^^^^^^^^^^^ help: use `dyn`: `dyn Send + Sync`
+   |
+note: the lint level is defined here
+  --> $DIR/bare-trait-object.rs:1:9
+   |
+1  | #![deny(bare_trait_objects)]
+   |         ^^^^^^^^^^^^^^^^^^

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 05c24928d08c46ee1a916b1dff7f6dd389b21c94
git clean -fd
