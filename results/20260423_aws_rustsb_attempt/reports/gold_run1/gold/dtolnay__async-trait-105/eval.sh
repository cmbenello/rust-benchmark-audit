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
@@ -556,10 +556,10 @@ pub mod issue45 {
 pub mod issue46 {
     use async_trait::async_trait;
 
-    macro_rules! implement_commands {
+    macro_rules! implement_commands_workaround {
         ($tyargs:tt : $ty:tt) => {
             #[async_trait]
-            pub trait AsyncCommands: Sized {
+            pub trait AsyncCommands1: Sized {
                 async fn f<$tyargs: $ty>(&mut self, x: $tyargs) {
                     self.f(x).await
                 }
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -567,7 +567,22 @@ pub mod issue46 {
         };
     }
 
-    implement_commands!(K: Send);
+    implement_commands_workaround!(K: Send);
+
+    macro_rules! implement_commands {
+        (
+            $tyargs:ident : $ty:ident
+        ) => {
+            #[async_trait]
+            pub trait AsyncCommands2: Sized {
+                async fn f<$tyargs: $ty>(&mut self, x: $tyargs) {
+                    self.f(x).await
+                }
+            }
+        };
+    }
+
+    implement_commands! { K: Send }
 }
 
 // https://github.com/dtolnay/async-trait/issues/53
diff --git a/tests/test.rs b/tests/test.rs
--- a/tests/test.rs
+++ b/tests/test.rs
@@ -867,3 +882,27 @@ pub mod issue92 {
         }
     }
 }
+
+mod issue104 {
+    use async_trait::async_trait;
+
+    #[async_trait]
+    trait T1 {
+        async fn id(&self) -> i32;
+    }
+
+    macro_rules! impl_t1 {
+        ($ty: ty, $id: expr) => {
+            #[async_trait]
+            impl T1 for $ty {
+                async fn id(&self) -> i32 {
+                    $id
+                }
+            }
+        };
+    }
+
+    struct Foo;
+
+    impl_t1!(Foo, 1);
+}

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard 5220bbd2bdf30fa43c78d08a8e9a9f311a239cf8
git clean -fd
