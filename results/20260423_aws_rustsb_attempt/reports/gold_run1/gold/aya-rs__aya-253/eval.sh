#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/aya/src/programs/links.rs b/aya/src/programs/links.rs
--- a/aya/src/programs/links.rs
+++ b/aya/src/programs/links.rs
@@ -153,7 +192,7 @@ pub(crate) use define_link_wrapper;
 mod tests {
     use std::{cell::RefCell, rc::Rc};
 
-    use crate::programs::ProgramError;
+    use crate::programs::{OwnedLink, ProgramError};
 
     use super::{Link, LinkMap};
 
diff --git a/aya/src/programs/links.rs b/aya/src/programs/links.rs
--- a/aya/src/programs/links.rs
+++ b/aya/src/programs/links.rs
@@ -257,4 +296,58 @@ mod tests {
         assert!(*l1_detached.borrow() == 1);
         assert!(*l2_detached.borrow() == 1);
     }
+
+    #[test]
+    fn test_owned_detach() {
+        let l1 = TestLink::new(1, 2);
+        let l1_detached = Rc::clone(&l1.detached);
+        let l2 = TestLink::new(1, 3);
+        let l2_detached = Rc::clone(&l2.detached);
+
+        let owned_l1 = {
+            let mut links = LinkMap::new();
+            let id1 = links.insert(l1).unwrap();
+            links.insert(l2).unwrap();
+            // manually forget one link
+            let owned_l1 = links.forget(id1);
+            assert!(*l1_detached.borrow() == 0);
+            assert!(*l2_detached.borrow() == 0);
+            owned_l1.unwrap()
+        };
+
+        // l2 is detached on `Drop`, but l1 is still alive
+        assert!(*l1_detached.borrow() == 0);
+        assert!(*l2_detached.borrow() == 1);
+
+        // manually detach l1
+        assert!(owned_l1.detach().is_ok());
+        assert!(*l1_detached.borrow() == 1);
+        assert!(*l2_detached.borrow() == 1);
+    }
+
+    #[test]
+    fn test_owned_drop() {
+        let l1 = TestLink::new(1, 2);
+        let l1_detached = Rc::clone(&l1.detached);
+        let l2 = TestLink::new(1, 3);
+        let l2_detached = Rc::clone(&l2.detached);
+
+        {
+            let mut links = LinkMap::new();
+            let id1 = links.insert(l1).unwrap();
+            links.insert(l2).unwrap();
+
+            // manually forget one link and wrap in OwnedLink
+            let _ = OwnedLink {
+                inner: Some(links.forget(id1).unwrap()),
+            };
+
+            // OwnedLink was dropped in the statement above
+            assert!(*l1_detached.borrow() == 1);
+            assert!(*l2_detached.borrow() == 0);
+        };
+
+        assert!(*l1_detached.borrow() == 1);
+        assert!(*l2_detached.borrow() == 1);
+    }
 }

EOF_114329324912
cd "aya"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard b9a544831c5b4cd8728e9cca6580c14a623b7793
git clean -fd
