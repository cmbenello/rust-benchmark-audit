#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_bytes_odd_alloc.rs b/tests/test_bytes_odd_alloc.rs
--- a/tests/test_bytes_odd_alloc.rs
+++ b/tests/test_bytes_odd_alloc.rs
@@ -41,7 +41,7 @@ unsafe impl GlobalAlloc for Odd {
             };
             System.dealloc(ptr.offset(-1), new_layout);
         } else {
-            System.alloc(layout);
+            System.dealloc(ptr, layout);
         }
     }
 }
diff --git /dev/null b/tests/test_bytes_vec_alloc.rs
new file mode 100644
--- /dev/null
+++ b/tests/test_bytes_vec_alloc.rs
@@ -0,0 +1,75 @@
+use std::alloc::{GlobalAlloc, Layout, System};
+use std::{mem, ptr};
+
+use bytes::{Buf, Bytes};
+
+#[global_allocator]
+static LEDGER: Ledger = Ledger;
+
+struct Ledger;
+
+const USIZE_SIZE: usize = mem::size_of::<usize>();
+
+unsafe impl GlobalAlloc for Ledger {
+    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
+        if layout.align() == 1 && layout.size() > 0 {
+            // Allocate extra space to stash a record of
+            // how much space there was.
+            let orig_size = layout.size();
+            let size = orig_size + USIZE_SIZE;
+            let new_layout = match Layout::from_size_align(size, 1) {
+                Ok(layout) => layout,
+                Err(_err) => return ptr::null_mut(),
+            };
+            let ptr = System.alloc(new_layout);
+            if !ptr.is_null() {
+                (ptr as *mut usize).write(orig_size);
+                let ptr = ptr.offset(USIZE_SIZE as isize);
+                ptr
+            } else {
+                ptr
+            }
+        } else {
+            System.alloc(layout)
+        }
+    }
+
+    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
+        if layout.align() == 1 && layout.size() > 0 {
+            let off_ptr = (ptr as *mut usize).offset(-1);
+            let orig_size = off_ptr.read();
+            if orig_size != layout.size() {
+                panic!("bad dealloc: alloc size was {}, dealloc size is {}", orig_size, layout.size());
+            }
+
+            let new_layout = match Layout::from_size_align(layout.size() + USIZE_SIZE, 1) {
+                Ok(layout) => layout,
+                Err(_err) => std::process::abort(),
+            };
+            System.dealloc(off_ptr as *mut u8, new_layout);
+        } else {
+            System.dealloc(ptr, layout);
+        }
+    }
+}
+#[test]
+fn test_bytes_advance() {
+    let mut bytes = Bytes::from(vec![10, 20, 30]);
+    bytes.advance(1);
+    drop(bytes);
+}
+
+#[test]
+fn test_bytes_truncate() {
+    let mut bytes = Bytes::from(vec![10, 20, 30]);
+    bytes.truncate(2);
+    drop(bytes);
+}
+
+#[test]
+fn test_bytes_truncate_and_advance() {
+    let mut bytes = Bytes::from(vec![10, 20, 30]);
+    bytes.truncate(2);
+    bytes.advance(1);
+    drop(bytes);
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 729bc7c2084a42fda2c62da6933951fa7ac875aa
git clean -fd
