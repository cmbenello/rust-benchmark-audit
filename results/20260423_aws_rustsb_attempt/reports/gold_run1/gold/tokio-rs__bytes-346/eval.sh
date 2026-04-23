#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/test_bytes_odd_alloc.rs
new file mode 100644
--- /dev/null
+++ b/tests/test_bytes_odd_alloc.rs
@@ -0,0 +1,67 @@
+//! Test using `Bytes` with an allocator that hands out "odd" pointers for
+//! vectors (pointers where the LSB is set).
+
+use std::alloc::{GlobalAlloc, Layout, System};
+use std::ptr;
+
+use bytes::Bytes;
+
+#[global_allocator]
+static ODD: Odd = Odd;
+
+struct Odd;
+
+unsafe impl GlobalAlloc for Odd {
+    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
+        if layout.align() == 1 && layout.size() > 0 {
+            // Allocate slightly bigger so that we can offset the pointer by 1
+            let size = layout.size() + 1;
+            let new_layout = match Layout::from_size_align(size, 1) {
+                Ok(layout) => layout,
+                Err(_err) => return ptr::null_mut(),
+            };
+            let ptr = System.alloc(new_layout);
+            if !ptr.is_null() {
+                let ptr = ptr.offset(1);
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
+            let size = layout.size() + 1;
+            let new_layout = match Layout::from_size_align(size, 1) {
+                Ok(layout) => layout,
+                Err(_err) => std::process::abort(),
+            };
+            System.dealloc(ptr.offset(-1), new_layout);
+        } else {
+            System.alloc(layout);
+        }
+    }
+}
+
+#[test]
+fn sanity_check_odd_allocator() {
+    let vec = vec![33u8; 1024];
+    let p = vec.as_ptr() as usize;
+    assert!(p & 0x1 == 0x1, "{:#b}", p);
+}
+
+#[test]
+fn test_bytes_from_vec_drop() {
+    let vec = vec![33u8; 1024];
+    let _b = Bytes::from(vec);
+}
+
+#[test]
+fn test_bytes_clone_drop() {
+    let vec = vec![33u8; 1024];
+    let b1 = Bytes::from(vec);
+    let _b2 = b1.clone();
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 8ae3bb2104fda9a02d55ac5635974ca1b5a49ebb
git clean -fd
