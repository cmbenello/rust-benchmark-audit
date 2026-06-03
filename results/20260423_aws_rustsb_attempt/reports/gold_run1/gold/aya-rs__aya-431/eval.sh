#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/aya/src/maps/hash_map/hash_map.rs b/aya/src/maps/hash_map/hash_map.rs
--- a/aya/src/maps/hash_map/hash_map.rs
+++ b/aya/src/maps/hash_map/hash_map.rs
@@ -311,6 +317,27 @@ mod tests {
         assert!(hm.insert(1, 42, 0).is_ok());
     }
 
+    #[test]
+    fn test_insert_boxed_ok() {
+        override_syscall(|call| match call {
+            Syscall::Bpf {
+                cmd: bpf_cmd::BPF_MAP_UPDATE_ELEM,
+                ..
+            } => Ok(1),
+            _ => sys_error(EFAULT),
+        });
+
+        let mut map = MapData {
+            obj: new_obj_map(),
+            fd: Some(42),
+            pinned: false,
+            btf_fd: None,
+        };
+        let mut hm = HashMap::<_, u32, u32>::new(&mut map).unwrap();
+
+        assert!(hm.insert(Box::new(1), Box::new(42), 0).is_ok());
+    }
+
     #[test]
     fn test_remove_syscall_error() {
         override_syscall(|_| sys_error(EFAULT));

EOF_114329324912
cd "aya"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 82773f46c8cff830b36b8b4f57100952eb6ec8d2
git clean -fd
