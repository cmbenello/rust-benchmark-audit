#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/aya/src/maps/hash_map/hash_map.rs b/aya/src/maps/hash_map/hash_map.rs
--- a/aya/src/maps/hash_map/hash_map.rs
+++ b/aya/src/maps/hash_map/hash_map.rs
@@ -108,43 +108,22 @@ mod tests {
     use libc::{EFAULT, ENOENT};
 
     use crate::{
-        bpf_map_def,
         generated::{
             bpf_attr, bpf_cmd,
             bpf_map_type::{BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_LRU_HASH},
         },
-        maps::{Map, MapData},
-        obj::{self, maps::LegacyMap, BpfSectionKind},
+        maps::Map,
+        obj,
         sys::{override_syscall, SysResult, Syscall},
     };
 
-    use super::*;
+    use super::{
+        super::test_utils::{self, new_map},
+        *,
+    };
 
     fn new_obj_map() -> obj::Map {
-        obj::Map::Legacy(LegacyMap {
-            def: bpf_map_def {
-                map_type: BPF_MAP_TYPE_HASH as u32,
-                key_size: 4,
-                value_size: 4,
-                max_entries: 1024,
-                ..Default::default()
-            },
-            section_index: 0,
-            section_kind: BpfSectionKind::Maps,
-            data: Vec::new(),
-            symbol_index: None,
-        })
-    }
-
-    fn new_map(obj: obj::Map) -> MapData {
-        override_syscall(|call| match call {
-            Syscall::Bpf {
-                cmd: bpf_cmd::BPF_MAP_CREATE,
-                ..
-            } => Ok(1337),
-            call => panic!("unexpected syscall {:?}", call),
-        });
-        MapData::create(obj, "foo", None).unwrap()
+        test_utils::new_obj_map(BPF_MAP_TYPE_HASH)
     }
 
     fn sys_error(value: i32) -> SysResult<c_long> {
diff --git a/aya/src/maps/hash_map/hash_map.rs b/aya/src/maps/hash_map/hash_map.rs
--- a/aya/src/maps/hash_map/hash_map.rs
+++ b/aya/src/maps/hash_map/hash_map.rs
@@ -213,21 +192,10 @@ mod tests {
 
     #[test]
     fn test_try_from_ok_lru() {
-        let map = new_map(obj::Map::Legacy(LegacyMap {
-            def: bpf_map_def {
-                map_type: BPF_MAP_TYPE_LRU_HASH as u32,
-                key_size: 4,
-                value_size: 4,
-                max_entries: 1024,
-                ..Default::default()
-            },
-            section_index: 0,
-            section_kind: BpfSectionKind::Maps,
-            symbol_index: None,
-            data: Vec::new(),
-        }));
-        let map = Map::HashMap(map);
-
+        let map_data = || new_map(test_utils::new_obj_map(BPF_MAP_TYPE_LRU_HASH));
+        let map = Map::HashMap(map_data());
+        assert!(HashMap::<_, u32, u32>::try_from(&map).is_ok());
+        let map = Map::LruHashMap(map_data());
         assert!(HashMap::<_, u32, u32>::try_from(&map).is_ok())
     }
 
diff --git a/aya/src/maps/hash_map/mod.rs b/aya/src/maps/hash_map/mod.rs
--- a/aya/src/maps/hash_map/mod.rs
+++ b/aya/src/maps/hash_map/mod.rs
@@ -41,3 +41,41 @@ pub(crate) fn remove<K: Pod>(map: &MapData, key: &K) -> Result<(), MapError> {
             .into()
         })
 }
+
+#[cfg(test)]
+mod test_utils {
+    use crate::{
+        bpf_map_def,
+        generated::{bpf_cmd, bpf_map_type},
+        maps::MapData,
+        obj::{self, maps::LegacyMap, BpfSectionKind},
+        sys::{override_syscall, Syscall},
+    };
+
+    pub(super) fn new_map(obj: obj::Map) -> MapData {
+        override_syscall(|call| match call {
+            Syscall::Bpf {
+                cmd: bpf_cmd::BPF_MAP_CREATE,
+                ..
+            } => Ok(1337),
+            call => panic!("unexpected syscall {:?}", call),
+        });
+        MapData::create(obj, "foo", None).unwrap()
+    }
+
+    pub(super) fn new_obj_map(map_type: bpf_map_type) -> obj::Map {
+        obj::Map::Legacy(LegacyMap {
+            def: bpf_map_def {
+                map_type: map_type as u32,
+                key_size: 4,
+                value_size: 4,
+                max_entries: 1024,
+                ..Default::default()
+            },
+            section_index: 0,
+            section_kind: BpfSectionKind::Maps,
+            data: Vec::new(),
+            symbol_index: None,
+        })
+    }
+}
diff --git a/aya/src/maps/hash_map/per_cpu_hash_map.rs b/aya/src/maps/hash_map/per_cpu_hash_map.rs
--- a/aya/src/maps/hash_map/per_cpu_hash_map.rs
+++ b/aya/src/maps/hash_map/per_cpu_hash_map.rs
@@ -146,3 +146,30 @@ impl<T: Borrow<MapData>, K: Pod, V: Pod> IterableMap<K, PerCpuValues<V>>
         Self::get(self, key, 0)
     }
 }
+
+#[cfg(test)]
+mod tests {
+    use crate::{
+        generated::bpf_map_type::{BPF_MAP_TYPE_LRU_PERCPU_HASH, BPF_MAP_TYPE_PERCPU_HASH},
+        maps::Map,
+    };
+
+    use super::{super::test_utils, *};
+
+    #[test]
+    fn test_try_from_ok() {
+        let map = Map::PerCpuHashMap(test_utils::new_map(test_utils::new_obj_map(
+            BPF_MAP_TYPE_PERCPU_HASH,
+        )));
+        assert!(PerCpuHashMap::<_, u32, u32>::try_from(&map).is_ok())
+    }
+    #[test]
+    fn test_try_from_ok_lru() {
+        let map_data =
+            || test_utils::new_map(test_utils::new_obj_map(BPF_MAP_TYPE_LRU_PERCPU_HASH));
+        let map = Map::PerCpuHashMap(map_data());
+        assert!(PerCpuHashMap::<_, u32, u32>::try_from(&map).is_ok());
+        let map = Map::PerCpuLruHashMap(map_data());
+        assert!(PerCpuHashMap::<_, u32, u32>::try_from(&map).is_ok())
+    }
+}

EOF_114329324912
cd "aya"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 792f467d40dc4c6c543c62dda7a608863fc364dc
git clean -fd
