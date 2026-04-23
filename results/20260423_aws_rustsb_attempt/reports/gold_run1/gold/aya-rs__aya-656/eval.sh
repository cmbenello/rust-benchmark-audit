#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/aya-obj/src/obj.rs b/aya-obj/src/obj.rs
--- a/aya-obj/src/obj.rs
+++ b/aya-obj/src/obj.rs
@@ -1486,7 +1486,7 @@ pub fn copy_instructions(data: &[u8]) -> Result<Vec<bpf_insn>, ParseError> {
 #[cfg(test)]
 mod tests {
     use alloc::vec;
-    use matches::assert_matches;
+    use assert_matches::assert_matches;
     use object::Endianness;
 
     use super::*;
diff --git a/aya/src/maps/bloom_filter.rs b/aya/src/maps/bloom_filter.rs
--- a/aya/src/maps/bloom_filter.rs
+++ b/aya/src/maps/bloom_filter.rs
@@ -88,8 +88,8 @@ mod tests {
         obj::{self, maps::LegacyMap, BpfSectionKind},
         sys::{override_syscall, SysResult, Syscall},
     };
+    use assert_matches::assert_matches;
     use libc::{EFAULT, ENOENT};
-    use matches::assert_matches;
     use std::io;
 
     fn new_obj_map() -> obj::Map {
diff --git a/aya/src/maps/hash_map/hash_map.rs b/aya/src/maps/hash_map/hash_map.rs
--- a/aya/src/maps/hash_map/hash_map.rs
+++ b/aya/src/maps/hash_map/hash_map.rs
@@ -107,8 +107,8 @@ impl<T: Borrow<MapData>, K: Pod, V: Pod> IterableMap<K, V> for HashMap<T, K, V>
 mod tests {
     use std::io;
 
+    use assert_matches::assert_matches;
     use libc::{EFAULT, ENOENT};
-    use matches::assert_matches;
 
     use crate::{
         bpf_map_def,
diff --git a/aya/src/maps/lpm_trie.rs b/aya/src/maps/lpm_trie.rs
--- a/aya/src/maps/lpm_trie.rs
+++ b/aya/src/maps/lpm_trie.rs
@@ -207,8 +207,8 @@ mod tests {
         obj::{self, maps::LegacyMap, BpfSectionKind},
         sys::{override_syscall, SysResult, Syscall},
     };
+    use assert_matches::assert_matches;
     use libc::{EFAULT, ENOENT};
-    use matches::assert_matches;
     use std::{io, mem, net::Ipv4Addr};
 
     fn new_obj_map() -> obj::Map {
diff --git a/aya/src/maps/mod.rs b/aya/src/maps/mod.rs
--- a/aya/src/maps/mod.rs
+++ b/aya/src/maps/mod.rs
@@ -842,8 +842,8 @@ impl<T: Pod> Deref for PerCpuValues<T> {
 
 #[cfg(test)]
 mod tests {
+    use assert_matches::assert_matches;
     use libc::EFAULT;
-    use matches::assert_matches;
 
     use crate::{
         bpf_map_def,
diff --git a/aya/src/maps/perf/perf_buffer.rs b/aya/src/maps/perf/perf_buffer.rs
--- a/aya/src/maps/perf/perf_buffer.rs
+++ b/aya/src/maps/perf/perf_buffer.rs
@@ -319,7 +319,7 @@ mod tests {
         generated::perf_event_mmap_page,
         sys::{override_syscall, Syscall, TEST_MMAP_RET},
     };
-    use matches::assert_matches;
+    use assert_matches::assert_matches;
     use std::{fmt::Debug, mem};
 
     const PAGE_SIZE: usize = 4096;
diff --git a/aya/src/programs/links.rs b/aya/src/programs/links.rs
--- a/aya/src/programs/links.rs
+++ b/aya/src/programs/links.rs
@@ -353,7 +353,7 @@ pub enum LinkError {
 
 #[cfg(test)]
 mod tests {
-    use matches::assert_matches;
+    use assert_matches::assert_matches;
     use std::{cell::RefCell, fs::File, mem, os::unix::io::AsRawFd, rc::Rc};
     use tempfile::tempdir;
 
diff --git a/aya/src/util.rs b/aya/src/util.rs
--- a/aya/src/util.rs
+++ b/aya/src/util.rs
@@ -331,6 +332,19 @@ pub(crate) fn bytes_of_slice<T: Pod>(val: &[T]) -> &[u8] {
 #[cfg(test)]
 mod tests {
     use super::*;
+    use assert_matches::assert_matches;
+
+    #[test]
+    fn test_parse_kernel_version_string() {
+        // WSL.
+        assert_matches!(KernelVersion::parse_kernel_version_string("5.15.90.1-microsoft-standard-WSL2"), Ok(kernel_version) => {
+            assert_eq!(kernel_version, KernelVersion::new(5, 15, 90))
+        });
+        // uname -r on Fedora.
+        assert_matches!(KernelVersion::parse_kernel_version_string("6.3.11-200.fc38.x86_64"), Ok(kernel_version) => {
+            assert_eq!(kernel_version, KernelVersion::new(6, 3, 11))
+        });
+    }
 
     #[test]
     fn test_parse_online_cpus() {
diff --git a/test/integration-test/Cargo.toml b/test/integration-test/Cargo.toml
--- a/test/integration-test/Cargo.toml
+++ b/test/integration-test/Cargo.toml
@@ -6,12 +6,12 @@ publish = false
 
 [dependencies]
 anyhow = "1"
+assert_matches = "1.5.0"
 aya = { path = "../../aya" }
 aya-log = { path = "../../aya-log" }
 aya-obj = { path = "../../aya-obj" }
 libc = { version = "0.2.105" }
 log = "0.4"
-matches = "0.1.8"
 object = { version = "0.31", default-features = false, features = [
     "elf",
     "read_core",
diff --git a/test/integration-test/src/tests/rbpf.rs b/test/integration-test/src/tests/rbpf.rs
--- a/test/integration-test/src/tests/rbpf.rs
+++ b/test/integration-test/src/tests/rbpf.rs
@@ -8,7 +8,7 @@ fn run_with_rbpf() {
     let object = Object::parse(crate::PASS).unwrap();
 
     assert_eq!(object.programs.len(), 1);
-    matches::assert_matches!(object.programs["pass"].section, ProgramSection::Xdp { .. });
+    assert_matches::assert_matches!(object.programs["pass"].section, ProgramSection::Xdp { .. });
     assert_eq!(object.programs["pass"].section.name(), "pass");
 
     let instructions = &object
diff --git a/test/integration-test/src/tests/rbpf.rs b/test/integration-test/src/tests/rbpf.rs
--- a/test/integration-test/src/tests/rbpf.rs
+++ b/test/integration-test/src/tests/rbpf.rs
@@ -35,7 +35,7 @@ fn use_map_with_rbpf() {
     let mut object = Object::parse(crate::MULTIMAP_BTF).unwrap();
 
     assert_eq!(object.programs.len(), 1);
-    matches::assert_matches!(
+    assert_matches::assert_matches!(
         object.programs["tracepoint"].section,
         ProgramSection::TracePoint { .. }
     );

EOF_114329324912
cd "aya-obj"
cargo test --no-fail-fast --all-features
cd ../
cd "aya"
cargo test --no-fail-fast --all-features
cd ../
cd "test/integration-test"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard 61608e64583f9dc599eef9b8db098f38a765b285
git clean -fd
