#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/federation.rs b/tests/federation.rs
--- a/tests/federation.rs
+++ b/tests/federation.rs
@@ -534,6 +534,8 @@ pub async fn test_entity_inaccessible() {
     assert!(schema_sdl.contains("input MyInputObjInaccessible @inaccessible"));
     // INPUT_FIELD_DEFINITION
     assert!(schema_sdl.contains("inputFieldInaccessibleA: Int! @inaccessible"));
+    // no trailing spaces
+    assert!(!schema_sdl.contains(" \n"));
 }
 
 #[tokio::test]
diff --git a/tests/federation.rs b/tests/federation.rs
--- a/tests/federation.rs
+++ b/tests/federation.rs
@@ -733,4 +735,6 @@ pub async fn test_entity_tag() {
     assert!(
         schema_sdl.contains(r#"inputFieldTaggedA: Int! @tag(name: "tagged_input_object_field")"#)
     );
+    // no trailing spaces
+    assert!(!schema_sdl.contains(" \n"));
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 337279f6c76885e1f792cff50e3e455efd664812
git clean -fd
