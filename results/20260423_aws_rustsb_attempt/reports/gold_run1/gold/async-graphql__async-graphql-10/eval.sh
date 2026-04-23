#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/http/mod.rs b/src/http/mod.rs
--- a/src/http/mod.rs
+++ b/src/http/mod.rs
@@ -259,6 +270,30 @@ mod tests {
         );
     }
 
+    #[test]
+    fn test_response_error_with_extension() {
+        let err = ExtendedError(
+            "MyErrorMessage".to_owned(),
+            json!({
+                "code": "MY_TEST_CODE"
+            }),
+        );
+
+        let resp = GQLResponse(Err(err.into()));
+
+        assert_eq!(
+            serde_json::to_value(resp).unwrap(),
+            json!({
+                "errors": [{
+                    "message":"MyErrorMessage",
+                    "extensions": {
+                        "code": "MY_TEST_CODE"
+                    }
+                }]
+            })
+        );
+    }
+
     #[test]
     fn test_response_error() {
         let resp = GQLResponse(Err(anyhow::anyhow!("error")));
diff --git a/src/http/mod.rs b/src/http/mod.rs
--- a/src/http/mod.rs
+++ b/src/http/mod.rs
@@ -293,3 +328,12 @@ mod tests {
         );
     }
 }
+
+fn get_error_extensions(err: &crate::Error) -> Option<&serde_json::Value> {
+    if let Some(extended_err) = err.downcast_ref::<ExtendedError>() {
+        if extended_err.1.is_object() {
+            return Some(&extended_err.1);
+        }
+    }
+    None
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard ba64ecc31d8c94b29b481a4c6a6573e397a1a132
git clean -fd
