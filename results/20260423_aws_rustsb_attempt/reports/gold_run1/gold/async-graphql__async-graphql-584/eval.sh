#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/http/playground_source.rs b/src/http/playground_source.rs
--- a/src/http/playground_source.rs
+++ b/src/http/playground_source.rs
@@ -608,3 +620,30 @@ impl<'a> GraphQLPlaygroundConfig<'a> {
         self
     }
 }
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+    use std::collections::BTreeMap;
+
+    #[test]
+    fn test_with_setting_can_use_any_json_value() {
+        let settings = GraphQLPlaygroundConfig::new("")
+            .with_setting("string", "string")
+            .with_setting("bool", false)
+            .with_setting("number", 10)
+            .with_setting("null", Value::Null)
+            .with_setting("array", Vec::from([1, 2, 3]))
+            .with_setting("object", BTreeMap::new());
+
+        let json = serde_json::to_value(settings).unwrap();
+        let settings = json["settings"].as_object().unwrap();
+
+        assert!(settings["string"].as_str().is_some());
+        assert!(settings["bool"].as_bool().is_some());
+        assert!(settings["number"].as_u64().is_some());
+        assert!(settings["null"].as_null().is_some());
+        assert!(settings["array"].as_array().is_some());
+        assert!(settings["object"].as_object().is_some());
+    }
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 06a5eb298365b741187baec5b7bc6aec0ad3abab
git clean -fd
