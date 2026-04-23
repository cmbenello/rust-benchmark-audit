#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/askama_shared/src/filters/mod.rs b/askama_shared/src/filters/mod.rs
--- a/askama_shared/src/filters/mod.rs
+++ b/askama_shared/src/filters/mod.rs
@@ -655,6 +649,9 @@ mod tests {
         assert_eq!(capitalize(&"").unwrap(), "".to_string());
         assert_eq!(capitalize(&"FoO").unwrap(), "Foo".to_string());
         assert_eq!(capitalize(&"foO BAR").unwrap(), "Foo bar".to_string());
+        assert_eq!(capitalize(&"äØÄÅÖ").unwrap(), "Äøäåö".to_string());
+        assert_eq!(capitalize(&"ß").unwrap(), "SS".to_string());
+        assert_eq!(capitalize(&"ßß").unwrap(), "SSß".to_string());
     }
 
     #[test]

EOF_114329324912
cd "askama_shared"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard b14982f97ffd20039286171d56e6fcfab21f56bc
git clean -fd
