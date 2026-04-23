#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/weekday.rs b/src/weekday.rs
--- a/src/weekday.rs
+++ b/src/weekday.rs
@@ -331,6 +331,16 @@ mod tests {
         }
     }
 
+    #[test]
+    fn test_formatting_alignment() {
+        // No exhaustive testing here as we just delegate the
+        // implementation to Formatter::pad. Just some basic smoke
+        // testing to ensure that it's in fact being done.
+        assert_eq!(format!("{:x>7}", Weekday::Mon), "xxxxMon");
+        assert_eq!(format!("{:^7}", Weekday::Mon), "  Mon  ");
+        assert_eq!(format!("{:Z<7}", Weekday::Mon), "MonZZZZ");
+    }
+
     #[test]
     #[cfg(feature = "serde")]
     fn test_serde_serialize() {

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard d8a177e4f5cc7512b8cbe8a5d27e68d6cfcfb8fd
git clean -fd
