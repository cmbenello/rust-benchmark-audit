#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/offset/mod.rs b/src/offset/mod.rs
--- a/src/offset/mod.rs
+++ b/src/offset/mod.rs
@@ -466,4 +488,25 @@ mod tests {
         let dt = Utc.timestamp_millis(-3600000);
         assert_eq!(dt.to_string(), "1969-12-31 23:00:00 UTC");
     }
+
+    #[test]
+    fn test_negative_nanos() {
+        let dt = Utc.timestamp_nanos(-1_000_000_000);
+        assert_eq!(dt.to_string(), "1969-12-31 23:59:59 UTC");
+        let dt = Utc.timestamp_nanos(-999_999_999);
+        assert_eq!(dt.to_string(), "1969-12-31 23:59:59.000000001 UTC");
+        let dt = Utc.timestamp_nanos(-1);
+        assert_eq!(dt.to_string(), "1969-12-31 23:59:59.999999999 UTC");
+        let dt = Utc.timestamp_nanos(-60_000_000_000);
+        assert_eq!(dt.to_string(), "1969-12-31 23:59:00 UTC");
+        let dt = Utc.timestamp_nanos(-3_600_000_000_000);
+        assert_eq!(dt.to_string(), "1969-12-31 23:00:00 UTC");
+    }
+
+    #[test]
+    fn test_nanos_never_panics() {
+        Utc.timestamp_nanos(i64::max_value());
+        Utc.timestamp_nanos(i64::default());
+        Utc.timestamp_nanos(i64::min_value());
+    }
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 77110ffecbc9831210335e40b46b0f6d00d41cd7
git clean -fd
