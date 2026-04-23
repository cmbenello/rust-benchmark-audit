#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/offset/mod.rs b/src/offset/mod.rs
--- a/src/offset/mod.rs
+++ b/src/offset/mod.rs
@@ -558,4 +572,18 @@ mod tests {
         Utc.timestamp_nanos(i64::default());
         Utc.timestamp_nanos(i64::min_value());
     }
+
+    #[test]
+    fn test_negative_micros() {
+        let dt = Utc.timestamp_micros(-1_000_000).unwrap();
+        assert_eq!(dt.to_string(), "1969-12-31 23:59:59 UTC");
+        let dt = Utc.timestamp_micros(-999_999).unwrap();
+        assert_eq!(dt.to_string(), "1969-12-31 23:59:59.000001 UTC");
+        let dt = Utc.timestamp_micros(-1).unwrap();
+        assert_eq!(dt.to_string(), "1969-12-31 23:59:59.999999 UTC");
+        let dt = Utc.timestamp_micros(-60_000_000).unwrap();
+        assert_eq!(dt.to_string(), "1969-12-31 23:59:00 UTC");
+        let dt = Utc.timestamp_micros(-3_600_000_000).unwrap();
+        assert_eq!(dt.to_string(), "1969-12-31 23:00:00 UTC");
+    }
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 21f9ccc5dc74004bf40f0fd79b92dd606a9cb670
git clean -fd
