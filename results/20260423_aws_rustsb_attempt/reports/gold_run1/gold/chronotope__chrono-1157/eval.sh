#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/offset/fixed.rs b/src/offset/fixed.rs
--- a/src/offset/fixed.rs
+++ b/src/offset/fixed.rs
@@ -246,6 +259,7 @@ impl<Tz: TimeZone> Sub<FixedOffset> for DateTime<Tz> {
 mod tests {
     use super::FixedOffset;
     use crate::offset::TimeZone;
+    use std::str::FromStr;
 
     #[test]
     fn test_date_extreme_offset() {
diff --git a/src/offset/fixed.rs b/src/offset/fixed.rs
--- a/src/offset/fixed.rs
+++ b/src/offset/fixed.rs
@@ -292,4 +306,14 @@ mod tests {
             "2012-03-04T05:06:07-23:59:59".to_string()
         );
     }
+
+    #[test]
+    fn test_parse_offset() {
+        let offset = FixedOffset::from_str("-0500").unwrap();
+        assert_eq!(offset.local_minus_utc, -5 * 3600);
+        let offset = FixedOffset::from_str("-08:00").unwrap();
+        assert_eq!(offset.local_minus_utc, -8 * 3600);
+        let offset = FixedOffset::from_str("+06:30").unwrap();
+        assert_eq!(offset.local_minus_utc, (6 * 3600) + 1800);
+    }
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard ea9398eb0d8fa55fe0064093af32f45443e5c0e0
git clean -fd
