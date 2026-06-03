#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/datetime.rs b/src/datetime.rs
--- a/src/datetime.rs
+++ b/src/datetime.rs
@@ -1721,4 +1721,34 @@ mod tests {
         assert_eq!(SystemTime::from(epoch.with_timezone(&FixedOffset::east(32400))), UNIX_EPOCH);
         assert_eq!(SystemTime::from(epoch.with_timezone(&FixedOffset::west(28800))), UNIX_EPOCH);
     }
+
+    #[test]
+    fn test_datetime_format_alignment() {
+        let datetime = Utc.ymd(2007, 01, 02);
+
+        // Item::Literal
+        let percent = datetime.format("%%");
+        assert_eq!("  %", format!("{:>3}", percent));
+        assert_eq!("%  ", format!("{:<3}", percent));
+        assert_eq!(" % ", format!("{:^3}", percent));
+
+        // Item::Numeric
+        let year = datetime.format("%Y");
+        assert_eq!("  2007", format!("{:>6}", year));
+        assert_eq!("2007  ", format!("{:<6}", year));
+        assert_eq!(" 2007 ", format!("{:^6}", year));
+
+        // Item::Fixed
+        let tz = datetime.format("%Z");
+        assert_eq!("  UTC", format!("{:>5}", tz));
+        assert_eq!("UTC  ", format!("{:<5}", tz));
+        assert_eq!(" UTC ", format!("{:^5}", tz));
+
+        // [Item::Numeric, Item::Space, Item::Literal, Item::Space, Item::Numeric]
+        let ymd = datetime.format("%Y %B %d");
+        let ymd_formatted = "2007 January 02";
+        assert_eq!(format!("  {}", ymd_formatted), format!("{:>17}", ymd));
+        assert_eq!(format!("{}  ", ymd_formatted), format!("{:<17}", ymd));
+        assert_eq!(format!(" {} ", ymd_formatted), format!("{:^17}", ymd));
+    }
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 96c451ec20f045eebcfb05dd770bd265cdd01c0e
git clean -fd
