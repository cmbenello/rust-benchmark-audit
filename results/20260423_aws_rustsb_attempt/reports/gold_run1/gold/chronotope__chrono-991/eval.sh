#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/naive/isoweek.rs b/src/naive/isoweek.rs
--- a/src/naive/isoweek.rs
+++ b/src/naive/isoweek.rs
@@ -164,4 +165,38 @@ mod tests {
         assert_eq!(maxweek.week0(), 0);
         assert_eq!(format!("{:?}", maxweek), NaiveDate::MAX.format("%G-W%V").to_string());
     }
+
+    #[test]
+    fn test_iso_week_equivalence_for_first_week() {
+        let monday = NaiveDate::from_ymd_opt(2024, 12, 30).unwrap();
+        let friday = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
+
+        assert_eq!(monday.iso_week(), friday.iso_week());
+    }
+
+    #[test]
+    fn test_iso_week_equivalence_for_last_week() {
+        let monday = NaiveDate::from_ymd_opt(2026, 12, 28).unwrap();
+        let friday = NaiveDate::from_ymd_opt(2027, 1, 1).unwrap();
+
+        assert_eq!(monday.iso_week(), friday.iso_week());
+    }
+
+    #[test]
+    fn test_iso_week_ordering_for_first_week() {
+        let monday = NaiveDate::from_ymd_opt(2024, 12, 30).unwrap();
+        let friday = NaiveDate::from_ymd_opt(2025, 1, 3).unwrap();
+
+        assert!(monday.iso_week() >= friday.iso_week());
+        assert!(monday.iso_week() <= friday.iso_week());
+    }
+
+    #[test]
+    fn test_iso_week_ordering_for_last_week() {
+        let monday = NaiveDate::from_ymd_opt(2026, 12, 28).unwrap();
+        let friday = NaiveDate::from_ymd_opt(2027, 1, 1).unwrap();
+
+        assert!(monday.iso_week() >= friday.iso_week());
+        assert!(monday.iso_week() <= friday.iso_week());
+    }
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard daa86a77d36d74f474913fd3b560a40f1424bd77
git clean -fd
