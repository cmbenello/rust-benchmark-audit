#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/round.rs b/src/round.rs
--- a/src/round.rs
+++ b/src/round.rs
@@ -395,6 +399,28 @@ mod tests {
             dt.duration_round(Duration::days(1)).unwrap().to_string(),
             "2012-12-13 00:00:00 UTC"
         );
+
+        // timezone east
+        let dt = FixedOffset::east(1 * 3600).ymd(2020, 10, 27).and_hms(15, 0, 0);
+        assert_eq!(
+            dt.duration_round(Duration::days(1)).unwrap().to_string(),
+            "2020-10-28 00:00:00 +01:00"
+        );
+        assert_eq!(
+            dt.duration_round(Duration::weeks(1)).unwrap().to_string(),
+            "2020-10-29 00:00:00 +01:00"
+        );
+
+        // timezone west
+        let dt = FixedOffset::west(1 * 3600).ymd(2020, 10, 27).and_hms(15, 0, 0);
+        assert_eq!(
+            dt.duration_round(Duration::days(1)).unwrap().to_string(),
+            "2020-10-28 00:00:00 -01:00"
+        );
+        assert_eq!(
+            dt.duration_round(Duration::weeks(1)).unwrap().to_string(),
+            "2020-10-29 00:00:00 -01:00"
+        );
     }
 
     #[test]
diff --git a/src/round.rs b/src/round.rs
--- a/src/round.rs
+++ b/src/round.rs
@@ -443,6 +469,28 @@ mod tests {
             dt.duration_trunc(Duration::days(1)).unwrap().to_string(),
             "2012-12-12 00:00:00 UTC"
         );
+
+        // timezone east
+        let dt = FixedOffset::east(1 * 3600).ymd(2020, 10, 27).and_hms(15, 0, 0);
+        assert_eq!(
+            dt.duration_trunc(Duration::days(1)).unwrap().to_string(),
+            "2020-10-27 00:00:00 +01:00"
+        );
+        assert_eq!(
+            dt.duration_trunc(Duration::weeks(1)).unwrap().to_string(),
+            "2020-10-22 00:00:00 +01:00"
+        );
+
+        // timezone west
+        let dt = FixedOffset::west(1 * 3600).ymd(2020, 10, 27).and_hms(15, 0, 0);
+        assert_eq!(
+            dt.duration_trunc(Duration::days(1)).unwrap().to_string(),
+            "2020-10-27 00:00:00 -01:00"
+        );
+        assert_eq!(
+            dt.duration_trunc(Duration::weeks(1)).unwrap().to_string(),
+            "2020-10-22 00:00:00 -01:00"
+        );
     }
 
     #[test]

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard 2e0936bc3dfe4799f49fec6c0e237e0adb2aa2c5
git clean -fd
