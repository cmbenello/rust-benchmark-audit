#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/naive/datetime/tests.rs b/src/naive/datetime/tests.rs
--- a/src/naive/datetime/tests.rs
+++ b/src/naive/datetime/tests.rs
@@ -1,7 +1,7 @@
 use super::NaiveDateTime;
 use crate::naive::{NaiveDate, MAX_DATE, MIN_DATE};
 use crate::oldtime::Duration;
-use crate::Datelike;
+use crate::{Datelike, FixedOffset, Utc};
 use std::i64;
 
 #[test]
diff --git a/src/naive/datetime/tests.rs b/src/naive/datetime/tests.rs
--- a/src/naive/datetime/tests.rs
+++ b/src/naive/datetime/tests.rs
@@ -240,3 +240,16 @@ fn test_nanosecond_range() {
         NaiveDateTime::from_timestamp(nanos / A_BILLION, (nanos % A_BILLION) as u32)
     );
 }
+
+#[test]
+fn test_and_timezone() {
+    let ndt = NaiveDate::from_ymd(2022, 6, 15).and_hms(18, 59, 36);
+    let dt_utc = ndt.and_local_timezone(Utc).unwrap();
+    assert_eq!(dt_utc.naive_local(), ndt);
+    assert_eq!(dt_utc.timezone(), Utc);
+
+    let offset_tz = FixedOffset::west(4 * 3600);
+    let dt_offset = ndt.and_local_timezone(offset_tz).unwrap();
+    assert_eq!(dt_offset.naive_local(), ndt);
+    assert_eq!(dt_offset.timezone(), offset_tz);
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 13e1d483657a2e4d0f06a8692432cb90c4e98e9a
git clean -fd
