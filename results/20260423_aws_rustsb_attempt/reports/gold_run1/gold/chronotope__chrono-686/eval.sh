#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -126,6 +126,7 @@ fn test_datetime_rfc2822_and_rfc3339() {
         DateTime::parse_from_rfc2822("Wed, 18 Feb 2015 23:59:60 +0500"),
         Ok(edt.ymd(2015, 2, 18).and_hms_milli(23, 59, 59, 1_000))
     );
+    assert!(DateTime::parse_from_rfc2822("31 DEC 262143 23:59 -2359").is_err());
     assert_eq!(
         DateTime::parse_from_rfc3339("2015-02-18T23:59:60.234567+05:00"),
         Ok(edt.ymd(2015, 2, 18).and_hms_micro(23, 59, 59, 1_234_567))

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 752e69ae1ff600304250a5da117a3dd40f99581a
git clean -fd
