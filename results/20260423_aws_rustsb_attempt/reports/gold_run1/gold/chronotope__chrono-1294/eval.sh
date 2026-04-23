#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -1486,3 +1486,36 @@ fn locale_decimal_point() {
     assert_eq!(dt.format_localized("%T%.6f", ar_SY).to_string(), "18:58:00.123456");
     assert_eq!(dt.format_localized("%T%.9f", ar_SY).to_string(), "18:58:00.123456780");
 }
+
+/// This is an extended test for <https://github.com/chronotope/chrono/issues/1289>.
+#[test]
+fn nano_roundrip() {
+    const BILLION: i64 = 1_000_000_000;
+
+    for nanos in [
+        i64::MIN,
+        i64::MIN + 1,
+        i64::MIN + 2,
+        i64::MIN + BILLION - 1,
+        i64::MIN + BILLION,
+        i64::MIN + BILLION + 1,
+        -BILLION - 1,
+        -BILLION,
+        -BILLION + 1,
+        0,
+        BILLION - 1,
+        BILLION,
+        BILLION + 1,
+        i64::MAX - BILLION - 1,
+        i64::MAX - BILLION,
+        i64::MAX - BILLION + 1,
+        i64::MAX - 2,
+        i64::MAX - 1,
+        i64::MAX,
+    ] {
+        println!("nanos: {}", nanos);
+        let dt = Utc.timestamp_nanos(nanos);
+        let nanos2 = dt.timestamp_nanos_opt().expect("value roundtrips");
+        assert_eq!(nanos, nanos2);
+    }
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 46ad2c2b2c901eb20e43a7fca025aac02605bda4
git clean -fd
