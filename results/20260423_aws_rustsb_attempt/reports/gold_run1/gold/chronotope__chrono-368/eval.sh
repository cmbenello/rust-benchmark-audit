#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
chmod -R 755 /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/datetime.rs b/src/datetime.rs
--- a/src/datetime.rs
+++ b/src/datetime.rs
@@ -2051,6 +2051,8 @@ mod tests {
 
         assert_eq!(DateTime::parse_from_rfc2822("Wed, 18 Feb 2015 23:16:09 +0000"),
                    Ok(FixedOffset::east(0).ymd(2015, 2, 18).and_hms(23, 16, 9)));
+        assert_eq!(DateTime::parse_from_rfc2822("Wed, 18 Feb 2015 23:16:09 -0000"),
+                   Ok(FixedOffset::east(0).ymd(2015, 2, 18).and_hms(23, 16, 9)));
         assert_eq!(DateTime::parse_from_rfc3339("2015-02-18T23:16:09Z"),
                    Ok(FixedOffset::east(0).ymd(2015, 2, 18).and_hms(23, 16, 9)));
         assert_eq!(DateTime::parse_from_rfc2822("Wed, 18 Feb 2015 23:59:60 +0500"),

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard 19dd051d22a223d908ed636d322ba2482adb216b
git clean -fd
