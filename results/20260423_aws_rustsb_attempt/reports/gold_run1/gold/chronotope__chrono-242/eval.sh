#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/format/parse.rs b/src/format/parse.rs
--- a/src/format/parse.rs
+++ b/src/format/parse.rs
@@ -570,6 +572,10 @@ fn test_parse() {
     check!("zulu",      [fix!(TimezoneOffsetZ), lit!("ulu")]; offset: 0);
     check!("+1234ulu",  [fix!(TimezoneOffsetZ), lit!("ulu")]; offset: 754 * 60);
     check!("+12:34ulu", [fix!(TimezoneOffsetZ), lit!("ulu")]; offset: 754 * 60);
+    check!("Z",         [internal_fix!(TimezoneOffsetPermissive)]; offset: 0);
+    check!("z",         [internal_fix!(TimezoneOffsetPermissive)]; offset: 0);
+    check!("+12:00",    [internal_fix!(TimezoneOffsetPermissive)]; offset: 12 * 60 * 60);
+    check!("+12",       [internal_fix!(TimezoneOffsetPermissive)]; offset: 12 * 60 * 60);
     check!("???",       [fix!(TimezoneName)]; BAD_FORMAT); // not allowed
 
     // some practical examples
diff --git a/src/format/strftime.rs b/src/format/strftime.rs
--- a/src/format/strftime.rs
+++ b/src/format/strftime.rs
@@ -368,6 +379,9 @@ fn test_strftime_items() {
     assert_eq!(parse_and_collect("%-e"), [num!(Day)]);
     assert_eq!(parse_and_collect("%0e"), [num0!(Day)]);
     assert_eq!(parse_and_collect("%_e"), [nums!(Day)]);
+    assert_eq!(parse_and_collect("%z"), [fix!(TimezoneOffset)]);
+    assert_eq!(parse_and_collect("%#z"), [internal_fix!(TimezoneOffsetPermissive)]);
+    assert_eq!(parse_and_collect("%#m"), [Item::Error]);
 }
 
 #[cfg(test)]

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 9276929c58d9e8434a57676967e34c1b58e31fca
git clean -fd
