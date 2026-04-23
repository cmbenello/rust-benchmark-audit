#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/datetime.rs b/src/datetime.rs
--- a/src/datetime.rs
+++ b/src/datetime.rs
@@ -21,7 +21,7 @@ use offset::Local;
 use offset::{TimeZone, Offset, Utc, FixedOffset};
 use naive::{NaiveTime, NaiveDateTime, IsoWeek};
 use Date;
-use format::{Item, Numeric, Pad, Fixed};
+use format::{Item, Fixed};
 use format::{parse, Parsed, ParseError, ParseResult, StrftimeItems};
 #[cfg(any(feature = "alloc", feature = "std", test))]
 use format::DelayedFormat;
diff --git a/src/datetime.rs b/src/datetime.rs
--- a/src/datetime.rs
+++ b/src/datetime.rs
@@ -2104,6 +2077,15 @@ mod tests {
 
     #[test]
     fn test_datetime_from_str() {
+        assert_eq!("2015-02-18T23:16:9.15Z".parse::<DateTime<FixedOffset>>(),
+                   Ok(FixedOffset::east(0).ymd(2015, 2, 18).and_hms_milli(23, 16, 9, 150)));
+        assert_eq!("2015-02-18T23:16:9.15Z".parse::<DateTime<Utc>>(),
+                   Ok(Utc.ymd(2015, 2, 18).and_hms_milli(23, 16, 9, 150)));
+        assert_eq!("2015-02-18T23:16:9.15 UTC".parse::<DateTime<Utc>>(),
+                   Ok(Utc.ymd(2015, 2, 18).and_hms_milli(23, 16, 9, 150)));
+        assert_eq!("2015-02-18T23:16:9.15UTC".parse::<DateTime<Utc>>(),
+                   Ok(Utc.ymd(2015, 2, 18).and_hms_milli(23, 16, 9, 150)));
+
         assert_eq!("2015-2-18T23:16:9.15Z".parse::<DateTime<FixedOffset>>(),
                    Ok(FixedOffset::east(0).ymd(2015, 2, 18).and_hms_milli(23, 16, 9, 150)));
         assert_eq!("2015-2-18T13:16:9.15-10:00".parse::<DateTime<FixedOffset>>(),
diff --git a/src/datetime.rs b/src/datetime.rs
--- a/src/datetime.rs
+++ b/src/datetime.rs
@@ -2132,6 +2114,25 @@ mod tests {
                    Ok(Utc.ymd(2013, 8, 9).and_hms(23, 54, 35)));
     }
 
+    #[test]
+    fn test_to_string_round_trip() {
+        let dt = Utc.ymd(2000, 1, 1).and_hms(0, 0, 0);
+        let _dt: DateTime<Utc> = dt.to_string().parse().unwrap();
+
+        let ndt_fixed = dt.with_timezone(&FixedOffset::east(3600));
+        let _dt: DateTime<FixedOffset> = ndt_fixed.to_string().parse().unwrap();
+
+        let ndt_fixed = dt.with_timezone(&FixedOffset::east(0));
+        let _dt: DateTime<FixedOffset> = ndt_fixed.to_string().parse().unwrap();
+    }
+
+    #[test]
+    #[cfg(feature="clock")]
+    fn test_to_string_round_trip_with_local() {
+        let ndt = Local::now();
+        let _dt: DateTime<FixedOffset> = ndt.to_string().parse().unwrap();
+    }
+
     #[test]
     #[cfg(feature="clock")]
     fn test_datetime_format_with_local() {
diff --git a/src/format/parse.rs b/src/format/parse.rs
--- a/src/format/parse.rs
+++ b/src/format/parse.rs
@@ -201,24 +202,39 @@ fn parse_rfc3339<'a>(parsed: &mut Parsed, mut s: &'a str) -> ParseResult<(&'a st
 ///   so one can prepend any number of whitespace then any number of zeroes before numbers.
 ///
 /// - (Still) obeying the intrinsic parsing width. This allows, for example, parsing `HHMMSS`.
-pub fn parse<'a, I, B>(parsed: &mut Parsed, mut s: &str, items: I) -> ParseResult<()>
+pub fn parse<'a, I, B>(parsed: &mut Parsed, s: &str, items: I) -> ParseResult<()>
         where I: Iterator<Item=B>, B: Borrow<Item<'a>> {
+    parse_internal(parsed, s, items).map(|_| ()).map_err(|(_s, e)| e)
+}
+
+fn parse_internal<'a, 'b, I, B>(
+    parsed: &mut Parsed, mut s: &'b str, items: I
+) -> Result<&'b str, (&'b str, ParseError)>
+where I: Iterator<Item=B>, B: Borrow<Item<'a>> {
     macro_rules! try_consume {
-        ($e:expr) => ({ let (s_, v) = $e?; s = s_; v })
+        ($e:expr) => ({
+            match $e {
+                Ok((s_, v)) => {
+                    s = s_;
+                    v
+                }
+                Err(e) => return Err((s, e))
+            }
+        })
     }
 
     for item in items {
         match item.borrow() {
             &Item::Literal(prefix) => {
-                if s.len() < prefix.len() { return Err(TOO_SHORT); }
-                if !s.starts_with(prefix) { return Err(INVALID); }
+                if s.len() < prefix.len() { return Err((s, TOO_SHORT)); }
+                if !s.starts_with(prefix) { return Err((s, INVALID)); }
                 s = &s[prefix.len()..];
             }
 
             #[cfg(any(feature = "alloc", feature = "std", test))]
             &Item::OwnedLiteral(ref prefix) => {
-                if s.len() < prefix.len() { return Err(TOO_SHORT); }
-                if !s.starts_with(&prefix[..]) { return Err(INVALID); }
+                if s.len() < prefix.len() { return Err((s, TOO_SHORT)); }
+                if !s.starts_with(&prefix[..]) { return Err((s, INVALID)); }
                 s = &s[prefix.len()..];
             }
 

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard b9cd0ce8039a03db54de9049b574aedcad2269c1
git clean -fd
