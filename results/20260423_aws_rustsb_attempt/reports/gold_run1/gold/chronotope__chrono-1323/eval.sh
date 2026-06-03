#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/.github/workflows/test.yml b/.github/workflows/test.yml
--- a/.github/workflows/test.yml
+++ b/.github/workflows/test.yml
@@ -74,7 +74,7 @@ jobs:
       - uses: Swatinem/rust-cache@v2
       - run: |
           cargo hack check --feature-powerset --optional-deps serde,rkyv \
-            --skip __internal_bench,__doctest,iana-time-zone,pure-rust-locales,libc,winapi \
+            --skip __internal_bench,iana-time-zone,pure-rust-locales,libc,winapi \
             --all-targets
         # run using `bash` on all platforms for consistent
         # line-continuation marks
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -58,7 +58,7 @@ bincode = { version = "1.3.0" }
 wasm-bindgen-test = "0.3"
 
 [package.metadata.docs.rs]
-features = ["serde"]
+features = ["arbitrary, rkyv, serde, unstable-locales"]
 rustdoc-args = ["--cfg", "docsrs"]
 
 [package.metadata.playground]
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -272,7 +272,7 @@ fn ymdhms_milli(
 
 // local helper function to easily create a DateTime<FixedOffset>
 #[allow(clippy::too_many_arguments)]
-#[cfg(any(feature = "alloc", feature = "std"))]
+#[cfg(feature = "alloc")]
 fn ymdhms_micro(
     fixedoffset: &FixedOffset,
     year: i32,
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -292,7 +292,7 @@ fn ymdhms_micro(
 
 // local helper function to easily create a DateTime<FixedOffset>
 #[allow(clippy::too_many_arguments)]
-#[cfg(any(feature = "alloc", feature = "std"))]
+#[cfg(feature = "alloc")]
 fn ymdhms_nano(
     fixedoffset: &FixedOffset,
     year: i32,
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -311,7 +311,7 @@ fn ymdhms_nano(
 }
 
 // local helper function to easily create a DateTime<Utc>
-#[cfg(any(feature = "alloc", feature = "std"))]
+#[cfg(feature = "alloc")]
 fn ymdhms_utc(year: i32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> DateTime<Utc> {
     Utc.with_ymd_and_hms(year, month, day, hour, min, sec).unwrap()
 }
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -450,7 +450,7 @@ fn test_datetime_with_timezone() {
 }
 
 #[test]
-#[cfg(any(feature = "alloc", feature = "std"))]
+#[cfg(feature = "alloc")]
 fn test_datetime_rfc2822() {
     let edt = FixedOffset::east_opt(5 * 60 * 60).unwrap();
 
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -459,6 +459,10 @@ fn test_datetime_rfc2822() {
         Utc.with_ymd_and_hms(2015, 2, 18, 23, 16, 9).unwrap().to_rfc2822(),
         "Wed, 18 Feb 2015 23:16:09 +0000"
     );
+    assert_eq!(
+        Utc.with_ymd_and_hms(2015, 2, 1, 23, 16, 9).unwrap().to_rfc2822(),
+        "Sun, 1 Feb 2015 23:16:09 +0000"
+    );
     // timezone +05
     assert_eq!(
         edt.from_local_datetime(
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -572,7 +576,7 @@ fn test_datetime_rfc2822() {
 }
 
 #[test]
-#[cfg(any(feature = "alloc", feature = "std"))]
+#[cfg(feature = "alloc")]
 fn test_datetime_rfc3339() {
     let edt5 = FixedOffset::east_opt(5 * 60 * 60).unwrap();
     let edt0 = FixedOffset::east_opt(0).unwrap();
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -658,7 +662,7 @@ fn test_datetime_rfc3339() {
 }
 
 #[test]
-#[cfg(any(feature = "alloc", feature = "std"))]
+#[cfg(feature = "alloc")]
 fn test_rfc3339_opts() {
     use crate::SecondsFormat::*;
     let pst = FixedOffset::east_opt(8 * 60 * 60).unwrap();
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -1461,7 +1465,7 @@ fn test_test_deprecated_from_offset() {
 }
 
 #[test]
-#[cfg(all(feature = "unstable-locales", any(feature = "alloc", feature = "std")))]
+#[cfg(all(feature = "unstable-locales", feature = "alloc"))]
 fn locale_decimal_point() {
     use crate::Locale::{ar_SY, nl_NL};
     let dt =
diff --git a/src/datetime/tests.rs b/src/datetime/tests.rs
--- a/src/datetime/tests.rs
+++ b/src/datetime/tests.rs
@@ -1477,3 +1481,36 @@ fn locale_decimal_point() {
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
diff --git a/src/format/formatting.rs b/src/format/formatting.rs
--- a/src/format/formatting.rs
+++ b/src/format/formatting.rs
@@ -674,15 +638,15 @@ pub(crate) fn write_hundreds(w: &mut impl Write, n: u8) -> fmt::Result {
 }
 
 #[cfg(test)]
-#[cfg(any(feature = "alloc", feature = "std"))]
+#[cfg(feature = "alloc")]
 mod tests {
     use super::{Colons, OffsetFormat, OffsetPrecision, Pad};
     use crate::FixedOffset;
-    #[cfg(any(feature = "alloc", feature = "std"))]
+    #[cfg(feature = "alloc")]
     use crate::{NaiveDate, NaiveTime, TimeZone, Timelike, Utc};
 
     #[test]
-    #[cfg(any(feature = "alloc", feature = "std"))]
+    #[cfg(feature = "alloc")]
     fn test_date_format() {
         let d = NaiveDate::from_ymd_opt(2012, 3, 4).unwrap();
         assert_eq!(d.format("%Y,%C,%y,%G,%g").to_string(), "2012,20,12,2012,12");
diff --git a/src/format/formatting.rs b/src/format/formatting.rs
--- a/src/format/formatting.rs
+++ b/src/format/formatting.rs
@@ -727,7 +691,7 @@ mod tests {
     }
 
     #[test]
-    #[cfg(any(feature = "alloc", feature = "std"))]
+    #[cfg(feature = "alloc")]
     fn test_time_format() {
         let t = NaiveTime::from_hms_nano_opt(3, 5, 7, 98765432).unwrap();
         assert_eq!(t.format("%H,%k,%I,%l,%P,%p").to_string(), "03, 3,03, 3,am,AM");
diff --git a/src/format/formatting.rs b/src/format/formatting.rs
--- a/src/format/formatting.rs
+++ b/src/format/formatting.rs
@@ -763,7 +727,7 @@ mod tests {
     }
 
     #[test]
-    #[cfg(any(feature = "alloc", feature = "std"))]
+    #[cfg(feature = "alloc")]
     fn test_datetime_format() {
         let dt =
             NaiveDate::from_ymd_opt(2010, 9, 8).unwrap().and_hms_milli_opt(7, 6, 54, 321).unwrap();
diff --git a/src/format/formatting.rs b/src/format/formatting.rs
--- a/src/format/formatting.rs
+++ b/src/format/formatting.rs
@@ -781,7 +745,7 @@ mod tests {
     }
 
     #[test]
-    #[cfg(any(feature = "alloc", feature = "std"))]
+    #[cfg(feature = "alloc")]
     fn test_datetime_format_alignment() {
         let datetime = Utc
             .with_ymd_and_hms(2007, 1, 2, 12, 34, 56)
diff --git a/src/format/parse.rs b/src/format/parse.rs
--- a/src/format/parse.rs
+++ b/src/format/parse.rs
@@ -1729,7 +1729,7 @@ mod tests {
         let dt = Utc.with_ymd_and_hms(1994, 11, 6, 8, 49, 37).unwrap();
 
         // Check that the format is what we expect
-        #[cfg(any(feature = "alloc", feature = "std"))]
+        #[cfg(feature = "alloc")]
         assert_eq!(dt.format(RFC850_FMT).to_string(), "Sunday, 06-Nov-94 08:49:37 GMT");
 
         // Check that it parses correctly
diff --git a/src/format/strftime.rs b/src/format/strftime.rs
--- a/src/format/strftime.rs
+++ b/src/format/strftime.rs
@@ -501,7 +500,7 @@ mod tests {
     use crate::format::Locale;
     use crate::format::{fixed, internal_fixed, num, num0, nums};
     use crate::format::{Fixed, InternalInternal, Numeric::*};
-    #[cfg(any(feature = "alloc", feature = "std"))]
+    #[cfg(feature = "alloc")]
     use crate::{DateTime, FixedOffset, NaiveDate, TimeZone, Timelike, Utc};
 
     #[test]
diff --git a/src/format/strftime.rs b/src/format/strftime.rs
--- a/src/format/strftime.rs
+++ b/src/format/strftime.rs
@@ -663,7 +662,7 @@ mod tests {
     }
 
     #[test]
-    #[cfg(any(feature = "alloc", feature = "std"))]
+    #[cfg(feature = "alloc")]
     fn test_strftime_docs() {
         let dt = FixedOffset::east_opt(34200)
             .unwrap()
diff --git a/src/format/strftime.rs b/src/format/strftime.rs
--- a/src/format/strftime.rs
+++ b/src/format/strftime.rs
@@ -774,7 +773,7 @@ mod tests {
     }
 
     #[test]
-    #[cfg(all(feature = "unstable-locales", any(feature = "alloc", feature = "std")))]
+    #[cfg(all(feature = "unstable-locales", feature = "alloc"))]
     fn test_strftime_docs_localized() {
         let dt = FixedOffset::east_opt(34200)
             .unwrap()
diff --git a/src/format/strftime.rs b/src/format/strftime.rs
--- a/src/format/strftime.rs
+++ b/src/format/strftime.rs
@@ -827,7 +826,7 @@ mod tests {
     ///
     /// See <https://github.com/chronotope/chrono/issues/1139>.
     #[test]
-    #[cfg(any(feature = "alloc", feature = "std"))]
+    #[cfg(feature = "alloc")]
     fn test_parse_only_timezone_offset_permissive_no_panic() {
         use crate::NaiveDate;
         use crate::{FixedOffset, TimeZone};
diff --git a/src/format/strftime.rs b/src/format/strftime.rs
--- a/src/format/strftime.rs
+++ b/src/format/strftime.rs
@@ -848,7 +847,7 @@ mod tests {
     }
 
     #[test]
-    #[cfg(all(feature = "unstable-locales", any(feature = "alloc", feature = "std")))]
+    #[cfg(all(feature = "unstable-locales", feature = "alloc"))]
     fn test_strftime_localized_korean() {
         let dt = FixedOffset::east_opt(34200)
             .unwrap()
diff --git a/src/format/strftime.rs b/src/format/strftime.rs
--- a/src/format/strftime.rs
+++ b/src/format/strftime.rs
@@ -877,7 +876,7 @@ mod tests {
     }
 
     #[test]
-    #[cfg(all(feature = "unstable-locales", any(feature = "alloc", feature = "std")))]
+    #[cfg(all(feature = "unstable-locales", feature = "alloc"))]
     fn test_strftime_localized_japanese() {
         let dt = FixedOffset::east_opt(34200)
             .unwrap()
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -219,7 +219,7 @@
 //! # #[allow(unused_imports)]
 //! use chrono::prelude::*;
 //!
-//! # #[cfg(feature = "unstable-locales")]
+//! # #[cfg(all(feature = "unstable-locales", feature = "alloc"))]
 //! # fn test() {
 //! let dt = Utc.with_ymd_and_hms(2014, 11, 28, 12, 0, 9).unwrap();
 //! assert_eq!(dt.format("%Y-%m-%d %H:%M:%S").to_string(), "2014-11-28 12:00:09");
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -236,9 +236,9 @@
 //! let dt_nano = NaiveDate::from_ymd_opt(2014, 11, 28).unwrap().and_hms_nano_opt(12, 0, 9, 1).unwrap().and_local_timezone(Utc).unwrap();
 //! assert_eq!(format!("{:?}", dt_nano), "2014-11-28T12:00:09.000000001Z");
 //! # }
-//! # #[cfg(not(feature = "unstable-locales"))]
+//! # #[cfg(not(all(feature = "unstable-locales", feature = "alloc")))]
 //! # fn test() {}
-//! # if cfg!(feature = "unstable-locales") {
+//! # if cfg!(all(feature = "unstable-locales", feature = "alloc")) {
 //! #    test();
 //! # }
 //! ```
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -458,7 +458,7 @@
 #![warn(unreachable_pub)]
 #![deny(clippy::tests_outside_test_module)]
 #![cfg_attr(not(any(feature = "std", test)), no_std)]
-#![cfg_attr(docsrs, feature(doc_cfg))]
+#![cfg_attr(docsrs, feature(doc_auto_cfg))]
 
 #[cfg(feature = "alloc")]
 extern crate alloc;
diff --git a/src/naive/date.rs b/src/naive/date.rs
--- a/src/naive/date.rs
+++ b/src/naive/date.rs
@@ -2359,13 +2364,14 @@ mod tests {
         let calculated_max = NaiveDate::from_ymd_opt(MAX_YEAR, 12, 31).unwrap();
         assert!(
             NaiveDate::MIN == calculated_min,
-            "`NaiveDate::MIN` should have a year flag {:?}",
+            "`NaiveDate::MIN` should have year flag {:?}",
             calculated_min.of().flags()
         );
         assert!(
             NaiveDate::MAX == calculated_max,
-            "`NaiveDate::MAX` should have a year flag {:?}",
-            calculated_max.of().flags()
+            "`NaiveDate::MAX` should have year flag {:?} and ordinal {}",
+            calculated_max.of().flags(),
+            calculated_max.of().ordinal()
         );
 
         // let's also check that the entire range do not exceed 2^44 seconds
diff --git a/src/naive/date.rs b/src/naive/date.rs
--- a/src/naive/date.rs
+++ b/src/naive/date.rs
@@ -3198,22 +3204,16 @@ mod tests {
     }
 
     //   MAX_YEAR-12-31 minus 0000-01-01
-    // = ((MAX_YEAR+1)-01-01 minus 0001-01-01) + (0001-01-01 minus 0000-01-01) - 1 day
-    // = ((MAX_YEAR+1)-01-01 minus 0001-01-01) + 365 days
-    // = MAX_YEAR * 365 + (# of leap years from 0001 to MAX_YEAR) + 365 days
+    // = (MAX_YEAR-12-31 minus 0000-12-31) + (0000-12-31 - 0000-01-01)
+    // = MAX_YEAR * 365 + (# of leap years from 0001 to MAX_YEAR) + 365
+    // = (MAX_YEAR + 1) * 365 + (# of leap years from 0001 to MAX_YEAR)
     const MAX_DAYS_FROM_YEAR_0: i32 =
-        MAX_YEAR * 365 + MAX_YEAR / 4 - MAX_YEAR / 100 + MAX_YEAR / 400 + 365;
+        (MAX_YEAR + 1) * 365 + MAX_YEAR / 4 - MAX_YEAR / 100 + MAX_YEAR / 400;
 
     //   MIN_YEAR-01-01 minus 0000-01-01
-    // = (MIN_YEAR+400n+1)-01-01 minus (400n+1)-01-01
-    // = ((MIN_YEAR+400n+1)-01-01 minus 0001-01-01) - ((400n+1)-01-01 minus 0001-01-01)
-    // = ((MIN_YEAR+400n+1)-01-01 minus 0001-01-01) - 146097n days
-    //
-    // n is set to 1000 for convenience.
-    const MIN_DAYS_FROM_YEAR_0: i32 = (MIN_YEAR + 400_000) * 365 + (MIN_YEAR + 400_000) / 4
-        - (MIN_YEAR + 400_000) / 100
-        + (MIN_YEAR + 400_000) / 400
-        - 146_097_000;
+    // = MIN_YEAR * 365 + (# of leap years from MIN_YEAR to 0000)
+    const MIN_DAYS_FROM_YEAR_0: i32 =
+        MIN_YEAR * 365 + MIN_YEAR / 4 - MIN_YEAR / 100 + MIN_YEAR / 400;
 
     // only used for testing, but duplicated in naive::datetime
     const MAX_BITS: usize = 44;
diff --git a/src/naive/datetime/tests.rs b/src/naive/datetime/tests.rs
--- a/src/naive/datetime/tests.rs
+++ b/src/naive/datetime/tests.rs
@@ -19,7 +19,7 @@ fn test_datetime_from_timestamp_millis() {
     for (timestamp_millis, _formatted) in valid_map.iter().copied() {
         let naive_datetime = NaiveDateTime::from_timestamp_millis(timestamp_millis);
         assert_eq!(timestamp_millis, naive_datetime.unwrap().timestamp_millis());
-        #[cfg(any(feature = "alloc", feature = "std"))]
+        #[cfg(feature = "alloc")]
         assert_eq!(naive_datetime.unwrap().format("%F %T%.9f").to_string(), _formatted);
     }
 
diff --git a/src/naive/datetime/tests.rs b/src/naive/datetime/tests.rs
--- a/src/naive/datetime/tests.rs
+++ b/src/naive/datetime/tests.rs
@@ -57,7 +57,7 @@ fn test_datetime_from_timestamp_micros() {
     for (timestamp_micros, _formatted) in valid_map.iter().copied() {
         let naive_datetime = NaiveDateTime::from_timestamp_micros(timestamp_micros);
         assert_eq!(timestamp_micros, naive_datetime.unwrap().timestamp_micros());
-        #[cfg(any(feature = "alloc", feature = "std"))]
+        #[cfg(feature = "alloc")]
         assert_eq!(naive_datetime.unwrap().format("%F %T%.9f").to_string(), _formatted);
     }
 
diff --git a/src/naive/datetime/tests.rs b/src/naive/datetime/tests.rs
--- a/src/naive/datetime/tests.rs
+++ b/src/naive/datetime/tests.rs
@@ -407,7 +407,7 @@ fn test_nanosecond_range() {
     const A_BILLION: i64 = 1_000_000_000;
     let maximum = "2262-04-11T23:47:16.854775804";
     let parsed: NaiveDateTime = maximum.parse().unwrap();
-    let nanos = parsed.timestamp_nanos();
+    let nanos = parsed.timestamp_nanos_opt().unwrap();
     assert_eq!(
         parsed,
         NaiveDateTime::from_timestamp_opt(nanos / A_BILLION, (nanos % A_BILLION) as u32).unwrap()
diff --git a/src/naive/datetime/tests.rs b/src/naive/datetime/tests.rs
--- a/src/naive/datetime/tests.rs
+++ b/src/naive/datetime/tests.rs
@@ -415,29 +415,23 @@ fn test_nanosecond_range() {
 
     let minimum = "1677-09-21T00:12:44.000000000";
     let parsed: NaiveDateTime = minimum.parse().unwrap();
-    let nanos = parsed.timestamp_nanos();
+    let nanos = parsed.timestamp_nanos_opt().unwrap();
     assert_eq!(
         parsed,
         NaiveDateTime::from_timestamp_opt(nanos / A_BILLION, (nanos % A_BILLION) as u32).unwrap()
     );
-}
 
-#[test]
-#[should_panic]
-fn test_nanosecond_just_beyond_range() {
+    // Just beyond range
     let maximum = "2262-04-11T23:47:16.854775804";
     let parsed: NaiveDateTime = maximum.parse().unwrap();
     let beyond_max = parsed + TimeDelta::milliseconds(300);
-    let _ = beyond_max.timestamp_nanos();
-}
+    assert!(beyond_max.timestamp_nanos_opt().is_none());
 
-#[test]
-#[should_panic]
-fn test_nanosecond_far_beyond_range() {
+    // Far beyond range
     let maximum = "2262-04-11T23:47:16.854775804";
     let parsed: NaiveDateTime = maximum.parse().unwrap();
     let beyond_max = parsed + TimeDelta::days(365);
-    let _ = beyond_max.timestamp_nanos();
+    assert!(beyond_max.timestamp_nanos_opt().is_none());
 }
 
 #[test]
diff --git a/src/naive/datetime/tests.rs b/src/naive/datetime/tests.rs
--- a/src/naive/datetime/tests.rs
+++ b/src/naive/datetime/tests.rs
@@ -460,3 +454,60 @@ fn test_and_utc() {
     assert_eq!(dt_utc.naive_local(), ndt);
     assert_eq!(dt_utc.timezone(), Utc);
 }
+
+#[test]
+fn test_checked_add_offset() {
+    let ymdhmsm = |y, m, d, h, mn, s, mi| {
+        NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_milli_opt(h, mn, s, mi)
+    };
+
+    let positive_offset = FixedOffset::east_opt(2 * 60 * 60).unwrap();
+    // regular date
+    let dt = ymdhmsm(2023, 5, 5, 20, 10, 0, 0).unwrap();
+    assert_eq!(dt.checked_add_offset(positive_offset), ymdhmsm(2023, 5, 5, 22, 10, 0, 0));
+    // leap second is preserved
+    let dt = ymdhmsm(2023, 6, 30, 23, 59, 59, 1_000).unwrap();
+    assert_eq!(dt.checked_add_offset(positive_offset), ymdhmsm(2023, 7, 1, 1, 59, 59, 1_000));
+    // out of range
+    assert!(NaiveDateTime::MAX.checked_add_offset(positive_offset).is_none());
+
+    let negative_offset = FixedOffset::west_opt(2 * 60 * 60).unwrap();
+    // regular date
+    let dt = ymdhmsm(2023, 5, 5, 20, 10, 0, 0).unwrap();
+    assert_eq!(dt.checked_add_offset(negative_offset), ymdhmsm(2023, 5, 5, 18, 10, 0, 0));
+    // leap second is preserved
+    let dt = ymdhmsm(2023, 6, 30, 23, 59, 59, 1_000).unwrap();
+    assert_eq!(dt.checked_add_offset(negative_offset), ymdhmsm(2023, 6, 30, 21, 59, 59, 1_000));
+    // out of range
+    assert!(NaiveDateTime::MIN.checked_add_offset(negative_offset).is_none());
+}
+
+#[test]
+fn test_checked_sub_offset() {
+    let ymdhmsm = |y, m, d, h, mn, s, mi| {
+        NaiveDate::from_ymd_opt(y, m, d).unwrap().and_hms_milli_opt(h, mn, s, mi)
+    };
+
+    let positive_offset = FixedOffset::east_opt(2 * 60 * 60).unwrap();
+    // regular date
+    let dt = ymdhmsm(2023, 5, 5, 20, 10, 0, 0).unwrap();
+    assert_eq!(dt.checked_sub_offset(positive_offset), ymdhmsm(2023, 5, 5, 18, 10, 0, 0));
+    // leap second is preserved
+    let dt = ymdhmsm(2023, 6, 30, 23, 59, 59, 1_000).unwrap();
+    assert_eq!(dt.checked_sub_offset(positive_offset), ymdhmsm(2023, 6, 30, 21, 59, 59, 1_000));
+    // out of range
+    assert!(NaiveDateTime::MIN.checked_sub_offset(positive_offset).is_none());
+
+    let negative_offset = FixedOffset::west_opt(2 * 60 * 60).unwrap();
+    // regular date
+    let dt = ymdhmsm(2023, 5, 5, 20, 10, 0, 0).unwrap();
+    assert_eq!(dt.checked_sub_offset(negative_offset), ymdhmsm(2023, 5, 5, 22, 10, 0, 0));
+    // leap second is preserved
+    let dt = ymdhmsm(2023, 6, 30, 23, 59, 59, 1_000).unwrap();
+    assert_eq!(dt.checked_sub_offset(negative_offset), ymdhmsm(2023, 7, 1, 1, 59, 59, 1_000));
+    // out of range
+    assert!(NaiveDateTime::MAX.checked_sub_offset(negative_offset).is_none());
+
+    assert_eq!(dt.checked_add_offset(positive_offset), Some(dt + positive_offset));
+    assert_eq!(dt.checked_sub_offset(positive_offset), Some(dt - positive_offset));
+}
diff --git a/src/naive/isoweek.rs b/src/naive/isoweek.rs
--- a/src/naive/isoweek.rs
+++ b/src/naive/isoweek.rs
@@ -158,13 +162,13 @@ mod tests {
         assert_eq!(minweek.year(), internals::MIN_YEAR);
         assert_eq!(minweek.week(), 1);
         assert_eq!(minweek.week0(), 0);
-        #[cfg(any(feature = "alloc", feature = "std"))]
+        #[cfg(feature = "alloc")]
         assert_eq!(format!("{:?}", minweek), NaiveDate::MIN.format("%G-W%V").to_string());
 
         assert_eq!(maxweek.year(), internals::MAX_YEAR + 1);
         assert_eq!(maxweek.week(), 1);
         assert_eq!(maxweek.week0(), 0);
-        #[cfg(any(feature = "alloc", feature = "std"))]
+        #[cfg(feature = "alloc")]
         assert_eq!(format!("{:?}", maxweek), NaiveDate::MAX.format("%G-W%V").to_string());
     }
 
diff --git a/src/naive/time/mod.rs b/src/naive/time/mod.rs
--- a/src/naive/time/mod.rs
+++ b/src/naive/time/mod.rs
@@ -167,28 +167,43 @@ mod tests;
 /// assert_eq!(format!("{:?}", dt), "2015-06-30T23:59:60Z");
 /// ```
 ///
-/// There are hypothetical leap seconds not on the minute boundary
-/// nevertheless supported by Chrono.
-/// They are allowed for the sake of completeness and consistency;
-/// there were several "exotic" time zone offsets with fractional minutes prior to UTC after all.
-/// For such cases the human-readable representation is ambiguous
-/// and would be read back to the next non-leap second.
+/// There are hypothetical leap seconds not on the minute boundary nevertheless supported by Chrono.
+/// They are allowed for the sake of completeness and consistency; there were several "exotic" time
+/// zone offsets with fractional minutes prior to UTC after all.
+/// For such cases the human-readable representation is ambiguous and would be read back to the next
+/// non-leap second.
 ///
-/// ```
-/// use chrono::{DateTime, Utc, TimeZone, NaiveDate};
-///
-/// let dt = NaiveDate::from_ymd_opt(2015, 6, 30).unwrap().and_hms_milli_opt(23, 56, 4, 1_000).unwrap().and_local_timezone(Utc).unwrap();
-/// assert_eq!(format!("{:?}", dt), "2015-06-30T23:56:05Z");
+/// A `NaiveTime` with a leap second that is not on a minute boundary can only be created from a
+/// [`DateTime`](crate::DateTime) with fractional minutes as offset, or using
+/// [`Timelike::with_nanosecond()`].
 ///
-/// let dt = Utc.with_ymd_and_hms(2015, 6, 30, 23, 56, 5).unwrap();
-/// assert_eq!(format!("{:?}", dt), "2015-06-30T23:56:05Z");
-/// assert_eq!(DateTime::parse_from_rfc3339("2015-06-30T23:56:05Z").unwrap(), dt);
+/// ```
+/// use chrono::{FixedOffset, NaiveDate, TimeZone};
+///
+/// let paramaribo_pre1945 = FixedOffset::east_opt(-13236).unwrap(); // -03:40:36
+/// let leap_sec_2015 =
+///     NaiveDate::from_ymd_opt(2015, 6, 30).unwrap().and_hms_milli_opt(23, 59, 59, 1_000).unwrap();
+/// let dt1 = paramaribo_pre1945.from_utc_datetime(&leap_sec_2015);
+/// assert_eq!(format!("{:?}", dt1), "2015-06-30T20:19:24-03:40:36");
+/// assert_eq!(format!("{:?}", dt1.time()), "20:19:24");
+///
+/// let next_sec = NaiveDate::from_ymd_opt(2015, 7, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
+/// let dt2 = paramaribo_pre1945.from_utc_datetime(&next_sec);
+/// assert_eq!(format!("{:?}", dt2), "2015-06-30T20:19:24-03:40:36");
+/// assert_eq!(format!("{:?}", dt2.time()), "20:19:24");
+///
+/// assert!(dt1.time() != dt2.time());
+/// assert!(dt1.time().to_string() == dt2.time().to_string());
 /// ```
 ///
 /// Since Chrono alone cannot determine any existence of leap seconds,
 /// **there is absolutely no guarantee that the leap second read has actually happened**.
 #[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Copy, Clone)]
 #[cfg_attr(feature = "rkyv", derive(Archive, Deserialize, Serialize))]
+#[cfg_attr(
+    feature = "rkyv",
+    archive_attr(derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash))
+)]
 pub struct NaiveTime {
     secs: u32,
     frac: u32,
diff --git a/src/naive/time/tests.rs b/src/naive/time/tests.rs
--- a/src/naive/time/tests.rs
+++ b/src/naive/time/tests.rs
@@ -1,5 +1,5 @@
 use super::NaiveTime;
-use crate::{TimeDelta, Timelike};
+use crate::{FixedOffset, TimeDelta, Timelike};
 
 #[test]
 fn test_time_from_hms_milli() {
diff --git a/src/naive/time/tests.rs b/src/naive/time/tests.rs
--- a/src/naive/time/tests.rs
+++ b/src/naive/time/tests.rs
@@ -12,12 +12,12 @@ fn test_time_from_hms_milli() {
         Some(NaiveTime::from_hms_nano_opt(3, 5, 7, 777_000_000).unwrap())
     );
     assert_eq!(
-        NaiveTime::from_hms_milli_opt(3, 5, 7, 1_999),
-        Some(NaiveTime::from_hms_nano_opt(3, 5, 7, 1_999_000_000).unwrap())
+        NaiveTime::from_hms_milli_opt(3, 5, 59, 1_999),
+        Some(NaiveTime::from_hms_nano_opt(3, 5, 59, 1_999_000_000).unwrap())
     );
-    assert_eq!(NaiveTime::from_hms_milli_opt(3, 5, 7, 2_000), None);
-    assert_eq!(NaiveTime::from_hms_milli_opt(3, 5, 7, 5_000), None); // overflow check
-    assert_eq!(NaiveTime::from_hms_milli_opt(3, 5, 7, u32::MAX), None);
+    assert_eq!(NaiveTime::from_hms_milli_opt(3, 5, 59, 2_000), None);
+    assert_eq!(NaiveTime::from_hms_milli_opt(3, 5, 59, 5_000), None); // overflow check
+    assert_eq!(NaiveTime::from_hms_milli_opt(3, 5, 59, u32::MAX), None);
 }
 
 #[test]
diff --git a/src/naive/time/tests.rs b/src/naive/time/tests.rs
--- a/src/naive/time/tests.rs
+++ b/src/naive/time/tests.rs
@@ -35,12 +35,12 @@ fn test_time_from_hms_micro() {
         Some(NaiveTime::from_hms_nano_opt(3, 5, 7, 777_777_000).unwrap())
     );
     assert_eq!(
-        NaiveTime::from_hms_micro_opt(3, 5, 7, 1_999_999),
-        Some(NaiveTime::from_hms_nano_opt(3, 5, 7, 1_999_999_000).unwrap())
+        NaiveTime::from_hms_micro_opt(3, 5, 59, 1_999_999),
+        Some(NaiveTime::from_hms_nano_opt(3, 5, 59, 1_999_999_000).unwrap())
     );
-    assert_eq!(NaiveTime::from_hms_micro_opt(3, 5, 7, 2_000_000), None);
-    assert_eq!(NaiveTime::from_hms_micro_opt(3, 5, 7, 5_000_000), None); // overflow check
-    assert_eq!(NaiveTime::from_hms_micro_opt(3, 5, 7, u32::MAX), None);
+    assert_eq!(NaiveTime::from_hms_micro_opt(3, 5, 59, 2_000_000), None);
+    assert_eq!(NaiveTime::from_hms_micro_opt(3, 5, 59, 5_000_000), None); // overflow check
+    assert_eq!(NaiveTime::from_hms_micro_opt(3, 5, 59, u32::MAX), None);
 }
 
 #[test]
diff --git a/src/naive/time/tests.rs b/src/naive/time/tests.rs
--- a/src/naive/time/tests.rs
+++ b/src/naive/time/tests.rs
@@ -93,19 +93,19 @@ fn test_time_add() {
 
     let hmsm = |h, m, s, ms| NaiveTime::from_hms_milli_opt(h, m, s, ms).unwrap();
 
-    check!(hmsm(3, 5, 7, 900), TimeDelta::zero(), hmsm(3, 5, 7, 900));
-    check!(hmsm(3, 5, 7, 900), TimeDelta::milliseconds(100), hmsm(3, 5, 8, 0));
-    check!(hmsm(3, 5, 7, 1_300), TimeDelta::milliseconds(-1800), hmsm(3, 5, 6, 500));
-    check!(hmsm(3, 5, 7, 1_300), TimeDelta::milliseconds(-800), hmsm(3, 5, 7, 500));
-    check!(hmsm(3, 5, 7, 1_300), TimeDelta::milliseconds(-100), hmsm(3, 5, 7, 1_200));
-    check!(hmsm(3, 5, 7, 1_300), TimeDelta::milliseconds(100), hmsm(3, 5, 7, 1_400));
-    check!(hmsm(3, 5, 7, 1_300), TimeDelta::milliseconds(800), hmsm(3, 5, 8, 100));
-    check!(hmsm(3, 5, 7, 1_300), TimeDelta::milliseconds(1800), hmsm(3, 5, 9, 100));
-    check!(hmsm(3, 5, 7, 900), TimeDelta::seconds(86399), hmsm(3, 5, 6, 900)); // overwrap
-    check!(hmsm(3, 5, 7, 900), TimeDelta::seconds(-86399), hmsm(3, 5, 8, 900));
-    check!(hmsm(3, 5, 7, 900), TimeDelta::days(12345), hmsm(3, 5, 7, 900));
-    check!(hmsm(3, 5, 7, 1_300), TimeDelta::days(1), hmsm(3, 5, 7, 300));
-    check!(hmsm(3, 5, 7, 1_300), TimeDelta::days(-1), hmsm(3, 5, 8, 300));
+    check!(hmsm(3, 5, 59, 900), TimeDelta::zero(), hmsm(3, 5, 59, 900));
+    check!(hmsm(3, 5, 59, 900), TimeDelta::milliseconds(100), hmsm(3, 6, 0, 0));
+    check!(hmsm(3, 5, 59, 1_300), TimeDelta::milliseconds(-1800), hmsm(3, 5, 58, 500));
+    check!(hmsm(3, 5, 59, 1_300), TimeDelta::milliseconds(-800), hmsm(3, 5, 59, 500));
+    check!(hmsm(3, 5, 59, 1_300), TimeDelta::milliseconds(-100), hmsm(3, 5, 59, 1_200));
+    check!(hmsm(3, 5, 59, 1_300), TimeDelta::milliseconds(100), hmsm(3, 5, 59, 1_400));
+    check!(hmsm(3, 5, 59, 1_300), TimeDelta::milliseconds(800), hmsm(3, 6, 0, 100));
+    check!(hmsm(3, 5, 59, 1_300), TimeDelta::milliseconds(1800), hmsm(3, 6, 1, 100));
+    check!(hmsm(3, 5, 59, 900), TimeDelta::seconds(86399), hmsm(3, 5, 58, 900)); // overwrap
+    check!(hmsm(3, 5, 59, 900), TimeDelta::seconds(-86399), hmsm(3, 6, 0, 900));
+    check!(hmsm(3, 5, 59, 900), TimeDelta::days(12345), hmsm(3, 5, 59, 900));
+    check!(hmsm(3, 5, 59, 1_300), TimeDelta::days(1), hmsm(3, 5, 59, 300));
+    check!(hmsm(3, 5, 59, 1_300), TimeDelta::days(-1), hmsm(3, 6, 0, 300));
 
     // regression tests for #37
     check!(hmsm(0, 0, 0, 0), TimeDelta::milliseconds(-990), hmsm(23, 59, 59, 10));
diff --git a/src/naive/time/tests.rs b/src/naive/time/tests.rs
--- a/src/naive/time/tests.rs
+++ b/src/naive/time/tests.rs
@@ -131,12 +131,12 @@ fn test_time_overflowing_add() {
 
     // overflowing_add_signed with leap seconds may be counter-intuitive
     assert_eq!(
-        hmsm(3, 4, 5, 1_678).overflowing_add_signed(TimeDelta::days(1)),
-        (hmsm(3, 4, 5, 678), 86_400)
+        hmsm(3, 4, 59, 1_678).overflowing_add_signed(TimeDelta::days(1)),
+        (hmsm(3, 4, 59, 678), 86_400)
     );
     assert_eq!(
-        hmsm(3, 4, 5, 1_678).overflowing_add_signed(TimeDelta::days(-1)),
-        (hmsm(3, 4, 6, 678), -86_400)
+        hmsm(3, 4, 59, 1_678).overflowing_add_signed(TimeDelta::days(-1)),
+        (hmsm(3, 5, 0, 678), -86_400)
     );
 }
 
diff --git a/src/naive/time/tests.rs b/src/naive/time/tests.rs
--- a/src/naive/time/tests.rs
+++ b/src/naive/time/tests.rs
@@ -183,14 +183,29 @@ fn test_time_sub() {
 
     // treats the leap second as if it coincides with the prior non-leap second,
     // as required by `time1 - time2 = duration` and `time2 - time1 = -duration` equivalence.
-    check!(hmsm(3, 5, 7, 200), hmsm(3, 5, 6, 1_800), TimeDelta::milliseconds(400));
-    check!(hmsm(3, 5, 7, 1_200), hmsm(3, 5, 6, 1_800), TimeDelta::milliseconds(1400));
-    check!(hmsm(3, 5, 7, 1_200), hmsm(3, 5, 6, 800), TimeDelta::milliseconds(1400));
+    check!(hmsm(3, 6, 0, 200), hmsm(3, 5, 59, 1_800), TimeDelta::milliseconds(400));
+    //check!(hmsm(3, 5, 7, 1_200), hmsm(3, 5, 6, 1_800), TimeDelta::milliseconds(1400));
+    //check!(hmsm(3, 5, 7, 1_200), hmsm(3, 5, 6, 800), TimeDelta::milliseconds(1400));
 
     // additional equality: `time1 + duration = time2` is equivalent to
     // `time2 - time1 = duration` IF AND ONLY IF `time2` represents a non-leap second.
     assert_eq!(hmsm(3, 5, 6, 800) + TimeDelta::milliseconds(400), hmsm(3, 5, 7, 200));
-    assert_eq!(hmsm(3, 5, 6, 1_800) + TimeDelta::milliseconds(400), hmsm(3, 5, 7, 200));
+    //assert_eq!(hmsm(3, 5, 6, 1_800) + TimeDelta::milliseconds(400), hmsm(3, 5, 7, 200));
+}
+
+#[test]
+fn test_core_duration_ops() {
+    use core::time::Duration;
+
+    let mut t = NaiveTime::from_hms_opt(11, 34, 23).unwrap();
+    let same = t + Duration::ZERO;
+    assert_eq!(t, same);
+
+    t += Duration::new(3600, 0);
+    assert_eq!(t, NaiveTime::from_hms_opt(12, 34, 23).unwrap());
+
+    t -= Duration::new(7200, 0);
+    assert_eq!(t, NaiveTime::from_hms_opt(10, 34, 23).unwrap());
 }
 
 #[test]
diff --git a/src/naive/time/tests.rs b/src/naive/time/tests.rs
--- a/src/naive/time/tests.rs
+++ b/src/naive/time/tests.rs
@@ -342,3 +357,29 @@ fn test_time_parse_from_str() {
     assert!(NaiveTime::parse_from_str("12:59  PM", "%H:%M %P").is_err());
     assert!(NaiveTime::parse_from_str("12:3456", "%H:%M:%S").is_err());
 }
+
+#[test]
+fn test_overflowing_offset() {
+    let hmsm = |h, m, s, n| NaiveTime::from_hms_milli_opt(h, m, s, n).unwrap();
+
+    let positive_offset = FixedOffset::east_opt(4 * 60 * 60).unwrap();
+    // regular time
+    let t = hmsm(5, 6, 7, 890);
+    assert_eq!(t.overflowing_add_offset(positive_offset), (hmsm(9, 6, 7, 890), 0));
+    assert_eq!(t.overflowing_sub_offset(positive_offset), (hmsm(1, 6, 7, 890), 0));
+    // leap second is preserved, and wrap to next day
+    let t = hmsm(23, 59, 59, 1_000);
+    assert_eq!(t.overflowing_add_offset(positive_offset), (hmsm(3, 59, 59, 1_000), 1));
+    assert_eq!(t.overflowing_sub_offset(positive_offset), (hmsm(19, 59, 59, 1_000), 0));
+    // wrap to previous day
+    let t = hmsm(1, 2, 3, 456);
+    assert_eq!(t.overflowing_sub_offset(positive_offset), (hmsm(21, 2, 3, 456), -1));
+    // an odd offset
+    let negative_offset = FixedOffset::west_opt(((2 * 60) + 3) * 60 + 4).unwrap();
+    let t = hmsm(5, 6, 7, 890);
+    assert_eq!(t.overflowing_add_offset(negative_offset), (hmsm(3, 3, 3, 890), 0));
+    assert_eq!(t.overflowing_sub_offset(negative_offset), (hmsm(7, 9, 11, 890), 0));
+
+    assert_eq!(t.overflowing_add_offset(positive_offset).0, t + positive_offset);
+    assert_eq!(t.overflowing_sub_offset(positive_offset).0, t - positive_offset);
+}
diff --git a/src/offset/fixed.rs b/src/offset/fixed.rs
--- a/src/offset/fixed.rs
+++ b/src/offset/fixed.rs
@@ -183,75 +181,6 @@ impl arbitrary::Arbitrary<'_> for FixedOffset {
     }
 }
 
-// addition or subtraction of FixedOffset to/from Timelike values is the same as
-// adding or subtracting the offset's local_minus_utc value
-// but keep keeps the leap second information.
-// this should be implemented more efficiently, but for the time being, this is generic right now.
-
-fn add_with_leapsecond<T>(lhs: &T, rhs: i32) -> T
-where
-    T: Timelike + Add<TimeDelta, Output = T>,
-{
-    // extract and temporarily remove the fractional part and later recover it
-    let nanos = lhs.nanosecond();
-    let lhs = lhs.with_nanosecond(0).unwrap();
-    (lhs + TimeDelta::seconds(i64::from(rhs))).with_nanosecond(nanos).unwrap()
-}
-
-impl Add<FixedOffset> for NaiveTime {
-    type Output = NaiveTime;
-
-    #[inline]
-    fn add(self, rhs: FixedOffset) -> NaiveTime {
-        add_with_leapsecond(&self, rhs.local_minus_utc)
-    }
-}
-
-impl Sub<FixedOffset> for NaiveTime {
-    type Output = NaiveTime;
-
-    #[inline]
-    fn sub(self, rhs: FixedOffset) -> NaiveTime {
-        add_with_leapsecond(&self, -rhs.local_minus_utc)
-    }
-}
-
-impl Add<FixedOffset> for NaiveDateTime {
-    type Output = NaiveDateTime;
-
-    #[inline]
-    fn add(self, rhs: FixedOffset) -> NaiveDateTime {
-        add_with_leapsecond(&self, rhs.local_minus_utc)
-    }
-}
-
-impl Sub<FixedOffset> for NaiveDateTime {
-    type Output = NaiveDateTime;
-
-    #[inline]
-    fn sub(self, rhs: FixedOffset) -> NaiveDateTime {
-        add_with_leapsecond(&self, -rhs.local_minus_utc)
-    }
-}
-
-impl<Tz: TimeZone> Add<FixedOffset> for DateTime<Tz> {
-    type Output = DateTime<Tz>;
-
-    #[inline]
-    fn add(self, rhs: FixedOffset) -> DateTime<Tz> {
-        add_with_leapsecond(&self, rhs.local_minus_utc)
-    }
-}
-
-impl<Tz: TimeZone> Sub<FixedOffset> for DateTime<Tz> {
-    type Output = DateTime<Tz>;
-
-    #[inline]
-    fn sub(self, rhs: FixedOffset) -> DateTime<Tz> {
-        add_with_leapsecond(&self, -rhs.local_minus_utc)
-    }
-}
-
 #[cfg(test)]
 mod tests {
     use super::FixedOffset;
diff --git a/src/round.rs b/src/round.rs
--- a/src/round.rs
+++ b/src/round.rs
@@ -769,16 +759,16 @@ mod tests {
 
     #[test]
     fn issue1010() {
-        let dt = NaiveDateTime::from_timestamp_opt(-4227854320, 1678774288).unwrap();
-        let span = TimeDelta::microseconds(-7019067213869040);
+        let dt = NaiveDateTime::from_timestamp_opt(-4_227_854_320, 678_774_288).unwrap();
+        let span = TimeDelta::microseconds(-7_019_067_213_869_040);
         assert_eq!(dt.duration_trunc(span), Err(RoundingError::DurationExceedsLimit));
 
-        let dt = NaiveDateTime::from_timestamp_opt(320041586, 1920103021).unwrap();
-        let span = TimeDelta::nanoseconds(-8923838508697114584);
+        let dt = NaiveDateTime::from_timestamp_opt(320_041_586, 920_103_021).unwrap();
+        let span = TimeDelta::nanoseconds(-8_923_838_508_697_114_584);
         assert_eq!(dt.duration_round(span), Err(RoundingError::DurationExceedsLimit));
 
-        let dt = NaiveDateTime::from_timestamp_opt(-2621440, 0).unwrap();
-        let span = TimeDelta::nanoseconds(-9223372036854771421);
+        let dt = NaiveDateTime::from_timestamp_opt(-2_621_440, 0).unwrap();
+        let span = TimeDelta::nanoseconds(-9_223_372_036_854_771_421);
         assert_eq!(dt.duration_round(span), Err(RoundingError::DurationExceedsLimit));
     }
 }
diff --git a/tests/dateutils.rs b/tests/dateutils.rs
--- a/tests/dateutils.rs
+++ b/tests/dateutils.rs
@@ -121,6 +121,7 @@ fn verify_against_date_command_format_local(path: &'static str, dt: NaiveDateTim
 
     let output = process::Command::new(path)
         .env("LANG", "c")
+        .env("LC_ALL", "c")
         .arg("-d")
         .arg(format!(
             "{}-{:02}-{:02} {:02}:{:02}:{:02}",

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard ce4644f5df6878f3c08a8dc8785433f16c9055c3
git clean -fd
