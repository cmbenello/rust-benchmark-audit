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
@@ -1,8 +1,15 @@
 // This is a part of Chrono.
 // See README.md and LICENSE.txt for details.
 
+use core::cmp::Ordering;
+use core::fmt;
+use core::marker::Sized;
 use core::ops::{Add, Sub};
+use datetime::DateTime;
 use oldtime::Duration;
+#[cfg(any(feature = "std", test))]
+use std;
+use TimeZone;
 use Timelike;
 
 /// Extension trait for subsecond rounding or truncation to a maximum number
diff --git a/src/round.rs b/src/round.rs
--- a/src/round.rs
+++ b/src/round.rs
@@ -85,14 +92,187 @@ fn span_for_digits(digits: u16) -> u32 {
     }
 }
 
+/// Extension trait for rounding or truncating a DateTime by a Duration.
+///
+/// # Limitations
+/// Both rounding and truncating are done via [`Duration::num_nanoseconds`] and
+/// [`DateTime::timestamp_nanos`]. This means that they will fail if either the
+/// `Duration` or the `DateTime` are too big to represented as nanoseconds. They
+/// will also fail if the `Duration` is bigger than the timestamp.
+pub trait DurationRound: Sized {
+    /// Error that can occur in rounding or truncating
+    #[cfg(any(feature = "std", test))]
+    type Err: std::error::Error;
+
+    /// Error that can occur in rounding or truncating
+    #[cfg(not(any(feature = "std", test)))]
+    type Err: fmt::Debug + fmt::Display;
+
+    /// Return a copy rounded by Duration.
+    ///
+    /// # Example
+    /// ``` rust
+    /// # use chrono::{DateTime, DurationRound, Duration, TimeZone, Utc};
+    /// let dt = Utc.ymd(2018, 1, 11).and_hms_milli(12, 0, 0, 154);
+    /// assert_eq!(
+    ///     dt.duration_round(Duration::milliseconds(10)).unwrap().to_string(),
+    ///     "2018-01-11 12:00:00.150 UTC"
+    /// );
+    /// assert_eq!(
+    ///     dt.duration_round(Duration::days(1)).unwrap().to_string(),
+    ///     "2018-01-12 00:00:00 UTC"
+    /// );
+    /// ```
+    fn duration_round(self, duration: Duration) -> Result<Self, Self::Err>;
+
+    /// Return a copy truncated by Duration.
+    ///
+    /// # Example
+    /// ``` rust
+    /// # use chrono::{DateTime, DurationRound, Duration, TimeZone, Utc};
+    /// let dt = Utc.ymd(2018, 1, 11).and_hms_milli(12, 0, 0, 154);
+    /// assert_eq!(
+    ///     dt.duration_trunc(Duration::milliseconds(10)).unwrap().to_string(),
+    ///     "2018-01-11 12:00:00.150 UTC"
+    /// );
+    /// assert_eq!(
+    ///     dt.duration_trunc(Duration::days(1)).unwrap().to_string(),
+    ///     "2018-01-11 00:00:00 UTC"
+    /// );
+    /// ```
+    fn duration_trunc(self, duration: Duration) -> Result<Self, Self::Err>;
+}
+
+/// The maximum number of seconds a DateTime can be to be represented as nanoseconds
+const MAX_SECONDS_TIMESTAMP_FOR_NANOS: i64 = 9_223_372_036;
+
+impl<Tz: TimeZone> DurationRound for DateTime<Tz> {
+    type Err = RoundingError;
+
+    fn duration_round(self, duration: Duration) -> Result<Self, Self::Err> {
+        if let Some(span) = duration.num_nanoseconds() {
+            if self.timestamp().abs() > MAX_SECONDS_TIMESTAMP_FOR_NANOS {
+                return Err(RoundingError::TimestampExceedsLimit);
+            }
+            let stamp = self.timestamp_nanos();
+            if span > stamp.abs() {
+                return Err(RoundingError::DurationExceedsTimestamp);
+            }
+            let delta_down = stamp % span;
+            if delta_down == 0 {
+                Ok(self)
+            } else {
+                let (delta_up, delta_down) = if delta_down < 0 {
+                    (delta_down.abs(), span - delta_down.abs())
+                } else {
+                    (span - delta_down, delta_down)
+                };
+                if delta_up <= delta_down {
+                    Ok(self + Duration::nanoseconds(delta_up))
+                } else {
+                    Ok(self - Duration::nanoseconds(delta_down))
+                }
+            }
+        } else {
+            Err(RoundingError::DurationExceedsLimit)
+        }
+    }
+
+    fn duration_trunc(self, duration: Duration) -> Result<Self, Self::Err> {
+        if let Some(span) = duration.num_nanoseconds() {
+            if self.timestamp().abs() > MAX_SECONDS_TIMESTAMP_FOR_NANOS {
+                return Err(RoundingError::TimestampExceedsLimit);
+            }
+            let stamp = self.timestamp_nanos();
+            if span > stamp.abs() {
+                return Err(RoundingError::DurationExceedsTimestamp);
+            }
+            let delta_down = stamp % span;
+            match delta_down.cmp(&0) {
+                Ordering::Equal => Ok(self),
+                Ordering::Greater => Ok(self - Duration::nanoseconds(delta_down)),
+                Ordering::Less => Ok(self - Duration::nanoseconds(span - delta_down.abs())),
+            }
+        } else {
+            Err(RoundingError::DurationExceedsLimit)
+        }
+    }
+}
+
+/// An error from rounding by `Duration`
+///
+/// See: [`DurationRound`]
+#[derive(Debug, Clone, PartialEq, Eq, Copy)]
+pub enum RoundingError {
+    /// Error when the Duration exceeds the Duration from or until the Unix epoch.
+    ///
+    /// ``` rust
+    /// # use chrono::{DateTime, DurationRound, Duration, RoundingError, TimeZone, Utc};
+    /// let dt = Utc.ymd(1970, 12, 12).and_hms(0, 0, 0);
+    ///
+    /// assert_eq!(
+    ///     dt.duration_round(Duration::days(365)),
+    ///     Err(RoundingError::DurationExceedsTimestamp),
+    /// );
+    /// ```
+    DurationExceedsTimestamp,
+
+    /// Error when `Duration.num_nanoseconds` exceeds the limit.
+    ///
+    /// ``` rust
+    /// # use chrono::{DateTime, DurationRound, Duration, RoundingError, TimeZone, Utc};
+    /// let dt = Utc.ymd(2260, 12, 31).and_hms_nano(23, 59, 59, 1_75_500_000);
+    ///
+    /// assert_eq!(
+    ///     dt.duration_round(Duration::days(300 * 365)),
+    ///     Err(RoundingError::DurationExceedsLimit)
+    /// );
+    /// ```
+    DurationExceedsLimit,
+
+    /// Error when `DateTime.timestamp_nanos` exceeds the limit.
+    ///
+    /// ``` rust
+    /// # use chrono::{DateTime, DurationRound, Duration, RoundingError, TimeZone, Utc};
+    /// let dt = Utc.ymd(2300, 12, 12).and_hms(0, 0, 0);
+    ///
+    /// assert_eq!(dt.duration_round(Duration::days(1)), Err(RoundingError::TimestampExceedsLimit),);
+    /// ```
+    TimestampExceedsLimit,
+}
+
+impl fmt::Display for RoundingError {
+    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
+        match *self {
+            RoundingError::DurationExceedsTimestamp => {
+                write!(f, "duration in nanoseconds exceeds timestamp")
+            }
+            RoundingError::DurationExceedsLimit => {
+                write!(f, "duration exceeds num_nanoseconds limit")
+            }
+            RoundingError::TimestampExceedsLimit => {
+                write!(f, "timestamp exceeds num_nanoseconds limit")
+            }
+        }
+    }
+}
+
+#[cfg(any(feature = "std", test))]
+impl std::error::Error for RoundingError {
+    #[allow(deprecated)]
+    fn description(&self) -> &str {
+        "error from rounding or truncating with DurationRound"
+    }
+}
+
 #[cfg(test)]
 mod tests {
-    use super::SubsecRound;
+    use super::{Duration, DurationRound, SubsecRound};
     use offset::{FixedOffset, TimeZone, Utc};
     use Timelike;
 
     #[test]
-    fn test_round() {
+    fn test_round_subsecs() {
         let pst = FixedOffset::east(8 * 60 * 60);
         let dt = pst.ymd(2018, 1, 11).and_hms_nano(10, 5, 13, 084_660_684);
 
diff --git a/src/round.rs b/src/round.rs
--- a/src/round.rs
+++ b/src/round.rs
@@ -135,7 +315,7 @@ mod tests {
     }
 
     #[test]
-    fn test_trunc() {
+    fn test_trunc_subsecs() {
         let pst = FixedOffset::east(8 * 60 * 60);
         let dt = pst.ymd(2018, 1, 11).and_hms_nano(10, 5, 13, 084_660_684);
 
diff --git a/src/round.rs b/src/round.rs
--- a/src/round.rs
+++ b/src/round.rs
@@ -176,4 +356,101 @@ mod tests {
         assert_eq!(dt.trunc_subsecs(0).nanosecond(), 1_000_000_000);
         assert_eq!(dt.trunc_subsecs(0).second(), 59);
     }
+
+    #[test]
+    fn test_duration_round() {
+        let dt = Utc.ymd(2016, 12, 31).and_hms_nano(23, 59, 59, 175_500_000);
+
+        assert_eq!(
+            dt.duration_round(Duration::milliseconds(10)).unwrap().to_string(),
+            "2016-12-31 23:59:59.180 UTC"
+        );
+
+        // round up
+        let dt = Utc.ymd(2012, 12, 12).and_hms_milli(18, 22, 30, 0);
+        assert_eq!(
+            dt.duration_round(Duration::minutes(5)).unwrap().to_string(),
+            "2012-12-12 18:25:00 UTC"
+        );
+        // round down
+        let dt = Utc.ymd(2012, 12, 12).and_hms_milli(18, 22, 29, 999);
+        assert_eq!(
+            dt.duration_round(Duration::minutes(5)).unwrap().to_string(),
+            "2012-12-12 18:20:00 UTC"
+        );
+
+        assert_eq!(
+            dt.duration_round(Duration::minutes(10)).unwrap().to_string(),
+            "2012-12-12 18:20:00 UTC"
+        );
+        assert_eq!(
+            dt.duration_round(Duration::minutes(30)).unwrap().to_string(),
+            "2012-12-12 18:30:00 UTC"
+        );
+        assert_eq!(
+            dt.duration_round(Duration::hours(1)).unwrap().to_string(),
+            "2012-12-12 18:00:00 UTC"
+        );
+        assert_eq!(
+            dt.duration_round(Duration::days(1)).unwrap().to_string(),
+            "2012-12-13 00:00:00 UTC"
+        );
+    }
+
+    #[test]
+    fn test_duration_round_pre_epoch() {
+        let dt = Utc.ymd(1969, 12, 12).and_hms(12, 12, 12);
+        assert_eq!(
+            dt.duration_round(Duration::minutes(10)).unwrap().to_string(),
+            "1969-12-12 12:10:00 UTC"
+        );
+    }
+
+    #[test]
+    fn test_duration_trunc() {
+        let dt = Utc.ymd(2016, 12, 31).and_hms_nano(23, 59, 59, 1_75_500_000);
+
+        assert_eq!(
+            dt.duration_trunc(Duration::milliseconds(10)).unwrap().to_string(),
+            "2016-12-31 23:59:59.170 UTC"
+        );
+
+        // would round up
+        let dt = Utc.ymd(2012, 12, 12).and_hms_milli(18, 22, 30, 0);
+        assert_eq!(
+            dt.duration_trunc(Duration::minutes(5)).unwrap().to_string(),
+            "2012-12-12 18:20:00 UTC"
+        );
+        // would round down
+        let dt = Utc.ymd(2012, 12, 12).and_hms_milli(18, 22, 29, 999);
+        assert_eq!(
+            dt.duration_trunc(Duration::minutes(5)).unwrap().to_string(),
+            "2012-12-12 18:20:00 UTC"
+        );
+        assert_eq!(
+            dt.duration_trunc(Duration::minutes(10)).unwrap().to_string(),
+            "2012-12-12 18:20:00 UTC"
+        );
+        assert_eq!(
+            dt.duration_trunc(Duration::minutes(30)).unwrap().to_string(),
+            "2012-12-12 18:00:00 UTC"
+        );
+        assert_eq!(
+            dt.duration_trunc(Duration::hours(1)).unwrap().to_string(),
+            "2012-12-12 18:00:00 UTC"
+        );
+        assert_eq!(
+            dt.duration_trunc(Duration::days(1)).unwrap().to_string(),
+            "2012-12-12 00:00:00 UTC"
+        );
+    }
+
+    #[test]
+    fn test_duration_trunc_pre_epoch() {
+        let dt = Utc.ymd(1969, 12, 12).and_hms(12, 12, 12);
+        assert_eq!(
+            dt.duration_trunc(Duration::minutes(10)).unwrap().to_string(),
+            "1969-12-12 12:10:00 UTC"
+        );
+    }
 }

EOF_114329324912
git status
git diff
cargo test --no-fail-fast
git status
git reset --hard e90050c852a8be1f1541dabb39294816d891239b
git clean -fd
