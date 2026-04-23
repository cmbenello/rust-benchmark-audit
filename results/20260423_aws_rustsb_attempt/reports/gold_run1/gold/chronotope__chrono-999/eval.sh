#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/month.rs b/src/month.rs
--- a/src/month.rs
+++ b/src/month.rs
@@ -352,4 +352,13 @@ mod tests {
         assert_eq!(Month::January.pred(), Month::December);
         assert_eq!(Month::February.pred(), Month::January);
     }
+
+    #[test]
+    fn test_month_partial_ord() {
+        assert!(Month::January <= Month::January);
+        assert!(Month::January < Month::February);
+        assert!(Month::January < Month::December);
+        assert!(Month::July >= Month::May);
+        assert!(Month::September > Month::March);
+    }
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 1f1e2f8ff0e166ffd80ae95218a80b54fe26e003
git clean -fd
