#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/config/src/combined.rs b/crates/config/src/combined.rs
--- a/crates/config/src/combined.rs
+++ b/crates/config/src/combined.rs
@@ -272,10 +321,10 @@ language: Tsx",
     let pre = scan.find(&root);
     assert_eq!(pre.suppressions.0.len(), 4);
     let scanned = scan.scan(&root, pre, false);
-    let matches = &scanned.matches[&0];
-    assert_eq!(matches.len(), 2);
-    assert_eq!(matches[0].text(), "console.log('no ignore')");
-    assert_eq!(matches[1].text(), "console.log('ignore another')");
+    let matches = &scanned.matches[0];
+    assert_eq!(matches.1.len(), 2);
+    assert_eq!(matches.1[0].text(), "console.log('no ignore')");
+    assert_eq!(matches.1[1].text(), "console.log('ignore another')");
   }
 
   #[test]
diff --git a/crates/config/src/combined.rs b/crates/config/src/combined.rs
--- a/crates/config/src/combined.rs
+++ b/crates/config/src/combined.rs
@@ -294,10 +343,10 @@ language: Tsx",
     let pre = scan.find(&root);
     assert_eq!(pre.suppressions.0.len(), 4);
     let scanned = scan.scan(&root, pre, false);
-    let matches = &scanned.matches[&0];
-    assert_eq!(matches.len(), 2);
-    assert_eq!(matches[0].text(), "console.log('no ignore')");
-    assert_eq!(matches[1].text(), "console.log('ignore another')");
+    let matches = &scanned.matches[0];
+    assert_eq!(matches.1.len(), 2);
+    assert_eq!(matches.1[0].text(), "console.log('no ignore')");
+    assert_eq!(matches.1[1].text(), "console.log('ignore another')");
   }
 
   #[test]
diff --git a/crates/config/src/combined.rs b/crates/config/src/combined.rs
--- a/crates/config/src/combined.rs
+++ b/crates/config/src/combined.rs
@@ -310,12 +359,14 @@ language: Tsx",
     let root = TypeScript::Tsx.ast_grep(source);
     let rule = create_rule();
     let rules = vec![&rule];
-    let scan = CombinedScan::new(rules);
+    let mut scan = CombinedScan::new(rules);
+    scan.set_unused_suppression_rule(&rule);
     let pre = scan.find(&root);
     assert_eq!(pre.suppressions.0.len(), 2);
     let scanned = scan.scan(&root, pre, false);
-    let unused = &scanned.unused_suppressions;
-    assert_eq!(unused.len(), 1);
-    assert_eq!(unused[0].text(), "// ast-grep-ignore: test");
+    assert_eq!(scanned.matches.len(), 2);
+    let unused = &scanned.matches[1];
+    assert_eq!(unused.1.len(), 1);
+    assert_eq!(unused.1[0].text(), "// ast-grep-ignore: test");
   }
 }

EOF_114329324912
cd "crates/config"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard 1c68280bfb9fd2614cab19b4cda47d2bf6570626
git clean -fd
