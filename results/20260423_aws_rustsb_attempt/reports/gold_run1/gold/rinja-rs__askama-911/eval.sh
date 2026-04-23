#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/testing/templates/match-enum-or.html
new file mode 100644
--- /dev/null
+++ b/testing/templates/match-enum-or.html
@@ -0,0 +1,8 @@
+The card is
+{%- match suit %}
+   {%- when Suit::Clubs or Suit::Spades -%}
+     {{ " black" }}
+   {%- when Suit::Diamonds or Suit::Hearts -%}
+     {{ " red" }}
+{%- endmatch %}
+
diff --git a/testing/tests/matches.rs b/testing/tests/matches.rs
--- a/testing/tests/matches.rs
+++ b/testing/tests/matches.rs
@@ -195,3 +195,32 @@ fn test_match_with_comment() {
     let s = MatchWithComment { good: false };
     assert_eq!(s.render().unwrap(), "bad");
 }
+
+enum Suit {
+    Clubs,
+    Diamonds,
+    Hearts,
+    Spades,
+}
+
+#[derive(Template)]
+#[template(path = "match-enum-or.html")]
+struct MatchEnumOrTemplate {
+    suit: Suit,
+}
+
+#[test]
+fn test_match_enum_or() {
+    let template = MatchEnumOrTemplate { suit: Suit::Clubs };
+    assert_eq!(template.render().unwrap(), "The card is black\n");
+    let template = MatchEnumOrTemplate { suit: Suit::Spades };
+    assert_eq!(template.render().unwrap(), "The card is black\n");
+
+    let template = MatchEnumOrTemplate { suit: Suit::Hearts };
+    assert_eq!(template.render().unwrap(), "The card is red\n");
+
+    let template = MatchEnumOrTemplate {
+        suit: Suit::Diamonds,
+    };
+    assert_eq!(template.render().unwrap(), "The card is red\n");
+}

EOF_114329324912
cd "testing"
cargo test --no-fail-fast --all-features --test "matches"
cd ../
git reset --hard 6cbfde04514a90d4e24350c21ef490c40666d820
git clean -fd
