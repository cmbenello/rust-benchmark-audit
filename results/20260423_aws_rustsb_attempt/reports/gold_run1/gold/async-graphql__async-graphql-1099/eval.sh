#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/http/graphiql_v2_source.rs b/src/http/graphiql_v2_source.rs
--- a/src/http/graphiql_v2_source.rs
+++ b/src/http/graphiql_v2_source.rs
@@ -275,6 +286,7 @@ mod tests {
             .endpoint("http://localhost:8000")
             .subscription_endpoint("ws://localhost:8000/ws")
             .header("Authorization", "Bearer <token>")
+            .title("Awesome GraphiQL IDE Test")
             .finish();
 
         assert_eq!(
diff --git a/src/http/graphiql_v2_source.rs b/src/http/graphiql_v2_source.rs
--- a/src/http/graphiql_v2_source.rs
+++ b/src/http/graphiql_v2_source.rs
@@ -288,7 +300,7 @@ mod tests {
     <meta name="viewport" content="width=device-width, initial-scale=1">
     <meta name="referrer" content="origin">
 
-    <title>GraphiQL IDE</title>
+    <title>Awesome GraphiQL IDE Test</title>
 
     <style>
       body {

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 49b6fa4f9f3a679a318fb489fe2d017c38b09237
git clean -fd
