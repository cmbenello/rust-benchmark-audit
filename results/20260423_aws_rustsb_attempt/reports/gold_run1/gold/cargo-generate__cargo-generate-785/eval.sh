#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/args.rs b/src/args.rs
--- a/src/args.rs
+++ b/src/args.rs
@@ -42,8 +42,8 @@ pub struct GenerateArgs {
             "bin",
             "define",
             "init",
-            "template-values-file",
-            "ssh-identity",
+            "template_values_file",
+            "ssh_identity",
             "test",
         ])
     )]

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 15b2143c1c5520cdd378c762769eacf55d2853e4
git clean -fd
