#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/integration/git.rs b/tests/integration/git.rs
--- a/tests/integration/git.rs
+++ b/tests/integration/git.rs
@@ -1,7 +1,7 @@
 use assert_cmd::prelude::*;
 use bstr::ByteSlice;
 use git2::Repository;
-use git_config::File as GitConfig;
+use gix_config::File as GitConfig;
 use predicates::prelude::*;
 use std::ops::Deref;
 

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard faf3d01b1e1af968ec2a17a80849193c70657f8a
git clean -fd
