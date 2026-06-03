#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/bin/upgrade/upgrade.rs b/src/bin/upgrade/upgrade.rs
--- a/src/bin/upgrade/upgrade.rs
+++ b/src/bin/upgrade/upgrade.rs
@@ -263,7 +268,7 @@ fn exec(args: UpgradeArgs) -> CargoResult<()> {
                             }
                         }
                         let is_prerelease = old_version_req.contains('-');
-                        get_latest_dependency(
+                        let new_version = get_latest_dependency(
                             &dependency.name,
                             is_prerelease,
                             &manifest_path,
diff --git a/tests/cmd/upgrade/skip_compatible.toml b/tests/cmd/upgrade/skip_compatible.toml
--- a/tests/cmd/upgrade/skip_compatible.toml
+++ b/tests/cmd/upgrade/skip_compatible.toml
@@ -1,5 +1,5 @@
 bin.name = "cargo-upgrade"
-args = ["upgrade", "--skip-compatible", "--verbose"]
+args = ["upgrade", "--verbose"]
 status = "success"
 stdout = ""
 stderr = """
diff --git a/tests/cmd/upgrade/skip_pinned.out/Cargo.toml b/tests/cmd/upgrade/skip_pinned.out/Cargo.toml
--- a/tests/cmd/upgrade/skip_pinned.out/Cargo.toml
+++ b/tests/cmd/upgrade/skip_pinned.out/Cargo.toml
@@ -9,6 +9,6 @@ lessthan = "<0.4"
 lessorequal = "<=3.0"
 caret = "^99999.0"
 tilde = "~99999.0.0"
-greaterthan = "99999.0.0"
-greaterorequal = "99999.0.0"
+greaterthan = ">2.0"
+greaterorequal = ">=2.1.0"
 wildcard = "99999.*"
diff --git a/tests/cmd/upgrade/skip_pinned.toml b/tests/cmd/upgrade/skip_pinned.toml
--- a/tests/cmd/upgrade/skip_pinned.toml
+++ b/tests/cmd/upgrade/skip_pinned.toml
@@ -10,8 +10,8 @@ warning: ignoring lessthan, version (<0.4) is pinned
 warning: ignoring lessorequal, version (<=3.0) is pinned
    Upgrading caret: ^3.0 -> ^99999.0
    Upgrading tilde: ~4.1.0 -> ~99999.0.0
-   Upgrading greaterthan: >2.0 -> v99999.0.0
-   Upgrading greaterorequal: >=2.1.0 -> v99999.0.0
+warning: ignoring greaterthan, version (>2.0) is compatible with 99999.0.0
+warning: ignoring greaterorequal, version (>=2.1.0) is compatible with 99999.0.0
    Upgrading wildcard: v3.* -> v99999.*
 """
 fs.sandbox = true

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard e5279e65c2664a01eb5ef26184743535824dac56
git clean -fd
