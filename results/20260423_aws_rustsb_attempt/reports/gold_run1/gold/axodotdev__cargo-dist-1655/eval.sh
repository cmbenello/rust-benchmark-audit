#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/cargo-dist/src/lib.rs b/cargo-dist/src/lib.rs
--- a/cargo-dist/src/lib.rs
+++ b/cargo-dist/src/lib.rs
@@ -255,7 +255,7 @@ fn run_build_step(
 }
 
 const AXOUPDATER_ASSET_ROOT: &str = "https://github.com/axodotdev/axoupdater/releases";
-const AXOUPDATER_MINIMUM_VERSION: &str = "0.7.0";
+const AXOUPDATER_MINIMUM_VERSION: &str = "0.9.0";
 
 fn axoupdater_latest_asset_root() -> String {
     format!("{AXOUPDATER_ASSET_ROOT}/latest/download")
diff --git a/cargo-dist/tests/cli-tests.rs b/cargo-dist/tests/cli-tests.rs
--- a/cargo-dist/tests/cli-tests.rs
+++ b/cargo-dist/tests/cli-tests.rs
@@ -273,6 +273,8 @@ fn test_self_update() {
         .map(|s| s == "selfupdate" || s == "all")
         .unwrap_or(false)
     {
+        std::env::remove_var("XDG_CONFIG_HOME");
+
         let mut args = RuntestArgs {
             app_name: "cargo-dist".to_owned(),
             package: "cargo-dist".to_owned(),
diff --git a/cargo-dist/tests/gallery/dist/powershell.rs b/cargo-dist/tests/gallery/dist/powershell.rs
--- a/cargo-dist/tests/gallery/dist/powershell.rs
+++ b/cargo-dist/tests/gallery/dist/powershell.rs
@@ -105,6 +105,7 @@ impl AppResult {
                     .env("LOCALAPPDATA", &appdata)
                     .env("MY_ENV_VAR", &app_home)
                     .env_remove("CARGO_HOME")
+                    .env_remove("XDG_CONFIG_HOME")
                     .env_remove("PSModulePath")
             })?;
             eprintln!(
diff --git a/cargo-dist/tests/gallery/dist/shell.rs b/cargo-dist/tests/gallery/dist/shell.rs
--- a/cargo-dist/tests/gallery/dist/shell.rs
+++ b/cargo-dist/tests/gallery/dist/shell.rs
@@ -48,6 +48,7 @@ impl AppResult {
                     .env("ZDOTDIR", &tempdir)
                     .env("MY_ENV_VAR", &app_home)
                     .env_remove("CARGO_HOME")
+                    .env_remove("XDG_CONFIG_HOME")
             })?;
             // we could theoretically look at the above output and parse out the `source` line...
 
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/akaikatana-repack"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/akaikatana-repack"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\akaikatana-repack"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\akaikatana-repack"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\akaikatana-repack"
+}
 
 if ($env:AKAIKATANA_REPACK_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/akaikatana_musl.snap b/cargo-dist/tests/snapshots/akaikatana_musl.snap
--- a/cargo-dist/tests/snapshots/akaikatana_musl.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_musl.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/akaikatana-repack"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/akaikatana-repack"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
--- a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/akaikatana-repack"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/akaikatana-repack"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
--- a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
@@ -1468,7 +1468,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\akaikatana-repack"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\akaikatana-repack"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\akaikatana-repack"
+}
 
 if ($env:AKAIKATANA_REPACK_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
--- a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/akaikatana-repack"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/akaikatana-repack"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
--- a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
@@ -1492,7 +1492,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\akaikatana-repack"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\akaikatana-repack"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\akaikatana-repack"
+}
 
 if ($env:AKAIKATANA_REPACK_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/akaikatana_updaters.snap b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
--- a/cargo-dist/tests/snapshots/akaikatana_updaters.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/akaikatana-repack"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/akaikatana-repack"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/akaikatana_updaters.snap b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
--- a/cargo-dist/tests/snapshots/akaikatana_updaters.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"akaikatana-repack","name":"akaikatana-repack","owner":"mistydemeo","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\akaikatana-repack"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\akaikatana-repack"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\akaikatana-repack"
+}
 
 if ($env:AKAIKATANA_REPACK_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -47,7 +47,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"axo"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"axo"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -47,7 +47,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"axo"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"axo"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias.snap b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias.snap b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
@@ -1468,7 +1468,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
@@ -1472,7 +1472,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1508,7 +1508,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
@@ -1443,7 +1443,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_checksum_blake2b.snap b/cargo-dist/tests/snapshots/axolotlsay_checksum_blake2b.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_checksum_blake2b.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_checksum_blake2b.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_checksum_blake2s.snap b/cargo-dist/tests/snapshots/axolotlsay_checksum_blake2s.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_checksum_blake2s.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_checksum_blake2s.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_checksum_sha3_256.snap b/cargo-dist/tests/snapshots/axolotlsay_checksum_sha3_256.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_checksum_sha3_256.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_checksum_sha3_256.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_checksum_sha3_512.snap b/cargo-dist/tests/snapshots/axolotlsay_checksum_sha3_512.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_checksum_sha3_512.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_checksum_sha3_512.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -1441,7 +1441,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross2.snap b/cargo-dist/tests/snapshots/axolotlsay_cross2.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross2.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross2.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross2.snap b/cargo-dist/tests/snapshots/axolotlsay_cross2.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross2.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross2.snap
@@ -1305,7 +1305,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
@@ -1375,7 +1375,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay-js","name":"axolotlsay-hybrid","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay-js"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay-js"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -1439,7 +1439,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay-js","name":"axolotlsay-hybrid","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay-js"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay-js"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay-js"
+}
 
 if ($env:AXOLOTLSAY_JS_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -1991,7 +1995,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay-hybrid","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -3375,7 +3379,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay-hybrid","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl.snap b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
@@ -1472,7 +1472,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -1375,7 +1375,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -1375,7 +1375,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -1440,7 +1440,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
--- a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
+++ b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
--- a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
+++ b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
@@ -1437,7 +1437,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -1423,7 +1423,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false
diff --git a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
--- a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
+++ b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
@@ -56,7 +56,7 @@ fi
 read -r RECEIPT <<EORECEIPT
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 EORECEIPT
-RECEIPT_HOME="${HOME}/.config/axolotlsay"
+RECEIPT_HOME="${XDG_CONFIG_HOME:-$HOME/.config}/axolotlsay"
 
 usage() {
     # print help (this cat/EOF stuff is a "heredoc" string)
diff --git a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
--- a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
+++ b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
@@ -1437,7 +1437,11 @@ if ($env:INSTALLER_DOWNLOAD_URL) {
 $receipt = @"
 {"binaries":["CARGO_DIST_BINS"],"binary_aliases":{},"cdylibs":["CARGO_DIST_DYLIBS"],"cstaticlibs":["CARGO_DIST_STATICLIBS"],"install_layout":"unspecified","install_prefix":"AXO_INSTALL_PREFIX","modify_path":true,"provider":{"source":"cargo-dist","version":"CENSORED"},"source":{"app_name":"axolotlsay","name":"axolotlsay","owner":"axodotdev","release_type":"github"},"version":"CENSORED"}
 "@
-$receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+if ($env:XDG_CONFIG_HOME) {
+  $receipt_home = "${env:XDG_CONFIG_HOME}\axolotlsay"
+} else {
+  $receipt_home = "${env:LOCALAPPDATA}\axolotlsay"
+}
 
 if ($env:AXOLOTLSAY_DISABLE_UPDATE) {
   $install_updater = $false

EOF_114329324912
cd "cargo-dist"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard aa437ee6ed8b01bf28cfcac1d2fa1a7fe95979a6
git clean -fd
