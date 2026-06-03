#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/cargo-dist/tests/integration-tests.rs b/cargo-dist/tests/integration-tests.rs
--- a/cargo-dist/tests/integration-tests.rs
+++ b/cargo-dist/tests/integration-tests.rs
@@ -37,7 +37,7 @@ cargo-dist-version = "{dist_version}"
 installers = ["shell", "powershell", "homebrew", "npm", "msi", "pkg"]
 tap = "axodotdev/homebrew-packages"
 publish-jobs = ["homebrew", "npm"]
-targets = ["x86_64-unknown-linux-gnu", "i686-unknown-linux-gnu", "x86_64-apple-darwin", "x86_64-pc-windows-msvc", "aarch64-apple-darwin"]
+targets = ["x86_64-unknown-linux-gnu", "i686-unknown-linux-gnu", "x86_64-apple-darwin", "x86_64-pc-windows-msvc", "x86_64-pc-windows-gnu", "aarch64-apple-darwin"]
 install-success-msg = ">o_o< everything's installed!"
 ci = ["github"]
 unix-archive = ".tar.gz"
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc.zip"
+      "bins" = @("akextract.exe", "akmetadata.exe", "akrepack.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".zip"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc.zip"
       "bins" = @("akextract.exe", "akmetadata.exe", "akrepack.exe")
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -2019,6 +2038,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/mistydemeo/akaikatana-repack/releases/download/v0.2.0/akaikatana-repack-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
--- a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
@@ -1512,6 +1512,17 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{"akextract.exe":["akextract-link.exe"]}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc.zip"
+      "bins" = @("akextract.exe", "akmetadata.exe", "akrepack.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".zip"
+      "aliases" = @{
+        "akextract.exe" = "akextract-link.exe"
+      }
+      "aliases_json" = '{"akextract.exe":["akextract-link.exe"]}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc.zip"
       "bins" = @("akextract.exe", "akmetadata.exe", "akrepack.exe")
diff --git a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
--- a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
@@ -1539,7 +1550,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
--- a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
@@ -1553,10 +1573,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
--- a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
@@ -1568,14 +1588,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
--- a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
@@ -1658,7 +1678,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
--- a/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_one_alias_among_many_binaries.snap
@@ -2049,6 +2069,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/mistydemeo/akaikatana-repack/releases/download/v0.2.0/akaikatana-repack-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
--- a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
@@ -1537,6 +1537,18 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{"akextract.exe":["akextract-link.exe"],"akmetadata.exe":["akmetadata-link.exe"]}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc.zip"
+      "bins" = @("akextract.exe", "akmetadata.exe", "akrepack.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".zip"
+      "aliases" = @{
+        "akextract.exe" = "akextract-link.exe"
+        "akmetadata.exe" = "akmetadata-link.exe"
+      }
+      "aliases_json" = '{"akextract.exe":["akextract-link.exe"],"akmetadata.exe":["akmetadata-link.exe"]}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc.zip"
       "bins" = @("akextract.exe", "akmetadata.exe", "akrepack.exe")
diff --git a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
--- a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
@@ -1565,7 +1577,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
--- a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
@@ -1579,10 +1600,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
--- a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
@@ -1594,14 +1615,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
--- a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
@@ -1684,7 +1705,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
--- a/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_two_bin_aliases.snap
@@ -2075,6 +2096,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/mistydemeo/akaikatana-repack/releases/download/v0.2.0/akaikatana-repack-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/akaikatana_updaters.snap b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
--- a/cargo-dist/tests/snapshots/akaikatana_updaters.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
@@ -1487,6 +1487,20 @@ function Install-Binary($install_args) {
         "bin" = "akaikatana-repack-x86_64-pc-windows-msvc-update"
       }
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc.zip"
+      "bins" = @("akextract.exe", "akmetadata.exe", "akrepack.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".zip"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+      "updater" = @{
+        "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc-update"
+        "bin" = "akaikatana-repack-x86_64-pc-windows-msvc-update"
+      }
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "akaikatana-repack-x86_64-pc-windows-msvc.zip"
       "bins" = @("akextract.exe", "akmetadata.exe", "akrepack.exe")
diff --git a/cargo-dist/tests/snapshots/akaikatana_updaters.snap b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
--- a/cargo-dist/tests/snapshots/akaikatana_updaters.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
@@ -1517,7 +1531,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/akaikatana_updaters.snap b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
--- a/cargo-dist/tests/snapshots/akaikatana_updaters.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
@@ -1531,10 +1554,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/akaikatana_updaters.snap b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
--- a/cargo-dist/tests/snapshots/akaikatana_updaters.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
@@ -1546,14 +1569,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/akaikatana_updaters.snap b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
--- a/cargo-dist/tests/snapshots/akaikatana_updaters.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
@@ -1636,7 +1659,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/akaikatana_updaters.snap b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
--- a/cargo-dist/tests/snapshots/akaikatana_updaters.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_updaters.snap
@@ -2038,6 +2061,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/mistydemeo/akaikatana-repack/releases/download/v0.2.0/akaikatana-repack-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -3480,6 +3499,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://fake.axo.dev/faker/axolotlsay/fake-id-do-not-upload/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -3473,6 +3492,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://fake.axo.dev/faker/axolotlsay/fake-id-do-not-upload/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias.snap b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
@@ -1512,6 +1512,17 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{"axolotlsay.exe":["axolotlsay-link.exe"]}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+        "axolotlsay.exe" = "axolotlsay-link.exe"
+      }
+      "aliases_json" = '{"axolotlsay.exe":["axolotlsay-link.exe"]}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias.snap b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
@@ -1539,7 +1550,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias.snap b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
@@ -1553,10 +1573,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias.snap b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
@@ -1568,14 +1588,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias.snap b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
@@ -1658,7 +1678,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias.snap b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias.snap
@@ -3521,6 +3541,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
@@ -1516,6 +1516,17 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{"nosuchbin.exe":["axolotlsay-link1.exe","axolotlsay-link2.exe"]}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+        "nosuchbin.exe" = "axolotlsay-link1.exe", "axolotlsay-link2.exe"
+      }
+      "aliases_json" = '{"nosuchbin.exe":["axolotlsay-link1.exe","axolotlsay-link2.exe"]}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
@@ -1543,7 +1554,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
@@ -1557,10 +1577,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
@@ -1572,14 +1592,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
@@ -1662,7 +1682,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_alias_ignores_missing_bins.snap
@@ -3523,6 +3543,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -201,6 +201,18 @@ download_binary_and_run_installer() {
             _updater_name=""
             _updater_bin=""
             ;;
+        "axolotlsay-x86_64-pc-windows-gnu.tar.gz")
+            _arch="x86_64-pc-windows-gnu"
+            _zip_ext=".tar.gz"
+            _bins="axolotlsay.exe"
+            _bins_js_array='"axolotlsay.exe"'
+            _libs=""
+            _libs_js_array=""
+            _staticlibs=""
+            _staticlibs_js_array=""
+            _updater_name=""
+            _updater_bin=""
+            ;;
         "axolotlsay-x86_64-pc-windows-msvc.tar.gz")
             _arch="x86_64-pc-windows-msvc"
             _zip_ext=".tar.gz"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -338,6 +350,9 @@ json_binary_aliases() {
     "aarch64-apple-darwin")
         echo '{}'
         ;;
+    "aarch64-pc-windows-gnu")
+        echo '{}'
+        ;;
     "i686-unknown-linux-gnu")
         echo '{}'
         ;;
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -368,6 +383,13 @@ aliases_for_binary() {
             ;;
         esac
         ;;
+    "aarch64-pc-windows-gnu")
+        case "$_bin" in
+        *)
+            echo ""
+            ;;
+        esac
+        ;;
     "i686-unknown-linux-gnu")
         case "$_bin" in
         *)
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -421,6 +443,13 @@ select_archive_for_arch() {
                 return 0
             fi
             ;;
+        "aarch64-pc-windows-gnu")
+            _archive="axolotlsay-x86_64-pc-windows-gnu.tar.gz"
+            if [ -n "$_archive" ]; then
+                echo "$_archive"
+                return 0
+            fi
+            ;;
         "aarch64-pc-windows-msvc")
             _archive="axolotlsay-x86_64-pc-windows-msvc.tar.gz"
             if [ -n "$_archive" ]; then
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -446,6 +475,11 @@ select_archive_for_arch() {
             fi
             ;;
         "x86_64-pc-windows-gnu")
+            _archive="axolotlsay-x86_64-pc-windows-gnu.tar.gz"
+            if [ -n "$_archive" ]; then
+                echo "$_archive"
+                return 0
+            fi
             _archive="axolotlsay-x86_64-pc-windows-msvc.tar.gz"
             if [ -n "$_archive" ]; then
                 echo "$_archive"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1370,6 +1404,7 @@ class Axolotlsay < Formula
 
   BINARY_ALIASES = {
     "aarch64-apple-darwin": {},
+    "aarch64-pc-windows-gnu": {},
     "i686-unknown-linux-gnu": {},
     "x86_64-apple-darwin": {},
     "x86_64-pc-windows-gnu": {},
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1506,6 +1541,16 @@ function Install-Binary($install_args) {
 
   # Platform info injected by dist
   $platforms = @{
+    "aarch64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-gnu.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "aarch64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1516,6 +1561,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-gnu.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1542,7 +1597,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1556,10 +1620,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1571,14 +1635,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1661,7 +1725,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3340,6 +3404,13 @@ install(false);
       },
       "zipExt": ".tar.gz"
     },
+    "aarch64-pc-windows-gnu": {
+      "artifactName": "axolotlsay-x86_64-pc-windows-gnu.tar.gz",
+      "bins": {
+        "axolotlsay": "axolotlsay.exe"
+      },
+      "zipExt": ".tar.gz"
+    },
     "aarch64-pc-windows-msvc": {
       "artifactName": "axolotlsay-x86_64-pc-windows-msvc.tar.gz",
       "bins": {
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3362,7 +3433,7 @@ install(false);
       "zipExt": ".tar.gz"
     },
     "x86_64-pc-windows-gnu": {
-      "artifactName": "axolotlsay-x86_64-pc-windows-msvc.tar.gz",
+      "artifactName": "axolotlsay-x86_64-pc-windows-gnu.tar.gz",
       "bins": {
         "axolotlsay": "axolotlsay.exe"
       },
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3408,7 +3479,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
   "announcement_is_prerelease": false,
   "announcement_title": "Version 0.2.2",
   "announcement_changelog": "```text\n         +----------------------------------+\n         | now with arm64 linux binaries!!! |\n         +----------------------------------+\n        /\n≽(◕ ᴗ ◕)≼\n```",
-  "announcement_github_body": "## Release Notes\n\n```text\n         +----------------------------------+\n         | now with arm64 linux binaries!!! |\n         +----------------------------------+\n        /\n≽(◕ ᴗ ◕)≼\n```\n\n## Install axolotlsay 0.2.2\n\n### Install prebuilt binaries via shell script\n\n```sh\ncurl --proto '=https' --tlsv1.2 -LsSf https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.sh | sh\n```\n\n### Install prebuilt binaries via powershell script\n\n```sh\npowershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"\n```\n\n### Install prebuilt binaries via Homebrew\n\n```sh\nbrew install axodotdev/packages/axolotlsay\n```\n\n### Install prebuilt binaries into your npm project\n\n```sh\nnpm install @axodotdev/axolotlsay@0.2.2\n```\n\n## Download axolotlsay 0.2.2\n\n|  File  | Platform | Checksum |\n|--------|----------|----------|\n| [axolotlsay-aarch64-apple-darwin.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-aarch64-apple-darwin.tar.gz) | Apple Silicon macOS | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-aarch64-apple-darwin.tar.gz.sha256) |\n| [axolotlsay-aarch64-apple-darwin.pkg](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-aarch64-apple-darwin.pkg) | Apple Silicon macOS | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-aarch64-apple-darwin.pkg.sha256) |\n| [axolotlsay-x86_64-apple-darwin.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-apple-darwin.tar.gz) | Intel macOS | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-apple-darwin.tar.gz.sha256) |\n| [axolotlsay-x86_64-apple-darwin.pkg](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-apple-darwin.pkg) | Intel macOS | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-apple-darwin.pkg.sha256) |\n| [axolotlsay-x86_64-pc-windows-msvc.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-msvc.tar.gz) | x64 Windows | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-msvc.tar.gz.sha256) |\n| [axolotlsay-x86_64-pc-windows-msvc.msi](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-msvc.msi) | x64 Windows | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-msvc.msi.sha256) |\n| [axolotlsay-i686-unknown-linux-gnu.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-i686-unknown-linux-gnu.tar.gz) | x86 Linux | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-i686-unknown-linux-gnu.tar.gz.sha256) |\n| [axolotlsay-x86_64-unknown-linux-gnu.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-unknown-linux-gnu.tar.gz) | x64 Linux | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-unknown-linux-gnu.tar.gz.sha256) |\n\n",
+  "announcement_github_body": "## Release Notes\n\n```text\n         +----------------------------------+\n         | now with arm64 linux binaries!!! |\n         +----------------------------------+\n        /\n≽(◕ ᴗ ◕)≼\n```\n\n## Install axolotlsay 0.2.2\n\n### Install prebuilt binaries via shell script\n\n```sh\ncurl --proto '=https' --tlsv1.2 -LsSf https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.sh | sh\n```\n\n### Install prebuilt binaries via powershell script\n\n```sh\npowershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"\n```\n\n### Install prebuilt binaries via Homebrew\n\n```sh\nbrew install axodotdev/packages/axolotlsay\n```\n\n### Install prebuilt binaries into your npm project\n\n```sh\nnpm install @axodotdev/axolotlsay@0.2.2\n```\n\n## Download axolotlsay 0.2.2\n\n|  File  | Platform | Checksum |\n|--------|----------|----------|\n| [axolotlsay-aarch64-apple-darwin.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-aarch64-apple-darwin.tar.gz) | Apple Silicon macOS | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-aarch64-apple-darwin.tar.gz.sha256) |\n| [axolotlsay-aarch64-apple-darwin.pkg](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-aarch64-apple-darwin.pkg) | Apple Silicon macOS | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-aarch64-apple-darwin.pkg.sha256) |\n| [axolotlsay-x86_64-apple-darwin.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-apple-darwin.tar.gz) | Intel macOS | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-apple-darwin.tar.gz.sha256) |\n| [axolotlsay-x86_64-apple-darwin.pkg](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-apple-darwin.pkg) | Intel macOS | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-apple-darwin.pkg.sha256) |\n| [axolotlsay-x86_64-pc-windows-gnu.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-gnu.tar.gz) | x64 MinGW | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-gnu.tar.gz.sha256) |\n| [axolotlsay-x86_64-pc-windows-gnu.msi](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-gnu.msi) | x64 MinGW | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-gnu.msi.sha256) |\n| [axolotlsay-x86_64-pc-windows-msvc.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-msvc.tar.gz) | x64 Windows | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-msvc.tar.gz.sha256) |\n| [axolotlsay-x86_64-pc-windows-msvc.msi](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-msvc.msi) | x64 Windows | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-pc-windows-msvc.msi.sha256) |\n| [axolotlsay-i686-unknown-linux-gnu.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-i686-unknown-linux-gnu.tar.gz) | x86 Linux | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-i686-unknown-linux-gnu.tar.gz.sha256) |\n| [axolotlsay-x86_64-unknown-linux-gnu.tar.gz](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-unknown-linux-gnu.tar.gz) | x64 Linux | [checksum](https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-x86_64-unknown-linux-gnu.tar.gz.sha256) |\n\n",
   "releases": [
     {
       "app_name": "axolotlsay",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3440,6 +3511,8 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
         "axolotlsay-i686-unknown-linux-gnu.tar.gz.omnibor",
         "axolotlsay-x86_64-apple-darwin.tar.gz.omnibor",
         "axolotlsay-x86_64-apple-darwin.pkg.omnibor",
+        "axolotlsay-x86_64-pc-windows-gnu.tar.gz.omnibor",
+        "axolotlsay-x86_64-pc-windows-gnu.msi.omnibor",
         "axolotlsay-x86_64-pc-windows-msvc.tar.gz.omnibor",
         "axolotlsay-x86_64-pc-windows-msvc.msi.omnibor",
         "axolotlsay-x86_64-unknown-linux-gnu.tar.gz.omnibor",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3455,6 +3528,10 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
         "axolotlsay-x86_64-apple-darwin.tar.gz.sha256",
         "axolotlsay-x86_64-apple-darwin.pkg",
         "axolotlsay-x86_64-apple-darwin.pkg.sha256",
+        "axolotlsay-x86_64-pc-windows-gnu.tar.gz",
+        "axolotlsay-x86_64-pc-windows-gnu.tar.gz.sha256",
+        "axolotlsay-x86_64-pc-windows-gnu.msi",
+        "axolotlsay-x86_64-pc-windows-gnu.msi.sha256",
         "axolotlsay-x86_64-pc-windows-msvc.tar.gz",
         "axolotlsay-x86_64-pc-windows-msvc.tar.gz.sha256",
         "axolotlsay-x86_64-pc-windows-msvc.msi",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3599,7 +3676,9 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "name": "axolotlsay-installer.ps1",
       "kind": "installer",
       "target_triples": [
+        "aarch64-pc-windows-gnu",
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3614,6 +3693,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-apple-darwin",
+        "aarch64-pc-windows-gnu",
         "i686-unknown-linux-gnu",
         "x86_64-apple-darwin",
         "x86_64-pc-windows-gnu",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3631,6 +3711,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-apple-darwin",
+        "aarch64-pc-windows-gnu",
         "aarch64-pc-windows-msvc",
         "i686-unknown-linux-gnu",
         "x86_64-apple-darwin",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3777,6 +3858,81 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
         "x86_64-apple-darwin"
       ]
     },
+    "axolotlsay-x86_64-pc-windows-gnu.msi": {
+      "name": "axolotlsay-x86_64-pc-windows-gnu.msi",
+      "kind": "installer",
+      "target_triples": [
+        "x86_64-pc-windows-gnu"
+      ],
+      "assets": [
+        {
+          "id": "axolotlsay-x86_64-pc-windows-gnu-exe-axolotlsay",
+          "name": "axolotlsay",
+          "path": "axolotlsay.exe",
+          "kind": "executable"
+        }
+      ],
+      "description": "install via msi",
+      "checksum": "axolotlsay-x86_64-pc-windows-gnu.msi.sha256"
+    },
+    "axolotlsay-x86_64-pc-windows-gnu.msi.omnibor": {
+      "name": "axolotlsay-x86_64-pc-windows-gnu.msi.omnibor",
+      "kind": "omnibor-artifact-id"
+    },
+    "axolotlsay-x86_64-pc-windows-gnu.msi.sha256": {
+      "name": "axolotlsay-x86_64-pc-windows-gnu.msi.sha256",
+      "kind": "checksum",
+      "target_triples": [
+        "x86_64-pc-windows-gnu"
+      ]
+    },
+    "axolotlsay-x86_64-pc-windows-gnu.tar.gz": {
+      "name": "axolotlsay-x86_64-pc-windows-gnu.tar.gz",
+      "kind": "executable-zip",
+      "target_triples": [
+        "x86_64-pc-windows-gnu"
+      ],
+      "assets": [
+        {
+          "name": "CHANGELOG.md",
+          "path": "CHANGELOG.md",
+          "kind": "changelog"
+        },
+        {
+          "name": "LICENSE-APACHE",
+          "path": "LICENSE-APACHE",
+          "kind": "license"
+        },
+        {
+          "name": "LICENSE-MIT",
+          "path": "LICENSE-MIT",
+          "kind": "license"
+        },
+        {
+          "name": "README.md",
+          "path": "README.md",
+          "kind": "readme"
+        },
+        {
+          "id": "axolotlsay-x86_64-pc-windows-gnu-exe-axolotlsay",
+          "name": "axolotlsay",
+          "path": "axolotlsay.exe",
+          "kind": "executable"
+        }
+      ],
+      "checksum": "axolotlsay-x86_64-pc-windows-gnu.tar.gz.sha256"
+    },
+    "axolotlsay-x86_64-pc-windows-gnu.tar.gz.omnibor": {
+      "name": "axolotlsay-x86_64-pc-windows-gnu.tar.gz.omnibor",
+      "kind": "omnibor-artifact-id"
+    },
+    "axolotlsay-x86_64-pc-windows-gnu.tar.gz.sha256": {
+      "name": "axolotlsay-x86_64-pc-windows-gnu.tar.gz.sha256",
+      "kind": "checksum",
+      "target_triples": [
+        "x86_64-pc-windows-gnu"
+      ]
+    },
     "axolotlsay-x86_64-pc-windows-msvc.msi": {
       "name": "axolotlsay-x86_64-pc-windows-msvc.msi",
       "kind": "installer",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -3908,6 +4064,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-apple-darwin",
+        "aarch64-pc-windows-gnu",
         "i686-unknown-linux-gnu",
         "x86_64-apple-darwin",
         "x86_64-pc-windows-gnu",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -4015,6 +4172,27 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
             },
             "cache_provider": "github"
           },
+          {
+            "runner": "windows-2019",
+            "host": "x86_64-pc-windows-msvc",
+            "install_dist": {
+              "shell": "pwsh",
+              "run": "irm https://github.com/axodotdev/cargo-dist/releases/download/vSOME_VERSION/cargo-dist-installer.ps1 | iex"
+            },
+            "dist_args": "--artifacts=local --target=x86_64-pc-windows-gnu",
+            "targets": [
+              "x86_64-pc-windows-gnu"
+            ],
+            "install_cargo_auditable": {
+              "shell": "pwsh",
+              "run": "powershell -c \"irm https://github.com/rust-secure-code/cargo-auditable/releases/latest/download/cargo-auditable-installer.ps1 | iex\""
+            },
+            "install_omnibor": {
+              "shell": "pwsh",
+              "run": "powershell -c \"irm https://github.com/omnibor/omnibor-rs/releases/download/omnibor-cli-v0.7.0/omnibor-cli-installer.ps1 | iex\""
+            },
+            "cache_provider": "github"
+          },
           {
             "runner": "windows-2019",
             "host": "x86_64-pc-windows-msvc",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
@@ -1486,6 +1486,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
@@ -1512,7 +1522,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
@@ -1526,10 +1545,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
@@ -1541,14 +1560,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
@@ -1631,7 +1650,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic_lies.snap
@@ -3505,6 +3524,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_build_setup_steps.snap
@@ -3489,6 +3508,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -1474,6 +1474,16 @@ function Install-Binary($install_args) {
 
   # Platform info injected by dist
   $platforms = @{
+    "aarch64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-aarch64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "aarch64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-aarch64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -1484,6 +1494,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -1510,7 +1530,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -1524,10 +1553,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -1539,14 +1568,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -1629,7 +1658,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_cross1.snap
@@ -2107,7 +2136,9 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "name": "axolotlsay-installer.ps1",
       "kind": "installer",
       "target_triples": [
+        "aarch64-pc-windows-gnu",
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_disable_source_tarball.snap
@@ -3486,6 +3505,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  axolotlsay-n
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
@@ -1418,6 +1418,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
@@ -1444,7 +1454,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
@@ -1458,10 +1477,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
@@ -1473,14 +1492,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
@@ -1563,7 +1582,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_dist_url_override.snap
@@ -1952,6 +1971,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -3459,6 +3478,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -1482,6 +1482,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-js-x86_64-pc-windows-msvc.zip"
+      "bins" = @("axolotlsay-js.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".zip"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-js-x86_64-pc-windows-msvc.zip"
       "bins" = @("axolotlsay-js.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -1508,7 +1518,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -1522,10 +1541,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -1537,14 +1556,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -1627,7 +1646,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -3399,6 +3418,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.zip"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".zip"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.zip"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -3425,7 +3454,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -3439,10 +3477,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -3454,14 +3492,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -3544,7 +3582,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -3971,6 +4009,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay-hybrid/releases/download/v0.10.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_generic_workspace_basic.snap
@@ -4031,6 +4070,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay-hybrid/releases/download/v0.10.2/axolotlsay-js-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_homebrew_packages.snap
@@ -3489,6 +3508,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -3459,6 +3478,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
@@ -1516,6 +1516,17 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{"axolotlsay.exe":["axolotlsay-link1.exe","axolotlsay-link2.exe"]}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+        "axolotlsay.exe" = "axolotlsay-link1.exe", "axolotlsay-link2.exe"
+      }
+      "aliases_json" = '{"axolotlsay.exe":["axolotlsay-link1.exe","axolotlsay-link2.exe"]}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
@@ -1543,7 +1554,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
@@ -1557,10 +1577,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
@@ -1572,14 +1592,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
@@ -1662,7 +1682,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_several_aliases.snap
@@ -3527,6 +3547,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -1418,6 +1418,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -1444,7 +1454,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -1458,10 +1477,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -1473,14 +1492,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -1563,7 +1582,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -1982,6 +2001,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -1418,6 +1418,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -1444,7 +1454,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -1458,10 +1477,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -1473,14 +1492,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -1563,7 +1582,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -1982,6 +2001,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
@@ -1487,6 +1487,20 @@ function Install-Binary($install_args) {
         "bin" = "axolotlsay-x86_64-pc-windows-msvc-update"
       }
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+      "updater" = @{
+        "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc-update"
+        "bin" = "axolotlsay-x86_64-pc-windows-msvc-update"
+      }
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
@@ -1517,7 +1531,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
@@ -1531,10 +1554,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
@@ -1546,14 +1569,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
@@ -1636,7 +1659,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_updaters.snap
@@ -3508,6 +3531,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -3459,6 +3478,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -3459,6 +3478,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -3459,6 +3478,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -3459,6 +3478,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -3459,6 +3478,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -1483,6 +1483,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -1509,7 +1519,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -1523,10 +1542,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -1538,14 +1557,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -1628,7 +1647,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -2018,6 +2037,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -1994,6 +2013,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -1994,6 +2013,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -1994,6 +2013,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -1994,6 +2013,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
--- a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
+++ b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
@@ -1480,6 +1480,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
--- a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
+++ b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
@@ -1506,7 +1516,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
--- a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
+++ b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
@@ -1520,10 +1539,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
--- a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
+++ b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
@@ -1535,14 +1554,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
--- a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
+++ b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
@@ -1625,7 +1644,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
--- a/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
+++ b/cargo-dist/tests/snapshots/install_path_fallback_no_env_var_set.snap
@@ -2017,6 +2036,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -1994,6 +2013,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -1994,6 +2013,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -1994,6 +2013,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -1466,6 +1466,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -1492,7 +1502,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -1506,10 +1525,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -1521,14 +1540,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -1611,7 +1630,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -1994,6 +2013,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
--- a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
+++ b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
@@ -1480,6 +1480,16 @@ function Install-Binary($install_args) {
       }
       "aliases_json" = '{}'
     }
+    "x86_64-pc-windows-gnu" = @{
+      "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
+      "bins" = @("axolotlsay.exe")
+      "libs" = @()
+      "staticlibs" = @()
+      "zip_ext" = ".tar.gz"
+      "aliases" = @{
+      }
+      "aliases_json" = '{}'
+    }
     "x86_64-pc-windows-msvc" = @{
       "artifact_name" = "axolotlsay-x86_64-pc-windows-msvc.tar.gz"
       "bins" = @("axolotlsay.exe")
diff --git a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
--- a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
+++ b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
@@ -1506,7 +1516,16 @@ $_
   }
 }
 
-function Get-TargetTriple() {
+function Get-TargetTriple($platforms) {
+  $double = Get-Arch
+  if ($platforms.Contains("$double-msvc")) {
+    return "$double-msvc"
+  } else {
+    return "$double-gnu"
+  }
+}
+
+function Get-Arch() {
   try {
     # NOTE: this might return X64 on ARM64 Windows, which is OK since emulation is available.
     # It works correctly starting in PowerShell Core 7.3 and Windows PowerShell in Win 11 22H2.
diff --git a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
--- a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
+++ b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
@@ -1520,10 +1539,10 @@ function Get-TargetTriple() {
     # Rust supported platforms: https://doc.rust-lang.org/stable/rustc/platform-support.html
     switch ($p.GetValue($null).ToString())
     {
-      "X86" { return "i686-pc-windows-msvc" }
-      "X64" { return "x86_64-pc-windows-msvc" }
-      "Arm" { return "thumbv7a-pc-windows-msvc" }
-      "Arm64" { return "aarch64-pc-windows-msvc" }
+      "X86" { return "i686-pc-windows" }
+      "X64" { return "x86_64-pc-windows" }
+      "Arm" { return "thumbv7a-pc-windows" }
+      "Arm64" { return "aarch64-pc-windows" }
     }
   } catch {
     # The above was added in .NET 4.7.1, so Windows PowerShell in versions of Windows
diff --git a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
--- a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
+++ b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
@@ -1535,14 +1554,14 @@ function Get-TargetTriple() {
   # This is available in .NET 4.0. We already checked for PS 5, which requires .NET 4.5.
   Write-Verbose("Get-TargetTriple: falling back to Is64BitOperatingSystem.")
   if ([System.Environment]::Is64BitOperatingSystem) {
-    return "x86_64-pc-windows-msvc"
+    return "x86_64-pc-windows"
   } else {
-    return "i686-pc-windows-msvc"
+    return "i686-pc-windows"
   }
 }
 
 function Download($download_url, $platforms) {
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
--- a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
+++ b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
@@ -1625,7 +1644,7 @@ function Download($download_url, $platforms) {
 
 function Invoke-Installer($artifacts, $platforms) {
   # Replaces the placeholder binary entry with the actual list of binaries
-  $arch = Get-TargetTriple
+  $arch = Get-TargetTriple $platforms
 
   if (-not $platforms.ContainsKey($arch)) {
     $platforms_json = ConvertTo-Json $platforms
diff --git a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
--- a/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
+++ b/cargo-dist/tests/snapshots/install_path_no_fallback_taken.snap
@@ -2017,6 +2036,7 @@ CENSORED (see https://github.com/axodotdev/cargo-dist/issues/1477)  source.tar.g
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://github.com/axodotdev/axolotlsay/releases/download/v0.2.2/axolotlsay-installer.ps1 | iex\"",
diff --git a/cargo-dist/tests/snapshots/manifest.snap b/cargo-dist/tests/snapshots/manifest.snap
--- a/cargo-dist/tests/snapshots/manifest.snap
+++ b/cargo-dist/tests/snapshots/manifest.snap
@@ -203,6 +203,7 @@ stdout:
       "kind": "installer",
       "target_triples": [
         "aarch64-pc-windows-msvc",
+        "x86_64-pc-windows-gnu",
         "x86_64-pc-windows-msvc"
       ],
       "install_hint": "powershell -ExecutionPolicy ByPass -c \"irm https://fake.axo.dev/faker/cargo-dist/fake-id-do-not-upload/cargo-dist-installer.ps1 | iex\"",

EOF_114329324912
cd "cargo-dist"
cargo test --no-fail-fast --all-features --test "integration-tests"
cd ../
git reset --hard 4c2cd562aac54b7a428a789a9f1c66388f024730
git clean -fd
