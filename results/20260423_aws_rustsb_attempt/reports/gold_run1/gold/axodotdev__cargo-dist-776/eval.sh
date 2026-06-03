#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/cargo-dist/tests/gallery/dist.rs b/cargo-dist/tests/gallery/dist.rs
--- a/cargo-dist/tests/gallery/dist.rs
+++ b/cargo-dist/tests/gallery/dist.rs
@@ -382,6 +382,7 @@ impl DistResult {
                 tempdir.join(".bash_profile"),
                 tempdir.join(".zshrc"),
             ];
+            let receipt_file = tempdir.join(format!(".config/{app_name}/{app_name}-receipt.json"));
             let expected_bin_dir = Utf8PathBuf::from(expected_bin_dir);
             let bin_dir = tempdir.join(&expected_bin_dir);
             let env_dir = if expected_bin_dir
diff --git a/cargo-dist/tests/gallery/dist.rs b/cargo-dist/tests/gallery/dist.rs
--- a/cargo-dist/tests/gallery/dist.rs
+++ b/cargo-dist/tests/gallery/dist.rs
@@ -439,6 +440,52 @@ impl DistResult {
                     "bin path wasn't right"
                 );
             }
+
+            // Check that the install receipt works
+            {
+                use serde::Deserialize;
+
+                #[derive(Deserialize)]
+                #[allow(dead_code)]
+                struct InstallReceipt {
+                    binaries: Vec<String>,
+                    install_prefix: String,
+                    provider: InstallReceiptProvider,
+                    source: InstallReceiptSource,
+                    version: String,
+                }
+                #[derive(Deserialize)]
+                #[allow(dead_code)]
+                struct InstallReceiptProvider {
+                    source: String,
+                    version: String,
+                }
+                #[derive(Deserialize)]
+                #[allow(dead_code)]
+                struct InstallReceiptSource {
+                    app_name: String,
+                    name: String,
+                    owner: String,
+                    release_type: String,
+                }
+
+                assert!(receipt_file.exists());
+                let receipt_src =
+                    SourceFile::load_local(receipt_file).expect("couldn't load receipt file");
+                let receipt: InstallReceipt = receipt_src.deserialize_json().unwrap();
+                assert_eq!(receipt.source.app_name, app_name);
+                assert_eq!(
+                    receipt.binaries,
+                    ctx.repo
+                        .bins
+                        .iter()
+                        .map(|s| s.to_owned())
+                        .collect::<Vec<_>>()
+                );
+                let receipt_bin_dir = receipt.install_prefix.trim_end_matches('/').to_owned();
+                let expected_bin_dir = bin_dir.to_string().trim_end_matches('/').to_owned();
+                assert_eq!(receipt_bin_dir, expected_bin_dir);
+            }
         }
         Ok(())
     }
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -1028,7 +1028,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/akaikatana_musl.snap b/cargo-dist/tests/snapshots/akaikatana_musl.snap
--- a/cargo-dist/tests/snapshots/akaikatana_musl.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_musl.snap
@@ -173,7 +173,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/akaikatana_musl.snap b/cargo-dist/tests/snapshots/akaikatana_musl.snap
--- a/cargo-dist/tests/snapshots/akaikatana_musl.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_musl.snap
@@ -309,7 +309,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
--- a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
--- a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
--- a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
@@ -1028,7 +1028,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl.snap b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
@@ -173,7 +173,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl.snap b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
@@ -309,7 +309,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
@@ -173,7 +173,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
@@ -309,7 +309,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -983,7 +983,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -983,7 +983,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_global_build_job.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_host_job.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_local_build_job.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_plan_job.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -297,7 +297,7 @@ install() {
         err "could not find your CARGO_HOME or HOME dir to install binaries to"
     fi
     # Replace the temporary cargo home with the calculated one
-    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_home,")
+    RECEIPT=$(echo "$RECEIPT" | sed "s,AXO_CARGO_HOME,$_install_dir,")
 
     say "installing to $_install_dir"
     ensure mkdir -p "$_install_dir"
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -1029,7 +1029,7 @@ function Invoke-Installer($bin_paths) {
   $dest_dir = Join-Path $root "bin"
 
   # The replace call here ensures proper escaping is inlined into the receipt
-  $receipt = $receipt.Replace('AXO_CARGO_HOME', $root.replace("\", "\\"))
+  $receipt = $receipt.Replace('AXO_CARGO_HOME', $dest_dir.replace("\", "\\"))
 
   $dest_dir = New-Item -Force -ItemType Directory -Path $dest_dir
   Write-Information "Installing to $dest_dir"
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -161,7 +161,7 @@ download_binary_and_run_installer() {
     esac
 
     # Replace the placeholder binaries with the calculated array from above
-    RECEIPT="$(echo "$RECEIPT" | sed s,'"CARGO_DIST_BINS"',"$_bins_js_array",)"
+    RECEIPT="$(echo "$RECEIPT" | sed s/'"CARGO_DIST_BINS"'/"$_bins_js_array"/)"
 
     # download the archive
     local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"

EOF_114329324912
cd "cargo-dist"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 4bf34bcf55a95c67b0b56708e33826a550bb2f1d
git clean -fd
