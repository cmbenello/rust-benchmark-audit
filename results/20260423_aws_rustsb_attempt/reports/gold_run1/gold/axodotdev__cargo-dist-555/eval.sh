#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/cargo-dist/tests/gallery/dist.rs b/cargo-dist/tests/gallery/dist.rs
--- a/cargo-dist/tests/gallery/dist.rs
+++ b/cargo-dist/tests/gallery/dist.rs
@@ -370,13 +370,18 @@ impl DistResult {
             let app_home = tempdir.join(format!(".{app_name}"));
             let _output = script.output_checked(|cmd| {
                 cmd.env("HOME", &tempdir)
+                    .env("ZDOTDIR", &tempdir)
                     .env("MY_ENV_VAR", &app_home)
                     .env_remove("CARGO_HOME")
             })?;
             // we could theoretically look at the above output and parse out the `source` line...
 
             // Check that the script wrote files where we expected
-            let rcfile = tempdir.join(".profile");
+            let rcfiles = &[
+                tempdir.join(".profile"),
+                tempdir.join(".bash_profile"),
+                tempdir.join(".zshrc"),
+            ];
             let expected_bin_dir = Utf8PathBuf::from(expected_bin_dir);
             let bin_dir = tempdir.join(&expected_bin_dir);
             let env_dir = if expected_bin_dir
diff --git a/cargo-dist/tests/gallery/dist.rs b/cargo-dist/tests/gallery/dist.rs
--- a/cargo-dist/tests/gallery/dist.rs
+++ b/cargo-dist/tests/gallery/dist.rs
@@ -390,7 +395,9 @@ impl DistResult {
             let env_script = env_dir.join("env");
 
             assert!(bin_dir.exists(), "bin dir wasn't created");
-            assert!(rcfile.exists(), ".profile wasn't created");
+            for rcfile in rcfiles {
+                assert!(rcfile.exists(), "{} wasn't created", rcfile);
+            }
             assert!(env_script.exists(), "env script wasn't created");
 
             // Check that all the binaries work
diff --git a/cargo-dist/tests/gallery/dist.rs b/cargo-dist/tests/gallery/dist.rs
--- a/cargo-dist/tests/gallery/dist.rs
+++ b/cargo-dist/tests/gallery/dist.rs
@@ -411,9 +418,10 @@ impl DistResult {
                 let test_script_text = format!(
                     r#"#!/bin/sh
 
-                . {rcfile}
+                . {}
                 which {bin_name}
-                "#
+                "#,
+                    rcfiles.first().expect("rcfiles was empty?!")
                 );
                 LocalAsset::write_new(&test_script_text, &test_script_path)?;
                 std::fs::set_permissions(&test_script_path, std::fs::Permissions::from_mode(0o755))
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/mistydemeo/akaikatana-repack/releases/download/v0.2.0
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     akaikatana-repack-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/akaikatana_basic.snap b/cargo-dist/tests/snapshots/akaikatana_basic.snap
--- a/cargo-dist/tests/snapshots/akaikatana_basic.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_basic.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/akaikatana_musl.snap b/cargo-dist/tests/snapshots/akaikatana_musl.snap
--- a/cargo-dist/tests/snapshots/akaikatana_musl.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_musl.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/mistydemeo/akaikatana-repack/releases/download/v0.2.0
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     akaikatana-repack-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/akaikatana_musl.snap b/cargo-dist/tests/snapshots/akaikatana_musl.snap
--- a/cargo-dist/tests/snapshots/akaikatana_musl.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_musl.snap
@@ -300,10 +300,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/akaikatana_musl.snap b/cargo-dist/tests/snapshots/akaikatana_musl.snap
--- a/cargo-dist/tests/snapshots/akaikatana_musl.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_musl.snap
@@ -315,8 +339,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/akaikatana_musl.snap b/cargo-dist/tests/snapshots/akaikatana_musl.snap
--- a/cargo-dist/tests/snapshots/akaikatana_musl.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_musl.snap
@@ -342,14 +391,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
--- a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/mistydemeo/akaikatana-repack/releases/download/v0.2.0
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     akaikatana-repack-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
--- a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
--- a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
--- a/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
+++ b/cargo-dist/tests/snapshots/akaikatana_repo_with_dot_git.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://fake.axo.dev/faker/axolotlsay/fake-id-do-not-upload
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://fake.axo.dev/faker/axolotlsay/fake-id-do-not-upload
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_abyss_only.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_basic.snap b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_basic.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_basic.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_edit_existing.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl.snap b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl.snap b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
@@ -300,10 +300,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl.snap b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
@@ -315,8 +339,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl.snap b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl.snap
@@ -342,14 +391,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
@@ -300,10 +300,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
@@ -315,8 +339,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_musl_no_gnu.snap
@@ -342,14 +391,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_no_homebrew_publish.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_ssldotcom_windows_sign_prod.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
--- a/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
+++ b/cargo-dist/tests/snapshots/axolotlsay_user_publish_job.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$CARGO_HOME/bin (\$HOME/.cargo/bin)
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -290,10 +290,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -305,8 +329,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_cargo_home.snap b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
--- a/cargo-dist/tests/snapshots/install_path_cargo_home.snap
+++ b/cargo-dist/tests/snapshots/install_path_cargo_home.snap
@@ -332,14 +381,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$MY_ENV_VAR/
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -273,10 +273,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -288,8 +312,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_no_subdir.snap
@@ -315,14 +364,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$MY_ENV_VAR/bin
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -273,10 +273,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -288,8 +312,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir.snap b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir.snap
@@ -315,14 +364,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$MY_ENV_VAR/My Axolotlsay Documents
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -273,10 +273,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -288,8 +312,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space.snap
@@ -315,14 +364,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$MY_ENV_VAR/My Axolotlsay Documents/bin
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -273,10 +273,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -288,8 +312,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_env_subdir_space_deeper.snap
@@ -315,14 +364,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$HOME/.axolotlsay/bins
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -273,10 +273,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -288,8 +312,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_deeper.snap
@@ -315,14 +364,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$HOME/.axolotlsay
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -273,10 +273,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -288,8 +312,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_min.snap
@@ -315,14 +364,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$HOME/My Axolotlsay Documents
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -273,10 +273,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -288,8 +312,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space.snap
@@ -315,14 +364,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -45,7 +45,7 @@ This script detects what platform you're on and fetches an appropriate archive f
 https://github.com/axodotdev/axolotlsay/releases/download/v0.2.1
 then unpacks the binaries and installs them to \$HOME/My Axolotlsay Documents/bin
 
-It will then add that dir to PATH by adding the appropriate line to \$HOME/.profile
+It will then add that dir to PATH by adding the appropriate line to your shell profiles.
 
 USAGE:
     axolotlsay-installer.sh [OPTIONS]
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -273,10 +273,34 @@ install() {
     say "everything's installed!"
 
     if [ "0" = "$NO_MODIFY_PATH" ]; then
-        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".profile"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".bash_profile .bash_login .bashrc"
+        add_install_dir_to_path "$_install_dir_expr" "$_env_script_path" "$_env_script_path_expr" ".zshrc .zshenv"
     fi
 }
 
+print_home_for_script() {
+    local script="$1"
+
+    local _home
+    case "$script" in
+        # zsh has a special ZDOTDIR directory, which if set
+        # should be considered instead of $HOME
+        .zsh*)
+            if [ -n "$ZDOTDIR" ]; then
+                _home="$ZDOTDIR"
+            else
+                _home="$HOME"
+            fi
+            ;;
+        *)
+            _home="$HOME"
+            ;;
+    esac
+
+    echo "$_home"
+}
+
 add_install_dir_to_path() {
     # Edit rcfiles ($HOME/.profile) to add install_dir to $PATH
     #
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -288,8 +312,33 @@ add_install_dir_to_path() {
     local _install_dir_expr="$1"
     local _env_script_path="$2"
     local _env_script_path_expr="$3"
+    local _rcfiles="$4"
+
     if [ -n "${HOME:-}" ]; then
-        local _rcfile="$HOME/.profile"
+        local _target
+        local _home
+
+        # Find the first file in the array that exists and choose
+        # that as our target to write to
+        for _rcfile_relative in $_rcfiles; do
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            local _rcfile="$_home/$_rcfile_relative"
+
+            if [ -f "$_rcfile" ]; then
+                _target="$_rcfile"
+                break
+            fi
+        done
+
+        # If we didn't find anything, pick the first entry in the
+        # list as the default to create and write to
+        if [ -z "${_target:-}" ]; then
+            local _rcfile_relative
+            _rcfile_relative="$(echo "$_rcfiles" | awk '{ print $1 }')"
+            _home="$(print_home_for_script "$_rcfile_relative")"
+            _target="$_home/$_rcfile_relative"
+        fi
+
         # `source x` is an alias for `. x`, and the latter is more portable/actually-posix.
         # This apparently comes up a lot on freebsd. It's easy enough to always add
         # the more robust line to rcfiles, but when telling the user to apply the change
diff --git a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
--- a/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
+++ b/cargo-dist/tests/snapshots/install_path_home_subdir_space_deeper.snap
@@ -315,14 +364,14 @@ add_install_dir_to_path() {
         # (on error we want to create the file, which >> conveniently does)
         #
         # We search for both kinds of line here just to do the right thing in more cases.
-        if ! grep -F "$_robust_line" "$_rcfile" > /dev/null 2>/dev/null && \
-           ! grep -F "$_pretty_line" "$_rcfile" > /dev/null 2>/dev/null
+        if ! grep -F "$_robust_line" "$_target" > /dev/null 2>/dev/null && \
+           ! grep -F "$_pretty_line" "$_target" > /dev/null 2>/dev/null
         then
             # If the script now exists, add the line to source it to the rcfile
             # (This will also create the rcfile if it doesn't exist)
             if [ -f "$_env_script_path" ]; then
-                say_verbose "adding $_robust_line to $_rcfile"
-                ensure echo "$_robust_line" >> "$_rcfile"
+                say_verbose "adding $_robust_line to $_target"
+                ensure echo "$_robust_line" >> "$_target"
                 say ""
                 say "To add $_install_dir_expr to your PATH, either restart your shell or run:"
                 say ""

EOF_114329324912
cd "cargo-dist"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard ec189bb700f4356d39ed4da5427490772e32d745
git clean -fd
