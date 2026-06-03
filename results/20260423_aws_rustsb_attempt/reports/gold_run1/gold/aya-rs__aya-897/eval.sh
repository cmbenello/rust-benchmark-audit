#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/aya-bpf-macros/src/cgroup_skb.rs b/aya-bpf-macros/src/cgroup_skb.rs
--- a/aya-bpf-macros/src/cgroup_skb.rs
+++ b/aya-bpf-macros/src/cgroup_skb.rs
@@ -66,7 +66,7 @@ mod tests {
         let expanded = prog.expand().unwrap();
         let expected = quote! {
             #[no_mangle]
-            #[link_section = "cgroup_skb"]
+            #[link_section = "cgroup/skb"]
             fn foo(ctx: *mut ::aya_bpf::bindings::__sk_buff) -> i32 {
                 return foo(::aya_bpf::programs::SkBuffContext::new(ctx));
 

EOF_114329324912
cd "aya-bpf-macros"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard b6a84b658ae00f23d0f1721c30d11f2e57f99eab
git clean -fd
