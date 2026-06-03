#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cairo-lang-semantic/src/expr/compute.rs b/crates/cairo-lang-semantic/src/expr/compute.rs
--- a/crates/cairo-lang-semantic/src/expr/compute.rs
+++ b/crates/cairo-lang-semantic/src/expr/compute.rs
@@ -2279,30 +2279,13 @@ fn member_access_expr(
 
     // Find MemberId.
     let member_name = expr_as_identifier(ctx, &rhs_syntax, syntax_db)?;
-    let ty = ctx.reduce_ty(lexpr.ty());
-    let (base_snapshots, mut long_ty) = peel_snapshots(ctx.db, ty);
-    if let TypeLongId::ImplType(impl_type_id) = long_ty {
-        let inference = &mut ctx.resolver.inference();
-        let Ok(ty) = inference.reduce_impl_ty(impl_type_id) else {
-            return Err(ctx
-                .diagnostics
-                .report(&rhs_syntax, InternalInferenceError(InferenceError::TypeNotInferred(ty))));
-        };
-        long_ty = ty.lookup_intern(ctx.db);
-    }
-    if matches!(long_ty, TypeLongId::Var(_)) {
-        // Save some work. ignore the result. The error, if any, will be reported later.
-        ctx.resolver.inference().solve().ok();
-        long_ty = ctx.resolver.inference().rewrite(long_ty).no_err();
-    }
-    let (additional_snapshots, long_ty) = peel_snapshots_ex(ctx.db, long_ty);
-    let n_snapshots = base_snapshots + additional_snapshots;
+    let (n_snapshots, long_ty) = finalized_snapshot_peeled_ty(ctx, lexpr.ty(), &rhs_syntax)?;
 
-    match long_ty {
+    match &long_ty {
         TypeLongId::Concrete(concrete) => match concrete {
             ConcreteTypeId::Struct(concrete_struct_id) => {
                 // TODO(lior): Add a diagnostic test when accessing a member of a missing type.
-                let members = ctx.db.concrete_struct_members(concrete_struct_id)?;
+                let members = ctx.db.concrete_struct_members(*concrete_struct_id)?;
                 let Some(member) = members.get(&member_name) else {
                     return Err(ctx.diagnostics.report(
                         &rhs_syntax,
diff --git a/crates/cairo-lang-semantic/src/expr/test_data/coupon b/crates/cairo-lang-semantic/src/expr/test_data/coupon
--- a/crates/cairo-lang-semantic/src/expr/test_data/coupon
+++ b/crates/cairo-lang-semantic/src/expr/test_data/coupon
@@ -110,7 +110,7 @@ error: Type "test::bar::<core::integer::u8, core::integer::u16>::Coupon" has no
     x.a;
       ^
 
-error: Type "@test::bar::<core::integer::u8, core::integer::u16>::Coupon" has no members.
+error: Type "test::bar::<core::integer::u8, core::integer::u16>::Coupon" has no members.
  --> lib.cairo:20:12
     x_snap.a;
            ^
diff --git /dev/null b/tests/bug_samples/issue5680.cairo
new file mode 100644
--- /dev/null
+++ b/tests/bug_samples/issue5680.cairo
@@ -0,0 +1,19 @@
+#[starknet::contract]
+mod c1 {
+    #[starknet::interface]
+    trait IMy<T> {
+        fn a(self: @T);
+    }
+
+    #[storage]
+    struct Storage {
+        v1: LegacyMap::<felt252, (u32, u32)>,
+    }
+
+    #[abi(embed_v0)]
+    impl My of IMy<ContractState> {
+        fn a(self: @ContractState) {
+            let (_one, _two) = self.v1.read(0);
+        }
+    }
+}
diff --git a/tests/bug_samples/lib.cairo b/tests/bug_samples/lib.cairo
--- a/tests/bug_samples/lib.cairo
+++ b/tests/bug_samples/lib.cairo
@@ -44,6 +44,7 @@ mod issue5043;
 mod issue5411;
 mod issue5438;
 mod issue5629;
+mod issue5680;
 mod loop_break_in_match;
 mod loop_only_change;
 mod partial_param_local;

EOF_114329324912
cd "crates/cairo-lang-semantic"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard 576ddd1b38abe25af1e204cb77ea039b7c46d05e
git clean -fd
