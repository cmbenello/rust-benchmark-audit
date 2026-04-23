#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cairo-lang-lowering/src/optimizations/match_optimizer.rs b/crates/cairo-lang-lowering/src/optimizations/match_optimizer.rs
--- a/crates/cairo-lang-lowering/src/optimizations/match_optimizer.rs
+++ b/crates/cairo-lang-lowering/src/optimizations/match_optimizer.rs
@@ -2,13 +2,14 @@
 #[path = "match_optimizer_test.rs"]
 mod test;
 
+use cairo_lang_utils::unordered_hash_set::UnorderedHashSet;
 use itertools::{zip_eq, Itertools};
 
 use crate::borrow_check::analysis::{Analyzer, BackAnalysis, StatementLocation};
 use crate::borrow_check::demand::DemandReporter;
 use crate::borrow_check::LoweredDemand;
 use crate::{
-    BlockId, FlatBlockEnd, FlatLowered, MatchArm, MatchEnumInfo, MatchInfo, Statement,
+    BlockId, FlatBlock, FlatBlockEnd, FlatLowered, MatchArm, MatchEnumInfo, MatchInfo, Statement,
     StatementEnumConstruct, VarRemapping, VariableId,
 };
 
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/arm_pattern_destructure b/crates/cairo-lang-lowering/src/optimizations/test_data/arm_pattern_destructure
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/arm_pattern_destructure
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/arm_pattern_destructure
@@ -236,14 +236,14 @@ blk9:
 Statements:
 End:
   Match(match_enum(v37) {
-    MyEnum::a(v39) => blk10,
-    MyEnum::b(v44) => blk11,
-    MyEnum::c(v47) => blk12,
-    MyEnum::d(v48) => blk13,
-    MyEnum::e(v51) => blk14,
-    MyEnum::f(v52) => blk15,
-    MyEnum::g(v54) => blk16,
-    MyEnum::h(v55) => blk17,
+    MyEnum::a(v39) => blk19,
+    MyEnum::b(v44) => blk20,
+    MyEnum::c(v47) => blk21,
+    MyEnum::d(v48) => blk22,
+    MyEnum::e(v51) => blk23,
+    MyEnum::f(v52) => blk24,
+    MyEnum::g(v54) => blk25,
+    MyEnum::h(v55) => blk26,
   })
 
 blk10:
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/arm_pattern_destructure b/crates/cairo-lang-lowering/src/optimizations/test_data/arm_pattern_destructure
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/arm_pattern_destructure
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/arm_pattern_destructure
@@ -291,3 +291,43 @@ Statements:
   (v56: ()) <- struct_construct()
 End:
   Return(v56)
+
+blk19:
+Statements:
+End:
+  Goto(blk10, {})
+
+blk20:
+Statements:
+End:
+  Goto(blk11, {})
+
+blk21:
+Statements:
+End:
+  Goto(blk12, {})
+
+blk22:
+Statements:
+End:
+  Goto(blk13, {})
+
+blk23:
+Statements:
+End:
+  Goto(blk14, {})
+
+blk24:
+Statements:
+End:
+  Goto(blk15, {})
+
+blk25:
+Statements:
+End:
+  Goto(blk16, {})
+
+blk26:
+Statements:
+End:
+  Goto(blk17, {})
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/option b/crates/cairo-lang-lowering/src/optimizations/test_data/option
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/option
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/option
@@ -97,8 +97,8 @@ blk3:
 Statements:
 End:
   Match(match_enum(v4) {
-    Option::Some(v5) => blk4,
-    Option::None(v8) => blk5,
+    Option::Some(v5) => blk7,
+    Option::None(v8) => blk8,
   })
 
 blk4:
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/option b/crates/cairo-lang-lowering/src/optimizations/test_data/option
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/option
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/option
@@ -120,6 +120,16 @@ Statements:
 End:
   Return(v11)
 
+blk7:
+Statements:
+End:
+  Goto(blk4, {})
+
+blk8:
+Statements:
+End:
+  Goto(blk5, {})
+
 //! > ==========================================================================
 
 //! > Test skipping of match optimization.
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/option b/crates/cairo-lang-lowering/src/optimizations/test_data/option
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/option
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/option
@@ -462,8 +472,8 @@ blk3:
 Statements:
 End:
   Match(match_enum(v4) {
-    Option::Some(v5) => blk4,
-    Option::None(v9) => blk6,
+    Option::Some(v5) => blk15,
+    Option::None(v9) => blk16,
   })
 
 blk4:
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/option b/crates/cairo-lang-lowering/src/optimizations/test_data/option
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/option
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/option
@@ -488,8 +498,8 @@ blk7:
 Statements:
 End:
   Match(match_enum(v12) {
-    Option::Some(v17) => blk8,
-    Option::None(v18) => blk12,
+    Option::Some(v17) => blk17,
+    Option::None(v18) => blk18,
   })
 
 blk8:
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/option b/crates/cairo-lang-lowering/src/optimizations/test_data/option
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/option
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/option
@@ -533,6 +543,26 @@ Statements:
 End:
   Return(v29)
 
+blk15:
+Statements:
+End:
+  Goto(blk4, {})
+
+blk16:
+Statements:
+End:
+  Goto(blk6, {})
+
+blk17:
+Statements:
+End:
+  Goto(blk8, {})
+
+blk18:
+Statements:
+End:
+  Goto(blk12, {})
+
 //! > ==========================================================================
 
 //! > withdraw_gas
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/option b/crates/cairo-lang-lowering/src/optimizations/test_data/option
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/option
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/option
@@ -650,8 +680,8 @@ blk3:
 Statements:
 End:
   Match(match_enum(v4) {
-    Option::Some(v8) => blk4,
-    Option::None(v9) => blk7,
+    Option::Some(v8) => blk10,
+    Option::None(v9) => blk11,
   })
 
 blk4:
diff --git a/crates/cairo-lang-lowering/src/optimizations/test_data/option b/crates/cairo-lang-lowering/src/optimizations/test_data/option
--- a/crates/cairo-lang-lowering/src/optimizations/test_data/option
+++ b/crates/cairo-lang-lowering/src/optimizations/test_data/option
@@ -690,3 +720,13 @@ Statements:
   (v20: core::PanicResult::<((),)>) <- PanicResult::Err(v16)
 End:
   Return(v20)
+
+blk10:
+Statements:
+End:
+  Goto(blk4, {})
+
+blk11:
+Statements:
+End:
+  Goto(blk7, {})
diff --git a/crates/cairo-lang-lowering/src/test.rs b/crates/cairo-lang-lowering/src/test.rs
--- a/crates/cairo-lang-lowering/src/test.rs
+++ b/crates/cairo-lang-lowering/src/test.rs
@@ -35,6 +35,7 @@ cairo_lang_test_utils::test_file_test!(
         extern_ :"extern",
         arm_pattern_destructure :"arm_pattern_destructure",
         if_ :"if",
+        implicits :"implicits",
         loop_ :"loop",
         match_ :"match",
         members :"members",
diff --git /dev/null b/crates/cairo-lang-lowering/src/test_data/implicits
new file mode 100644
--- /dev/null
+++ b/crates/cairo-lang-lowering/src/test_data/implicits
@@ -0,0 +1,124 @@
+//! > Test implicits with multiple jumps to arm blocks.
+
+//! > test_runner_name
+test_function_lowering
+
+//! > function
+fn foo(a: u256) -> u64 {
+    a.try_into().unwrap()
+}
+
+//! > function_name
+foo
+
+//! > module_code
+use array::ArrayTrait;
+use core::integer::u128;
+use core::integer::Felt252TryIntoU128;
+use traits::{Into, TryInto, Default, Felt252DictValue};
+use option::OptionTrait;
+
+impl U256TryIntoU64 of TryInto<u256, u64> {
+    #[inline(always)]
+    fn try_into(self: u256) -> Option<u64> {
+        if (self.high == 0) {
+            self.low.try_into()
+        } else {
+            Option::None(())
+        }
+    }
+}
+
+//! > semantic_diagnostics
+
+//! > lowering_diagnostics
+
+//! > lowering_flat
+Parameters: v33: core::RangeCheck, v0: core::integer::u256
+blk0 (root):
+Statements:
+  (v3: core::integer::u128, v4: core::integer::u128) <- struct_destructure(v0)
+  (v5: core::integer::u128) <- 0u
+End:
+  Match(match core::integer::u128_eq(v4, v5) {
+    bool::False => blk1,
+    bool::True => blk2,
+  })
+
+blk1:
+Statements:
+End:
+  Goto(blk5, {v33 -> v37})
+
+blk2:
+Statements:
+  (v44: core::RangeCheck, v9: core::option::Option::<core::integer::u64>) <- core::integer::U128TryIntoU64::try_into(v33, v3)
+End:
+  Match(match_enum(v9) {
+    Option::Some(v20) => blk3,
+    Option::None(v21) => blk4,
+  })
+
+blk3:
+Statements:
+  (v30: (core::integer::u64,)) <- struct_construct(v20)
+  (v31: core::PanicResult::<(core::integer::u64,)>) <- PanicResult::Ok(v30)
+End:
+  Return(v44, v31)
+
+blk4:
+Statements:
+End:
+  Goto(blk5, {v44 -> v37})
+
+blk5:
+Statements:
+  (v27: core::array::Array::<core::felt252>) <- core::array::array_new::<core::felt252>()
+  (v13: core::felt252) <- 29721761890975875353235833581453094220424382983267374u
+  (v28: core::array::Array::<core::felt252>) <- core::array::array_append::<core::felt252>(v27, v13)
+  (v32: core::PanicResult::<(core::integer::u64,)>) <- PanicResult::Err(v28)
+End:
+  Return(v37, v32)
+
+//! > lowering
+Main:
+Parameters:
+blk0 (root):
+Statements:
+  (v0: core::felt252) <- 5u
+  (v2: core::felt252, v1: core::bool) <- foo[expr14](v0)
+End:
+  Return(v1)
+
+
+Generated:
+Parameters: v0: core::felt252
+blk0 (root):
+Statements:
+  (v1: core::felt252) <- 1u
+  (v2: core::felt252) <- core::Felt252Add::add(v0, v1)
+  (v3: core::felt252) <- 10u
+  (v4: core::felt252) <- core::Felt252Sub::sub(v2, v3)
+End:
+  Match(match core::felt252_is_zero(v4) {
+    IsZeroResult::Zero => blk1,
+    IsZeroResult::NonZero(v7) => blk2,
+  })
+
+blk1:
+Statements:
+  (v5: ()) <- struct_construct()
+  (v6: core::bool) <- bool::True(v5)
+End:
+  Return(v2, v6)
+
+blk2:
+Statements:
+End:
+  Goto(blk3, {})
+
+blk3:
+Statements:
+  (v9: core::felt252, v8: core::bool) <- foo[expr14](v2)
+End:
+  Return(v9, v8)
diff --git /dev/null b/tests/bug_samples/issue3211.cairo
new file mode 100644
--- /dev/null
+++ b/tests/bug_samples/issue3211.cairo
@@ -0,0 +1,22 @@
+use core::integer::u128;
+use core::integer::Felt252TryIntoU128;
+use traits::{Into, TryInto};
+use option::OptionTrait;
+
+impl U256TryIntoU64 of TryInto<u256, u64> {
+    #[inline(always)]
+    fn try_into(self: u256) -> Option<u64> {
+        if (self.high == 0) {
+            self.low.try_into()
+        } else {
+            Option::None(())
+        }
+    }
+}
+
+#[test]
+fn test_u256_tryinto_u64() {
+    let a = u256 { low: 64, high: 0 };
+    let b: u64 = a.try_into().unwrap();
+    assert(b == 64, 'b conv');
+}
diff --git a/tests/bug_samples/lib.cairo b/tests/bug_samples/lib.cairo
--- a/tests/bug_samples/lib.cairo
+++ b/tests/bug_samples/lib.cairo
@@ -17,6 +17,7 @@ mod issue2961;
 mod issue2964;
 mod issue3153;
 mod issue3192;
+mod issue3211;
 mod loop_only_change;
 mod inconsistent_gas;
 mod partial_param_local;

EOF_114329324912
cd "crates/cairo-lang-lowering"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard fecc9dc533ba78bb333d8f444d86ad9dd9b61e82
git clean -fd
