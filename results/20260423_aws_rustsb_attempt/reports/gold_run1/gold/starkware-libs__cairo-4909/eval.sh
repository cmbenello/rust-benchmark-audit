#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cairo-lang-starknet/src/plugin/starknet_module/contract.rs b/crates/cairo-lang-starknet/src/plugin/starknet_module/contract.rs
--- a/crates/cairo-lang-starknet/src/plugin/starknet_module/contract.rs
+++ b/crates/cairo-lang-starknet/src/plugin/starknet_module/contract.rs
@@ -302,12 +310,13 @@ pub(super) fn generate_contract_specific_code(
     common_data: StarknetModuleCommonGenerationData,
     body: &ast::ModuleBody,
     module_ast: &ast::ItemModule,
+    metadata: &MacroPluginMetadata<'_>,
     event_variants: Vec<SmolStr>,
 ) -> RewriteNode {
     let mut generation_data = ContractGenerationData { common: common_data, ..Default::default() };
     generation_data.specific.components_data.nested_event_variants = event_variants;
-    for item in body.items(db).elements(db) {
-        handle_contract_item(db, diagnostics, &item, &mut generation_data);
+    for item in body.iter_items_in_cfg(db, metadata.cfg_set) {
+        handle_contract_item(db, diagnostics, &item, metadata, &mut generation_data);
     }
 
     let test_class_hash = format!(
diff --git /dev/null b/tests/bug_samples/issue4897.cairo
new file mode 100644
--- /dev/null
+++ b/tests/bug_samples/issue4897.cairo
@@ -0,0 +1,22 @@
+#[cfg(missing_cfg)]
+#[starknet::interface]
+trait ITests<TContractState> {
+    fn set_value(ref self: TContractState, value: felt252);
+}
+
+
+#[starknet::contract]
+mod MyContract {
+    #[storage]
+    struct Storage {
+        value: felt252
+    }
+
+    #[cfg(missing_cfg)]
+    #[abi(embed_v0)]
+    impl TestsImpl of super::ITests<ContractState> {
+        fn set_value(ref self: ContractState, value: felt252) {
+            self.value.write(value);
+        }
+    }
+}
diff --git a/tests/bug_samples/lib.cairo b/tests/bug_samples/lib.cairo
--- a/tests/bug_samples/lib.cairo
+++ b/tests/bug_samples/lib.cairo
@@ -36,6 +36,7 @@ mod issue4109;
 mod issue4314;
 mod issue4318;
 mod issue4380;
+mod issue4897;
 mod loop_only_change;
 mod inconsistent_gas;
 mod partial_param_local;

EOF_114329324912
cd "crates/cairo-lang-starknet"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard 7e0dceb788114a0f7e201801a03c034b52c19679
git clean -fd
