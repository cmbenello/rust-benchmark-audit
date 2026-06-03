#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cairo-lang-sierra-generator/src/store_variables/test.rs b/crates/cairo-lang-sierra-generator/src/store_variables/test.rs
--- a/crates/cairo-lang-sierra-generator/src/store_variables/test.rs
+++ b/crates/cairo-lang-sierra-generator/src/store_variables/test.rs
@@ -207,6 +207,11 @@ fn get_lib_func_signature(db: &dyn SierraGenGroup, libfunc: ConcreteLibfuncId) -
             ],
             SierraApChange::Known { new_vars_only: true },
         ),
+        "make_local" => LibfuncSignature::new_non_branch(
+            vec![felt252_ty.clone()],
+            vec![OutputVarInfo { ty: felt252_ty, ref_info: OutputVarReferenceInfo::NewLocalVar }],
+            SierraApChange::Known { new_vars_only: true },
+        ),
         _ => panic!("get_branch_signatures() is not implemented for '{name}'."),
     }
 }
diff --git a/crates/cairo-lang-sierra-generator/src/store_variables/test.rs b/crates/cairo-lang-sierra-generator/src/store_variables/test.rs
--- a/crates/cairo-lang-sierra-generator/src/store_variables/test.rs
+++ b/crates/cairo-lang-sierra-generator/src/store_variables/test.rs
@@ -841,3 +846,27 @@ fn consecutive_appends_with_branch() {
         ]
     );
 }
+
+#[test]
+fn push_values_with_hole() {
+    let db = SierraGenDatabaseForTesting::default();
+    let statements: Vec<pre_sierra::Statement> = vec![
+        dummy_push_values(&db, &[("0", "100"), ("1", "101"), ("2", "102")]),
+        dummy_simple_statement(&db, "make_local", &["102"], &["102"]),
+        dummy_push_values(&db, &[("100", "200"), ("101", "201")]),
+        dummy_return_statement(&["201"]),
+    ];
+
+    assert_eq!(
+        test_add_store_statements(&db, statements, LocalVariables::default(), &["0", "1", "2"]),
+        vec![
+            "store_temp<felt252>(0) -> (100)",
+            "store_temp<felt252>(1) -> (101)",
+            "store_temp<felt252>(2) -> (102)",
+            "make_local(102) -> (102)",
+            "store_temp<felt252>(100) -> (200)",
+            "store_temp<felt252>(101) -> (201)",
+            "return(201)",
+        ]
+    );
+}

EOF_114329324912
cd "crates/cairo-lang-sierra-generator"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard 75f779e8553d28e4bfff3b23f140ccc781c9dab0
git clean -fd
