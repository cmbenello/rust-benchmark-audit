#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/variables.rs b/tests/variables.rs
--- a/tests/variables.rs
+++ b/tests/variables.rs
@@ -71,6 +71,69 @@ pub async fn test_variable_default_value() {
     );
 }
 
+#[async_std::test]
+pub async fn test_variable_no_value() {
+    struct QueryRoot;
+
+    #[Object]
+    impl QueryRoot {
+        pub async fn int_val(&self, value: Option<i32>) -> i32 {
+            value.unwrap_or(10)
+        }
+    }
+
+    let schema = Schema::new(QueryRoot, EmptyMutation, EmptySubscription);
+    let query = QueryBuilder::new(
+        r#"
+            query QueryWithVariables($intVal: Int) {
+                intVal(value: $intVal)
+            }
+        "#,
+    )
+    .variables(Variables::parse_from_json(serde_json::json!({})).unwrap());
+    let resp = query.execute(&schema).await.unwrap();
+    assert_eq!(
+        resp.data,
+        serde_json::json!({
+            "intVal": 10,
+        })
+    );
+}
+
+#[async_std::test]
+pub async fn test_variable_null() {
+    struct QueryRoot;
+
+    #[Object]
+    impl QueryRoot {
+        pub async fn int_val(&self, value: Option<i32>) -> i32 {
+            value.unwrap_or(10)
+        }
+    }
+
+    let schema = Schema::new(QueryRoot, EmptyMutation, EmptySubscription);
+    let query = QueryBuilder::new(
+        r#"
+            query QueryWithVariables($intVal: Int) {
+                intVal(value: $intVal)
+            }
+        "#,
+    )
+    .variables(
+        Variables::parse_from_json(serde_json::json!({
+            "intVal": null,
+        }))
+        .unwrap(),
+    );
+    let resp = query.execute(&schema).await.unwrap();
+    assert_eq!(
+        resp.data,
+        serde_json::json!({
+            "intVal": 10,
+        })
+    );
+}
+
 #[async_std::test]
 pub async fn test_variable_in_input_object() {
     #[InputObject]

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 2e9557ff1c1482acce1337508b81ccdde09009a9
git clean -fd
