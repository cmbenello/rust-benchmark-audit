#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/validation/rules/default_values_of_correct_type.rs b/src/validation/rules/default_values_of_correct_type.rs
--- a/src/validation/rules/default_values_of_correct_type.rs
+++ b/src/validation/rules/default_values_of_correct_type.rs
@@ -101,8 +96,8 @@ mod tests {
     }
 
     #[test]
-    fn no_required_variables_with_default_values() {
-        expect_fails_rule!(
+    fn required_variables_with_default_values() {
+        expect_passes_rule!(
             factory,
             r#"
           query UnreachableDefaultValues($a: Int! = 3, $b: String! = "default") {
diff --git a/tests/variables.rs b/tests/variables.rs
--- a/tests/variables.rs
+++ b/tests/variables.rs
@@ -131,6 +131,59 @@ pub async fn test_variable_null() {
     );
 }
 
+#[tokio::test]
+pub async fn test_required_variable_with_default() {
+    struct Query;
+
+    #[Object]
+    impl Query {
+        pub async fn int_val(&self, value: i32) -> i32 {
+            value
+        }
+    }
+
+    let schema = Schema::new(Query, EmptyMutation, EmptySubscription);
+
+    // test variable default
+    {
+        let query = Request::new(
+            r#"
+            query QueryWithVariables($intVal: Int! = 10) {
+                intVal(value: $intVal)
+            }
+            "#,
+        );
+        let resp = schema.execute(query).await;
+        assert_eq!(
+            resp.data,
+            value!({
+                "intVal": 10,
+            }),
+            "{}",
+            resp.data
+        );
+    }
+
+    // test variable null
+    {
+        let query = Request::new(
+            r#"
+            query QueryWithVariables($intVal: Int! = 10) {
+                intVal(value: $intVal)
+            }
+            "#,
+        )
+        .variables(Variables::from_value(value!({
+            "intVal": null,
+        })));
+        let resp = schema.execute(query).await;
+        assert_eq!(
+            resp.errors.first().map(|v| v.message.as_str()),
+            Some("Invalid value for argument \"value\", expected type \"Int\"")
+        );
+    }
+}
+
 #[tokio::test]
 pub async fn test_variable_in_input_object() {
     #[derive(InputObject)]

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 759fb0e594606aecf4a239b00d6f3f8aeb22d885
git clean -fd
