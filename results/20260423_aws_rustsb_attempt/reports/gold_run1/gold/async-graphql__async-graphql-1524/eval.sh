#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/directive.rs b/tests/directive.rs
--- a/tests/directive.rs
+++ b/tests/directive.rs
@@ -1,4 +1,5 @@
 use async_graphql::*;
+use serde::{Deserialize, Serialize};
 
 #[tokio::test]
 pub async fn test_directive_skip() {
diff --git a/tests/directive.rs b/tests/directive.rs
--- a/tests/directive.rs
+++ b/tests/directive.rs
@@ -127,3 +128,95 @@ pub async fn test_custom_directive() {
         value!({ "value": "&abc*" })
     );
 }
+
+#[tokio::test]
+pub async fn test_no_unused_directives() {
+    struct Query;
+
+    #[Object]
+    impl Query {
+        pub async fn a(&self) -> String {
+            "a".into()
+        }
+    }
+
+    let sdl = Schema::new(Query, EmptyMutation, EmptySubscription).sdl();
+
+    assert!(!sdl.contains("directive @deprecated"));
+    assert!(!sdl.contains("directive @specifiedBy"));
+    assert!(!sdl.contains("directive @oneOf"));
+}
+
+#[tokio::test]
+pub async fn test_includes_deprecated_directive() {
+    #[derive(SimpleObject)]
+    struct A {
+        #[graphql(deprecation = "Use `Foo` instead")]
+        a: String,
+    }
+
+    struct Query;
+
+    #[Object]
+    impl Query {
+        pub async fn a(&self) -> A {
+            A { a: "a".into() }
+        }
+    }
+
+    let schema = Schema::new(Query, EmptyMutation, EmptySubscription);
+
+    assert!(schema.sdl().contains(r#"directive @deprecated(reason: String = "No longer supported") on FIELD_DEFINITION | ARGUMENT_DEFINITION | INPUT_FIELD_DEFINITION | ENUM_VALUE"#))
+}
+
+#[tokio::test]
+pub async fn test_includes_specified_by_directive() {
+    #[derive(Serialize, Deserialize)]
+    struct A {
+        a: String,
+    }
+
+    scalar!(
+        A,
+        "A",
+        "This is A",
+        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
+    );
+
+    struct Query;
+
+    #[Object]
+    impl Query {
+        pub async fn a(&self) -> A {
+            A { a: "a".into() }
+        }
+    }
+
+    let schema = Schema::new(Query, EmptyMutation, EmptySubscription);
+
+    assert!(schema
+        .sdl()
+        .contains(r#"directive @specifiedBy(url: String!) on SCALAR"#))
+}
+
+#[tokio::test]
+pub async fn test_includes_one_of_directive() {
+    #[derive(OneofObject)]
+    enum AB {
+        A(String),
+        B(i64),
+    }
+
+    struct Query;
+
+    #[Object]
+    impl Query {
+        pub async fn ab(&self, _input: AB) -> bool {
+            true
+        }
+    }
+
+    let schema = Schema::new(Query, EmptyMutation, EmptySubscription);
+
+    assert!(schema.sdl().contains(r#"directive @oneOf on INPUT_OBJECT"#))
+}
diff --git a/tests/introspection.rs b/tests/introspection.rs
--- a/tests/introspection.rs
+++ b/tests/introspection.rs
@@ -1514,3 +1514,146 @@ pub async fn test_introspection_default() {
     let res = schema.execute(query).await.into_result().unwrap().data;
     assert_eq!(res, res_json);
 }
+
+#[tokio::test]
+pub async fn test_introspection_directives() {
+    struct Query;
+
+    #[Object]
+    impl Query {
+        pub async fn a(&self) -> String {
+            "a".into()
+        }
+    }
+
+    let schema = Schema::new(Query, EmptyMutation, EmptySubscription);
+    let query = r#"
+      query IntrospectionQuery {
+        __schema {
+          directives {
+            name
+            locations
+            args {
+              ...InputValue
+            }
+          }
+        }
+      }
+      
+      fragment InputValue on __InputValue {
+        name
+        type {
+          ...TypeRef
+        }
+        defaultValue
+      }
+      
+      fragment TypeRef on __Type {
+        kind
+        name
+        ofType {
+          kind
+          name
+        }
+      }
+    "#;
+
+    let res_json = value!({"__schema": {
+      "directives": [
+        {
+          "name": "deprecated",
+          "locations": [
+            "FIELD_DEFINITION",
+            "ARGUMENT_DEFINITION",
+            "INPUT_FIELD_DEFINITION",
+            "ENUM_VALUE"
+          ],
+          "args": [
+            {
+              "name": "reason",
+              "type": {
+                "kind": "SCALAR",
+                "name": "String",
+                "ofType": null
+              },
+              "defaultValue": "\"No longer supported\""
+            }
+          ]
+        },
+        {
+          "name": "include",
+          "locations": [
+            "FIELD",
+            "FRAGMENT_SPREAD",
+            "INLINE_FRAGMENT"
+          ],
+          "args": [
+            {
+              "name": "if",
+              "type": {
+                "kind": "NON_NULL",
+                "name": null,
+                "ofType": {
+                  "kind": "SCALAR",
+                  "name": "Boolean"
+                }
+              },
+              "defaultValue": null
+            }
+          ]
+        },
+        {
+          "name": "oneOf",
+          "locations": [
+            "INPUT_OBJECT"
+          ],
+          "args": []
+        },
+        {
+          "name": "skip",
+          "locations": [
+            "FIELD",
+            "FRAGMENT_SPREAD",
+            "INLINE_FRAGMENT"
+          ],
+          "args": [
+            {
+              "name": "if",
+              "type": {
+                "kind": "NON_NULL",
+                "name": null,
+                "ofType": {
+                  "kind": "SCALAR",
+                  "name": "Boolean"
+                }
+              },
+              "defaultValue": null
+            }
+          ]
+        },
+        {
+          "name": "specifiedBy",
+          "locations": [
+            "SCALAR"
+          ],
+          "args": [
+            {
+              "name": "url",
+              "type": {
+                "kind": "NON_NULL",
+                "name": null,
+                "ofType": {
+                  "kind": "SCALAR",
+                  "name": "String"
+                }
+              },
+              "defaultValue": null
+            }
+          ]
+        }
+      ]
+    }});
+    let res = schema.execute(query).await.into_result().unwrap().data;
+
+    assert_eq!(res, res_json);
+}
diff --git a/tests/schemas/test_fed2_compose_2.schema.graphql b/tests/schemas/test_fed2_compose_2.schema.graphql
--- a/tests/schemas/test_fed2_compose_2.schema.graphql
+++ b/tests/schemas/test_fed2_compose_2.schema.graphql
@@ -1,5 +1,3 @@
-directive @oneOf on INPUT_OBJECT
-
 
 
 
diff --git a/tests/schemas/test_fed2_compose_2.schema.graphql b/tests/schemas/test_fed2_compose_2.schema.graphql
--- a/tests/schemas/test_fed2_compose_2.schema.graphql
+++ b/tests/schemas/test_fed2_compose_2.schema.graphql
@@ -47,6 +45,7 @@ type TestSimpleObject @type_directive_object(description: "This is OBJECT in Sim
 }
 
 directive @include(if: Boolean!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT
+directive @oneOf on INPUT_OBJECT
 directive @skip(if: Boolean!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT
 directive @type_directive_argument_definition(description: String!) on ARGUMENT_DEFINITION
 directive @type_directive_enum(description: String!) on ENUM

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 9d1befde1444a3d6a9dfdc62750ba271159c7173
git clean -fd
