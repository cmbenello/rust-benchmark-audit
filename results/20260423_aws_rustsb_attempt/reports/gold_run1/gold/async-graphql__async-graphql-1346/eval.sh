#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/registry/export_sdl.rs b/src/registry/export_sdl.rs
--- a/src/registry/export_sdl.rs
+++ b/src/registry/export_sdl.rs
@@ -644,8 +649,8 @@ mod tests {
     #[test]
     fn test_compose_directive_dsl() {
         let expected = r#"extend schema @link(
-	url: "https://specs.apollo.dev/federation/v2.1",
-	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective"]
+	url: "https://specs.apollo.dev/federation/v2.3",
+	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective", "@interfaceObject"]
 )
 
 extend schema @link(
diff --git a/tests/federation.rs b/tests/federation.rs
--- a/tests/federation.rs
+++ b/tests/federation.rs
@@ -826,3 +826,62 @@ pub async fn test_entity_tag() {
         panic!("schema was not up-to-date. rerun")
     }
 }
+
+#[tokio::test]
+pub async fn test_interface_object() {
+    #[derive(SimpleObject)]
+    struct VariantA {
+        pub id: u64,
+    }
+
+    #[derive(Interface)]
+    #[graphql(field(name = "id", ty = "&u64"))]
+    enum MyInterface {
+        VariantA(VariantA),
+    }
+
+    #[derive(SimpleObject)]
+    #[graphql(interface_object)]
+    struct MyInterfaceObject1 {
+        pub id: u64,
+    }
+
+    struct MyInterfaceObject2;
+
+    #[Object(interface_object)]
+    impl MyInterfaceObject2 {
+        pub async fn id(&self) -> u64 {
+            todo!()
+        }
+    }
+
+    struct Query;
+
+    #[Object(extends)]
+    impl Query {
+        #[graphql(entity)]
+        async fn my_interface(&self, _id: u64) -> MyInterface {
+            todo!()
+        }
+
+        #[graphql(entity)]
+        async fn my_interface_object1(&self, _id: u64) -> MyInterfaceObject1 {
+            todo!()
+        }
+
+        #[graphql(entity)]
+        async fn my_interface_object2(&self, _id: u64) -> MyInterfaceObject2 {
+            todo!()
+        }
+    }
+
+    let schema_sdl = Schema::new(Query, EmptyMutation, EmptySubscription)
+        .sdl_with_options(SDLExportOptions::new().federation());
+
+    // Interface with @key directive
+    assert!(schema_sdl.contains("interface MyInterface @key(fields: \"id\")"));
+
+    // Object with @interfaceObject directive
+    assert!(schema_sdl.contains("type MyInterfaceObject1 @key(fields: \"id\") @interfaceObject"));
+    assert!(schema_sdl.contains("type MyInterfaceObject2 @key(fields: \"id\") @interfaceObject"));
+}
diff --git a/tests/schemas/test_entity_inaccessible.schema.graphql b/tests/schemas/test_entity_inaccessible.schema.graphql
--- a/tests/schemas/test_entity_inaccessible.schema.graphql
+++ b/tests/schemas/test_entity_inaccessible.schema.graphql
@@ -66,8 +66,8 @@ extend type Query {
 
 
 extend schema @link(
-	url: "https://specs.apollo.dev/federation/v2.1",
-	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective"]
+	url: "https://specs.apollo.dev/federation/v2.3",
+	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective", "@interfaceObject"]
 )
 directive @include(if: Boolean!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT
 directive @skip(if: Boolean!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT
diff --git a/tests/schemas/test_entity_tag.schema.graphql b/tests/schemas/test_entity_tag.schema.graphql
--- a/tests/schemas/test_entity_tag.schema.graphql
+++ b/tests/schemas/test_entity_tag.schema.graphql
@@ -66,8 +66,8 @@ extend type Query {
 
 
 extend schema @link(
-	url: "https://specs.apollo.dev/federation/v2.1",
-	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective"]
+	url: "https://specs.apollo.dev/federation/v2.3",
+	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective", "@interfaceObject"]
 )
 directive @include(if: Boolean!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT
 directive @skip(if: Boolean!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT
diff --git a/tests/schemas/test_fed2_compose.schema.graphql b/tests/schemas/test_fed2_compose.schema.graphql
--- a/tests/schemas/test_fed2_compose.schema.graphql
+++ b/tests/schemas/test_fed2_compose.schema.graphql
@@ -13,8 +13,8 @@ type SimpleValue @testDirective(scope: "simple object type", input: 1, opt: 3) {
 
 
 extend schema @link(
-	url: "https://specs.apollo.dev/federation/v2.1",
-	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective"]
+	url: "https://specs.apollo.dev/federation/v2.3",
+	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective", "@interfaceObject"]
 )
 
 extend schema @link(
diff --git a/tests/schemas/test_fed2_link.schema.graphqls b/tests/schemas/test_fed2_link.schema.graphqls
--- a/tests/schemas/test_fed2_link.schema.graphqls
+++ b/tests/schemas/test_fed2_link.schema.graphqls
@@ -21,8 +21,8 @@ extend type User @key(fields: "id") {
 }
 
 extend schema @link(
-	url: "https://specs.apollo.dev/federation/v2.1",
-	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective"]
+	url: "https://specs.apollo.dev/federation/v2.3",
+	import: ["@key", "@tag", "@shareable", "@inaccessible", "@override", "@external", "@provides", "@requires", "@composeDirective", "@interfaceObject"]
 )
 directive @include(if: Boolean!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT
 directive @skip(if: Boolean!) on FIELD | FRAGMENT_SPREAD | INLINE_FRAGMENT

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard b0e2c7494bf7c8bfdc00692a9cf41f4ca7d5a676
git clean -fd
