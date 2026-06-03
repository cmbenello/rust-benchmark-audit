#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/schemas/test_fed2_compose_2.schema.graphql b/tests/schemas/test_fed2_compose_2.schema.graphql
--- a/tests/schemas/test_fed2_compose_2.schema.graphql
+++ b/tests/schemas/test_fed2_compose_2.schema.graphql
@@ -3,7 +3,7 @@
 
 
 type Query {
-	testArgument(arg: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Object")): String!
+	testArgument(arg1: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Object.arg1"), arg2: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Object.arg2")): String!
 	testInputObject(arg: TestInput!): String!
 	testComplexObject: TestComplexObject!
 	testSimpleObject: TestSimpleObject!
diff --git a/tests/schemas/test_fed2_compose_2.schema.graphql b/tests/schemas/test_fed2_compose_2.schema.graphql
--- a/tests/schemas/test_fed2_compose_2.schema.graphql
+++ b/tests/schemas/test_fed2_compose_2.schema.graphql
@@ -15,7 +15,7 @@ type Query {
 
 type TestComplexObject @type_directive_object(description: "This is OBJECT in (Complex / Simple)Object") {
 	field: String! @type_directive_field_definition(description: "This is FIELD_DEFINITION in (Complex / Simple)Object")
-	test(arg: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in ComplexObject")): String!
+	test(arg1: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in ComplexObject.arg1"), arg2: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in ComplexObject.arg2")): String!
 }
 
 enum TestEnum @type_directive_enum(description: "This is ENUM in Enum") {
diff --git a/tests/schemas/test_fed2_compose_2.schema.graphql b/tests/schemas/test_fed2_compose_2.schema.graphql
--- a/tests/schemas/test_fed2_compose_2.schema.graphql
+++ b/tests/schemas/test_fed2_compose_2.schema.graphql
@@ -28,11 +28,11 @@ input TestInput @type_directive_input_object(description: "This is INPUT_OBJECT
 }
 
 interface TestInterface @type_directive_interface(description: "This is INTERFACE in Interface") {
-	field(arg: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Interface")): String! @type_directive_field_definition(description: "This is INTERFACE in Interface")
+	field(arg1: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Interface.arg1"), arg2: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Interface.arg2")): String! @type_directive_field_definition(description: "This is INTERFACE in Interface")
 }
 
 type TestObjectForInterface implements TestInterface {
-	field(arg: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Interface")): String!
+	field(arg1: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Interface.arg1"), arg2: String! @type_directive_argument_definition(description: "This is ARGUMENT_DEFINITION in Interface.arg2")): String!
 }
 
 input TestOneOfObject @oneOf @type_directive_input_object(description: "This is INPUT_OBJECT in OneofObject") {
diff --git a/tests/type_directive.rs b/tests/type_directive.rs
--- a/tests/type_directive.rs
+++ b/tests/type_directive.rs
@@ -118,8 +118,10 @@ fn test_type_directive_2() {
     impl TestComplexObject {
         async fn test(
             &self,
-            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in ComplexObject".to_string()))]
-            _arg: String,
+            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in ComplexObject.arg1".to_string()))]
+            _arg1: String,
+            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in ComplexObject.arg2".to_string()))]
+            _arg2: String,
         ) -> &'static str {
             "test"
         }
diff --git a/tests/type_directive.rs b/tests/type_directive.rs
--- a/tests/type_directive.rs
+++ b/tests/type_directive.rs
@@ -140,8 +142,10 @@ fn test_type_directive_2() {
     impl TestObjectForInterface {
         async fn field(
             &self,
-            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Interface".to_string()))]
-            _arg: String,
+            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Interface.arg1".to_string()))]
+            _arg1: String,
+            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Interface.arg2".to_string()))]
+            _arg2: String,
         ) -> &'static str {
             "hello"
         }
diff --git a/tests/type_directive.rs b/tests/type_directive.rs
--- a/tests/type_directive.rs
+++ b/tests/type_directive.rs
@@ -153,9 +157,14 @@ fn test_type_directive_2() {
             ty = "String",
             directive = type_directive_field_definition::apply("This is INTERFACE in Interface".to_string()),
             arg(
-                name = "_arg",
+                name = "_arg1",
                 ty = "String",
-                directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Interface".to_string())
+                directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Interface.arg1".to_string())
+            ),
+            arg(
+                name = "_arg2",
+                ty = "String",
+                directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Interface.arg2".to_string())
             )
         ),
         directive = type_directive_interface::apply("This is INTERFACE in Interface".to_string())
diff --git a/tests/type_directive.rs b/tests/type_directive.rs
--- a/tests/type_directive.rs
+++ b/tests/type_directive.rs
@@ -170,8 +179,10 @@ fn test_type_directive_2() {
     impl Query {
         pub async fn test_argument(
             &self,
-            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Object".to_string()))]
-            _arg: String,
+            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Object.arg1".to_string()))]
+            _arg1: String,
+            #[graphql(directive = type_directive_argument_definition::apply("This is ARGUMENT_DEFINITION in Object.arg2".to_string()))]
+            _arg2: String,
         ) -> &'static str {
             "hello"
         }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard faa78d071cfae795dacc39fd46900fdfb6157d0d
git clean -fd
