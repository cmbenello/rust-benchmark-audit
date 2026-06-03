#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/federation.rs b/tests/federation.rs
--- a/tests/federation.rs
+++ b/tests/federation.rs
@@ -536,6 +536,15 @@ pub async fn test_entity_inaccessible() {
     assert!(schema_sdl.contains("inputFieldInaccessibleA: Int! @inaccessible"));
     // no trailing spaces
     assert!(!schema_sdl.contains(" \n"));
+
+    // compare to expected schema
+    let path = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
+        .join("tests/schemas/test_entity_inaccessible.schema.graphql");
+    let expected_schema = std::fs::read_to_string(&path).unwrap();
+    if schema_sdl != expected_schema {
+        std::fs::write(path, schema_sdl).unwrap();
+        panic!("schema was not up-to-date. rerun")
+    }
 }
 
 #[tokio::test]
diff --git a/tests/federation.rs b/tests/federation.rs
--- a/tests/federation.rs
+++ b/tests/federation.rs
@@ -737,4 +746,13 @@ pub async fn test_entity_tag() {
     );
     // no trailing spaces
     assert!(!schema_sdl.contains(" \n"));
+
+    // compare to expected schema
+    let path = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
+        .join("tests/schemas/test_entity_tag.schema.graphql");
+    let expected_schema = std::fs::read_to_string(&path).unwrap();
+    if schema_sdl != expected_schema {
+        std::fs::write(path, schema_sdl).unwrap();
+        panic!("schema was not up-to-date. rerun")
+    }
 }
diff --git /dev/null b/tests/schemas/test_entity_inaccessible.schema.graphql
new file mode 100644
--- /dev/null
+++ b/tests/schemas/test_entity_inaccessible.schema.graphql
@@ -0,0 +1,67 @@
+
+
+
+
+type MyCustomObjInaccessible @inaccessible {
+	a: Int!
+	customObjectInaccessible: Int! @inaccessible
+}
+
+enum MyEnumInaccessible @inaccessible {
+	OPTION_A
+	OPTION_B
+	OPTION_C
+}
+
+enum MyEnumVariantInaccessible {
+	OPTION_A_INACCESSIBLE @inaccessible
+	OPTION_B
+	OPTION_C
+}
+
+input MyInputObjFieldInaccessible {
+	inputFieldInaccessibleA: Int! @inaccessible
+}
+
+input MyInputObjInaccessible @inaccessible {
+	a: Int!
+}
+
+interface MyInterfaceInaccessible @inaccessible {
+	inaccessibleInterfaceValue: String! @inaccessible
+}
+
+type MyInterfaceObjA implements MyInterfaceInaccessible {
+	inaccessibleInterfaceValue: String!
+}
+
+type MyInterfaceObjB implements MyInterfaceInaccessible @inaccessible {
+	inaccessibleInterfaceValue: String!
+}
+
+scalar MyNumberInaccessible @inaccessible
+
+type MyObjFieldInaccessible @key(fields: "id") {
+	objFieldInaccessibleA: Int! @inaccessible
+}
+
+type MyObjInaccessible @key(fields: "id") @inaccessible {
+	a: Int!
+}
+
+union MyUnionInaccessible @inaccessible = MyInterfaceObjA | MyInterfaceObjB
+
+extend type Query {
+	enumVariantInaccessible(id: Int!): MyEnumVariantInaccessible!
+	enumInaccessible(id: Int!): MyEnumInaccessible!
+	inaccessibleField(id: Int!): Int! @inaccessible
+	inaccessibleArgument(id: Int! @inaccessible): Int!
+	inaccessibleInterface: MyInterfaceInaccessible!
+	inaccessibleUnion: MyUnionInaccessible!
+	inaccessibleScalar: MyNumberInaccessible!
+	inaccessibleInputField(value: MyInputObjFieldInaccessible!): Int!
+	inaccessibleInput(value: MyInputObjInaccessible!): Int!
+	inaccessibleCustomObject: MyCustomObjInaccessible!
+}
+
+
diff --git /dev/null b/tests/schemas/test_entity_tag.schema.graphql
new file mode 100644
--- /dev/null
+++ b/tests/schemas/test_entity_tag.schema.graphql
@@ -0,0 +1,67 @@
+
+
+
+
+type MyCustomObjTagged @tag(name: "tagged") @tag(name: "object") @tag(name: "with") @tag(name: "multiple") @tag(name: "tags") {
+	a: Int!
+	customObjectTagged: Int! @tag(name: "tagged_custom_object_field")
+}
+
+enum MyEnumTagged @tag(name: "tagged_num") {
+	OPTION_A
+	OPTION_B
+	OPTION_C
+}
+
+enum MyEnumVariantTagged {
+	OPTION_A_TAGGED @tag(name: "tagged_enum_option")
+	OPTION_B
+	OPTION_C
+}
+
+input MyInputObjFieldTagged {
+	inputFieldTaggedA: Int! @tag(name: "tagged_input_object_field")
+}
+
+input MyInputObjTagged @tag(name: "input_object_tag") {
+	a: Int!
+}
+
+type MyInterfaceObjA implements MyInterfaceTagged {
+	taggedInterfaceValue: String!
+}
+
+type MyInterfaceObjB implements MyInterfaceTagged @tag(name: "interface_object") {
+	taggedInterfaceValue: String!
+}
+
+interface MyInterfaceTagged @tag(name: "tagged_interface") {
+	taggedInterfaceValue: String! @tag(name: "tagged_interface_field")
+}
+
+scalar MyNumberTagged @tag(name: "tagged_scalar")
+
+type MyObjFieldTagged @key(fields: "id") {
+	objFieldTaggedA: Int! @tag(name: "tagged_field")
+}
+
+type MyObjTagged @key(fields: "id") @tag(name: "tagged_simple_object") {
+	a: Int!
+}
+
+union MyUnionTagged @tag(name: "tagged_union") = MyInterfaceObjA | MyInterfaceObjB
+
+extend type Query {
+	enumVariantTagged(id: Int!): MyEnumVariantTagged!
+	enumTagged(id: Int!): MyEnumTagged!
+	taggedField(id: Int!): Int! @tag(name: "tagged_\"field\"")
+	taggedArgument(id: Int! @tag(name: "tagged_argument")): Int!
+	taggedInterface: MyInterfaceTagged!
+	taggedUnion: MyUnionTagged!
+	taggedScalar: MyNumberTagged!
+	taggedInputField(value: MyInputObjFieldTagged!): Int!
+	taggedInput(value: MyInputObjTagged!): Int!
+	taggedCustomObject: MyCustomObjTagged!
+}
+
+

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard b2acefdf602e91d451ba01730c8a38a8f9d30abc
git clean -fd
