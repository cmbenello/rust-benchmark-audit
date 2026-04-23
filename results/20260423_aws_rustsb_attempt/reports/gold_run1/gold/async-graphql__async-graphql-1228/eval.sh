#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/dynamic/interface.rs b/src/dynamic/interface.rs
--- a/src/dynamic/interface.rs
+++ b/src/dynamic/interface.rs
@@ -389,4 +389,47 @@ mod tests {
             }]
         );
     }
+    #[tokio::test]
+    async fn query_type_condition() {
+        struct MyObjA;
+        let obj_a = Object::new("MyObjA")
+            .implement("MyInterface")
+            .field(Field::new("a", TypeRef::named(TypeRef::INT), |_| {
+                FieldFuture::new(async { Ok(Some(Value::from(100))) })
+            }))
+            .field(Field::new("b", TypeRef::named(TypeRef::INT), |_| {
+                FieldFuture::new(async { Ok(Some(Value::from(200))) })
+            }));
+        let interface = Interface::new("MyInterface")
+            .field(InterfaceField::new("a", TypeRef::named(TypeRef::INT)));
+        let query = Object::new("Query");
+        let query = query.field(Field::new(
+            "valueA",
+            TypeRef::named_nn(obj_a.type_name()),
+            |_| FieldFuture::new(async { Ok(Some(FieldValue::owned_any(MyObjA))) }),
+        ));
+        let schema = Schema::build(query.type_name(), None, None)
+            .register(obj_a)
+            .register(interface)
+            .register(query)
+            .finish()
+            .unwrap();
+        let query = r#"
+        {
+            valueA { __typename
+            b
+            ... on MyInterface { a } }
+        }
+        "#;
+        assert_eq!(
+            schema.execute(query).await.into_result().unwrap().data,
+            value!({
+                "valueA": {
+                    "__typename": "MyObjA",
+                    "b": 200,
+                    "a": 100,
+                }
+            })
+        );
+    }
 }
diff --git a/src/dynamic/union.rs b/src/dynamic/union.rs
--- a/src/dynamic/union.rs
+++ b/src/dynamic/union.rs
@@ -137,7 +137,7 @@ impl Union {
 mod tests {
     use async_graphql_parser::Pos;
 
-    use crate::{dynamic::*, value, PathSegment, ServerError, Value};
+    use crate::{dynamic::*, value, PathSegment, Request, ServerError, Value};
 
     #[tokio::test]
     async fn basic_union() {
diff --git a/src/dynamic/union.rs b/src/dynamic/union.rs
--- a/src/dynamic/union.rs
+++ b/src/dynamic/union.rs
@@ -258,4 +258,136 @@ mod tests {
             }]
         );
     }
+
+    #[tokio::test]
+    async fn test_query() {
+        struct Dog;
+        struct Cat;
+        struct Snake;
+        // enum
+        #[allow(dead_code)]
+        enum Animal {
+            Dog(Dog),
+            Cat(Cat),
+            Snake(Snake),
+        }
+        struct Query {
+            pet: Animal,
+        }
+
+        impl Animal {
+            fn to_field_value(&self) -> FieldValue {
+                match self {
+                    Animal::Dog(dog) => FieldValue::borrowed_any(dog).with_type("Dog"),
+                    Animal::Cat(cat) => FieldValue::borrowed_any(cat).with_type("Cat"),
+                    Animal::Snake(snake) => FieldValue::borrowed_any(snake).with_type("Snake"),
+                }
+            }
+        }
+        fn create_schema() -> Schema {
+            // interface
+            let named = Interface::new("Named");
+            let named = named.field(InterfaceField::new(
+                "name",
+                TypeRef::named_nn(TypeRef::STRING),
+            ));
+            // dog
+            let dog = Object::new("Dog");
+            let dog = dog.field(Field::new(
+                "name",
+                TypeRef::named_nn(TypeRef::STRING),
+                |_ctx| FieldFuture::new(async move { Ok(Some(Value::from("dog"))) }),
+            ));
+            let dog = dog.field(Field::new(
+                "power",
+                TypeRef::named_nn(TypeRef::INT),
+                |_ctx| FieldFuture::new(async move { Ok(Some(Value::from(100))) }),
+            ));
+            let dog = dog.implement("Named");
+            // cat
+            let cat = Object::new("Cat");
+            let cat = cat.field(Field::new(
+                "name",
+                TypeRef::named_nn(TypeRef::STRING),
+                |_ctx| FieldFuture::new(async move { Ok(Some(Value::from("cat"))) }),
+            ));
+            let cat = cat.field(Field::new(
+                "life",
+                TypeRef::named_nn(TypeRef::INT),
+                |_ctx| FieldFuture::new(async move { Ok(Some(Value::from(9))) }),
+            ));
+            let cat = cat.implement("Named");
+            // snake
+            let snake = Object::new("Snake");
+            let snake = snake.field(Field::new(
+                "length",
+                TypeRef::named_nn(TypeRef::INT),
+                |_ctx| FieldFuture::new(async move { Ok(Some(Value::from(200))) }),
+            ));
+            // animal
+            let animal = Union::new("Animal");
+            let animal = animal.possible_type("Dog");
+            let animal = animal.possible_type("Cat");
+            let animal = animal.possible_type("Snake");
+            // query
+
+            let query = Object::new("Query");
+            let query = query.field(Field::new("pet", TypeRef::named_nn("Animal"), |ctx| {
+                FieldFuture::new(async move {
+                    let query = ctx.parent_value.try_downcast_ref::<Query>()?;
+                    Ok(Some(query.pet.to_field_value()))
+                })
+            }));
+
+            let schema = Schema::build(query.type_name(), None, None);
+            let schema = schema
+                .register(query)
+                .register(named)
+                .register(dog)
+                .register(cat)
+                .register(snake)
+                .register(animal);
+
+            schema.finish().unwrap()
+        }
+
+        let schema = create_schema();
+        let query = r#"
+            query {
+                dog: pet {
+                    ... on Dog {
+                        __dog_typename: __typename
+                        name
+                        power
+                    }
+                }
+                named: pet {
+                    ... on Named {
+                        __named_typename: __typename
+                        name
+                    }
+                }
+            }
+        "#;
+        let root = Query {
+            pet: Animal::Dog(Dog),
+        };
+        let req = Request::new(query).root_value(FieldValue::owned_any(root));
+        let res = schema.execute(req).await;
+
+        assert_eq!(
+            res.data.into_json().unwrap(),
+            serde_json::json!({
+                "dog": {
+                    "__dog_typename": "Dog",
+                    "name": "dog",
+                    "power": 100
+                },
+                "named": {
+                    "__named_typename": "Dog",
+                    "name": "dog"
+                }
+            })
+        );
+    }
 }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard dc140a55012ab4fd671fc4974f3aed7b0de23c7a
git clean -fd
