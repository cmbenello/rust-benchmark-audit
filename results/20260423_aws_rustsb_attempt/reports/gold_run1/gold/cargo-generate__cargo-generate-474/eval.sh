#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/hooks/variable_mod.rs b/src/hooks/variable_mod.rs
--- a/src/hooks/variable_mod.rs
+++ b/src/hooks/variable_mod.rs
@@ -203,3 +220,61 @@ impl GetNamedValue for Rc<RefCell<Object>> {
         }
     }
 }
+
+fn rhai_to_liquid_value(val: Dynamic) -> Result<Value> {
+    val.as_bool()
+        .map(Into::into)
+        .map(Value::Scalar)
+        .or_else(|_| val.clone().into_string().map(Into::into).map(Value::Scalar))
+        .or_else(|_| {
+            val.clone()
+                .try_cast::<Array>()
+                .ok_or_else(|| {
+                    format!(
+                        "expecting type to be string, bool or array but found a '{}' instead",
+                        val.type_name()
+                    )
+                    .into()
+                })
+                .and_then(|arr| {
+                    arr.into_iter()
+                        .map(rhai_to_liquid_value)
+                        .collect::<Result<_>>()
+                        .map(Value::Array)
+                })
+        })
+}
+
+#[cfg(test)]
+mod tests {
+    use super::*;
+
+    #[test]
+    fn test_rhai_set() {
+        let mut engine = rhai::Engine::new();
+        let liquid_object = Rc::new(RefCell::new(liquid::Object::new()));
+
+        let module = create_module(liquid_object.clone());
+        engine.register_static_module("variable", module.into());
+
+        engine
+            .eval::<()>(
+                r#"
+            let dependencies = ["some_dep", "other_dep"];
+
+            variable::set("dependencies", dependencies);
+        "#,
+            )
+            .unwrap();
+
+        let liquid_object = liquid_object.borrow();
+
+        assert_eq!(
+            liquid_object.get("dependencies"),
+            Some(&Value::Array(vec![
+                Value::Scalar("some_dep".into()),
+                Value::Scalar("other_dep".into())
+            ]))
+        );
+    }
+}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 86bd44b49a0b9acf0fa09bdbdd49db7fcc88abf8
git clean -fd
