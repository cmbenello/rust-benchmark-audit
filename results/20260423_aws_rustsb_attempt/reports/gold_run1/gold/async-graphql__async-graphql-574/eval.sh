#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/look_ahead.rs b/src/look_ahead.rs
--- a/src/look_ahead.rs
+++ b/src/look_ahead.rs
@@ -45,34 +48,36 @@ impl<'a> From<SelectionField<'a>> for Lookahead<'a> {
     fn from(selection_field: SelectionField<'a>) -> Self {
         Lookahead {
             fragments: selection_field.fragments,
-            field: Some(selection_field.field),
+            fields: vec![selection_field.field],
         }
     }
 }
 
-fn find<'a>(
+fn filter<'a>(
+    fields: &mut Vec<&'a Field>,
     fragments: &'a HashMap<Name, Positioned<FragmentDefinition>>,
     selection_set: &'a SelectionSet,
     name: &str,
-) -> Option<&'a Field> {
-    selection_set
-        .items
-        .iter()
-        .find_map(|item| match &item.node {
+) {
+    for item in &selection_set.items {
+        // doing this imperatively is a bit nasty, but using iterators would
+        // require a boxed return type (I believe) as its recusive
+        match &item.node {
             Selection::Field(field) => {
                 if field.node.name.node == name {
-                    Some(&field.node)
-                } else {
-                    None
+                    fields.push(&field.node)
                 }
             }
             Selection::InlineFragment(fragment) => {
-                find(fragments, &fragment.node.selection_set.node, name)
+                filter(fields, fragments, &fragment.node.selection_set.node, name)
+            }
+            Selection::FragmentSpread(spread) => {
+                if let Some(fragment) = fragments.get(&spread.node.fragment_name.node) {
+                    filter(fields, fragments, &fragment.node.selection_set.node, name)
+                }
             }
-            Selection::FragmentSpread(spread) => fragments
-                .get(&spread.node.fragment_name.node)
-                .and_then(|fragment| find(fragments, &fragment.node.selection_set.node, name)),
-        })
+        }
+    }
 }
 
 #[cfg(test)]
diff --git a/src/look_ahead.rs b/src/look_ahead.rs
--- a/src/look_ahead.rs
+++ b/src/look_ahead.rs
@@ -104,12 +109,17 @@ mod tests {
                 if ctx.look_ahead().field("a").exists() {
                     // This is a query like `obj { a }`
                     assert_eq!(n, 1);
-                } else if ctx.look_ahead().field("detail").field("c").exists() {
+                } else if ctx.look_ahead().field("detail").field("c").exists()
+                    && ctx.look_ahead().field("detail").field("d").exists()
+                {
                     // This is a query like `obj { detail { c } }`
                     assert_eq!(n, 2);
+                } else if ctx.look_ahead().field("detail").field("c").exists() {
+                    // This is a query like `obj { detail { c } }`
+                    assert_eq!(n, 3);
                 } else {
                     // This query doesn't have `a`
-                    assert_eq!(n, 3);
+                    assert_eq!(n, 4);
                 }
                 MyObj {
                     a: 0,
diff --git a/src/look_ahead.rs b/src/look_ahead.rs
--- a/src/look_ahead.rs
+++ b/src/look_ahead.rs
@@ -143,10 +153,27 @@ mod tests {
             .await
             .is_ok());
 
+        assert!(schema
+            .execute(
+                r#"{
+            obj(n: 3) {
+                detail {
+                    c
+                }
+            }
+        }"#,
+            )
+            .await
+            .is_ok());
+
         assert!(schema
             .execute(
                 r#"{
             obj(n: 2) {
+                detail {
+                    d
+                }
+
                 detail {
                     c
                 }
diff --git a/src/look_ahead.rs b/src/look_ahead.rs
--- a/src/look_ahead.rs
+++ b/src/look_ahead.rs
@@ -159,7 +186,7 @@ mod tests {
         assert!(schema
             .execute(
                 r#"{
-            obj(n: 3) {
+            obj(n: 4) {
                 b
             }
         }"#,
diff --git a/src/look_ahead.rs b/src/look_ahead.rs
--- a/src/look_ahead.rs
+++ b/src/look_ahead.rs
@@ -180,11 +207,30 @@ mod tests {
             .await
             .is_ok());
 
+        assert!(schema
+            .execute(
+                r#"{
+            obj(n: 3) {
+                ... {
+                    detail {
+                        c
+                    }
+                }
+            }
+        }"#,
+            )
+            .await
+            .is_ok());
+
         assert!(schema
             .execute(
                 r#"{
             obj(n: 2) {
                 ... {
+                    detail {
+                        d
+                    }
+
                     detail {
                         c
                     }
diff --git a/src/look_ahead.rs b/src/look_ahead.rs
--- a/src/look_ahead.rs
+++ b/src/look_ahead.rs
@@ -210,15 +256,39 @@ mod tests {
             .await
             .is_ok());
 
+        assert!(schema
+            .execute(
+                r#"{
+            obj(n: 3) {
+                ... A
+            }
+        }
+        
+        fragment A on MyObj {
+            detail {
+                c
+            }
+        }"#,
+            )
+            .await
+            .is_ok());
+
         assert!(schema
             .execute(
                 r#"{
             obj(n: 2) {
                 ... A
+                ... B
             }
         }
         
         fragment A on MyObj {
+            detail {
+                d
+            }
+        }
+        
+        fragment B on MyObj {
             detail {
                 c
             }

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard c1f651254ec0c92cc632ae86c2ac2060ba8f678d
git clean -fd
