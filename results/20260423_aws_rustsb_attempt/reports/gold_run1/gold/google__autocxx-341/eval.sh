#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/engine/src/conversion/parse/parse_foreign_mod.rs b/engine/src/conversion/parse/parse_foreign_mod.rs
--- a/engine/src/conversion/parse/parse_foreign_mod.rs
+++ b/engine/src/conversion/parse/parse_foreign_mod.rs
@@ -123,3 +126,34 @@ impl ParseForeignMod {
         }
     }
 }
+
+/// bindgen sometimes generates an impl fn called a which calls
+/// a function called a1(), if it's dealing with conflicting names.
+/// We actually care about the name a1, so we have to parse the
+/// name of the actual function call inside the block's body.
+fn get_called_function(block: &Block) -> Option<&Ident> {
+    match block.stmts.first() {
+        Some(Stmt::Expr(Expr::Call(ExprCall { func, .. }))) => match **func {
+            Expr::Path(ref exp) => exp.path.segments.first().map(|ps| &ps.ident),
+            _ => None,
+        },
+        _ => None,
+    }
+}
+
+#[cfg(test)]
+mod test {
+    use super::get_called_function;
+    use syn::parse_quote;
+    use syn::Block;
+
+    #[test]
+    fn test_get_called_function() {
+        let b: Block = parse_quote! {
+            {
+                call_foo()
+            }
+        };
+        assert_eq!(get_called_function(&b).unwrap().to_string(), "call_foo");
+    }
+}
diff --git a/engine/src/integration_tests.rs b/engine/src/integration_tests.rs
--- a/engine/src/integration_tests.rs
+++ b/engine/src/integration_tests.rs
@@ -2510,18 +2510,6 @@ fn test_give_nonpod_typedef_by_value() {
 }
 
 #[test]
-#[ignore] // bindgen creates functions called create and create1,
-          // then impl blocks for Bob and Fred which call them from methods each
-          // called simply 'create'. Because of this mismatch we have no way to know
-          // that 'create1' is a static function, and this fails. We _could_ parse
-          // the contents of the impl block methods in order to find out what is
-          // actually being called, but currently bindgen in fact gets this wrong,
-          // and calls 'create' from both of them. The first step to getting this
-          // working would therefore be to make a minimal test case and file it
-          // against bindgen. TODO. Then after that was fixed we could look at
-          // whether it's worth fixing this by parsing the contents of the impl'ed
-          // methods.
-          // https://github.com/google/autocxx/issues/331
 fn test_conflicting_static_functions() {
     let cxx = indoc! {"
         Bob Bob::create() { Bob a; return a; }

EOF_114329324912
cd "engine"
cargo test --no-fail-fast --all-features
cd ../
git reset --hard 7aaf7b65498a73d2cdc12a81390739e091710bab
git clean -fd
