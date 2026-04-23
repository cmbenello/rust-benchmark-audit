#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/data.rs b/src/data.rs
--- a/src/data.rs
+++ b/src/data.rs
@@ -205,15 +211,13 @@ mod tests {
         let agg = Aggregate::new(
             &["kc1".to_string(), "kc2".to_string()],
             "count".to_string(),
-            &[
-                (
-                    hashmap!{
-                        "kc1".to_string() => "k1".to_string(),
-                        "kc2".to_string() => "k2".to_string()
-                    },
-                    Value::Int(100),
-                ),
-            ],
+            &[(
+                hashmap! {
+                    "kc1".to_string() => "k1".to_string(),
+                    "kc2".to_string() => "k2".to_string()
+                },
+                Value::Int(100),
+            )],
         );
         assert_eq!(agg.data.len(), 1);
     }
diff --git a/src/data.rs b/src/data.rs
--- a/src/data.rs
+++ b/src/data.rs
@@ -224,14 +228,12 @@ mod tests {
         Aggregate::new(
             &["k1".to_string(), "k2".to_string()],
             "count".to_string(),
-            &[
-                (
-                    hashmap!{
-                        "kc2".to_string() => "k2".to_string()
-                    },
-                    Value::Int(100),
-                ),
-            ],
+            &[(
+                hashmap! {
+                    "kc2".to_string() => "k2".to_string()
+                },
+                Value::Int(100),
+            )],
         );
     }
 
diff --git a/src/lang.rs b/src/lang.rs
--- a/src/lang.rs
+++ b/src/lang.rs
@@ -657,9 +675,10 @@ mod tests {
                                 right: Box::new(Expr::Value(data::Value::Int(123))),
                             },
                         ],
-                        aggregate_functions: vec![
-                            ("_count".to_string(), AggregateFunction::Count {}),
-                        ],
+                        aggregate_functions: vec![(
+                            "_count".to_string(),
+                            AggregateFunction::Count {}
+                        ),],
                     }),
                     Operator::Sort(SortOperator {
                         sort_cols: vec!["foo".to_string()],
diff --git a/src/operator.rs b/src/operator.rs
--- a/src/operator.rs
+++ b/src/operator.rs
@@ -721,8 +812,8 @@ impl UnaryPreAggOperator for ParseJson {
 #[cfg(test)]
 mod tests {
     use super::*;
-    use crate::lang;
     use crate::data::Value;
+    use crate::lang;
     //use crate::operator::itertools::Itertools;
 
     impl From<String> for Expr {
diff --git a/src/operator.rs b/src/operator.rs
--- a/src/operator.rs
+++ b/src/operator.rs
@@ -802,7 +893,8 @@ mod tests {
                 "length".to_string(),
             ],
             None,
-        ).unwrap();
+        )
+        .unwrap();
         let rec = parser.process(rec).unwrap().unwrap();
         assert_eq!(
             rec.data.get("sender").unwrap(),
diff --git a/src/operator.rs b/src/operator.rs
--- a/src/operator.rs
+++ b/src/operator.rs
@@ -823,7 +915,8 @@ mod tests {
             lang::Keyword::new_wildcard("[*=*]".to_string()).to_regex(),
             vec!["key".to_string(), "value".to_string()],
             Some("from_col".to_string()),
-        ).unwrap();
+        )
+        .unwrap();
         let rec = parser.process(rec).unwrap().unwrap();
         assert_eq!(
             rec.data.get("key").unwrap(),
diff --git a/src/operator.rs b/src/operator.rs
--- a/src/operator.rs
+++ b/src/operator.rs
@@ -1037,4 +1130,40 @@ mod tests {
         let _: () = adapted.process(Row::Aggregate(agg.clone()));
         assert_eq!(adapted.emit(), agg.clone());
     }
+
+    #[test]
+    fn test_total() {
+        let mut total_op = Total::new(Expr::Column("count".to_string()), "_total".to_string());
+        let agg = Aggregate::new(
+            &["kc1".to_string(), "kc2".to_string()],
+            "count".to_string(),
+            &[
+                (
+                    hashmap! {
+                        "kc1".to_string() => "k1".to_string(),
+                        "kc2".to_string() => "k2".to_string()
+                    },
+                    Value::Int(100),
+                ),
+                (
+                    hashmap! {
+                        "kc1".to_string() => "k300".to_string(),
+                        "kc2".to_string() => "k40000".to_string()
+                    },
+                    Value::Int(500),
+                ),
+            ],
+        );
+        total_op.process(Row::Aggregate(agg.clone()));
+        let result = total_op.emit().data;
+        assert_eq!(result[0].get("_total").unwrap(), &Value::from_float(100.0));
+        assert_eq!(result[1].get("_total").unwrap(), &Value::from_float(600.0));
+        assert_eq!(result.len(), 2);
+        total_op.process(Row::Aggregate(agg.clone()));
+        let result = total_op.emit().data;
+        assert_eq!(result[0].get("_total").unwrap(), &Value::from_float(100.0));
+        assert_eq!(result[1].get("_total").unwrap(), &Value::from_float(600.0));
+        assert_eq!(result.len(), 2);
+        //assert_eq!(, agg.clone());
+    }
 }
diff --git a/src/render.rs b/src/render.rs
--- a/src/render.rs
+++ b/src/render.rs
@@ -322,14 +324,14 @@ mod tests {
             "count".to_string(),
             &[
                 (
-                    hashmap!{
+                    hashmap! {
                         "kc1".to_string() => "k1".to_string(),
                         "kc2".to_string() => "k2".to_string()
                     },
                     Value::Int(100),
                 ),
                 (
-                    hashmap!{
+                    hashmap! {
                         "kc1".to_string() => "k300".to_string(),
                         "kc2".to_string() => "k40000".to_string()
                     },
diff --git a/tests/integration.rs b/tests/integration.rs
--- a/tests/integration.rs
+++ b/tests/integration.rs
@@ -21,8 +21,8 @@ mod integration {
     use super::*;
     use ag::pipeline::Pipeline;
     use assert_cli;
-    use toml;
     use std::borrow::Borrow;
+    use toml;
 
     fn structured_test(s: &str) {
         let conf: TestDefinition = toml::from_str(s).unwrap();
diff --git a/tests/integration.rs b/tests/integration.rs
--- a/tests/integration.rs
+++ b/tests/integration.rs
@@ -57,6 +57,11 @@ mod integration {
         structured_test(include_str!("structured_tests/sort_order.toml"));
     }
 
+    #[test]
+    fn total() {
+        structured_test(include_str!("structured_tests/total.toml"));
+    }
+
     #[test]
     fn no_args() {
         assert_cli::Assert::main_binary()
diff --git a/tests/integration.rs b/tests/integration.rs
--- a/tests/integration.rs
+++ b/tests/integration.rs
@@ -185,20 +190,12 @@ $None$       1")
     #[test]
     fn filter_wildcard() {
         assert_cli::Assert::main_binary()
-            .with_args(&[
-                r#""*STAR*""#,
-                "--file",
-                "test_files/filter_test.log",
-            ])
+            .with_args(&[r#""*STAR*""#, "--file", "test_files/filter_test.log"])
             .stdout()
             .is("[INFO] Match a *STAR*!")
             .unwrap();
         assert_cli::Assert::main_binary()
-            .with_args(&[
-                r#"*STAR*"#,
-                "--file",
-                "test_files/filter_test.log",
-            ])
+            .with_args(&[r#"*STAR*"#, "--file", "test_files/filter_test.log"])
             .stdout()
             .is("[INFO] Match a *STAR*!
 [INFO] Not a STAR!")
diff --git a/tests/integration.rs b/tests/integration.rs
--- a/tests/integration.rs
+++ b/tests/integration.rs
@@ -210,6 +207,7 @@ $None$       1")
             "Query: `{}` from the README should have parsed",
             query
         ));
+        println!("validated {}", query);
     }
 
     #[test]
diff --git /dev/null b/tests/structured_tests/total.toml
new file mode 100644
--- /dev/null
+++ b/tests/structured_tests/total.toml
@@ -0,0 +1,24 @@
+query = "* | json | total(num_things)"
+input = """
+{"level": "info", "message": "A thing happened", "num_things": 1.5}
+{"level": "info", "message": "A thing happened", "num_things": 1103}
+{"level": "info", "message": "A thing happened", "num_things": 1105}
+{"level": "info", "message": "A thing happened", "num_things": "not_a_number"}
+{"level": "info", "message": "A thing happened"}
+{"level": "info", "message": "A thing happened", "num_things": 1105}
+{"level": "info", "message": "A thing happened", "num_things": 1105}
+{"level": "info", "message": "A thing happened", "num_things": 1105}
+{"level": "info", "message": "A thing happened", "num_things": 1105.5}
+"""
+output = """_total        level        message                 num_things
+---------------------------------------------------------------------
+1.50          info         A thing happened        1.50
+1104.50       info         A thing happened        1103
+2209.50       info         A thing happened        1105
+2209.50       info         A thing happened        not_a_number
+2209.50       info         A thing happened        $None$
+3314.50       info         A thing happened        1105
+4419.50       info         A thing happened        1105
+5524.50       info         A thing happened        1105
+6630          info         A thing happened        1105.50
+"""
diff --git a/tests/structured_tests/where-3.toml b/tests/structured_tests/where-3.toml
--- a/tests/structured_tests/where-3.toml
+++ b/tests/structured_tests/where-3.toml
@@ -11,7 +11,7 @@ input = """
 """
 output = """
 [thing_a=6]              [thing_b=5]
-[thing_a=1.00]           [thing_b=0.10]
+[thing_a=1]              [thing_b=0.10]
 [thing_a=hello]          [thing_b=goodbye]
 [thing_a=true]           [thing_b=false]
 """

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard c3a24bd096813cb9d2fe2c051ec01024182e1134
git clean -fd
