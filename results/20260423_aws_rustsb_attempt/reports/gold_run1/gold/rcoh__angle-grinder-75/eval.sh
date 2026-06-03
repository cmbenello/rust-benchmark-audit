#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/src/lang.rs b/src/lang.rs
--- a/src/lang.rs
+++ b/src/lang.rs
@@ -939,6 +950,15 @@ mod tests {
                 value: InlineOperator::Json { input_column: None }
             })
         );
+        expect!(
+            operator,
+            "  logfmt",
+            Operator::Inline(Positioned {
+                start_pos: QueryPosition(2),
+                end_pos: QueryPosition(8),
+                value: InlineOperator::Logfmt { input_column: None }
+            })
+        );
         expect!(
             operator,
             r#" parse "[key=*]" from field as v "#,
diff --git a/src/lang.rs b/src/lang.rs
--- a/src/lang.rs
+++ b/src/lang.rs
@@ -1219,4 +1239,29 @@ mod tests {
             }
         );
     }
+
+    #[test]
+    fn logfmt_operator() {
+        let query_str = r#"* | logfmt from col | sort by foo dsc "#;
+        expect!(
+            query,
+            query_str,
+            Query {
+                search: Search::And(vec![]),
+                operators: vec![
+                    Operator::Inline(Positioned {
+                        start_pos: QueryPosition(4),
+                        end_pos: QueryPosition(20),
+                        value: InlineOperator::Logfmt {
+                            input_column: Some("col".to_string()),
+                        }
+                    }),
+                    Operator::Sort(SortOperator {
+                        sort_cols: vec!["foo".to_string()],
+                        direction: SortMode::Descending,
+                    }),
+                ],
+            }
+        );
+    }
 }
diff --git a/src/operator.rs b/src/operator.rs
--- a/src/operator.rs
+++ b/src/operator.rs
@@ -1072,6 +1104,22 @@ mod tests {
         );
     }
 
+    #[test]
+    fn logfmt() {
+        let rec = Record::new(&(r#"k1=5 k2=5.5 k3="a str" k4="#.to_string() + "\n"));
+        let parser = ParseLogfmt::new(None);
+        let rec = parser.process(rec).unwrap().unwrap();
+        assert_eq!(
+            rec.data,
+            hashmap! {
+                "k1".to_string() => Value::Int(5),
+                "k2".to_string() => Value::from_float(5.5),
+                "k3".to_string() => Value::Str("a str".to_string()),
+                "k4".to_string() => Value::Str("".to_string())
+            }
+        );
+    }
+
     #[test]
     fn fields_only() {
         let rec = Record::new("");
diff --git /dev/null b/test_files/test_logfmt.log
new file mode 100644
--- /dev/null
+++ b/test_files/test_logfmt.log
@@ -0,0 +1,3 @@
+level=info msg="Stopping all fetchers" tag=stopping_fetchers id=ConsumerFetcherManager-1382721708341 module=kafka.consumer.ConsumerFetcherManager
+level=info msg="Starting all fetchers" tag=starting_fetchers id=ConsumerFetcherManager-1382721708342 module=kafka.consumer.ConsumerFetcherManager
+level=warn msg="Fetcher failed to start" tag=errored_fetchers id=ConsumerFetcherManager-1382721708342 module=kafka.consumer.ConsumerFetcherManager
diff --git /dev/null b/test_files/test_nested_logfmt.log
new file mode 100644
--- /dev/null
+++ b/test_files/test_nested_logfmt.log
@@ -0,0 +1,1 @@
+{"key": "blah", "nested_key": "some=logfmt data=more"}
diff --git a/tests/integration.rs b/tests/integration.rs
--- a/tests/integration.rs
+++ b/tests/integration.rs
@@ -74,6 +74,11 @@ mod integration {
         structured_test(include_str!("structured_tests/parse_nodrop.toml"));
     }
 
+    #[test]
+    fn logfmt_operator() {
+        structured_test(include_str!("structured_tests/logfmt.toml"));
+    }
+
     #[test]
     fn sum_operator() {
         structured_test(include_str!("structured_tests/sum.toml"));
diff --git /dev/null b/tests/structured_tests/logfmt.toml
new file mode 100644
--- /dev/null
+++ b/tests/structured_tests/logfmt.toml
@@ -0,0 +1,11 @@
+query = """* | logfmt | fields thing_a, thing_b"""
+input = """
+thing_a=5 thing_b=red
+thing_a=6 thing_b=yellow
+thing_a=7 thing_b=blue
+"""
+output = """
+[thing_a=5]              [thing_b=red]
+[thing_a=6]              [thing_b=yellow]
+[thing_a=7]              [thing_b=blue]
+"""

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 4c67c65113ddb7d3975a94e5bbfe71b19ac74abb
git clean -fd
