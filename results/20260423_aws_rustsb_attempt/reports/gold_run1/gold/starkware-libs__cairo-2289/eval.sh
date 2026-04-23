#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/crates/cairo-lang-parser/src/parser_test.rs b/crates/cairo-lang-parser/src/parser_test.rs
--- a/crates/cairo-lang-parser/src/parser_test.rs
+++ b/crates/cairo-lang-parser/src/parser_test.rs
@@ -62,14 +62,30 @@ const TEST_test2_tree_with_trivia: ParserTreeTestParams = ParserTreeTestParams {
     print_colors: false,
     print_trivia: true,
 };
+const TEST_test3_tree_no_trivia: ParserTreeTestParams = ParserTreeTestParams {
+    cairo_filename: "test_data/cairo_files/test3.cairo",
+    expected_output_filename: "test_data/expected_results/test3_tree_no_trivia",
+    print_diagnostics: true,
+    print_colors: false,
+    print_trivia: false,
+};
+const TEST_test3_tree_with_trivia: ParserTreeTestParams = ParserTreeTestParams {
+    cairo_filename: "test_data/cairo_files/test3.cairo",
+    expected_output_filename: "test_data/expected_results/test3_tree_with_trivia",
+    print_diagnostics: false,
+    print_colors: false,
+    print_trivia: true,
+};
 #[cfg(feature = "fix_parser_tests")]
-static TREE_TEST_CASES: [&ParserTreeTestParams; 6] = [
+static TREE_TEST_CASES: [&ParserTreeTestParams; 8] = [
     &TEST_short_tree_uncolored,
     &TEST_short_tree_colored,
     &TEST_test1_tree_no_trivia,
     &TEST_test1_tree_with_trivia,
     &TEST_test2_tree_no_trivia,
     &TEST_test2_tree_with_trivia,
+    &TEST_test3_tree_no_trivia,
+    &TEST_test3_tree_with_trivia,
 ];
 
 /// Parse the cairo file, print it, and compare with the expected result.
diff --git a/crates/cairo-lang-parser/src/parser_test.rs b/crates/cairo-lang-parser/src/parser_test.rs
--- a/crates/cairo-lang-parser/src/parser_test.rs
+++ b/crates/cairo-lang-parser/src/parser_test.rs
@@ -79,6 +95,8 @@ static TREE_TEST_CASES: [&ParserTreeTestParams; 6] = [
 #[test_case(&TEST_test1_tree_with_trivia; "test1_tree_with_trivia")]
 #[test_case(&TEST_test2_tree_no_trivia; "test2_tree_no_trivia")]
 #[test_case(&TEST_test2_tree_with_trivia; "test2_tree_with_trivia")]
+#[test_case(&TEST_test3_tree_no_trivia; "test3_tree_no_trivia")]
+#[test_case(&TEST_test3_tree_with_trivia; "test3_tree_with_trivia")]
 fn parse_and_compare_tree(test_params: &ParserTreeTestParams) {
     parse_and_compare_tree_maybe_fix(test_params, false);
 }
diff --git a/crates/cairo-lang-parser/src/parser_test.rs b/crates/cairo-lang-parser/src/parser_test.rs
--- a/crates/cairo-lang-parser/src/parser_test.rs
+++ b/crates/cairo-lang-parser/src/parser_test.rs
@@ -327,6 +345,7 @@ cairo_lang_test_utils::test_file_test!(
     "src/parser_test_data",
     {
         path: "path_with_trivia",
+        path_compat: "path_with_trivia_compat",
     },
     test_partial_parser_tree_with_trivia
 );
diff --git a/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia b/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
--- a/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
+++ b/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
@@ -1,4 +1,4 @@
-//! > Missing :: in path.
+//! > Test typed path without ::
 
 //! > test_runner_name
 test_partial_parser_tree_with_trivia
diff --git a/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia b/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
--- a/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
+++ b/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
@@ -12,20 +12,6 @@ FunctionSignature
 //! > ignored_kinds
 
 //! > expected_diagnostics
-error: Missing token TerminalComma.
- --> dummy_file.cairo:1:17
-fn foo(a: Option<felt>) {}
-                ^
-
-error: Skipped tokens. Expected: parameter.
- --> dummy_file.cairo:1:17
-fn foo(a: Option<felt>) {}
-                ^
-
-error: Unexpected token, expected ':' followed by a type.
- --> dummy_file.cairo:1:22
-fn foo(a: Option<felt>) {}
-                     ^
 
 //! > expected_tree
 └── Top level kind: FunctionSignature
diff --git a/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia b/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
--- a/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
+++ b/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia
@@ -34,47 +20,46 @@ fn foo(a: Option<felt>) {}
     │   ├── token (kind: TokenLParen): '('
     │   └── trailing_trivia (kind: Trivia) []
     ├── parameters (kind: ParamList)
-    │   ├── item #0 (kind: Param)
-    │   │   ├── modifiers (kind: ModifierList) []
-    │   │   ├── name (kind: TerminalIdentifier)
-    │   │   │   ├── leading_trivia (kind: Trivia) []
-    │   │   │   ├── token (kind: TokenIdentifier): 'a'
-    │   │   │   └── trailing_trivia (kind: Trivia) []
-    │   │   └── type_clause (kind: TypeClause)
-    │   │       ├── colon (kind: TerminalColon)
-    │   │       │   ├── leading_trivia (kind: Trivia) []
-    │   │       │   ├── token (kind: TokenColon): ':'
-    │   │       │   └── trailing_trivia (kind: Trivia)
-    │   │       │       └── child #0 (kind: TokenWhitespace).
-    │   │       └── ty (kind: ExprPath)
-    │   │           └── item #0 (kind: PathSegmentSimple)
-    │   │               └── ident (kind: TerminalIdentifier)
-    │   │                   ├── leading_trivia (kind: Trivia) []
-    │   │                   ├── token (kind: TokenIdentifier): 'Option'
-    │   │                   └── trailing_trivia (kind: Trivia) []
-    │   ├── separator #0 (kind: TerminalComma)
-    │   │   ├── leading_trivia (kind: Trivia) []
-    │   │   ├── token: Missing
-    │   │   └── trailing_trivia (kind: Trivia) []
-    │   └── item #1 (kind: Param)
+    │   └── item #0 (kind: Param)
     │       ├── modifiers (kind: ModifierList) []
     │       ├── name (kind: TerminalIdentifier)
-    │       │   ├── leading_trivia (kind: Trivia)
-    │       │   │   └── child #0 (kind: TokenSkipped): '<'
-    │       │   ├── token (kind: TokenIdentifier): 'felt'
+    │       │   ├── leading_trivia (kind: Trivia) []
+    │       │   ├── token (kind: TokenIdentifier): 'a'
     │       │   └── trailing_trivia (kind: Trivia) []
     │       └── type_clause (kind: TypeClause)
     │           ├── colon (kind: TerminalColon)
     │           │   ├── leading_trivia (kind: Trivia) []
-    │           │   ├── token: Missing
-    │           │   └── trailing_trivia (kind: Trivia) []
-    │           └── ty: Missing []
+    │           │   ├── token (kind: TokenColon): ':'
+    │           │   └── trailing_trivia (kind: Trivia)
+    │           │       └── child #0 (kind: TokenWhitespace).
+    │           └── ty (kind: ExprPath)
+    │               └── item #0 (kind: PathSegmentWithGenericArgs)
+    │                   ├── ident (kind: TerminalIdentifier)
+    │                   │   ├── leading_trivia (kind: Trivia) []
+    │                   │   ├── token (kind: TokenIdentifier): 'Option'
+    │                   │   └── trailing_trivia (kind: Trivia) []
+    │                   ├── separator (kind: OptionTerminalColonColonEmpty) []
+    │                   └── generic_args (kind: GenericArgs)
+    │                       ├── langle (kind: TerminalLT)
+    │                       │   ├── leading_trivia (kind: Trivia) []
+    │                       │   ├── token (kind: TokenLT): '<'
+    │                       │   └── trailing_trivia (kind: Trivia) []
+    │                       ├── generic_args (kind: GenericArgList)
+    │                       │   └── item #0 (kind: ExprPath)
+    │                       │       └── item #0 (kind: PathSegmentSimple)
+    │                       │           └── ident (kind: TerminalIdentifier)
+    │                       │               ├── leading_trivia (kind: Trivia) []
+    │                       │               ├── token (kind: TokenIdentifier): 'felt'
+    │                       │               └── trailing_trivia (kind: Trivia) []
+    │                       └── rangle (kind: TerminalGT)
+    │                           ├── leading_trivia (kind: Trivia) []
+    │                           ├── token (kind: TokenGT): '>'
+    │                           └── trailing_trivia (kind: Trivia) []
     ├── rparen (kind: TerminalRParen)
-    │   ├── leading_trivia (kind: Trivia)
-    │   │   └── child #0 (kind: TokenSkipped): '>'
+    │   ├── leading_trivia (kind: Trivia) []
     │   ├── token (kind: TokenRParen): ')'
     │   └── trailing_trivia (kind: Trivia)
     │       └── child #0 (kind: TokenWhitespace).
     ├── ret_ty (kind: OptionReturnTypeClauseEmpty) []
     ├── implicits_clause (kind: OptionImplicitsClauseEmpty) []
-    └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
+    └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
\ No newline at end of file
diff --git /dev/null b/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia_compat
new file mode 100644
--- /dev/null
+++ b/crates/cairo-lang-parser/src/parser_test_data/path_with_trivia_compat
@@ -0,0 +1,68 @@
+//! > Test typed path with ::
+
+//! > test_runner_name
+test_partial_parser_tree_with_trivia
+
+//! > cairo_code
+fn foo(a: Option::<felt>) {}
+
+//! > top_level_kind
+FunctionSignature
+
+//! > ignored_kinds
+
+//! > expected_diagnostics
+
+//! > expected_tree
+└── Top level kind: FunctionSignature
+    ├── lparen (kind: TerminalLParen)
+    │   ├── leading_trivia (kind: Trivia) []
+    │   ├── token (kind: TokenLParen): '('
+    │   └── trailing_trivia (kind: Trivia) []
+    ├── parameters (kind: ParamList)
+    │   └── item #0 (kind: Param)
+    │       ├── modifiers (kind: ModifierList) []
+    │       ├── name (kind: TerminalIdentifier)
+    │       │   ├── leading_trivia (kind: Trivia) []
+    │       │   ├── token (kind: TokenIdentifier): 'a'
+    │       │   └── trailing_trivia (kind: Trivia) []
+    │       └── type_clause (kind: TypeClause)
+    │           ├── colon (kind: TerminalColon)
+    │           │   ├── leading_trivia (kind: Trivia) []
+    │           │   ├── token (kind: TokenColon): ':'
+    │           │   └── trailing_trivia (kind: Trivia)
+    │           │       └── child #0 (kind: TokenWhitespace).
+    │           └── ty (kind: ExprPath)
+    │               └── item #0 (kind: PathSegmentWithGenericArgs)
+    │                   ├── ident (kind: TerminalIdentifier)
+    │                   │   ├── leading_trivia (kind: Trivia) []
+    │                   │   ├── token (kind: TokenIdentifier): 'Option'
+    │                   │   └── trailing_trivia (kind: Trivia) []
+    │                   ├── separator (kind: TerminalColonColon)
+    │                   │   ├── leading_trivia (kind: Trivia) []
+    │                   │   ├── token (kind: TokenColonColon): '::'
+    │                   │   └── trailing_trivia (kind: Trivia) []
+    │                   └── generic_args (kind: GenericArgs)
+    │                       ├── langle (kind: TerminalLT)
+    │                       │   ├── leading_trivia (kind: Trivia) []
+    │                       │   ├── token (kind: TokenLT): '<'
+    │                       │   └── trailing_trivia (kind: Trivia) []
+    │                       ├── generic_args (kind: GenericArgList)
+    │                       │   └── item #0 (kind: ExprPath)
+    │                       │       └── item #0 (kind: PathSegmentSimple)
+    │                       │           └── ident (kind: TerminalIdentifier)
+    │                       │               ├── leading_trivia (kind: Trivia) []
+    │                       │               ├── token (kind: TokenIdentifier): 'felt'
+    │                       │               └── trailing_trivia (kind: Trivia) []
+    │                       └── rangle (kind: TerminalGT)
+    │                           ├── leading_trivia (kind: Trivia) []
+    │                           ├── token (kind: TokenGT): '>'
+    │                           └── trailing_trivia (kind: Trivia) []
+    ├── rparen (kind: TerminalRParen)
+    │   ├── leading_trivia (kind: Trivia) []
+    │   ├── token (kind: TokenRParen): ')'
+    │   └── trailing_trivia (kind: Trivia)
+    │       └── child #0 (kind: TokenWhitespace).
+    ├── ret_ty (kind: OptionReturnTypeClauseEmpty) []
+    ├── implicits_clause (kind: OptionImplicitsClauseEmpty) []
+    └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
\ No newline at end of file
diff --git /dev/null b/crates/cairo-lang-parser/test_data/cairo_files/test3.cairo
new file mode 100644
--- /dev/null
+++ b/crates/cairo-lang-parser/test_data/cairo_files/test3.cairo
@@ -0,0 +1,12 @@
+fn main() -> Option<felt> {
+    fib(1, 1, 13)
+}
+
+/// Calculates fib...
+fn fib(a: felt, b: felt, n: felt) -> Option<felt> {
+    get_gas()?;
+    match n {
+        0 => Option::<felt>::Some(a),
+        _ => fib(b, a + b, n - 1),
+    }
+}
diff --git a/crates/cairo-lang-parser/test_data/expected_results/test1_tree_no_trivia b/crates/cairo-lang-parser/test_data/expected_results/test1_tree_no_trivia
--- a/crates/cairo-lang-parser/test_data/expected_results/test1_tree_no_trivia
+++ b/crates/cairo-lang-parser/test_data/expected_results/test1_tree_no_trivia
@@ -456,11 +456,19 @@
     │   │   │       │       ├── item #0 (kind: PathSegmentSimple)
     │   │   │       │       │   └── ident (kind: TokenIdentifier): 'crate'
     │   │   │       │       ├── separator #0 (kind: TokenColonColon): '::'
-    │   │   │       │       └── item #1 (kind: PathSegmentSimple)
-    │   │   │       │           └── ident (kind: TokenIdentifier): 'S'
+    │   │   │       │       └── item #1 (kind: PathSegmentWithGenericArgs)
+    │   │   │       │           ├── ident (kind: TokenIdentifier): 'S'
+    │   │   │       │           ├── separator (kind: OptionTerminalColonColonEmpty) []
+    │   │   │       │           └── generic_args (kind: GenericArgs)
+    │   │   │       │               ├── langle (kind: TokenLT): '<'
+    │   │   │       │               ├── generic_args (kind: GenericArgList)
+    │   │   │       │               │   └── item #0 (kind: ExprPath)
+    │   │   │       │               │       └── item #0 (kind: PathSegmentSimple)
+    │   │   │       │               │           └── ident (kind: TokenIdentifier): 'int'
+    │   │   │       │               └── rangle (kind: TokenGT): '>'
     │   │   │       ├── implicits_clause (kind: OptionImplicitsClauseEmpty) []
-    │   │   │       └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
-    │   │   └── semicolon: Missing
+    │   │   │       └── optional_no_panic (kind: TokenNoPanic): 'nopanic'
+    │   │   └── semicolon (kind: TokenSemicolon): ';'
     │   ├── child #6 (kind: ItemStruct)
     │   │   ├── attributes (kind: AttributeList) []
     │   │   ├── struct_kw (kind: TokenStruct): 'struct'
diff --git a/crates/cairo-lang-parser/test_data/expected_results/test1_tree_no_trivia b/crates/cairo-lang-parser/test_data/expected_results/test1_tree_no_trivia
--- a/crates/cairo-lang-parser/test_data/expected_results/test1_tree_no_trivia
+++ b/crates/cairo-lang-parser/test_data/expected_results/test1_tree_no_trivia
@@ -572,33 +580,3 @@ error: Missing token TerminalRBrace.
     return x;
              ^
 
-error: Missing token TerminalSemicolon.
- --> test1.cairo:35:45
-extern fn glee<A, b>(var1: int,) -> crate::S<int> nopanic;
-                                            ^
-
-error: Skipped tokens. Expected: Module/Use/FreeFunction/ExternFunction/ExternType/Trait/Impl/Struct/Enum or an attribute.
- --> test1.cairo:35:45
-extern fn glee<A, b>(var1: int,) -> crate::S<int> nopanic;
-                                            ^
-
-error: Skipped tokens. Expected: Module/Use/FreeFunction/ExternFunction/ExternType/Trait/Impl/Struct/Enum or an attribute.
- --> test1.cairo:35:46
-extern fn glee<A, b>(var1: int,) -> crate::S<int> nopanic;
-                                             ^*^
-
-error: Skipped tokens. Expected: Module/Use/FreeFunction/ExternFunction/ExternType/Trait/Impl/Struct/Enum or an attribute.
- --> test1.cairo:35:49
-extern fn glee<A, b>(var1: int,) -> crate::S<int> nopanic;
-                                                ^
-
-error: Skipped tokens. Expected: Module/Use/FreeFunction/ExternFunction/ExternType/Trait/Impl/Struct/Enum or an attribute.
- --> test1.cairo:35:51
-extern fn glee<A, b>(var1: int,) -> crate::S<int> nopanic;
-                                                  ^*****^
-
-error: Skipped tokens. Expected: Module/Use/FreeFunction/ExternFunction/ExternType/Trait/Impl/Struct/Enum or an attribute.
- --> test1.cairo:35:58
-extern fn glee<A, b>(var1: int,) -> crate::S<int> nopanic;
-                                                         ^
-
diff --git a/crates/cairo-lang-parser/test_data/expected_results/test1_tree_with_trivia b/crates/cairo-lang-parser/test_data/expected_results/test1_tree_with_trivia
--- a/crates/cairo-lang-parser/test_data/expected_results/test1_tree_with_trivia
+++ b/crates/cairo-lang-parser/test_data/expected_results/test1_tree_with_trivia
@@ -1263,29 +1263,44 @@
     │   │   │       │       │   ├── leading_trivia (kind: Trivia) []
     │   │   │       │       │   ├── token (kind: TokenColonColon): '::'
     │   │   │       │       │   └── trailing_trivia (kind: Trivia) []
-    │   │   │       │       └── item #1 (kind: PathSegmentSimple)
-    │   │   │       │           └── ident (kind: TerminalIdentifier)
-    │   │   │       │               ├── leading_trivia (kind: Trivia) []
-    │   │   │       │               ├── token (kind: TokenIdentifier): 'S'
-    │   │   │       │               └── trailing_trivia (kind: Trivia) []
+    │   │   │       │       └── item #1 (kind: PathSegmentWithGenericArgs)
+    │   │   │       │           ├── ident (kind: TerminalIdentifier)
+    │   │   │       │           │   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │           │   ├── token (kind: TokenIdentifier): 'S'
+    │   │   │       │           │   └── trailing_trivia (kind: Trivia) []
+    │   │   │       │           ├── separator (kind: OptionTerminalColonColonEmpty) []
+    │   │   │       │           └── generic_args (kind: GenericArgs)
+    │   │   │       │               ├── langle (kind: TerminalLT)
+    │   │   │       │               │   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │               │   ├── token (kind: TokenLT): '<'
+    │   │   │       │               │   └── trailing_trivia (kind: Trivia) []
+    │   │   │       │               ├── generic_args (kind: GenericArgList)
+    │   │   │       │               │   └── item #0 (kind: ExprPath)
+    │   │   │       │               │       └── item #0 (kind: PathSegmentSimple)
+    │   │   │       │               │           └── ident (kind: TerminalIdentifier)
+    │   │   │       │               │               ├── leading_trivia (kind: Trivia) []
+    │   │   │       │               │               ├── token (kind: TokenIdentifier): 'int'
+    │   │   │       │               │               └── trailing_trivia (kind: Trivia) []
+    │   │   │       │               └── rangle (kind: TerminalGT)
+    │   │   │       │                   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │                   ├── token (kind: TokenGT): '>'
+    │   │   │       │                   └── trailing_trivia (kind: Trivia)
+    │   │   │       │                       └── child #0 (kind: TokenWhitespace).
     │   │   │       ├── implicits_clause (kind: OptionImplicitsClauseEmpty) []
-    │   │   │       └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
+    │   │   │       └── optional_no_panic (kind: TerminalNoPanic)
+    │   │   │           ├── leading_trivia (kind: Trivia) []
+    │   │   │           ├── token (kind: TokenNoPanic): 'nopanic'
+    │   │   │           └── trailing_trivia (kind: Trivia) []
     │   │   └── semicolon (kind: TerminalSemicolon)
     │   │       ├── leading_trivia (kind: Trivia) []
-    │   │       ├── token: Missing
-    │   │       └── trailing_trivia (kind: Trivia) []
+    │   │       ├── token (kind: TokenSemicolon): ';'
+    │   │       └── trailing_trivia (kind: Trivia)
+    │   │           └── child #0 (kind: TokenNewline).
     │   ├── child #6 (kind: ItemStruct)
     │   │   ├── attributes (kind: AttributeList) []
     │   │   ├── struct_kw (kind: TerminalStruct)
     │   │   │   ├── leading_trivia (kind: Trivia)
-    │   │   │   │   ├── child #0 (kind: TokenSkipped): '<'
-    │   │   │   │   ├── child #1 (kind: TokenSkipped): 'int'
-    │   │   │   │   ├── child #2 (kind: TokenSkipped): '>'
-    │   │   │   │   ├── child #3 (kind: TokenWhitespace).
-    │   │   │   │   ├── child #4 (kind: TokenSkipped): 'nopanic'
-    │   │   │   │   ├── child #5 (kind: TokenSkipped): ';'
-    │   │   │   │   ├── child #6 (kind: TokenNewline).
-    │   │   │   │   └── child #7 (kind: TokenNewline).
+    │   │   │   │   └── child #0 (kind: TokenNewline).
     │   │   │   ├── token (kind: TokenStruct): 'struct'
     │   │   │   └── trailing_trivia (kind: Trivia)
     │   │   │       └── child #0 (kind: TokenWhitespace).
diff --git /dev/null b/crates/cairo-lang-parser/test_data/expected_results/test3_tree_no_trivia
new file mode 100644
--- /dev/null
+++ b/crates/cairo-lang-parser/test_data/expected_results/test3_tree_no_trivia
@@ -0,0 +1,203 @@
+└── root (kind: SyntaxFile)
+    ├── items (kind: ItemList)
+    │   ├── child #0 (kind: FunctionWithBody)
+    │   │   ├── attributes (kind: AttributeList) []
+    │   │   ├── declaration (kind: FunctionDeclaration)
+    │   │   │   ├── function_kw (kind: TokenFunction): 'fn'
+    │   │   │   ├── name (kind: TokenIdentifier): 'main'
+    │   │   │   ├── generic_params (kind: OptionWrappedGenericParamListEmpty) []
+    │   │   │   └── signature (kind: FunctionSignature)
+    │   │   │       ├── lparen (kind: TokenLParen): '('
+    │   │   │       ├── parameters (kind: ParamList) []
+    │   │   │       ├── rparen (kind: TokenRParen): ')'
+    │   │   │       ├── ret_ty (kind: ReturnTypeClause)
+    │   │   │       │   ├── arrow (kind: TokenArrow): '->'
+    │   │   │       │   └── ty (kind: ExprPath)
+    │   │   │       │       └── item #0 (kind: PathSegmentWithGenericArgs)
+    │   │   │       │           ├── ident (kind: TokenIdentifier): 'Option'
+    │   │   │       │           ├── separator (kind: OptionTerminalColonColonEmpty) []
+    │   │   │       │           └── generic_args (kind: GenericArgs)
+    │   │   │       │               ├── langle (kind: TokenLT): '<'
+    │   │   │       │               ├── generic_args (kind: GenericArgList)
+    │   │   │       │               │   └── item #0 (kind: ExprPath)
+    │   │   │       │               │       └── item #0 (kind: PathSegmentSimple)
+    │   │   │       │               │           └── ident (kind: TokenIdentifier): 'felt'
+    │   │   │       │               └── rangle (kind: TokenGT): '>'
+    │   │   │       ├── implicits_clause (kind: OptionImplicitsClauseEmpty) []
+    │   │   │       └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
+    │   │   └── body (kind: ExprBlock)
+    │   │       ├── lbrace (kind: TokenLBrace): '{'
+    │   │       ├── statements (kind: StatementList)
+    │   │       │   └── child #0 (kind: StatementExpr)
+    │   │       │       ├── expr (kind: ExprFunctionCall)
+    │   │       │       │   ├── path (kind: ExprPath)
+    │   │       │       │   │   └── item #0 (kind: PathSegmentSimple)
+    │   │       │       │   │       └── ident (kind: TokenIdentifier): 'fib'
+    │   │       │       │   └── arguments (kind: ArgListParenthesized)
+    │   │       │       │       ├── lparen (kind: TokenLParen): '('
+    │   │       │       │       ├── args (kind: ArgList)
+    │   │       │       │       │   ├── item #0 (kind: Arg)
+    │   │       │       │       │   │   ├── modifiers (kind: ModifierList) []
+    │   │       │       │       │   │   └── arg_clause (kind: ArgClauseUnnamed)
+    │   │       │       │       │   │       └── value (kind: TokenLiteralNumber): '1'
+    │   │       │       │       │   ├── separator #0 (kind: TokenComma): ','
+    │   │       │       │       │   ├── item #1 (kind: Arg)
+    │   │       │       │       │   │   ├── modifiers (kind: ModifierList) []
+    │   │       │       │       │   │   └── arg_clause (kind: ArgClauseUnnamed)
+    │   │       │       │       │   │       └── value (kind: TokenLiteralNumber): '1'
+    │   │       │       │       │   ├── separator #1 (kind: TokenComma): ','
+    │   │       │       │       │   └── item #2 (kind: Arg)
+    │   │       │       │       │       ├── modifiers (kind: ModifierList) []
+    │   │       │       │       │       └── arg_clause (kind: ArgClauseUnnamed)
+    │   │       │       │       │           └── value (kind: TokenLiteralNumber): '13'
+    │   │       │       │       └── rparen (kind: TokenRParen): ')'
+    │   │       │       └── semicolon (kind: OptionTerminalSemicolonEmpty) []
+    │   │       └── rbrace (kind: TokenRBrace): '}'
+    │   └── child #1 (kind: FunctionWithBody)
+    │       ├── attributes (kind: AttributeList) []
+    │       ├── declaration (kind: FunctionDeclaration)
+    │       │   ├── function_kw (kind: TokenFunction): 'fn'
+    │       │   ├── name (kind: TokenIdentifier): 'fib'
+    │       │   ├── generic_params (kind: OptionWrappedGenericParamListEmpty) []
+    │       │   └── signature (kind: FunctionSignature)
+    │       │       ├── lparen (kind: TokenLParen): '('
+    │       │       ├── parameters (kind: ParamList)
+    │       │       │   ├── item #0 (kind: Param)
+    │       │       │   │   ├── modifiers (kind: ModifierList) []
+    │       │       │   │   ├── name (kind: TokenIdentifier): 'a'
+    │       │       │   │   └── type_clause (kind: TypeClause)
+    │       │       │   │       ├── colon (kind: TokenColon): ':'
+    │       │       │   │       └── ty (kind: ExprPath)
+    │       │       │   │           └── item #0 (kind: PathSegmentSimple)
+    │       │       │   │               └── ident (kind: TokenIdentifier): 'felt'
+    │       │       │   ├── separator #0 (kind: TokenComma): ','
+    │       │       │   ├── item #1 (kind: Param)
+    │       │       │   │   ├── modifiers (kind: ModifierList) []
+    │       │       │   │   ├── name (kind: TokenIdentifier): 'b'
+    │       │       │   │   └── type_clause (kind: TypeClause)
+    │       │       │   │       ├── colon (kind: TokenColon): ':'
+    │       │       │   │       └── ty (kind: ExprPath)
+    │       │       │   │           └── item #0 (kind: PathSegmentSimple)
+    │       │       │   │               └── ident (kind: TokenIdentifier): 'felt'
+    │       │       │   ├── separator #1 (kind: TokenComma): ','
+    │       │       │   └── item #2 (kind: Param)
+    │       │       │       ├── modifiers (kind: ModifierList) []
+    │       │       │       ├── name (kind: TokenIdentifier): 'n'
+    │       │       │       └── type_clause (kind: TypeClause)
+    │       │       │           ├── colon (kind: TokenColon): ':'
+    │       │       │           └── ty (kind: ExprPath)
+    │       │       │               └── item #0 (kind: PathSegmentSimple)
+    │       │       │                   └── ident (kind: TokenIdentifier): 'felt'
+    │       │       ├── rparen (kind: TokenRParen): ')'
+    │       │       ├── ret_ty (kind: ReturnTypeClause)
+    │       │       │   ├── arrow (kind: TokenArrow): '->'
+    │       │       │   └── ty (kind: ExprPath)
+    │       │       │       └── item #0 (kind: PathSegmentWithGenericArgs)
+    │       │       │           ├── ident (kind: TokenIdentifier): 'Option'
+    │       │       │           ├── separator (kind: OptionTerminalColonColonEmpty) []
+    │       │       │           └── generic_args (kind: GenericArgs)
+    │       │       │               ├── langle (kind: TokenLT): '<'
+    │       │       │               ├── generic_args (kind: GenericArgList)
+    │       │       │               │   └── item #0 (kind: ExprPath)
+    │       │       │               │       └── item #0 (kind: PathSegmentSimple)
+    │       │       │               │           └── ident (kind: TokenIdentifier): 'felt'
+    │       │       │               └── rangle (kind: TokenGT): '>'
+    │       │       ├── implicits_clause (kind: OptionImplicitsClauseEmpty) []
+    │       │       └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
+    │       └── body (kind: ExprBlock)
+    │           ├── lbrace (kind: TokenLBrace): '{'
+    │           ├── statements (kind: StatementList)
+    │           │   ├── child #0 (kind: StatementExpr)
+    │           │   │   ├── expr (kind: ExprErrorPropagate)
+    │           │   │   │   ├── expr (kind: ExprFunctionCall)
+    │           │   │   │   │   ├── path (kind: ExprPath)
+    │           │   │   │   │   │   └── item #0 (kind: PathSegmentSimple)
+    │           │   │   │   │   │       └── ident (kind: TokenIdentifier): 'get_gas'
+    │           │   │   │   │   └── arguments (kind: ArgListParenthesized)
+    │           │   │   │   │       ├── lparen (kind: TokenLParen): '('
+    │           │   │   │   │       ├── args (kind: ArgList) []
+    │           │   │   │   │       └── rparen (kind: TokenRParen): ')'
+    │           │   │   │   └── op (kind: TokenQuestionMark): '?'
+    │           │   │   └── semicolon (kind: TokenSemicolon): ';'
+    │           │   └── child #1 (kind: StatementExpr)
+    │           │       ├── expr (kind: ExprMatch)
+    │           │       │   ├── match_kw (kind: TokenMatch): 'match'
+    │           │       │   ├── expr (kind: ExprPath)
+    │           │       │   │   └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │       └── ident (kind: TokenIdentifier): 'n'
+    │           │       │   ├── lbrace (kind: TokenLBrace): '{'
+    │           │       │   ├── arms (kind: MatchArms)
+    │           │       │   │   ├── item #0 (kind: MatchArm)
+    │           │       │   │   │   ├── pattern (kind: TokenLiteralNumber): '0'
+    │           │       │   │   │   ├── arrow (kind: TokenMatchArrow): '=>'
+    │           │       │   │   │   └── expression (kind: ExprFunctionCall)
+    │           │       │   │   │       ├── path (kind: ExprPath)
+    │           │       │   │   │       │   ├── item #0 (kind: PathSegmentWithGenericArgs)
+    │           │       │   │   │       │   │   ├── ident (kind: TokenIdentifier): 'Option'
+    │           │       │   │   │       │   │   ├── separator (kind: TokenColonColon): '::'
+    │           │       │   │   │       │   │   └── generic_args (kind: GenericArgs)
+    │           │       │   │   │       │   │       ├── langle (kind: TokenLT): '<'
+    │           │       │   │   │       │   │       ├── generic_args (kind: GenericArgList)
+    │           │       │   │   │       │   │       │   └── item #0 (kind: ExprPath)
+    │           │       │   │   │       │   │       │       └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │       │   │       │           └── ident (kind: TokenIdentifier): 'felt'
+    │           │       │   │   │       │   │       └── rangle (kind: TokenGT): '>'
+    │           │       │   │   │       │   ├── separator #0 (kind: TokenColonColon): '::'
+    │           │       │   │   │       │   └── item #1 (kind: PathSegmentSimple)
+    │           │       │   │   │       │       └── ident (kind: TokenIdentifier): 'Some'
+    │           │       │   │   │       └── arguments (kind: ArgListParenthesized)
+    │           │       │   │   │           ├── lparen (kind: TokenLParen): '('
+    │           │       │   │   │           ├── args (kind: ArgList)
+    │           │       │   │   │           │   └── item #0 (kind: Arg)
+    │           │       │   │   │           │       ├── modifiers (kind: ModifierList) []
+    │           │       │   │   │           │       └── arg_clause (kind: ArgClauseUnnamed)
+    │           │       │   │   │           │           └── value (kind: ExprPath)
+    │           │       │   │   │           │               └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │                   └── ident (kind: TokenIdentifier): 'a'
+    │           │       │   │   │           └── rparen (kind: TokenRParen): ')'
+    │           │       │   │   ├── separator #0 (kind: TokenComma): ','
+    │           │       │   │   ├── item #1 (kind: MatchArm)
+    │           │       │   │   │   ├── pattern (kind: TokenUnderscore): '_'
+    │           │       │   │   │   ├── arrow (kind: TokenMatchArrow): '=>'
+    │           │       │   │   │   └── expression (kind: ExprFunctionCall)
+    │           │       │   │   │       ├── path (kind: ExprPath)
+    │           │       │   │   │       │   └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │       │       └── ident (kind: TokenIdentifier): 'fib'
+    │           │       │   │   │       └── arguments (kind: ArgListParenthesized)
+    │           │       │   │   │           ├── lparen (kind: TokenLParen): '('
+    │           │       │   │   │           ├── args (kind: ArgList)
+    │           │       │   │   │           │   ├── item #0 (kind: Arg)
+    │           │       │   │   │           │   │   ├── modifiers (kind: ModifierList) []
+    │           │       │   │   │           │   │   └── arg_clause (kind: ArgClauseUnnamed)
+    │           │       │   │   │           │   │       └── value (kind: ExprPath)
+    │           │       │   │   │           │   │           └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │   │               └── ident (kind: TokenIdentifier): 'b'
+    │           │       │   │   │           │   ├── separator #0 (kind: TokenComma): ','
+    │           │       │   │   │           │   ├── item #1 (kind: Arg)
+    │           │       │   │   │           │   │   ├── modifiers (kind: ModifierList) []
+    │           │       │   │   │           │   │   └── arg_clause (kind: ArgClauseUnnamed)
+    │           │       │   │   │           │   │       └── value (kind: ExprBinary)
+    │           │       │   │   │           │   │           ├── lhs (kind: ExprPath)
+    │           │       │   │   │           │   │           │   └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │   │           │       └── ident (kind: TokenIdentifier): 'a'
+    │           │       │   │   │           │   │           ├── op (kind: TokenPlus): '+'
+    │           │       │   │   │           │   │           └── rhs (kind: ExprPath)
+    │           │       │   │   │           │   │               └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │   │                   └── ident (kind: TokenIdentifier): 'b'
+    │           │       │   │   │           │   ├── separator #1 (kind: TokenComma): ','
+    │           │       │   │   │           │   └── item #2 (kind: Arg)
+    │           │       │   │   │           │       ├── modifiers (kind: ModifierList) []
+    │           │       │   │   │           │       └── arg_clause (kind: ArgClauseUnnamed)
+    │           │       │   │   │           │           └── value (kind: ExprBinary)
+    │           │       │   │   │           │               ├── lhs (kind: ExprPath)
+    │           │       │   │   │           │               │   └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │               │       └── ident (kind: TokenIdentifier): 'n'
+    │           │       │   │   │           │               ├── op (kind: TokenMinus): '-'
+    │           │       │   │   │           │               └── rhs (kind: TokenLiteralNumber): '1'
+    │           │       │   │   │           └── rparen (kind: TokenRParen): ')'
+    │           │       │   │   └── separator #1 (kind: TokenComma): ','
+    │           │       │   └── rbrace (kind: TokenRBrace): '}'
+    │           │       └── semicolon (kind: OptionTerminalSemicolonEmpty) []
+    │           └── rbrace (kind: TokenRBrace): '}'
+    └── eof (kind: TokenEndOfFile).
+--------------------
diff --git /dev/null b/crates/cairo-lang-parser/test_data/expected_results/test3_tree_with_trivia
new file mode 100644
--- /dev/null
+++ b/crates/cairo-lang-parser/test_data/expected_results/test3_tree_with_trivia
@@ -0,0 +1,485 @@
+└── root (kind: SyntaxFile)
+    ├── items (kind: ItemList)
+    │   ├── child #0 (kind: FunctionWithBody)
+    │   │   ├── attributes (kind: AttributeList) []
+    │   │   ├── declaration (kind: FunctionDeclaration)
+    │   │   │   ├── function_kw (kind: TerminalFunction)
+    │   │   │   │   ├── leading_trivia (kind: Trivia) []
+    │   │   │   │   ├── token (kind: TokenFunction): 'fn'
+    │   │   │   │   └── trailing_trivia (kind: Trivia)
+    │   │   │   │       └── child #0 (kind: TokenWhitespace).
+    │   │   │   ├── name (kind: TerminalIdentifier)
+    │   │   │   │   ├── leading_trivia (kind: Trivia) []
+    │   │   │   │   ├── token (kind: TokenIdentifier): 'main'
+    │   │   │   │   └── trailing_trivia (kind: Trivia) []
+    │   │   │   ├── generic_params (kind: OptionWrappedGenericParamListEmpty) []
+    │   │   │   └── signature (kind: FunctionSignature)
+    │   │   │       ├── lparen (kind: TerminalLParen)
+    │   │   │       │   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │   ├── token (kind: TokenLParen): '('
+    │   │   │       │   └── trailing_trivia (kind: Trivia) []
+    │   │   │       ├── parameters (kind: ParamList) []
+    │   │   │       ├── rparen (kind: TerminalRParen)
+    │   │   │       │   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │   ├── token (kind: TokenRParen): ')'
+    │   │   │       │   └── trailing_trivia (kind: Trivia)
+    │   │   │       │       └── child #0 (kind: TokenWhitespace).
+    │   │   │       ├── ret_ty (kind: ReturnTypeClause)
+    │   │   │       │   ├── arrow (kind: TerminalArrow)
+    │   │   │       │   │   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │   │   ├── token (kind: TokenArrow): '->'
+    │   │   │       │   │   └── trailing_trivia (kind: Trivia)
+    │   │   │       │   │       └── child #0 (kind: TokenWhitespace).
+    │   │   │       │   └── ty (kind: ExprPath)
+    │   │   │       │       └── item #0 (kind: PathSegmentWithGenericArgs)
+    │   │   │       │           ├── ident (kind: TerminalIdentifier)
+    │   │   │       │           │   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │           │   ├── token (kind: TokenIdentifier): 'Option'
+    │   │   │       │           │   └── trailing_trivia (kind: Trivia) []
+    │   │   │       │           ├── separator (kind: OptionTerminalColonColonEmpty) []
+    │   │   │       │           └── generic_args (kind: GenericArgs)
+    │   │   │       │               ├── langle (kind: TerminalLT)
+    │   │   │       │               │   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │               │   ├── token (kind: TokenLT): '<'
+    │   │   │       │               │   └── trailing_trivia (kind: Trivia) []
+    │   │   │       │               ├── generic_args (kind: GenericArgList)
+    │   │   │       │               │   └── item #0 (kind: ExprPath)
+    │   │   │       │               │       └── item #0 (kind: PathSegmentSimple)
+    │   │   │       │               │           └── ident (kind: TerminalIdentifier)
+    │   │   │       │               │               ├── leading_trivia (kind: Trivia) []
+    │   │   │       │               │               ├── token (kind: TokenIdentifier): 'felt'
+    │   │   │       │               │               └── trailing_trivia (kind: Trivia) []
+    │   │   │       │               └── rangle (kind: TerminalGT)
+    │   │   │       │                   ├── leading_trivia (kind: Trivia) []
+    │   │   │       │                   ├── token (kind: TokenGT): '>'
+    │   │   │       │                   └── trailing_trivia (kind: Trivia)
+    │   │   │       │                       └── child #0 (kind: TokenWhitespace).
+    │   │   │       ├── implicits_clause (kind: OptionImplicitsClauseEmpty) []
+    │   │   │       └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
+    │   │   └── body (kind: ExprBlock)
+    │   │       ├── lbrace (kind: TerminalLBrace)
+    │   │       │   ├── leading_trivia (kind: Trivia) []
+    │   │       │   ├── token (kind: TokenLBrace): '{'
+    │   │       │   └── trailing_trivia (kind: Trivia)
+    │   │       │       └── child #0 (kind: TokenNewline).
+    │   │       ├── statements (kind: StatementList)
+    │   │       │   └── child #0 (kind: StatementExpr)
+    │   │       │       ├── expr (kind: ExprFunctionCall)
+    │   │       │       │   ├── path (kind: ExprPath)
+    │   │       │       │   │   └── item #0 (kind: PathSegmentSimple)
+    │   │       │       │   │       └── ident (kind: TerminalIdentifier)
+    │   │       │       │   │           ├── leading_trivia (kind: Trivia)
+    │   │       │       │   │           │   └── child #0 (kind: TokenWhitespace).
+    │   │       │       │   │           ├── token (kind: TokenIdentifier): 'fib'
+    │   │       │       │   │           └── trailing_trivia (kind: Trivia) []
+    │   │       │       │   └── arguments (kind: ArgListParenthesized)
+    │   │       │       │       ├── lparen (kind: TerminalLParen)
+    │   │       │       │       │   ├── leading_trivia (kind: Trivia) []
+    │   │       │       │       │   ├── token (kind: TokenLParen): '('
+    │   │       │       │       │   └── trailing_trivia (kind: Trivia) []
+    │   │       │       │       ├── args (kind: ArgList)
+    │   │       │       │       │   ├── item #0 (kind: Arg)
+    │   │       │       │       │   │   ├── modifiers (kind: ModifierList) []
+    │   │       │       │       │   │   └── arg_clause (kind: ArgClauseUnnamed)
+    │   │       │       │       │   │       └── value (kind: TerminalLiteralNumber)
+    │   │       │       │       │   │           ├── leading_trivia (kind: Trivia) []
+    │   │       │       │       │   │           ├── token (kind: TokenLiteralNumber): '1'
+    │   │       │       │       │   │           └── trailing_trivia (kind: Trivia) []
+    │   │       │       │       │   ├── separator #0 (kind: TerminalComma)
+    │   │       │       │       │   │   ├── leading_trivia (kind: Trivia) []
+    │   │       │       │       │   │   ├── token (kind: TokenComma): ','
+    │   │       │       │       │   │   └── trailing_trivia (kind: Trivia)
+    │   │       │       │       │   │       └── child #0 (kind: TokenWhitespace).
+    │   │       │       │       │   ├── item #1 (kind: Arg)
+    │   │       │       │       │   │   ├── modifiers (kind: ModifierList) []
+    │   │       │       │       │   │   └── arg_clause (kind: ArgClauseUnnamed)
+    │   │       │       │       │   │       └── value (kind: TerminalLiteralNumber)
+    │   │       │       │       │   │           ├── leading_trivia (kind: Trivia) []
+    │   │       │       │       │   │           ├── token (kind: TokenLiteralNumber): '1'
+    │   │       │       │       │   │           └── trailing_trivia (kind: Trivia) []
+    │   │       │       │       │   ├── separator #1 (kind: TerminalComma)
+    │   │       │       │       │   │   ├── leading_trivia (kind: Trivia) []
+    │   │       │       │       │   │   ├── token (kind: TokenComma): ','
+    │   │       │       │       │   │   └── trailing_trivia (kind: Trivia)
+    │   │       │       │       │   │       └── child #0 (kind: TokenWhitespace).
+    │   │       │       │       │   └── item #2 (kind: Arg)
+    │   │       │       │       │       ├── modifiers (kind: ModifierList) []
+    │   │       │       │       │       └── arg_clause (kind: ArgClauseUnnamed)
+    │   │       │       │       │           └── value (kind: TerminalLiteralNumber)
+    │   │       │       │       │               ├── leading_trivia (kind: Trivia) []
+    │   │       │       │       │               ├── token (kind: TokenLiteralNumber): '13'
+    │   │       │       │       │               └── trailing_trivia (kind: Trivia) []
+    │   │       │       │       └── rparen (kind: TerminalRParen)
+    │   │       │       │           ├── leading_trivia (kind: Trivia) []
+    │   │       │       │           ├── token (kind: TokenRParen): ')'
+    │   │       │       │           └── trailing_trivia (kind: Trivia)
+    │   │       │       │               └── child #0 (kind: TokenNewline).
+    │   │       │       └── semicolon (kind: OptionTerminalSemicolonEmpty) []
+    │   │       └── rbrace (kind: TerminalRBrace)
+    │   │           ├── leading_trivia (kind: Trivia) []
+    │   │           ├── token (kind: TokenRBrace): '}'
+    │   │           └── trailing_trivia (kind: Trivia)
+    │   │               └── child #0 (kind: TokenNewline).
+    │   └── child #1 (kind: FunctionWithBody)
+    │       ├── attributes (kind: AttributeList) []
+    │       ├── declaration (kind: FunctionDeclaration)
+    │       │   ├── function_kw (kind: TerminalFunction)
+    │       │   │   ├── leading_trivia (kind: Trivia)
+    │       │   │   │   ├── child #0 (kind: TokenNewline).
+    │       │   │   │   ├── child #1 (kind: TokenSingleLineComment): '/// Calculates fib...'
+    │       │   │   │   └── child #2 (kind: TokenNewline).
+    │       │   │   ├── token (kind: TokenFunction): 'fn'
+    │       │   │   └── trailing_trivia (kind: Trivia)
+    │       │   │       └── child #0 (kind: TokenWhitespace).
+    │       │   ├── name (kind: TerminalIdentifier)
+    │       │   │   ├── leading_trivia (kind: Trivia) []
+    │       │   │   ├── token (kind: TokenIdentifier): 'fib'
+    │       │   │   └── trailing_trivia (kind: Trivia) []
+    │       │   ├── generic_params (kind: OptionWrappedGenericParamListEmpty) []
+    │       │   └── signature (kind: FunctionSignature)
+    │       │       ├── lparen (kind: TerminalLParen)
+    │       │       │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   ├── token (kind: TokenLParen): '('
+    │       │       │   └── trailing_trivia (kind: Trivia) []
+    │       │       ├── parameters (kind: ParamList)
+    │       │       │   ├── item #0 (kind: Param)
+    │       │       │   │   ├── modifiers (kind: ModifierList) []
+    │       │       │   │   ├── name (kind: TerminalIdentifier)
+    │       │       │   │   │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │   │   ├── token (kind: TokenIdentifier): 'a'
+    │       │       │   │   │   └── trailing_trivia (kind: Trivia) []
+    │       │       │   │   └── type_clause (kind: TypeClause)
+    │       │       │   │       ├── colon (kind: TerminalColon)
+    │       │       │   │       │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │       │   ├── token (kind: TokenColon): ':'
+    │       │       │   │       │   └── trailing_trivia (kind: Trivia)
+    │       │       │   │       │       └── child #0 (kind: TokenWhitespace).
+    │       │       │   │       └── ty (kind: ExprPath)
+    │       │       │   │           └── item #0 (kind: PathSegmentSimple)
+    │       │       │   │               └── ident (kind: TerminalIdentifier)
+    │       │       │   │                   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │                   ├── token (kind: TokenIdentifier): 'felt'
+    │       │       │   │                   └── trailing_trivia (kind: Trivia) []
+    │       │       │   ├── separator #0 (kind: TerminalComma)
+    │       │       │   │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │   ├── token (kind: TokenComma): ','
+    │       │       │   │   └── trailing_trivia (kind: Trivia)
+    │       │       │   │       └── child #0 (kind: TokenWhitespace).
+    │       │       │   ├── item #1 (kind: Param)
+    │       │       │   │   ├── modifiers (kind: ModifierList) []
+    │       │       │   │   ├── name (kind: TerminalIdentifier)
+    │       │       │   │   │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │   │   ├── token (kind: TokenIdentifier): 'b'
+    │       │       │   │   │   └── trailing_trivia (kind: Trivia) []
+    │       │       │   │   └── type_clause (kind: TypeClause)
+    │       │       │   │       ├── colon (kind: TerminalColon)
+    │       │       │   │       │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │       │   ├── token (kind: TokenColon): ':'
+    │       │       │   │       │   └── trailing_trivia (kind: Trivia)
+    │       │       │   │       │       └── child #0 (kind: TokenWhitespace).
+    │       │       │   │       └── ty (kind: ExprPath)
+    │       │       │   │           └── item #0 (kind: PathSegmentSimple)
+    │       │       │   │               └── ident (kind: TerminalIdentifier)
+    │       │       │   │                   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │                   ├── token (kind: TokenIdentifier): 'felt'
+    │       │       │   │                   └── trailing_trivia (kind: Trivia) []
+    │       │       │   ├── separator #1 (kind: TerminalComma)
+    │       │       │   │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │   ├── token (kind: TokenComma): ','
+    │       │       │   │   └── trailing_trivia (kind: Trivia)
+    │       │       │   │       └── child #0 (kind: TokenWhitespace).
+    │       │       │   └── item #2 (kind: Param)
+    │       │       │       ├── modifiers (kind: ModifierList) []
+    │       │       │       ├── name (kind: TerminalIdentifier)
+    │       │       │       │   ├── leading_trivia (kind: Trivia) []
+    │       │       │       │   ├── token (kind: TokenIdentifier): 'n'
+    │       │       │       │   └── trailing_trivia (kind: Trivia) []
+    │       │       │       └── type_clause (kind: TypeClause)
+    │       │       │           ├── colon (kind: TerminalColon)
+    │       │       │           │   ├── leading_trivia (kind: Trivia) []
+    │       │       │           │   ├── token (kind: TokenColon): ':'
+    │       │       │           │   └── trailing_trivia (kind: Trivia)
+    │       │       │           │       └── child #0 (kind: TokenWhitespace).
+    │       │       │           └── ty (kind: ExprPath)
+    │       │       │               └── item #0 (kind: PathSegmentSimple)
+    │       │       │                   └── ident (kind: TerminalIdentifier)
+    │       │       │                       ├── leading_trivia (kind: Trivia) []
+    │       │       │                       ├── token (kind: TokenIdentifier): 'felt'
+    │       │       │                       └── trailing_trivia (kind: Trivia) []
+    │       │       ├── rparen (kind: TerminalRParen)
+    │       │       │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   ├── token (kind: TokenRParen): ')'
+    │       │       │   └── trailing_trivia (kind: Trivia)
+    │       │       │       └── child #0 (kind: TokenWhitespace).
+    │       │       ├── ret_ty (kind: ReturnTypeClause)
+    │       │       │   ├── arrow (kind: TerminalArrow)
+    │       │       │   │   ├── leading_trivia (kind: Trivia) []
+    │       │       │   │   ├── token (kind: TokenArrow): '->'
+    │       │       │   │   └── trailing_trivia (kind: Trivia)
+    │       │       │   │       └── child #0 (kind: TokenWhitespace).
+    │       │       │   └── ty (kind: ExprPath)
+    │       │       │       └── item #0 (kind: PathSegmentWithGenericArgs)
+    │       │       │           ├── ident (kind: TerminalIdentifier)
+    │       │       │           │   ├── leading_trivia (kind: Trivia) []
+    │       │       │           │   ├── token (kind: TokenIdentifier): 'Option'
+    │       │       │           │   └── trailing_trivia (kind: Trivia) []
+    │       │       │           ├── separator (kind: OptionTerminalColonColonEmpty) []
+    │       │       │           └── generic_args (kind: GenericArgs)
+    │       │       │               ├── langle (kind: TerminalLT)
+    │       │       │               │   ├── leading_trivia (kind: Trivia) []
+    │       │       │               │   ├── token (kind: TokenLT): '<'
+    │       │       │               │   └── trailing_trivia (kind: Trivia) []
+    │       │       │               ├── generic_args (kind: GenericArgList)
+    │       │       │               │   └── item #0 (kind: ExprPath)
+    │       │       │               │       └── item #0 (kind: PathSegmentSimple)
+    │       │       │               │           └── ident (kind: TerminalIdentifier)
+    │       │       │               │               ├── leading_trivia (kind: Trivia) []
+    │       │       │               │               ├── token (kind: TokenIdentifier): 'felt'
+    │       │       │               │               └── trailing_trivia (kind: Trivia) []
+    │       │       │               └── rangle (kind: TerminalGT)
+    │       │       │                   ├── leading_trivia (kind: Trivia) []
+    │       │       │                   ├── token (kind: TokenGT): '>'
+    │       │       │                   └── trailing_trivia (kind: Trivia)
+    │       │       │                       └── child #0 (kind: TokenWhitespace).
+    │       │       ├── implicits_clause (kind: OptionImplicitsClauseEmpty) []
+    │       │       └── optional_no_panic (kind: OptionTerminalNoPanicEmpty) []
+    │       └── body (kind: ExprBlock)
+    │           ├── lbrace (kind: TerminalLBrace)
+    │           │   ├── leading_trivia (kind: Trivia) []
+    │           │   ├── token (kind: TokenLBrace): '{'
+    │           │   └── trailing_trivia (kind: Trivia)
+    │           │       └── child #0 (kind: TokenNewline).
+    │           ├── statements (kind: StatementList)
+    │           │   ├── child #0 (kind: StatementExpr)
+    │           │   │   ├── expr (kind: ExprErrorPropagate)
+    │           │   │   │   ├── expr (kind: ExprFunctionCall)
+    │           │   │   │   │   ├── path (kind: ExprPath)
+    │           │   │   │   │   │   └── item #0 (kind: PathSegmentSimple)
+    │           │   │   │   │   │       └── ident (kind: TerminalIdentifier)
+    │           │   │   │   │   │           ├── leading_trivia (kind: Trivia)
+    │           │   │   │   │   │           │   └── child #0 (kind: TokenWhitespace).
+    │           │   │   │   │   │           ├── token (kind: TokenIdentifier): 'get_gas'
+    │           │   │   │   │   │           └── trailing_trivia (kind: Trivia) []
+    │           │   │   │   │   └── arguments (kind: ArgListParenthesized)
+    │           │   │   │   │       ├── lparen (kind: TerminalLParen)
+    │           │   │   │   │       │   ├── leading_trivia (kind: Trivia) []
+    │           │   │   │   │       │   ├── token (kind: TokenLParen): '('
+    │           │   │   │   │       │   └── trailing_trivia (kind: Trivia) []
+    │           │   │   │   │       ├── args (kind: ArgList) []
+    │           │   │   │   │       └── rparen (kind: TerminalRParen)
+    │           │   │   │   │           ├── leading_trivia (kind: Trivia) []
+    │           │   │   │   │           ├── token (kind: TokenRParen): ')'
+    │           │   │   │   │           └── trailing_trivia (kind: Trivia) []
+    │           │   │   │   └── op (kind: TerminalQuestionMark)
+    │           │   │   │       ├── leading_trivia (kind: Trivia) []
+    │           │   │   │       ├── token (kind: TokenQuestionMark): '?'
+    │           │   │   │       └── trailing_trivia (kind: Trivia) []
+    │           │   │   └── semicolon (kind: TerminalSemicolon)
+    │           │   │       ├── leading_trivia (kind: Trivia) []
+    │           │   │       ├── token (kind: TokenSemicolon): ';'
+    │           │   │       └── trailing_trivia (kind: Trivia)
+    │           │   │           └── child #0 (kind: TokenNewline).
+    │           │   └── child #1 (kind: StatementExpr)
+    │           │       ├── expr (kind: ExprMatch)
+    │           │       │   ├── match_kw (kind: TerminalMatch)
+    │           │       │   │   ├── leading_trivia (kind: Trivia)
+    │           │       │   │   │   └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   ├── token (kind: TokenMatch): 'match'
+    │           │       │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   ├── expr (kind: ExprPath)
+    │           │       │   │   └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │       └── ident (kind: TerminalIdentifier)
+    │           │       │   │           ├── leading_trivia (kind: Trivia) []
+    │           │       │   │           ├── token (kind: TokenIdentifier): 'n'
+    │           │       │   │           └── trailing_trivia (kind: Trivia)
+    │           │       │   │               └── child #0 (kind: TokenWhitespace).
+    │           │       │   ├── lbrace (kind: TerminalLBrace)
+    │           │       │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   ├── token (kind: TokenLBrace): '{'
+    │           │       │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │       └── child #0 (kind: TokenNewline).
+    │           │       │   ├── arms (kind: MatchArms)
+    │           │       │   │   ├── item #0 (kind: MatchArm)
+    │           │       │   │   │   ├── pattern (kind: TerminalLiteralNumber)
+    │           │       │   │   │   │   ├── leading_trivia (kind: Trivia)
+    │           │       │   │   │   │   │   └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │   │   ├── token (kind: TokenLiteralNumber): '0'
+    │           │       │   │   │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │   │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │   ├── arrow (kind: TerminalMatchArrow)
+    │           │       │   │   │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │   │   ├── token (kind: TokenMatchArrow): '=>'
+    │           │       │   │   │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │   │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │   └── expression (kind: ExprFunctionCall)
+    │           │       │   │   │       ├── path (kind: ExprPath)
+    │           │       │   │   │       │   ├── item #0 (kind: PathSegmentWithGenericArgs)
+    │           │       │   │   │       │   │   ├── ident (kind: TerminalIdentifier)
+    │           │       │   │   │       │   │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │   │   ├── token (kind: TokenIdentifier): 'Option'
+    │           │       │   │   │       │   │   │   └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │   ├── separator (kind: TerminalColonColon)
+    │           │       │   │   │       │   │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │   │   ├── token (kind: TokenColonColon): '::'
+    │           │       │   │   │       │   │   │   └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │   └── generic_args (kind: GenericArgs)
+    │           │       │   │   │       │   │       ├── langle (kind: TerminalLT)
+    │           │       │   │   │       │   │       │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │       │   ├── token (kind: TokenLT): '<'
+    │           │       │   │   │       │   │       │   └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │       ├── generic_args (kind: GenericArgList)
+    │           │       │   │   │       │   │       │   └── item #0 (kind: ExprPath)
+    │           │       │   │   │       │   │       │       └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │       │   │       │           └── ident (kind: TerminalIdentifier)
+    │           │       │   │   │       │   │       │               ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │       │               ├── token (kind: TokenIdentifier): 'felt'
+    │           │       │   │   │       │   │       │               └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │       └── rangle (kind: TerminalGT)
+    │           │       │   │   │       │   │           ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │           ├── token (kind: TokenGT): '>'
+    │           │       │   │   │       │   │           └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │       │   ├── separator #0 (kind: TerminalColonColon)
+    │           │       │   │   │       │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │       │   │   ├── token (kind: TokenColonColon): '::'
+    │           │       │   │   │       │   │   └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │       │   └── item #1 (kind: PathSegmentSimple)
+    │           │       │   │   │       │       └── ident (kind: TerminalIdentifier)
+    │           │       │   │   │       │           ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │       │           ├── token (kind: TokenIdentifier): 'Some'
+    │           │       │   │   │       │           └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │       └── arguments (kind: ArgListParenthesized)
+    │           │       │   │   │           ├── lparen (kind: TerminalLParen)
+    │           │       │   │   │           │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │   ├── token (kind: TokenLParen): '('
+    │           │       │   │   │           │   └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │           ├── args (kind: ArgList)
+    │           │       │   │   │           │   └── item #0 (kind: Arg)
+    │           │       │   │   │           │       ├── modifiers (kind: ModifierList) []
+    │           │       │   │   │           │       └── arg_clause (kind: ArgClauseUnnamed)
+    │           │       │   │   │           │           └── value (kind: ExprPath)
+    │           │       │   │   │           │               └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │                   └── ident (kind: TerminalIdentifier)
+    │           │       │   │   │           │                       ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │                       ├── token (kind: TokenIdentifier): 'a'
+    │           │       │   │   │           │                       └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │           └── rparen (kind: TerminalRParen)
+    │           │       │   │   │               ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │               ├── token (kind: TokenRParen): ')'
+    │           │       │   │   │               └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   ├── separator #0 (kind: TerminalComma)
+    │           │       │   │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │   ├── token (kind: TokenComma): ','
+    │           │       │   │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │       └── child #0 (kind: TokenNewline).
+    │           │       │   │   ├── item #1 (kind: MatchArm)
+    │           │       │   │   │   ├── pattern (kind: TerminalUnderscore)
+    │           │       │   │   │   │   ├── leading_trivia (kind: Trivia)
+    │           │       │   │   │   │   │   └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │   │   ├── token (kind: TokenUnderscore): '_'
+    │           │       │   │   │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │   │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │   ├── arrow (kind: TerminalMatchArrow)
+    │           │       │   │   │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │   │   ├── token (kind: TokenMatchArrow): '=>'
+    │           │       │   │   │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │   │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │   └── expression (kind: ExprFunctionCall)
+    │           │       │   │   │       ├── path (kind: ExprPath)
+    │           │       │   │   │       │   └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │       │       └── ident (kind: TerminalIdentifier)
+    │           │       │   │   │       │           ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │       │           ├── token (kind: TokenIdentifier): 'fib'
+    │           │       │   │   │       │           └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │       └── arguments (kind: ArgListParenthesized)
+    │           │       │   │   │           ├── lparen (kind: TerminalLParen)
+    │           │       │   │   │           │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │   ├── token (kind: TokenLParen): '('
+    │           │       │   │   │           │   └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │           ├── args (kind: ArgList)
+    │           │       │   │   │           │   ├── item #0 (kind: Arg)
+    │           │       │   │   │           │   │   ├── modifiers (kind: ModifierList) []
+    │           │       │   │   │           │   │   └── arg_clause (kind: ArgClauseUnnamed)
+    │           │       │   │   │           │   │       └── value (kind: ExprPath)
+    │           │       │   │   │           │   │           └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │   │               └── ident (kind: TerminalIdentifier)
+    │           │       │   │   │           │   │                   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │   │                   ├── token (kind: TokenIdentifier): 'b'
+    │           │       │   │   │           │   │                   └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │           │   ├── separator #0 (kind: TerminalComma)
+    │           │       │   │   │           │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │   │   ├── token (kind: TokenComma): ','
+    │           │       │   │   │           │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │           │   │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │           │   ├── item #1 (kind: Arg)
+    │           │       │   │   │           │   │   ├── modifiers (kind: ModifierList) []
+    │           │       │   │   │           │   │   └── arg_clause (kind: ArgClauseUnnamed)
+    │           │       │   │   │           │   │       └── value (kind: ExprBinary)
+    │           │       │   │   │           │   │           ├── lhs (kind: ExprPath)
+    │           │       │   │   │           │   │           │   └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │   │           │       └── ident (kind: TerminalIdentifier)
+    │           │       │   │   │           │   │           │           ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │   │           │           ├── token (kind: TokenIdentifier): 'a'
+    │           │       │   │   │           │   │           │           └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │           │   │           │               └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │           │   │           ├── op (kind: TerminalPlus)
+    │           │       │   │   │           │   │           │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │   │           │   ├── token (kind: TokenPlus): '+'
+    │           │       │   │   │           │   │           │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │           │   │           │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │           │   │           └── rhs (kind: ExprPath)
+    │           │       │   │   │           │   │               └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │   │                   └── ident (kind: TerminalIdentifier)
+    │           │       │   │   │           │   │                       ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │   │                       ├── token (kind: TokenIdentifier): 'b'
+    │           │       │   │   │           │   │                       └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │           │   ├── separator #1 (kind: TerminalComma)
+    │           │       │   │   │           │   │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │   │   ├── token (kind: TokenComma): ','
+    │           │       │   │   │           │   │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │           │   │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │           │   └── item #2 (kind: Arg)
+    │           │       │   │   │           │       ├── modifiers (kind: ModifierList) []
+    │           │       │   │   │           │       └── arg_clause (kind: ArgClauseUnnamed)
+    │           │       │   │   │           │           └── value (kind: ExprBinary)
+    │           │       │   │   │           │               ├── lhs (kind: ExprPath)
+    │           │       │   │   │           │               │   └── item #0 (kind: PathSegmentSimple)
+    │           │       │   │   │           │               │       └── ident (kind: TerminalIdentifier)
+    │           │       │   │   │           │               │           ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │               │           ├── token (kind: TokenIdentifier): 'n'
+    │           │       │   │   │           │               │           └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │           │               │               └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │           │               ├── op (kind: TerminalMinus)
+    │           │       │   │   │           │               │   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │               │   ├── token (kind: TokenMinus): '-'
+    │           │       │   │   │           │               │   └── trailing_trivia (kind: Trivia)
+    │           │       │   │   │           │               │       └── child #0 (kind: TokenWhitespace).
+    │           │       │   │   │           │               └── rhs (kind: TerminalLiteralNumber)
+    │           │       │   │   │           │                   ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │           │                   ├── token (kind: TokenLiteralNumber): '1'
+    │           │       │   │   │           │                   └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   │           └── rparen (kind: TerminalRParen)
+    │           │       │   │   │               ├── leading_trivia (kind: Trivia) []
+    │           │       │   │   │               ├── token (kind: TokenRParen): ')'
+    │           │       │   │   │               └── trailing_trivia (kind: Trivia) []
+    │           │       │   │   └── separator #1 (kind: TerminalComma)
+    │           │       │   │       ├── leading_trivia (kind: Trivia) []
+    │           │       │   │       ├── token (kind: TokenComma): ','
+    │           │       │   │       └── trailing_trivia (kind: Trivia)
+    │           │       │   │           └── child #0 (kind: TokenNewline).
+    │           │       │   └── rbrace (kind: TerminalRBrace)
+    │           │       │       ├── leading_trivia (kind: Trivia)
+    │           │       │       │   └── child #0 (kind: TokenWhitespace).
+    │           │       │       ├── token (kind: TokenRBrace): '}'
+    │           │       │       └── trailing_trivia (kind: Trivia)
+    │           │       │           └── child #0 (kind: TokenNewline).
+    │           │       └── semicolon (kind: OptionTerminalSemicolonEmpty) []
+    │           └── rbrace (kind: TerminalRBrace)
+    │               ├── leading_trivia (kind: Trivia) []
+    │               ├── token (kind: TokenRBrace): '}'
+    │               └── trailing_trivia (kind: Trivia)
+    │                   └── child #0 (kind: TokenNewline).
+    └── eof (kind: TerminalEndOfFile)
+        ├── leading_trivia (kind: Trivia) []
+        ├── token (kind: TokenEndOfFile).
+        └── trailing_trivia (kind: Trivia) []
diff --git a/crates/cairo-lang-semantic/src/diagnostic_test_data/tests b/crates/cairo-lang-semantic/src/diagnostic_test_data/tests
--- a/crates/cairo-lang-semantic/src/diagnostic_test_data/tests
+++ b/crates/cairo-lang-semantic/src/diagnostic_test_data/tests
@@ -211,43 +211,3 @@ error: Cycle detected while resolving 'use' items.
  --> lib.cairo:1:5
 use self;
     ^**^
-
-//! > ==========================================================================
-
-//! > Test missing `::`
-
-//! > test_runner_name
-test_expr_diagnostics
-
-//! > expr_code
-{
-}
-
-//! > module_code
-fn foo(a: Box<u256>) -> u128 {
-  let val: u256 = unbox(a);
-  val.high
-}
-
-//! > function_body
-
-//! > expected_diagnostics
-error: Missing token TerminalComma.
- --> lib.cairo:1:14
-fn foo(a: Box<u256>) -> u128 {
-             ^
-
-error: Skipped tokens. Expected: parameter.
- --> lib.cairo:1:14
-fn foo(a: Box<u256>) -> u128 {
-             ^
-
-error: Unexpected token, expected ':' followed by a type.
- --> lib.cairo:1:19
-fn foo(a: Box<u256>) -> u128 {
-                  ^
-
-error: Unknown type.
- --> lib.cairo:1:19
-fn foo(a: Box<u256>) -> u128 {
-                  ^

EOF_114329324912
cd "crates/cairo-lang-parser"
cargo test --no-fail-fast --all-features
cd ../../
git reset --hard 8f433e752a78afc017d0f08c9e21b966acfe1c11
git clean -fd
