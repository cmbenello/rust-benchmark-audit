#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git /dev/null b/tests/expectations/both/opaque.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/opaque.c
@@ -0,0 +1,28 @@
+#ifdef __cplusplus
+// These could be added as opaque types I guess.
+template <typename T>
+struct BuildHasherDefault;
+
+struct DefaultHasher;
+#endif
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct HashMap_i32__i32__BuildHasherDefault_DefaultHasher HashMap_i32__i32__BuildHasherDefault_DefaultHasher;
+
+typedef struct Result_Foo Result_Foo;
+
+/**
+ * Fast hash map used internally.
+ */
+typedef HashMap_i32__i32__BuildHasherDefault_DefaultHasher FastHashMap_i32__i32;
+
+typedef FastHashMap_i32__i32 Foo;
+
+typedef Result_Foo Bar;
+
+void root(const Foo *a, const Bar *b);
diff --git /dev/null b/tests/expectations/both/opaque.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/both/opaque.compat.c
@@ -0,0 +1,36 @@
+#ifdef __cplusplus
+// These could be added as opaque types I guess.
+template <typename T>
+struct BuildHasherDefault;
+
+struct DefaultHasher;
+#endif
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct HashMap_i32__i32__BuildHasherDefault_DefaultHasher HashMap_i32__i32__BuildHasherDefault_DefaultHasher;
+
+typedef struct Result_Foo Result_Foo;
+
+/**
+ * Fast hash map used internally.
+ */
+typedef HashMap_i32__i32__BuildHasherDefault_DefaultHasher FastHashMap_i32__i32;
+
+typedef FastHashMap_i32__i32 Foo;
+
+typedef Result_Foo Bar;
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(const Foo *a, const Bar *b);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git a/tests/expectations/cell.cpp b/tests/expectations/cell.cpp
--- a/tests/expectations/cell.cpp
+++ b/tests/expectations/cell.cpp
@@ -3,10 +3,10 @@
 #include <cstdlib>
 #include <new>
 
-template<typename T>
+template<typename T = void>
 struct NotReprC;
 
-template<typename T>
+template<typename T = void>
 struct RefCell;
 
 using Foo = NotReprC<RefCell<int32_t>>;
diff --git a/tests/expectations/monomorph-1.cpp b/tests/expectations/monomorph-1.cpp
--- a/tests/expectations/monomorph-1.cpp
+++ b/tests/expectations/monomorph-1.cpp
@@ -3,7 +3,7 @@
 #include <cstdlib>
 #include <new>
 
-template<typename T>
+template<typename T = void>
 struct Bar;
 
 template<typename T>
diff --git a/tests/expectations/monomorph-3.cpp b/tests/expectations/monomorph-3.cpp
--- a/tests/expectations/monomorph-3.cpp
+++ b/tests/expectations/monomorph-3.cpp
@@ -3,7 +3,7 @@
 #include <cstdlib>
 #include <new>
 
-template<typename T>
+template<typename T = void>
 struct Bar;
 
 template<typename T>
diff --git /dev/null b/tests/expectations/opaque.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/opaque.c
@@ -0,0 +1,28 @@
+#ifdef __cplusplus
+// These could be added as opaque types I guess.
+template <typename T>
+struct BuildHasherDefault;
+
+struct DefaultHasher;
+#endif
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct HashMap_i32__i32__BuildHasherDefault_DefaultHasher HashMap_i32__i32__BuildHasherDefault_DefaultHasher;
+
+typedef struct Result_Foo Result_Foo;
+
+/**
+ * Fast hash map used internally.
+ */
+typedef HashMap_i32__i32__BuildHasherDefault_DefaultHasher FastHashMap_i32__i32;
+
+typedef FastHashMap_i32__i32 Foo;
+
+typedef Result_Foo Bar;
+
+void root(const Foo *a, const Bar *b);
diff --git /dev/null b/tests/expectations/opaque.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/opaque.compat.c
@@ -0,0 +1,36 @@
+#ifdef __cplusplus
+// These could be added as opaque types I guess.
+template <typename T>
+struct BuildHasherDefault;
+
+struct DefaultHasher;
+#endif
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+typedef struct HashMap_i32__i32__BuildHasherDefault_DefaultHasher HashMap_i32__i32__BuildHasherDefault_DefaultHasher;
+
+typedef struct Result_Foo Result_Foo;
+
+/**
+ * Fast hash map used internally.
+ */
+typedef HashMap_i32__i32__BuildHasherDefault_DefaultHasher FastHashMap_i32__i32;
+
+typedef FastHashMap_i32__i32 Foo;
+
+typedef Result_Foo Bar;
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(const Foo *a, const Bar *b);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/expectations/opaque.cpp
new file mode 100644
--- /dev/null
+++ b/tests/expectations/opaque.cpp
@@ -0,0 +1,33 @@
+#ifdef __cplusplus
+// These could be added as opaque types I guess.
+template <typename T>
+struct BuildHasherDefault;
+
+struct DefaultHasher;
+#endif
+
+
+#include <cstdarg>
+#include <cstdint>
+#include <cstdlib>
+#include <new>
+
+template<typename K = void, typename V = void, typename Hasher = void>
+struct HashMap;
+
+template<typename T = void, typename E = void>
+struct Result;
+
+/// Fast hash map used internally.
+template<typename K, typename V>
+using FastHashMap = HashMap<K, V, BuildHasherDefault<DefaultHasher>>;
+
+using Foo = FastHashMap<int32_t, int32_t>;
+
+using Bar = Result<Foo>;
+
+extern "C" {
+
+void root(const Foo *a, const Bar *b);
+
+} // extern "C"
diff --git a/tests/expectations/std_lib.cpp b/tests/expectations/std_lib.cpp
--- a/tests/expectations/std_lib.cpp
+++ b/tests/expectations/std_lib.cpp
@@ -3,15 +3,15 @@
 #include <cstdlib>
 #include <new>
 
-template<typename T>
+template<typename T = void>
 struct Option;
 
-template<typename T, typename E>
+template<typename T = void, typename E = void>
 struct Result;
 
 struct String;
 
-template<typename T>
+template<typename T = void>
 struct Vec;
 
 extern "C" {
diff --git a/tests/expectations/swift_name.cpp b/tests/expectations/swift_name.cpp
--- a/tests/expectations/swift_name.cpp
+++ b/tests/expectations/swift_name.cpp
@@ -5,7 +5,7 @@
 #include <cstdlib>
 #include <new>
 
-template<typename T>
+template<typename T = void>
 struct Box;
 
 struct Opaque;
diff --git /dev/null b/tests/expectations/tag/opaque.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/opaque.c
@@ -0,0 +1,28 @@
+#ifdef __cplusplus
+// These could be added as opaque types I guess.
+template <typename T>
+struct BuildHasherDefault;
+
+struct DefaultHasher;
+#endif
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+struct HashMap_i32__i32__BuildHasherDefault_DefaultHasher;
+
+struct Result_Foo;
+
+/**
+ * Fast hash map used internally.
+ */
+typedef struct HashMap_i32__i32__BuildHasherDefault_DefaultHasher FastHashMap_i32__i32;
+
+typedef FastHashMap_i32__i32 Foo;
+
+typedef struct Result_Foo Bar;
+
+void root(const Foo *a, const Bar *b);
diff --git /dev/null b/tests/expectations/tag/opaque.compat.c
new file mode 100644
--- /dev/null
+++ b/tests/expectations/tag/opaque.compat.c
@@ -0,0 +1,36 @@
+#ifdef __cplusplus
+// These could be added as opaque types I guess.
+template <typename T>
+struct BuildHasherDefault;
+
+struct DefaultHasher;
+#endif
+
+
+#include <stdarg.h>
+#include <stdbool.h>
+#include <stdint.h>
+#include <stdlib.h>
+
+struct HashMap_i32__i32__BuildHasherDefault_DefaultHasher;
+
+struct Result_Foo;
+
+/**
+ * Fast hash map used internally.
+ */
+typedef struct HashMap_i32__i32__BuildHasherDefault_DefaultHasher FastHashMap_i32__i32;
+
+typedef FastHashMap_i32__i32 Foo;
+
+typedef struct Result_Foo Bar;
+
+#ifdef __cplusplus
+extern "C" {
+#endif // __cplusplus
+
+void root(const Foo *a, const Bar *b);
+
+#ifdef __cplusplus
+} // extern "C"
+#endif // __cplusplus
diff --git /dev/null b/tests/rust/opaque.rs
new file mode 100644
--- /dev/null
+++ b/tests/rust/opaque.rs
@@ -0,0 +1,10 @@
+/// Fast hash map used internally.
+type FastHashMap<K, V> =
+    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;
+
+pub type Foo = FastHashMap<i32, i32>;
+
+pub type Bar = Result<Foo, ()>;
+
+#[no_mangle]
+pub extern "C" fn root(a: &Foo, b: &Bar) {}
diff --git /dev/null b/tests/rust/opaque.toml
new file mode 100644
--- /dev/null
+++ b/tests/rust/opaque.toml
@@ -0,0 +1,9 @@
+header = """
+#ifdef __cplusplus
+// These could be added as opaque types I guess.
+template <typename T>
+struct BuildHasherDefault;
+
+struct DefaultHasher;
+#endif
+"""

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 6b4181540c146fff75efd500bfb75a2d60403b4c
git clean -fd
