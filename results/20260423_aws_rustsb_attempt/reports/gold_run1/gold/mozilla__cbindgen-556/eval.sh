#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/expectations/associated_in_body.c b/tests/expectations/associated_in_body.c
--- a/tests/expectations/associated_in_body.c
+++ b/tests/expectations/associated_in_body.c
@@ -14,22 +14,22 @@ typedef struct {
 /**
  * 'auto'
  */
-#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (1 << 1) }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (1 << 2) }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (1 << 3) }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 3) }
 
 void root(StyleAlignFlags flags);
diff --git a/tests/expectations/associated_in_body.compat.c b/tests/expectations/associated_in_body.compat.c
--- a/tests/expectations/associated_in_body.compat.c
+++ b/tests/expectations/associated_in_body.compat.c
@@ -14,23 +14,23 @@ typedef struct {
 /**
  * 'auto'
  */
-#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (1 << 1) }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (1 << 2) }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (1 << 3) }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 3) }
 
 #ifdef __cplusplus
 extern "C" {
diff --git a/tests/expectations/associated_in_body.cpp b/tests/expectations/associated_in_body.cpp
--- a/tests/expectations/associated_in_body.cpp
+++ b/tests/expectations/associated_in_body.cpp
@@ -43,15 +43,15 @@ struct StyleAlignFlags {
   static const StyleAlignFlags FLEX_START;
 };
 /// 'auto'
-inline const StyleAlignFlags StyleAlignFlags::AUTO = StyleAlignFlags{ /* .bits = */ 0 };
+inline const StyleAlignFlags StyleAlignFlags::AUTO = StyleAlignFlags{ /* .bits = */ (uint8_t)0 };
 /// 'normal'
-inline const StyleAlignFlags StyleAlignFlags::NORMAL = StyleAlignFlags{ /* .bits = */ 1 };
+inline const StyleAlignFlags StyleAlignFlags::NORMAL = StyleAlignFlags{ /* .bits = */ (uint8_t)1 };
 /// 'start'
-inline const StyleAlignFlags StyleAlignFlags::START = StyleAlignFlags{ /* .bits = */ (1 << 1) };
+inline const StyleAlignFlags StyleAlignFlags::START = StyleAlignFlags{ /* .bits = */ (uint8_t)(1 << 1) };
 /// 'end'
-inline const StyleAlignFlags StyleAlignFlags::END = StyleAlignFlags{ /* .bits = */ (1 << 2) };
+inline const StyleAlignFlags StyleAlignFlags::END = StyleAlignFlags{ /* .bits = */ (uint8_t)(1 << 2) };
 /// 'flex-start'
-inline const StyleAlignFlags StyleAlignFlags::FLEX_START = StyleAlignFlags{ /* .bits = */ (1 << 3) };
+inline const StyleAlignFlags StyleAlignFlags::FLEX_START = StyleAlignFlags{ /* .bits = */ (uint8_t)(1 << 3) };
 
 extern "C" {
 
diff --git a/tests/expectations/bitflags.c b/tests/expectations/bitflags.c
--- a/tests/expectations/bitflags.c
+++ b/tests/expectations/bitflags.c
@@ -14,22 +14,30 @@ typedef struct {
 /**
  * 'auto'
  */
-#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_AUTO (AlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define AlignFlags_START (AlignFlags){ .bits = (1 << 1) }
+#define AlignFlags_START (AlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define AlignFlags_END (AlignFlags){ .bits = (1 << 2) }
+#define AlignFlags_END (AlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define AlignFlags_FLEX_START (AlignFlags){ .bits = (1 << 3) }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = (uint8_t)(1 << 3) }
 
-void root(AlignFlags flags);
+typedef struct {
+  uint32_t bits;
+} DebugFlags;
+/**
+ * Flag with the topmost bit set of the u32
+ */
+#define DebugFlags_BIGGEST_ALLOWED (DebugFlags){ .bits = (uint32_t)(1 << 31) }
+
+void root(AlignFlags flags, DebugFlags bigger_flags);
diff --git a/tests/expectations/bitflags.compat.c b/tests/expectations/bitflags.compat.c
--- a/tests/expectations/bitflags.compat.c
+++ b/tests/expectations/bitflags.compat.c
@@ -14,29 +14,37 @@ typedef struct {
 /**
  * 'auto'
  */
-#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_AUTO (AlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define AlignFlags_START (AlignFlags){ .bits = (1 << 1) }
+#define AlignFlags_START (AlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define AlignFlags_END (AlignFlags){ .bits = (1 << 2) }
+#define AlignFlags_END (AlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define AlignFlags_FLEX_START (AlignFlags){ .bits = (1 << 3) }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = (uint8_t)(1 << 3) }
+
+typedef struct {
+  uint32_t bits;
+} DebugFlags;
+/**
+ * Flag with the topmost bit set of the u32
+ */
+#define DebugFlags_BIGGEST_ALLOWED (DebugFlags){ .bits = (uint32_t)(1 << 31) }
 
 #ifdef __cplusplus
 extern "C" {
 #endif // __cplusplus
 
-void root(AlignFlags flags);
+void root(AlignFlags flags, DebugFlags bigger_flags);
 
 #ifdef __cplusplus
 } // extern "C"
diff --git a/tests/expectations/bitflags.cpp b/tests/expectations/bitflags.cpp
--- a/tests/expectations/bitflags.cpp
+++ b/tests/expectations/bitflags.cpp
@@ -38,18 +38,52 @@ struct AlignFlags {
   }
 };
 /// 'auto'
-static const AlignFlags AlignFlags_AUTO = AlignFlags{ /* .bits = */ 0 };
+static const AlignFlags AlignFlags_AUTO = AlignFlags{ /* .bits = */ (uint8_t)0 };
 /// 'normal'
-static const AlignFlags AlignFlags_NORMAL = AlignFlags{ /* .bits = */ 1 };
+static const AlignFlags AlignFlags_NORMAL = AlignFlags{ /* .bits = */ (uint8_t)1 };
 /// 'start'
-static const AlignFlags AlignFlags_START = AlignFlags{ /* .bits = */ (1 << 1) };
+static const AlignFlags AlignFlags_START = AlignFlags{ /* .bits = */ (uint8_t)(1 << 1) };
 /// 'end'
-static const AlignFlags AlignFlags_END = AlignFlags{ /* .bits = */ (1 << 2) };
+static const AlignFlags AlignFlags_END = AlignFlags{ /* .bits = */ (uint8_t)(1 << 2) };
 /// 'flex-start'
-static const AlignFlags AlignFlags_FLEX_START = AlignFlags{ /* .bits = */ (1 << 3) };
+static const AlignFlags AlignFlags_FLEX_START = AlignFlags{ /* .bits = */ (uint8_t)(1 << 3) };
+
+struct DebugFlags {
+  uint32_t bits;
+
+  explicit operator bool() const {
+    return !!bits;
+  }
+  DebugFlags operator~() const {
+    return {static_cast<decltype(bits)>(~bits)};
+  }
+  DebugFlags operator|(const DebugFlags& other) const {
+    return {static_cast<decltype(bits)>(this->bits | other.bits)};
+  }
+  DebugFlags& operator|=(const DebugFlags& other) {
+    *this = (*this | other);
+    return *this;
+  }
+  DebugFlags operator&(const DebugFlags& other) const {
+    return {static_cast<decltype(bits)>(this->bits & other.bits)};
+  }
+  DebugFlags& operator&=(const DebugFlags& other) {
+    *this = (*this & other);
+    return *this;
+  }
+  DebugFlags operator^(const DebugFlags& other) const {
+    return {static_cast<decltype(bits)>(this->bits ^ other.bits)};
+  }
+  DebugFlags& operator^=(const DebugFlags& other) {
+    *this = (*this ^ other);
+    return *this;
+  }
+};
+/// Flag with the topmost bit set of the u32
+static const DebugFlags DebugFlags_BIGGEST_ALLOWED = DebugFlags{ /* .bits = */ (uint32_t)(1 << 31) };
 
 extern "C" {
 
-void root(AlignFlags flags);
+void root(AlignFlags flags, DebugFlags bigger_flags);
 
 } // extern "C"
diff --git a/tests/expectations/both/associated_in_body.c b/tests/expectations/both/associated_in_body.c
--- a/tests/expectations/both/associated_in_body.c
+++ b/tests/expectations/both/associated_in_body.c
@@ -14,22 +14,22 @@ typedef struct StyleAlignFlags {
 /**
  * 'auto'
  */
-#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (1 << 1) }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (1 << 2) }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (1 << 3) }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 3) }
 
 void root(StyleAlignFlags flags);
diff --git a/tests/expectations/both/associated_in_body.compat.c b/tests/expectations/both/associated_in_body.compat.c
--- a/tests/expectations/both/associated_in_body.compat.c
+++ b/tests/expectations/both/associated_in_body.compat.c
@@ -14,23 +14,23 @@ typedef struct StyleAlignFlags {
 /**
  * 'auto'
  */
-#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (1 << 1) }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (1 << 2) }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (1 << 3) }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 3) }
 
 #ifdef __cplusplus
 extern "C" {
diff --git a/tests/expectations/both/bitflags.c b/tests/expectations/both/bitflags.c
--- a/tests/expectations/both/bitflags.c
+++ b/tests/expectations/both/bitflags.c
@@ -14,22 +14,30 @@ typedef struct AlignFlags {
 /**
  * 'auto'
  */
-#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_AUTO (AlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define AlignFlags_START (AlignFlags){ .bits = (1 << 1) }
+#define AlignFlags_START (AlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define AlignFlags_END (AlignFlags){ .bits = (1 << 2) }
+#define AlignFlags_END (AlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define AlignFlags_FLEX_START (AlignFlags){ .bits = (1 << 3) }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = (uint8_t)(1 << 3) }
 
-void root(AlignFlags flags);
+typedef struct DebugFlags {
+  uint32_t bits;
+} DebugFlags;
+/**
+ * Flag with the topmost bit set of the u32
+ */
+#define DebugFlags_BIGGEST_ALLOWED (DebugFlags){ .bits = (uint32_t)(1 << 31) }
+
+void root(AlignFlags flags, DebugFlags bigger_flags);
diff --git a/tests/expectations/both/bitflags.compat.c b/tests/expectations/both/bitflags.compat.c
--- a/tests/expectations/both/bitflags.compat.c
+++ b/tests/expectations/both/bitflags.compat.c
@@ -14,29 +14,37 @@ typedef struct AlignFlags {
 /**
  * 'auto'
  */
-#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_AUTO (AlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define AlignFlags_START (AlignFlags){ .bits = (1 << 1) }
+#define AlignFlags_START (AlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define AlignFlags_END (AlignFlags){ .bits = (1 << 2) }
+#define AlignFlags_END (AlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define AlignFlags_FLEX_START (AlignFlags){ .bits = (1 << 3) }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = (uint8_t)(1 << 3) }
+
+typedef struct DebugFlags {
+  uint32_t bits;
+} DebugFlags;
+/**
+ * Flag with the topmost bit set of the u32
+ */
+#define DebugFlags_BIGGEST_ALLOWED (DebugFlags){ .bits = (uint32_t)(1 << 31) }
 
 #ifdef __cplusplus
 extern "C" {
 #endif // __cplusplus
 
-void root(AlignFlags flags);
+void root(AlignFlags flags, DebugFlags bigger_flags);
 
 #ifdef __cplusplus
 } // extern "C"
diff --git a/tests/expectations/tag/associated_in_body.c b/tests/expectations/tag/associated_in_body.c
--- a/tests/expectations/tag/associated_in_body.c
+++ b/tests/expectations/tag/associated_in_body.c
@@ -14,22 +14,22 @@ struct StyleAlignFlags {
 /**
  * 'auto'
  */
-#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (1 << 1) }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (1 << 2) }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (1 << 3) }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 3) }
 
 void root(struct StyleAlignFlags flags);
diff --git a/tests/expectations/tag/associated_in_body.compat.c b/tests/expectations/tag/associated_in_body.compat.c
--- a/tests/expectations/tag/associated_in_body.compat.c
+++ b/tests/expectations/tag/associated_in_body.compat.c
@@ -14,23 +14,23 @@ struct StyleAlignFlags {
 /**
  * 'auto'
  */
-#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = 0 }
+#define StyleAlignFlags_AUTO (StyleAlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = 1 }
+#define StyleAlignFlags_NORMAL (StyleAlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (1 << 1) }
+#define StyleAlignFlags_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (1 << 2) }
+#define StyleAlignFlags_END (StyleAlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (1 << 3) }
+#define StyleAlignFlags_FLEX_START (StyleAlignFlags){ .bits = (uint8_t)(1 << 3) }
 
 #ifdef __cplusplus
 extern "C" {
diff --git a/tests/expectations/tag/bitflags.c b/tests/expectations/tag/bitflags.c
--- a/tests/expectations/tag/bitflags.c
+++ b/tests/expectations/tag/bitflags.c
@@ -14,22 +14,30 @@ struct AlignFlags {
 /**
  * 'auto'
  */
-#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_AUTO (AlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define AlignFlags_START (AlignFlags){ .bits = (1 << 1) }
+#define AlignFlags_START (AlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define AlignFlags_END (AlignFlags){ .bits = (1 << 2) }
+#define AlignFlags_END (AlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define AlignFlags_FLEX_START (AlignFlags){ .bits = (1 << 3) }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = (uint8_t)(1 << 3) }
 
-void root(struct AlignFlags flags);
+struct DebugFlags {
+  uint32_t bits;
+};
+/**
+ * Flag with the topmost bit set of the u32
+ */
+#define DebugFlags_BIGGEST_ALLOWED (DebugFlags){ .bits = (uint32_t)(1 << 31) }
+
+void root(struct AlignFlags flags, struct DebugFlags bigger_flags);
diff --git a/tests/expectations/tag/bitflags.compat.c b/tests/expectations/tag/bitflags.compat.c
--- a/tests/expectations/tag/bitflags.compat.c
+++ b/tests/expectations/tag/bitflags.compat.c
@@ -14,29 +14,37 @@ struct AlignFlags {
 /**
  * 'auto'
  */
-#define AlignFlags_AUTO (AlignFlags){ .bits = 0 }
+#define AlignFlags_AUTO (AlignFlags){ .bits = (uint8_t)0 }
 /**
  * 'normal'
  */
-#define AlignFlags_NORMAL (AlignFlags){ .bits = 1 }
+#define AlignFlags_NORMAL (AlignFlags){ .bits = (uint8_t)1 }
 /**
  * 'start'
  */
-#define AlignFlags_START (AlignFlags){ .bits = (1 << 1) }
+#define AlignFlags_START (AlignFlags){ .bits = (uint8_t)(1 << 1) }
 /**
  * 'end'
  */
-#define AlignFlags_END (AlignFlags){ .bits = (1 << 2) }
+#define AlignFlags_END (AlignFlags){ .bits = (uint8_t)(1 << 2) }
 /**
  * 'flex-start'
  */
-#define AlignFlags_FLEX_START (AlignFlags){ .bits = (1 << 3) }
+#define AlignFlags_FLEX_START (AlignFlags){ .bits = (uint8_t)(1 << 3) }
+
+struct DebugFlags {
+  uint32_t bits;
+};
+/**
+ * Flag with the topmost bit set of the u32
+ */
+#define DebugFlags_BIGGEST_ALLOWED (DebugFlags){ .bits = (uint32_t)(1 << 31) }
 
 #ifdef __cplusplus
 extern "C" {
 #endif // __cplusplus
 
-void root(struct AlignFlags flags);
+void root(struct AlignFlags flags, struct DebugFlags bigger_flags);
 
 #ifdef __cplusplus
 } // extern "C"
diff --git a/tests/rust/bitflags.rs b/tests/rust/bitflags.rs
--- a/tests/rust/bitflags.rs
+++ b/tests/rust/bitflags.rs
@@ -18,5 +18,13 @@ bitflags! {
     }
 }
 
+bitflags! {
+    #[repr(C)]
+    pub struct DebugFlags: u32 {
+        /// Flag with the topmost bit set of the u32
+        const BIGGEST_ALLOWED = 1 << 31;
+    }
+}
+
 #[no_mangle]
-pub extern "C" fn root(flags: AlignFlags) {}
+pub extern "C" fn root(flags: AlignFlags, bigger_flags: DebugFlags) {}

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard 6ba31b49f445290a4ac1d3141f2a783306b23a88
git clean -fd
