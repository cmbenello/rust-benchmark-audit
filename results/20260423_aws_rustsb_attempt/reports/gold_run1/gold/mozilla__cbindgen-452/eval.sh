#!/bin/bash
set -uxo pipefail
cp -r /testbed/. /workspace
git config --global --add safe.directory /workspace
cd /workspace
git apply -v - <<'EOF_114329324912'
diff --git a/tests/expectations/body.c b/tests/expectations/body.c
--- a/tests/expectations/body.c
+++ b/tests/expectations/body.c
@@ -9,6 +9,12 @@ typedef enum {
   Baz1,
 } MyCLikeEnum;
 
+typedef enum {
+  Foo1_Prepended,
+  Bar1_Prepended,
+  Baz1_Prepended,
+} MyCLikeEnum_Prepended;
+
 typedef struct {
   int32_t i;
 #ifdef __cplusplus
diff --git a/tests/expectations/body.c b/tests/expectations/body.c
--- a/tests/expectations/body.c
+++ b/tests/expectations/body.c
@@ -47,4 +53,49 @@ typedef union {
   int32_t extra_member; // yolo
 } MyUnion;
 
-void root(MyFancyStruct s, MyFancyEnum e, MyCLikeEnum c, MyUnion u);
+typedef struct {
+#ifdef __cplusplus
+  inline void prepended_wohoo();
+#endif
+  int32_t i;
+} MyFancyStruct_Prepended;
+
+typedef enum {
+  Foo_Prepended,
+  Bar_Prepended,
+  Baz_Prepended,
+} MyFancyEnum_Prepended_Tag;
+
+typedef struct {
+  int32_t _0;
+} Bar_Prepended_Body;
+
+typedef struct {
+  int32_t _0;
+} Baz_Prepended_Body;
+
+typedef struct {
+  #ifdef __cplusplus
+    inline void wohoo();
+  #endif
+  MyFancyEnum_Prepended_Tag tag;
+  union {
+    Bar_Prepended_Body bar_prepended;
+    Baz_Prepended_Body baz_prepended;
+  };
+} MyFancyEnum_Prepended;
+
+typedef union {
+  int32_t extra_member; // yolo
+  float f;
+  uint32_t u;
+} MyUnion_Prepended;
+
+void root(MyFancyStruct s,
+          MyFancyEnum e,
+          MyCLikeEnum c,
+          MyUnion u,
+          MyFancyStruct_Prepended sp,
+          MyFancyEnum_Prepended ep,
+          MyCLikeEnum_Prepended cp,
+          MyUnion_Prepended up);
diff --git a/tests/expectations/body.compat.c b/tests/expectations/body.compat.c
--- a/tests/expectations/body.compat.c
+++ b/tests/expectations/body.compat.c
@@ -9,6 +9,12 @@ typedef enum {
   Baz1,
 } MyCLikeEnum;
 
+typedef enum {
+  Foo1_Prepended,
+  Bar1_Prepended,
+  Baz1_Prepended,
+} MyCLikeEnum_Prepended;
+
 typedef struct {
   int32_t i;
 #ifdef __cplusplus
diff --git a/tests/expectations/body.compat.c b/tests/expectations/body.compat.c
--- a/tests/expectations/body.compat.c
+++ b/tests/expectations/body.compat.c
@@ -47,11 +53,56 @@ typedef union {
   int32_t extra_member; // yolo
 } MyUnion;
 
+typedef struct {
+#ifdef __cplusplus
+  inline void prepended_wohoo();
+#endif
+  int32_t i;
+} MyFancyStruct_Prepended;
+
+typedef enum {
+  Foo_Prepended,
+  Bar_Prepended,
+  Baz_Prepended,
+} MyFancyEnum_Prepended_Tag;
+
+typedef struct {
+  int32_t _0;
+} Bar_Prepended_Body;
+
+typedef struct {
+  int32_t _0;
+} Baz_Prepended_Body;
+
+typedef struct {
+  #ifdef __cplusplus
+    inline void wohoo();
+  #endif
+  MyFancyEnum_Prepended_Tag tag;
+  union {
+    Bar_Prepended_Body bar_prepended;
+    Baz_Prepended_Body baz_prepended;
+  };
+} MyFancyEnum_Prepended;
+
+typedef union {
+  int32_t extra_member; // yolo
+  float f;
+  uint32_t u;
+} MyUnion_Prepended;
+
 #ifdef __cplusplus
 extern "C" {
 #endif // __cplusplus
 
-void root(MyFancyStruct s, MyFancyEnum e, MyCLikeEnum c, MyUnion u);
+void root(MyFancyStruct s,
+          MyFancyEnum e,
+          MyCLikeEnum c,
+          MyUnion u,
+          MyFancyStruct_Prepended sp,
+          MyFancyEnum_Prepended ep,
+          MyCLikeEnum_Prepended cp,
+          MyUnion_Prepended up);
 
 #ifdef __cplusplus
 } // extern "C"
diff --git a/tests/expectations/body.cpp b/tests/expectations/body.cpp
--- a/tests/expectations/body.cpp
+++ b/tests/expectations/body.cpp
@@ -9,6 +9,12 @@ enum class MyCLikeEnum {
   Baz1,
 };
 
+enum class MyCLikeEnum_Prepended {
+  Foo1_Prepended,
+  Bar1_Prepended,
+  Baz1_Prepended,
+};
+
 struct MyFancyStruct {
   int32_t i;
 #ifdef __cplusplus
diff --git a/tests/expectations/body.cpp b/tests/expectations/body.cpp
--- a/tests/expectations/body.cpp
+++ b/tests/expectations/body.cpp
@@ -47,8 +53,53 @@ union MyUnion {
   int32_t extra_member; // yolo
 };
 
+struct MyFancyStruct_Prepended {
+#ifdef __cplusplus
+  inline void prepended_wohoo();
+#endif
+  int32_t i;
+};
+
+struct MyFancyEnum_Prepended {
+  #ifdef __cplusplus
+    inline void wohoo();
+  #endif
+  enum class Tag {
+    Foo_Prepended,
+    Bar_Prepended,
+    Baz_Prepended,
+  };
+
+  struct Bar_Prepended_Body {
+    int32_t _0;
+  };
+
+  struct Baz_Prepended_Body {
+    int32_t _0;
+  };
+
+  Tag tag;
+  union {
+    Bar_Prepended_Body bar_prepended;
+    Baz_Prepended_Body baz_prepended;
+  };
+};
+
+union MyUnion_Prepended {
+  int32_t extra_member; // yolo
+  float f;
+  uint32_t u;
+};
+
 extern "C" {
 
-void root(MyFancyStruct s, MyFancyEnum e, MyCLikeEnum c, MyUnion u);
+void root(MyFancyStruct s,
+          MyFancyEnum e,
+          MyCLikeEnum c,
+          MyUnion u,
+          MyFancyStruct_Prepended sp,
+          MyFancyEnum_Prepended ep,
+          MyCLikeEnum_Prepended cp,
+          MyUnion_Prepended up);
 
 } // extern "C"
diff --git a/tests/expectations/both/body.c b/tests/expectations/both/body.c
--- a/tests/expectations/both/body.c
+++ b/tests/expectations/both/body.c
@@ -9,6 +9,12 @@ typedef enum MyCLikeEnum {
   Baz1,
 } MyCLikeEnum;
 
+typedef enum MyCLikeEnum_Prepended {
+  Foo1_Prepended,
+  Bar1_Prepended,
+  Baz1_Prepended,
+} MyCLikeEnum_Prepended;
+
 typedef struct MyFancyStruct {
   int32_t i;
 #ifdef __cplusplus
diff --git a/tests/expectations/both/body.c b/tests/expectations/both/body.c
--- a/tests/expectations/both/body.c
+++ b/tests/expectations/both/body.c
@@ -47,4 +53,49 @@ typedef union MyUnion {
   int32_t extra_member; // yolo
 } MyUnion;
 
-void root(MyFancyStruct s, MyFancyEnum e, MyCLikeEnum c, MyUnion u);
+typedef struct MyFancyStruct_Prepended {
+#ifdef __cplusplus
+  inline void prepended_wohoo();
+#endif
+  int32_t i;
+} MyFancyStruct_Prepended;
+
+typedef enum MyFancyEnum_Prepended_Tag {
+  Foo_Prepended,
+  Bar_Prepended,
+  Baz_Prepended,
+} MyFancyEnum_Prepended_Tag;
+
+typedef struct Bar_Prepended_Body {
+  int32_t _0;
+} Bar_Prepended_Body;
+
+typedef struct Baz_Prepended_Body {
+  int32_t _0;
+} Baz_Prepended_Body;
+
+typedef struct MyFancyEnum_Prepended {
+  #ifdef __cplusplus
+    inline void wohoo();
+  #endif
+  MyFancyEnum_Prepended_Tag tag;
+  union {
+    Bar_Prepended_Body bar_prepended;
+    Baz_Prepended_Body baz_prepended;
+  };
+} MyFancyEnum_Prepended;
+
+typedef union MyUnion_Prepended {
+  int32_t extra_member; // yolo
+  float f;
+  uint32_t u;
+} MyUnion_Prepended;
+
+void root(MyFancyStruct s,
+          MyFancyEnum e,
+          MyCLikeEnum c,
+          MyUnion u,
+          MyFancyStruct_Prepended sp,
+          MyFancyEnum_Prepended ep,
+          MyCLikeEnum_Prepended cp,
+          MyUnion_Prepended up);
diff --git a/tests/expectations/both/body.compat.c b/tests/expectations/both/body.compat.c
--- a/tests/expectations/both/body.compat.c
+++ b/tests/expectations/both/body.compat.c
@@ -9,6 +9,12 @@ typedef enum MyCLikeEnum {
   Baz1,
 } MyCLikeEnum;
 
+typedef enum MyCLikeEnum_Prepended {
+  Foo1_Prepended,
+  Bar1_Prepended,
+  Baz1_Prepended,
+} MyCLikeEnum_Prepended;
+
 typedef struct MyFancyStruct {
   int32_t i;
 #ifdef __cplusplus
diff --git a/tests/expectations/both/body.compat.c b/tests/expectations/both/body.compat.c
--- a/tests/expectations/both/body.compat.c
+++ b/tests/expectations/both/body.compat.c
@@ -47,11 +53,56 @@ typedef union MyUnion {
   int32_t extra_member; // yolo
 } MyUnion;
 
+typedef struct MyFancyStruct_Prepended {
+#ifdef __cplusplus
+  inline void prepended_wohoo();
+#endif
+  int32_t i;
+} MyFancyStruct_Prepended;
+
+typedef enum MyFancyEnum_Prepended_Tag {
+  Foo_Prepended,
+  Bar_Prepended,
+  Baz_Prepended,
+} MyFancyEnum_Prepended_Tag;
+
+typedef struct Bar_Prepended_Body {
+  int32_t _0;
+} Bar_Prepended_Body;
+
+typedef struct Baz_Prepended_Body {
+  int32_t _0;
+} Baz_Prepended_Body;
+
+typedef struct MyFancyEnum_Prepended {
+  #ifdef __cplusplus
+    inline void wohoo();
+  #endif
+  MyFancyEnum_Prepended_Tag tag;
+  union {
+    Bar_Prepended_Body bar_prepended;
+    Baz_Prepended_Body baz_prepended;
+  };
+} MyFancyEnum_Prepended;
+
+typedef union MyUnion_Prepended {
+  int32_t extra_member; // yolo
+  float f;
+  uint32_t u;
+} MyUnion_Prepended;
+
 #ifdef __cplusplus
 extern "C" {
 #endif // __cplusplus
 
-void root(MyFancyStruct s, MyFancyEnum e, MyCLikeEnum c, MyUnion u);
+void root(MyFancyStruct s,
+          MyFancyEnum e,
+          MyCLikeEnum c,
+          MyUnion u,
+          MyFancyStruct_Prepended sp,
+          MyFancyEnum_Prepended ep,
+          MyCLikeEnum_Prepended cp,
+          MyUnion_Prepended up);
 
 #ifdef __cplusplus
 } // extern "C"
diff --git a/tests/expectations/tag/body.c b/tests/expectations/tag/body.c
--- a/tests/expectations/tag/body.c
+++ b/tests/expectations/tag/body.c
@@ -9,6 +9,12 @@ enum MyCLikeEnum {
   Baz1,
 };
 
+enum MyCLikeEnum_Prepended {
+  Foo1_Prepended,
+  Bar1_Prepended,
+  Baz1_Prepended,
+};
+
 struct MyFancyStruct {
   int32_t i;
 #ifdef __cplusplus
diff --git a/tests/expectations/tag/body.c b/tests/expectations/tag/body.c
--- a/tests/expectations/tag/body.c
+++ b/tests/expectations/tag/body.c
@@ -47,4 +53,49 @@ union MyUnion {
   int32_t extra_member; // yolo
 };
 
-void root(struct MyFancyStruct s, struct MyFancyEnum e, enum MyCLikeEnum c, union MyUnion u);
+struct MyFancyStruct_Prepended {
+#ifdef __cplusplus
+  inline void prepended_wohoo();
+#endif
+  int32_t i;
+};
+
+enum MyFancyEnum_Prepended_Tag {
+  Foo_Prepended,
+  Bar_Prepended,
+  Baz_Prepended,
+};
+
+struct Bar_Prepended_Body {
+  int32_t _0;
+};
+
+struct Baz_Prepended_Body {
+  int32_t _0;
+};
+
+struct MyFancyEnum_Prepended {
+  #ifdef __cplusplus
+    inline void wohoo();
+  #endif
+  enum MyFancyEnum_Prepended_Tag tag;
+  union {
+    struct Bar_Prepended_Body bar_prepended;
+    struct Baz_Prepended_Body baz_prepended;
+  };
+};
+
+union MyUnion_Prepended {
+  int32_t extra_member; // yolo
+  float f;
+  uint32_t u;
+};
+
+void root(struct MyFancyStruct s,
+          struct MyFancyEnum e,
+          enum MyCLikeEnum c,
+          union MyUnion u,
+          struct MyFancyStruct_Prepended sp,
+          struct MyFancyEnum_Prepended ep,
+          enum MyCLikeEnum_Prepended cp,
+          union MyUnion_Prepended up);
diff --git a/tests/expectations/tag/body.compat.c b/tests/expectations/tag/body.compat.c
--- a/tests/expectations/tag/body.compat.c
+++ b/tests/expectations/tag/body.compat.c
@@ -9,6 +9,12 @@ enum MyCLikeEnum {
   Baz1,
 };
 
+enum MyCLikeEnum_Prepended {
+  Foo1_Prepended,
+  Bar1_Prepended,
+  Baz1_Prepended,
+};
+
 struct MyFancyStruct {
   int32_t i;
 #ifdef __cplusplus
diff --git a/tests/expectations/tag/body.compat.c b/tests/expectations/tag/body.compat.c
--- a/tests/expectations/tag/body.compat.c
+++ b/tests/expectations/tag/body.compat.c
@@ -47,11 +53,56 @@ union MyUnion {
   int32_t extra_member; // yolo
 };
 
+struct MyFancyStruct_Prepended {
+#ifdef __cplusplus
+  inline void prepended_wohoo();
+#endif
+  int32_t i;
+};
+
+enum MyFancyEnum_Prepended_Tag {
+  Foo_Prepended,
+  Bar_Prepended,
+  Baz_Prepended,
+};
+
+struct Bar_Prepended_Body {
+  int32_t _0;
+};
+
+struct Baz_Prepended_Body {
+  int32_t _0;
+};
+
+struct MyFancyEnum_Prepended {
+  #ifdef __cplusplus
+    inline void wohoo();
+  #endif
+  enum MyFancyEnum_Prepended_Tag tag;
+  union {
+    struct Bar_Prepended_Body bar_prepended;
+    struct Baz_Prepended_Body baz_prepended;
+  };
+};
+
+union MyUnion_Prepended {
+  int32_t extra_member; // yolo
+  float f;
+  uint32_t u;
+};
+
 #ifdef __cplusplus
 extern "C" {
 #endif // __cplusplus
 
-void root(struct MyFancyStruct s, struct MyFancyEnum e, enum MyCLikeEnum c, union MyUnion u);
+void root(struct MyFancyStruct s,
+          struct MyFancyEnum e,
+          enum MyCLikeEnum c,
+          union MyUnion u,
+          struct MyFancyStruct_Prepended sp,
+          struct MyFancyEnum_Prepended ep,
+          enum MyCLikeEnum_Prepended cp,
+          union MyUnion_Prepended up);
 
 #ifdef __cplusplus
 } // extern "C"
diff --git a/tests/rust/body.rs b/tests/rust/body.rs
--- a/tests/rust/body.rs
+++ b/tests/rust/body.rs
@@ -24,5 +24,32 @@ pub union MyUnion {
     pub u: u32,
 }
 
+
+#[repr(C)]
+pub struct MyFancyStruct_Prepended {
+    i: i32,
+}
+
+#[repr(C)]
+pub enum MyFancyEnum_Prepended {
+    Foo_Prepended,
+    Bar_Prepended(i32),
+    Baz_Prepended(i32),
+}
+
+#[repr(C)]
+pub enum MyCLikeEnum_Prepended {
+    Foo1_Prepended,
+    Bar1_Prepended,
+    Baz1_Prepended,
+}
+
+#[repr(C)]
+pub union MyUnion_Prepended {
+    pub f: f32,
+    pub u: u32,
+}
+
+
 #[no_mangle]
-pub extern "C" fn root(s: MyFancyStruct, e: MyFancyEnum, c: MyCLikeEnum, u: MyUnion) {}
+pub extern "C" fn root(s: MyFancyStruct, e: MyFancyEnum, c: MyCLikeEnum, u: MyUnion, sp: MyFancyStruct_Prepended, ep: MyFancyEnum_Prepended, cp: MyCLikeEnum_Prepended, up: MyUnion_Prepended) {}
diff --git a/tests/rust/body.toml b/tests/rust/body.toml
--- a/tests/rust/body.toml
+++ b/tests/rust/body.toml
@@ -18,3 +18,25 @@
 "MyUnion" = """
   int32_t extra_member; // yolo
 """
+
+[export.pre_body]
+"MyFancyStruct_Prepended" = """
+#ifdef __cplusplus
+  inline void prepended_wohoo();
+#endif
+"""
+
+"MyFancyEnum_Prepended" = """
+  #ifdef __cplusplus
+    inline void wohoo();
+  #endif
+"""
+
+"MyCLikeEnum_Prepended" = """
+  BogusVariantForSerializationForExample,
+"""
+
+"MyUnion_Prepended" = """
+  int32_t extra_member; // yolo
+"""
+

EOF_114329324912
cargo test --no-fail-fast --all-features
git reset --hard ac1a7d47e87658cf36cb7e56edad7fa5f935dddd
git clean -fd
