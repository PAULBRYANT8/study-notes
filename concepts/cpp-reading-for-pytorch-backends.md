---
title: 面向 ATen 与 torch_npu 源码阅读的 C++ 基础
type: concept
created: 2026-07-26
updated: 2026-07-26
tags: [cpp, aten, torch-npu, 模板, raii, 智能指针, 宏]
sources:
  - https://eel.is/c++draft/temp
  - https://eel.is/c++draft/cpp.pre
  - https://en.cppreference.com/w/cpp/language/templates
  - https://en.cppreference.com/w/cpp/language/raii
  - https://en.cppreference.com/w/cpp/memory
  - https://en.cppreference.com/w/cpp/preprocessor
  - https://docs.pytorch.org/cppdocs/
  - https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
---

**读 PyTorch 后端 C++ 的目标不是掌握所有现代 C++ 特性，而是能沿着“算子声明 → 注册 → 分派 → kernel → 资源释放”还原真实控制流和所有权。**

ATen 与设备后端的 C++ 代码难读，通常不是因为某一行算法复杂，而是因为一段行为被模板、宏、代码生成、静态注册和多层类型包装分散到了不同文件。有效的阅读顺序是先找运行时入口和数据流，再按需补语言细节。

---

## 一、先建立阅读地图

### 1. 一次算子调用大致包含哪些角色

```text
Python API
  → Python/C++ binding
  → ATen operator schema
  → dispatcher 根据 DispatchKey 选实现
  → 后端注册的 C++ kernel
  → 设备运行时 / 算子库
  → stream 上异步执行
```

真实实现会因算子、版本、eager/compile 路径而不同，但阅读时始终问：

1. 算子的稳定名字和 schema 是什么？
2. 哪段代码把实现注册到了哪个 dispatch key？
3. 被注册的函数签名是什么？
4. 输入的 dtype、device、layout、shape 在哪里校验或转换？
5. 哪一行真正调用设备算子或 runtime？
6. 错误如何变成 C++ 异常，再如何传回 Python？
7. 临时资源由谁持有，在哪个作用域释放？

ATen 是 PyTorch C++ 层的基础 tensor 库；PyTorch C++ API 的分层介绍见 [PyTorch C++ API](https://docs.pytorch.org/cppdocs/)。自定义 C++ 算子和注册的官方入口见 [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。

### 2. 声明、定义、实例化、注册不是一回事

- **声明**告诉编译器“名字和类型存在”，如头文件中的函数签名。
- **定义**给出函数体或对象存储。
- **模板定义**只是生成代码的配方，使用具体模板参数时才可能实例化。
- **宏展开**发生在编译前，展开结果才是编译器真正看到的 token。
- **代码生成**可能在构建阶段从 YAML/schema 生成 C++ 文件。
- **注册**通常在程序加载动态库时执行，把 schema 或函数指针放进全局 registry。

因此 `rg "foo_kernel"` 找不到 Python 调用名的同名函数并不奇怪：名字可能由代码生成或 token 拼接产生，真正入口也可能是注册表中的函数指针。

---

## 二、读 C++ 前必须熟悉的类型语法

### 1. `const`、指针与引用

```cpp
const Tensor& x;      // 对 Tensor 的只读左值引用，不复制对象
Tensor& out;          // 可修改的左值引用，常用于 out/in-place
Tensor&& tmp;         // 右值引用，可能参与移动或完美转发
const Tensor* p;      // 指向 const Tensor 的指针
Tensor* const p2;     // 指针本身不能改指向，Tensor 可改
```

阅读函数签名时先标记：

- 参数是按值、引用还是指针传递。
- 是否 `const`。
- 返回值是拥有对象、借用引用，还是可空指针。
- 函数是否可能修改输入或输出。

C++ 引用和 cv 限定规则见 [References](https://en.cppreference.com/w/cpp/language/reference) 与 [cv type qualifiers](https://en.cppreference.com/w/cpp/language/cv)。

### 2. `auto` 不代表“没有类型”

```cpp
auto x = expr;          // 通常会去掉顶层 const 和引用
auto& x = expr;         // 保留引用
const auto& x = expr;   // 只读引用，也能延长某些临时对象的生命周期
decltype(expr) x;       // 按 decltype 规则取得类型
decltype(auto) x = expr;// 保留表达式的引用性
```

遇到 `auto` 时不要猜类型，沿右侧表达式或让 IDE/编译器显示。`decltype(auto)` 返回局部变量引用可能造成悬空引用，阅读工具函数时要特别小心。

对应文档：[Placeholder type specifiers (`auto`)](https://en.cppreference.com/w/cpp/language/auto) · [`decltype`](https://en.cppreference.com/w/cpp/language/decltype)。

### 3. 值类别决定复制、移动和生命周期

实用层面只需先分：

- **左值**：有稳定身份，可取地址，常绑定到 `T&`。
- **右值/临时值**：通常即将销毁，可绑定到 `T&&` 或 `const T&`。
- `std::move(x)` 本身不移动数据，只把表达式转换成可被移动构造/赋值接受的右值。
- 移动后的对象仍然有效，但具体值通常只能假设为“有效但未指定”。

这能解释为什么某些容器 `push_back(std::move(x))` 后 `x` 不再保存原内容，也能帮助识别临时 tensor handle 和资源 wrapper 的生命周期。对应文档：[Value categories](https://en.cppreference.com/w/cpp/language/value_category) · [`std::move`](https://en.cppreference.com/w/cpp/utility/move)。

### 4. namespace、别名与限定查找

```cpp
namespace at::native {
using TensorList = at::ITensorListRef;

auto y = at::empty(...);   // 限定名
auto z = empty(...);       // 依赖当前作用域和 using 的非限定查找
}
```

同名函数在 `at`、`at::native`、`c10`、后端命名空间里可能完全不同。看到非限定调用时，先检查：

- 当前 namespace。
- 文件顶部的 `using`。
- 参数类型触发的 ADL（argument-dependent lookup）。
- 重载集合中最终匹配哪个签名。

对应文档：[Qualified name lookup](https://en.cppreference.com/w/cpp/language/qualified_lookup) · [ADL](https://en.cppreference.com/w/cpp/language/adl)。

---

## 三、模板：读懂“这段代码会为哪些类型生成”

### 1. 模板的最小模型

```cpp
template <typename scalar_t>
scalar_t square(scalar_t x) {
    return x * x;
}

auto a = square<float>(2.0f); // 实例化 float 版本
auto b = square(3);           // 推导并实例化 int 版本
```

模板定义的是一族函数或类型。模板参数可以是：

- 类型参数：`typename T` / `class T`。
- 非类型参数：`template <int N>`。
- 模板参数：参数本身是另一个模板。

模板的概览、实例化与特化见 [Templates](https://en.cppreference.com/w/cpp/language/templates)。

### 2. 函数模板与类模板

```cpp
template <typename T>
struct Holder {
    T value;
};

template <typename T>
void run(const Holder<T>& h);
```

阅读 `Foo<Bar<Baz>>` 时从最内层向外拆：

```text
Baz
Bar<Baz>
Foo<Bar<Baz>>
```

对 PyTorch 代码，模板常用来：

- 为不同 `scalar_t` 生成 dtype 专用 kernel。
- 接受不同设备、layout、索引类型或 functor。
- 在编译期选择实现，避免运行时分支。
- 通过类型 traits 判断某个类型是否满足条件。

### 3. 模板实参推导

```cpp
template <typename T>
void f(T x);

template <typename T>
void g(T& x);

template <typename T>
void h(T&& x);  // 若 T 被推导，这可能是 forwarding reference
```

`T` 的推导会受引用折叠、数组到指针退化、顶层 `const` 等规则影响。阅读后端源码时，最实用的问题是：

- `T` 最终是什么？
- 是复制了对象，还是保留引用？
- `std::forward<T>(x)` 会把值类别恢复成什么？

对应文档：[Template argument deduction](https://en.cppreference.com/w/cpp/language/template_argument_deduction) · [Reference initialization](https://en.cppreference.com/w/cpp/language/reference_initialization)。

### 4. 特化与 `if constexpr`

```cpp
template <typename T>
void convert(T x) {
    if constexpr (std::is_same_v<T, float>) {
        // 只为 float 参与编译
    } else {
        // 其他类型
    }
}
```

- **显式特化**：为某个具体类型提供单独实现。
- **偏特化**：只适用于类/变量模板，为一组类型提供实现。
- `if constexpr`：条件在编译期决定，未选分支不实例化。

这类代码解释了“同一个源码函数对 fp16 和 fp32 行为为何不同”。对应文档：[Template specialization](https://en.cppreference.com/w/cpp/language/template_specialization) · [`if constexpr`](https://en.cppreference.com/w/cpp/language/if)。

### 5. SFINAE、traits 和 concepts：先会识别，不必会炫技

常见形状：

```cpp
template <
    typename T,
    std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
void f(T x);
```

或 C++20：

```cpp
template <typename T>
requires std::floating_point<T>
void f(T x);
```

它们的目的都是限制候选重载。读报错时先找：

1. 哪个模板在实例化。
2. 模板参数被推导成什么。
3. 哪个 constraint/trait 不满足。
4. 第一条真正与业务类型相关的错误，而不是最后几十屏连锁错误。

对应文档：[SFINAE](https://en.cppreference.com/w/cpp/language/sfinae) · [Constraints and concepts](https://en.cppreference.com/w/cpp/language/constraints)。

### 6. PyTorch 中的 dtype dispatch 宏

后端常见逻辑是：运行时取得 tensor dtype，再通过宏展开成 `switch`，每个 case 给 `scalar_t` 绑定具体 C++ 类型并实例化模板/lambda。阅读时应手工展开成：

```text
运行时 dtype
  → switch case
  → using scalar_t = ...
  → 调用 kernel<scalar_t>(...)
```

这能解释：

- 为什么“支持 dtype 的列表”藏在宏名里。
- 为什么编译错误会出现在某个 `scalar_t` 实例化上。
- 为什么新 dtype 既要加入 dispatch 集合，也要保证 kernel 表达式对该类型合法。

具体宏集合会随 PyTorch 版本变化，应以当前源码定义为准，不把某个宏名当稳定 API。

---

## 四、RAII：先看资源属于哪个对象

### 1. 核心规则

RAII 把资源的获取与对象初始化绑定，把释放放进析构函数；离开作用域时，无论正常返回还是异常栈展开，已构造对象都会按逆序析构。

```cpp
class File {
public:
    explicit File(const char* path) : fp_(std::fopen(path, "r")) {
        if (!fp_) {
            throw std::runtime_error("open failed");
        }
    }

    ~File() {
        std::fclose(fp_);
    }

    File(const File&) = delete;
    File& operator=(const File&) = delete;

private:
    std::FILE* fp_;
};
```

对应文档：[RAII](https://en.cppreference.com/w/cpp/language/raii)。

### 2. 后端代码中的资源不只有内存

RAII 可管理：

- heap 内存。
- 文件描述符和动态库 handle。
- mutex lock。
- stream/event/设备 handle。
- 临时工作区。
- profiler range。
- 临时切换的设备、stream 或 dispatch 状态。

阅读一个 wrapper 时优先找：

- 构造函数获取了什么。
- 析构函数释放或恢复了什么。
- 是否可复制、可移动。
- 异常发生时是否仍安全。
- 所持资源能否比异步设备任务更早释放。

最后一点对设备后端尤其重要：host wrapper 离开作用域，并不自动证明设备已经不再使用其指向的内存；异步生命周期通常还需要 allocator、stream 记录或 event 协调。

### 3. Rule of zero / three / five

如果类直接管理资源，就要明确复制构造、复制赋值、析构，以及 C++11 后的移动构造、移动赋值；更推荐用标准 RAII 成员，让类遵循 rule of zero。

对应文档：[Rule of three/five/zero](https://en.cppreference.com/w/cpp/language/rule_of_three)。

### 4. scope guard 与状态恢复

以下模式常见于框架内部：

```cpp
DeviceGuard guard(device);
NoGradGuard no_grad;
```

对象构造时切换状态，析构时恢复先前状态。不要把它误读为“只创建了一个没用的局部变量”；这个局部变量的生命周期本身就是行为。

---

## 五、智能指针与所有权

### 1. 先区分 owning 和 non-owning

```text
T* / T&                 通常不表达所有权，需要看约定
std::unique_ptr<T>      独占所有权
std::shared_ptr<T>      共享所有权，控制块计数
std::weak_ptr<T>        观察 shared_ptr 对象，不延长生命周期
c10::intrusive_ptr<T>   引用计数嵌入对象，PyTorch 内部常见
```

C++ 标准智能指针总览见 [Dynamic memory management](https://en.cppreference.com/w/cpp/memory)。

### 2. `unique_ptr`

```cpp
auto p = std::make_unique<State>();
consume(std::move(p));
```

- 不可复制，可以移动。
- `std::move(p)` 后所有权转移，原 `p` 通常为空。
- 默认用 `delete`，也可带自定义 deleter 管理 C handle。

对应文档：[`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr)。

### 3. `shared_ptr` 与 `weak_ptr`

```cpp
auto p = std::make_shared<State>();
std::weak_ptr<State> observer = p;

if (auto locked = observer.lock()) {
    use(*locked);
}
```

- 拷贝 `shared_ptr` 增加共享计数。
- `use_count()` 适合调试，不适合做并发正确性的控制逻辑。
- 两个对象互持 `shared_ptr` 会形成环；其中一边通常应为 `weak_ptr`。
- `shared_ptr<T>` 的引用计数线程安全不等于 `T` 的成员访问线程安全。

对应文档：[`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr) · [`std::weak_ptr`](https://en.cppreference.com/w/cpp/memory/weak_ptr)。

### 4. PyTorch 的 `intrusive_ptr`

PyTorch 内部常见 `c10::intrusive_ptr`：引用计数存放在被管理对象内，而不是单独控制块。阅读时仍使用同一套问题：

- 谁持有强引用。
- 哪一次拷贝增加引用计数。
- 哪次 reset/析构可能让计数归零。
- 裸指针或借用引用是否跨越了 owner 的生命周期。

`Tensor` 本身是轻量 handle，内部持有实现对象；复制 `Tensor` 通常不是复制整块 tensor 数据。ATen 的总体角色见 [PyTorch C++ API: ATen](https://docs.pytorch.org/cppdocs/#aten)；具体内部类型以当前源码为准。

### 5. 常见 bug 模式

- 从临时 `unique_ptr`/`shared_ptr` 取 `.get()` 后长期保存裸指针。
- lambda 按引用捕获局部变量，异步执行时局部变量已销毁。
- 回调持有 `shared_ptr`，被管理对象又持有回调，形成环。
- 自定义 deleter 与实际分配方式不匹配。
- 设备异步任务仍使用 buffer，但 host owner 已析构。

---

## 六、宏：先看展开结果，再猜行为

### 1. 宏是 token 替换

```cpp
#define CHECK_OK(expr) \
    do { \
        auto status = (expr); \
        if (status != 0) throw Error(status); \
    } while (false)
```

它不是普通函数：

- 参数可能被展开多次。
- 不遵守 C++ 类型系统。
- `#` 可字符串化参数。
- `##` 可拼接 token，制造新标识符。
- `__VA_ARGS__` 接受可变参数。
- 展开后的 `__FILE__`、`__LINE__` 常用于错误信息和注册位置。

对应文档：[Preprocessor](https://en.cppreference.com/w/cpp/preprocessor) · [Replacing text macros](https://en.cppreference.com/w/cpp/preprocessor/replace)。

### 2. 为什么常用 `do { ... } while (false)`

这样多语句宏在语法上表现得像单条语句，可以安全写在：

```cpp
if (cond)
    CHECK_OK(call());
else
    fallback();
```

若宏只是裸 `{ ... }` 或多条语句，调用点的分号和 `else` 可能产生意外解析。

### 3. 宏参数的副作用

```cpp
#define BAD_MAX(a, b) ((a) > (b) ? (a) : (b))
int m = BAD_MAX(i++, j++);
```

某个参数可能被求值两次。读后端检查/dispatch 宏时确认实参是否含函数调用、自增或资源获取。

### 4. 查看预处理结果

```bash
c++ -E source.cpp > source.ii
c++ -E -dD source.cpp > source-with-macros.ii
clang++ -E source.cpp > source.ii
```

真实工程需要补齐与构建相同的 `-I`、`-D`、语言标准和编译器参数。最可靠的来源通常是 `compile_commands.json` 或详细构建日志。

只看局部展开可使用 IDE 的 macro expansion 功能，或在预处理输出中搜索调用点附近的独特字符串。GDB 也能在带宏调试信息的构建中查看宏；见 [GDB C Preprocessor Macros](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Macros.html)。

### 5. 注册宏为什么“没有调用点”

类似注册代码常在 namespace 作用域：

```cpp
TORCH_LIBRARY_IMPL(my_namespace, PrivateUse1, m) {
    m.impl("my_op", &my_kernel);
}
```

概念上，宏会生成静态初始化代码：动态库加载时把 `my_kernel` 放进 dispatcher registry。于是：

- 注册行为发生在 `import`/动态库加载阶段。
- 没有业务代码显式调用“注册函数”。
- 动态库未加载、链接器裁剪或注册到错误 key，都可能表现为“实现不存在”。

PyTorch 自定义算子的注册路径见 [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) 与 [Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html)。

---

## 七、编译、链接和动态库：源码存在不等于运行时用了它

### 1. 编译与链接的最小模型

```text
.cpp
  → 预处理
  → 编译为 .o
多个 .o / 静态库 / 动态库
  → 链接为可执行文件或 .so
运行时 loader
  → 按搜索路径加载依赖 .so
```

典型问题：

- 改了源码但没有重编对应 target。
- 编译了新 `.so`，运行时却加载旧路径中的同名库。
- 符号被隐藏、没有导出或 ABI 不匹配。
- 未定义符号到 import 时才暴露。
- C++ name mangling 使符号名看起来陌生。

### 2. 常用只读工具

```bash
ldd path/to/extension.so
readelf -d path/to/extension.so
readelf -Ws path/to/extension.so
nm -C path/to/extension.so
objdump -T path/to/extension.so
c++filt '_Z...'
```

- `ldd`：查看运行时依赖解析到哪里。
- `readelf -d`：看 `NEEDED`、RPATH/RUNPATH。
- `readelf -Ws` / `nm -C`：看符号是否存在及是否未定义。
- `c++filt`：反解 C++ mangled name。

工具文档可从 GNU Binutils 手册进入：[GNU Binutils](https://sourceware.org/binutils/docs/)；动态链接器行为见 [`ld.so(8)`](https://man7.org/linux/man-pages/man8/ld.so.8.html)。

### 3. ABI 和模板错误的阅读方法

长编译错误从上到下找：

1. 第一个 `required from here` 或 `in instantiation of`。
2. 业务源码首次出现的位置。
3. 模板参数的实际类型。
4. 真正失败的操作：没有匹配重载、删除的复制构造、const 不匹配等。

链接错误则问：

- 是 undefined reference 还是 duplicate symbol？
- 声明与定义的 namespace、参数、const、ABI 是否完全相同？
- 定义所在目标是否真的被链接？
- 动态库是否按运行时路径加载？

---

## 八、读 ATen / torch_npu 的具体方法

### 1. 从 schema 或报错中的算子名开始

推荐路径：

```text
算子名
  → schema / 生成声明
  → 后端注册块
  → 注册的函数或 wrapper
  → dtype/layout/shape 检查
  → 真正的设备 API
```

不要从一个看起来像 kernel 的函数开始向上猜，因为同名函数可能从未被目标 dispatch key 注册。

### 2. 搜索策略

```bash
rg -n '"namespace::op"' .
rg -n 'TORCH_LIBRARY|TORCH_LIBRARY_IMPL' .
rg -n 'PrivateUse1|AutogradPrivateUse1' .
rg -n 'op_name|op_name_' .
rg -n '#define +MACRO_NAME' .
```

若名字由宏拼出：

- 搜 schema 字符串。
- 搜设备 API 名。
- 搜注册宏，而不是只搜函数定义。
- 查看生成目录和构建规则。
- 用预处理输出确认宏结果。

### 3. 画四列笔记

| 层 | 文件/符号 | 输入输出 | 关键行为 |
|---|---|---|---|
| schema | `namespace::op` | Tensor, scalar → Tensor | alias/mutation contract |
| registration | 某 `TORCH_LIBRARY_IMPL` | dispatch key → fn | 选择后端 |
| wrapper | `op_npu` | 校验后的 tensors | format/cast/workspace |
| device call | runtime/aclnn/... | handle/descriptor | 异步下发 |

每读一层，只记录能影响行为的事实。模板辅助类和宏实现按需展开，不要一开始把所有依赖都读完。

### 4. 识别常见 PyTorch 类型和习惯

- `at::Tensor`：tensor handle，不等于整块数据的值拷贝。
- `TensorList` / `ArrayRef<T>`：常是非 owning view，生命周期由原容器保证。
- `c10::optional<T>` / `std::optional<T>`：值可能不存在。
- `IntArrayRef` / `SymInt` 相关类型：shape/stride 可能是借用视图或符号整数。
- `TORCH_CHECK` 一类检查：失败时构造框架异常，通常最终转成 Python 异常。
- guard 类：构造/析构会切换和恢复设备、stream、grad 或 dispatch 状态。

这些类型的精确定义可能变化，遇到生命周期问题必须回到当前版本头文件确认。

### 5. 读“输出参数”和原地算子

看到：

```cpp
Tensor& op_out(..., Tensor& out);
Tensor& op_(Tensor& self, ...);
```

要核对：

- `out` 是否需要 resize。
- alias contract 是否允许输入输出指向同一 storage。
- contiguous/format 转换是否创建了临时输出。
- 临时输出如何回写原 `out`。
- autograd 对原地修改的 version counter 有何要求。

算子的 schema 不只是类型签名，也描述 mutation/alias 行为；自定义算子契约见 [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)。

### 6. 读异步调用时检查生命周期

设备 API 返回成功通常只表示“成功下发”，不保证 kernel 已完成。检查：

- 输入/output storage 是否由 tensor 持有到任务完成。
- workspace 谁分配，allocator 是否感知 stream。
- descriptor/host 参数是同步复制还是异步读取。
- 错误是在 launch 时返回，还是下一次同步才返回。

这也是 [[linux-debugging-for-npu-adaptation]] 中“报错位置不一定是失败位置”的 C++ 根源。

---

## 九、常见阅读误区

1. **看到裸指针就判定泄漏。** 它可能是 non-owning view，必须找 owner。
2. **看到 `std::move` 就认为数据已移动。** 真正移动发生在接收方构造/赋值。
3. **把宏当函数。** 宏可能重复求值、拼接名字或生成静态对象。
4. **只读 `.cpp`，忽略 YAML 和生成文件。** PyTorch 很多声明与 wrapper 来自代码生成。
5. **以为复制 `Tensor` 会复制设备数据。** 通常只是 handle/引用计数变化。
6. **认为 shared_ptr 解决了并发。** 它只管理生命周期，不保护对象内部状态。
7. **沿 include 树无限下钻。** 应围绕控制流、数据流和所有权按需阅读。
8. **用最新文档解释旧分支。** 内部 API 变化快，精确行为以当前 checkout 源码为准。

---

## 十、两周学习安排

### 第 1–2 天：类型与生命周期

- 引用、`const`、`auto`、值类别。
- 构造/析构、复制/移动。
- 练习：给一个函数签名标注 owning、borrowed、mutable。

### 第 3–4 天：RAII 与智能指针

- `unique_ptr`、`shared_ptr`、`weak_ptr`。
- 读一个 guard 类和一个 handle wrapper。
- 练习：找出一次异常路径上资源如何释放。

### 第 5–7 天：模板

- 函数/类模板、推导、特化、`if constexpr`。
- 能识别 traits、SFINAE/concepts。
- 练习：把一个 dtype dispatch 手工展开成两种具体类型。

### 第 8–9 天：宏和代码生成

- `#`、`##`、可变参数、预处理输出。
- 练习：展开一个注册宏，标出静态初始化行为。

### 第 10–12 天：编译与链接

- `.o`、`.so`、符号、RPATH、name mangling。
- 练习：用 `ldd`、`readelf`、`nm -C` 解释一个扩展如何加载。

### 第 13–14 天：端到端源码追踪

- 选一个简单 ATen 算子。
- 从 schema 追到 CPU kernel，再对照一个设备后端实现。
- 输出四列表：schema、registration、wrapper、device call。

---

## 十一、自测题

- `const Tensor&` 与按值传 `Tensor` 在语义和成本上可能有什么差别？
- `std::move(x)` 是否保证已经发生内存移动？
- `shared_ptr` 为什么仍会泄漏？`weak_ptr` 如何打破环？
- RAII guard 看似“未使用”，为什么不能删？
- 一个宏实参中含 `i++`，你必须检查什么？
- 为什么算子源码存在，dispatcher 仍可能报告没有该后端实现？
- `ldd` 显示加载了另一个目录的 `.so`，这如何解释“修改没有生效”？
- 模板报错几百行时，应该从哪些锚点开始读？
- 异步设备调用返回后，workspace 为什么不能立即按普通局部内存理解？

---

## 相关

[[npu-training-adaptation-learning-path]] · [[linux-debugging-for-npu-adaptation]] · [[python-advanced-mechanisms-for-pytorch]] · [[torch-dispatcher]] · [[deep-learning-training-numerics]]
