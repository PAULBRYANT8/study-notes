---
title: PyTorch 适配所需的 Python 进阶机制
type: concept
created: 2026-07-26
updated: 2026-07-27
tags: [python, pytorch, 装饰器, 上下文管理器, torch-function, c-extension]
sources:
  - https://docs.python.org/3/glossary.html#term-decorator
  - https://docs.python.org/3/reference/compound_stmts.html#with
  - https://docs.python.org/3/library/contextlib.html
  - https://docs.python.org/3/extending/extending.html
  - https://docs.python.org/3/c-api/intro.html
  - https://docs.pytorch.org/docs/stable/notes/extending.html
  - https://docs.pytorch.org/docs/stable/torch.overrides.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
  - https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
---

**读 PyTorch 适配代码时，Python 进阶的关键是识别“调用被谁包了一层、临时状态由谁管理、Tensor 操作在哪个协议被截获，以及 Python 如何跨进 C++”。**

这四类机制经常叠在一起：一个被装饰器包装的训练函数，在上下文管理器中运行，tensor 调用先经过 `__torch_function__`/mode，再进入 C++ 扩展与 dispatcher。只盯着表面那一行 `torch.add(x, y)`，很容易漏掉真正执行的代码。

---

## 一、先掌握 Python 的对象与调用模型

### 1. 函数也是对象

```python
def add(a, b):
    return a + b

f = add
functions = {"add": add}
result = functions["add"](1, 2)
```

函数可以被赋值、传参、返回和保存在容器中。装饰器、callback、hook、registry 的基础都是“把 callable 当普通对象操作”。

### 2. callable 不只包括函数

实现 `__call__` 的实例也可被调用：

```python
class Counter:
    def __init__(self):
        self.calls = 0

    def __call__(self, fn):
        self.calls += 1
        return fn()
```

看到某变量后面有 `(...)`，不要默认它是函数；先看其类型和 `__call__`。Python callable 的定义见 [Python glossary: callable](https://docs.python.org/3/glossary.html#term-callable)。

### 3. 闭包捕获外层变量

```python
def make_logger(prefix):
    count = 0

    def log(message):
        nonlocal count
        count += 1
        print(prefix, count, message)

    return log
```

`log` 离开 `make_logger` 后仍持有 `prefix` 和 `count`。装饰器常用闭包保存配置和状态。调试时可检查：

```python
print(log.__closure__)
print(log.__code__.co_freevars)
```

循环中创建闭包要小心 late binding：

```python
hooks = []
for i in range(3):
    hooks.append(lambda x, i=i: (i, x))  # 用默认参数冻结当前 i
```

对应文档：[Execution model: binding of names](https://docs.python.org/3/reference/executionmodel.html#binding-of-names)。

### 4. 方法绑定来自 descriptor

```python
class A:
    def f(self, x):
        return x

A.f       # 函数/descriptor
A().f     # 已绑定方法，实例会作为 self 传入
```

这能帮助理解为什么：

- `@classmethod` 收到 `cls`。
- `@staticmethod` 不发生实例绑定。
- 某些 hook 定义成 classmethod 后签名发生变化。

descriptor 是函数、method、property 等行为的底层机制。对应文档：[Descriptor HowTo Guide](https://docs.python.org/3/howto/descriptor.html)。

---

## 二、装饰器：定义完成后替换对象

### 1. 语法糖的真实含义

```python
@decorator
def train_step(batch):
    ...
```

等价于：

```python
def train_step(batch):
    ...

train_step = decorator(train_step)
```

装饰器在函数定义执行时调用，而不是每次调用 `train_step` 时重新“装饰”。官方定义见 [Python glossary: decorator](https://docs.python.org/3/glossary.html#term-decorator)。

### 2. 最小函数装饰器

```python
from functools import wraps


def trace_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print("enter", fn.__qualname__)
        try:
            return fn(*args, **kwargs)
        finally:
            print("exit", fn.__qualname__)

    return wrapper
```

`functools.wraps` 会复制常见元数据并设置 `__wrapped__`，有利于 `inspect`、help、测试框架和调试器找到原函数。对应文档：[`functools.wraps`](https://docs.python.org/3/library/functools.html#functools.wraps)。

### 3. 带参数的装饰器是三层调用

```python
def retry(max_attempts):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for _ in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except RuntimeError as error:
                    last_error = error
            raise last_error

        return wrapper

    return decorate


@retry(max_attempts=3)
def run():
    ...
```

执行顺序：

```text
retry(max_attempts=3)  → 返回 decorate
decorate(run)          → 返回 wrapper
run = wrapper
run()                  → 执行 wrapper
```

排查参数来自哪里时，分清“装饰器工厂参数”“被装饰函数参数”和“闭包保存的状态”。

### 4. 多个装饰器的顺序

```python
@outer
@inner
def f():
    ...
```

等价于：

```python
f = outer(inner(f))
```

调用时通常先进入 `outer` 的 wrapper，再进入 `inner`。若两个装饰器都改变 grad mode、autocast、日志或异常，顺序会影响行为。

### 5. 类装饰器

```python
class CountCalls:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)
```

类装饰器把原函数替换成实例。它容易携带状态，但也可能影响 descriptor 绑定、序列化和多进程复制；装饰方法时必须验证绑定行为。

### 6. 调试被装饰的函数

```python
import inspect

print(train_step)
print(train_step.__wrapped__)
print(inspect.unwrap(train_step))
print(inspect.signature(train_step))
print(inspect.getsource(inspect.unwrap(train_step)))
```

若 wrapper 没用 `wraps`，`__name__`、签名和 traceback 可能都只显示 `wrapper`，这会显著增加排障成本。

### 7. 装饰器常见陷阱

- wrapper 忘记 `return fn(...)`，导致返回值变成 `None`。
- 没有 `*args, **kwargs`，破坏原签名兼容性。
- 捕获 `Exception` 后吞掉异常，丢失原 traceback。
- 在 import/定义阶段做昂贵或依赖设备的初始化。
- 把可变状态存在全局闭包，多线程/多 rank 下不安全。
- wrapper 内部打印 tensor，意外触发设备同步。
- 多个装饰器改变上下文状态，顺序不清。

---

## 三、上下文管理器：把进入、退出和异常路径绑在一起

### 1. `with` 的概念展开

```python
with manager as value:
    body()
```

概念上相当于：

```python
mgr = manager
value = mgr.__enter__()
try:
    body()
except BaseException as error:
    suppress = mgr.__exit__(
        type(error),
        error,
        error.__traceback__,
    )
    if not suppress:
        raise
else:
    mgr.__exit__(None, None, None)
```

语言规范中的精确执行顺序见 [`with` statement](https://docs.python.org/3/reference/compound_stmts.html#with)。

### 2. 类形式

```python
class TemporaryMode:
    def __init__(self, state, value):
        self.state = state
        self.value = value
        self.previous = None

    def __enter__(self):
        self.previous = self.state.mode
        self.state.mode = self.value
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.state.mode = self.previous
        return False
```

`__exit__` 返回 truthy 会抑制异常；除非上下文管理器的明确职责就是处理特定异常，否则通常返回 `False` 或 `None`。

### 3. `contextlib.contextmanager`

```python
from contextlib import contextmanager


@contextmanager
def temporary_mode(state, value):
    previous = state.mode
    state.mode = value
    try:
        yield
    finally:
        state.mode = previous
```

`yield` 前对应 `__enter__`，`yield` 后对应 `__exit__`。必须用 `try/finally` 保证 body 抛异常时也恢复状态。对应文档：[`contextlib.contextmanager`](https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager)。

### 4. 多个 context 的进入与退出顺序

```python
with A() as a, B() as b:
    body()
```

进入顺序为 A → B，退出顺序为 B → A。它类似嵌套：

```python
with A() as a:
    with B() as b:
        body()
```

这对 device guard、autocast、grad mode、profiler、临时 hook 的组合很重要。

### 5. PyTorch 中常见用途

- `torch.no_grad()` / `torch.inference_mode()`：改变 autograd 记录语义。
- `torch.autocast(...)`：在作用域内启用混合精度策略。
- stream/device context：改变当前下发目标。
- profiler context：只记录指定区间。
- `TorchFunctionMode` / `TorchDispatchMode`：作用域内拦截操作。

不同 context 管理的是不同层的状态，不能因为语法都叫 `with` 就认为它们可互换。

### 6. `ExitStack` 管理动态数量资源

当 context 数量运行时才知道：

```python
from contextlib import ExitStack


with ExitStack() as stack:
    files = [
        stack.enter_context(open(path))
        for path in paths
    ]
    process(files)
```

已进入的 context 会按逆序退出，即使中途某次 enter 失败。对应文档：[`contextlib.ExitStack`](https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack)。

### 7. 常见陷阱

- 在 `__enter__` 改状态后抛异常，却没恢复一半修改。
- `__exit__` 意外返回 truthy，吞掉真正的设备错误。
- generator context 忘记 `try/finally`。
- context 跨线程/异步任务使用，但状态其实是 thread-local 或 task-local。
- 在 context 外使用只在作用域内有效的资源。
- 嵌套相同 context 时只保存一个全局旧值，退出顺序一乱就恢复错误。

---

## 四、Hook 与协议：调用为何会被“拦截”

“hook”不是一种统一机制。常见类别：

| 机制 | 触发位置 | 典型用途 |
|---|---|---|
| Python decorator | 函数定义/调用外层 | 日志、缓存、重试、模式切换 |
| `nn.Module` forward hook | module 前向边界 | dump 激活、统计、调试 |
| Tensor/autograd hook | 反向或梯度累积 | 观察/修改梯度 |
| `__torch_function__` | 公共 Python `torch` API | Tensor-like 类型与高层 API override |
| `__torch_dispatch__` | dispatcher 的 Python 覆盖层 | 更底层地观察/改写 ATen 运算 |
| dispatcher registration | C++/operator registry | 把 schema + dispatch key 映射到 kernel |

### `nn.Module` 的 forward pre-hook 与 forward hook

`model(x)` 先进入 `nn.Module.__call__`，再经过前向 hook 和 `forward`；直接写 `model.forward(x)` 会绕过这层包装。可将调用链简化为：

```text
args/kwargs
  → forward_pre_hooks （观察或修改输入）
  → forward(*args, **kwargs)
  → forward_hooks （观察或修改输出）
  → 返回 output
```

| hook | 默认签名 | 执行时机 | 返回非 `None` 时 |
|---|---|---|---|
| `register_forward_pre_hook` | `hook(module, args)` | `forward` 之前 | 用返回值作为新输入 |
| `register_forward_hook` | `hook(module, args, output)` | `forward` 之后 | 用返回值替换输出 |

pre-hook 返回新输入后，`forward` 会收到新值；forward hook 返回新输出后，调用者会收到新值。两者返回 `None` 都表示只观察。需要关键字参数时使用 `with_kwargs=True`，此时 pre-hook 应返回 `(new_args, new_kwargs)`。完整参数约定参见 [PyTorch Module hooks 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)。

```python
def add_one_before(module, args):
    (x,) = args
    return (x + 1,)


def clamp_after(module, args, output):
    return output.clamp(max=5)


handle_pre = module.register_forward_pre_hook(add_one_before)
handle_post = module.register_forward_hook(clamp_after)
```

它们的主要价值是把横切逻辑插到模块边界，而不必改写 `forward`：

- 记录输入/输出的 shape、dtype、device 和数值范围，定位 `NaN` 或维度错误。
- 收集中间 activation 做可视化、特征分析或知识蒸馏。
- 在计算前做输入适配，在计算后做输出替换或限制。
- 用前后两个 hook 记录单个模块的执行耗时。

几个边界条件：

- forward hook 执行时 `forward` 已经结束，修改它收到的输入不会影响本次计算；要改输入用 pre-hook。
- 仅保存 activation 时通常保存 `output.detach()`，否则可能把计算图一并保留而造成显存增长。
- 尽量避免在 hook 中原地改 Tensor；要改结果时返回新 Tensor。
- `register_*` 返回的 handle 用完后调用 `handle.remove()`，否则重复注册会重复执行。
- 多个 hook 默认按注册顺序执行；`prepend=True` 可将当前 hook 放到同类已有 hook 前面。`always_call=True` 可让 forward hook 在 `forward` 抛异常时也执行，适合清理和诊断。
- `register_full_backward_hook` 是反向传播阶段的另一类 hook，发生在 `.backward()` 期间，不在上面的前向调用链中。

不要把它们混为一谈。一个模块 forward hook 看不到所有函数式调用；`__torch_function__` 也不是设备 kernel 注册。

---

## 五、`__torch_function__`：公共 PyTorch API 的覆盖协议

### 1. 它解决什么问题

当自定义 Python 类型或 `Tensor` 子类作为参数传给可覆盖的 PyTorch API 时，PyTorch 可以调用该类型的 `__torch_function__`。它适合：

- Tensor wrapper 携带元数据。
- 记录公共 `torch` API 调用。
- 为 Tensor-like duck type 提供兼容实现。
- 在高层 API 维持自定义返回类型。

官方说明见 [Extending `torch` Python API](https://docs.pytorch.org/docs/stable/notes/extending.html#extending-torch) 和 [`torch.overrides`](https://docs.pytorch.org/docs/stable/torch.overrides.html)。

### 2. 基本签名

```python
class LoggingTensor(torch.Tensor):
    @classmethod
    def __torch_function__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}
        print(func.__qualname__)
        return super().__torch_function__(
            func,
            types,
            args,
            kwargs,
        )
```

参数含义：

- `func`：被覆盖的 PyTorch callable。
- `types`：本次参与分派、实现此协议的不同类型。
- `args` / `kwargs`：原调用参数。

Tensor 子类通常通过 `super().__torch_function__` 继续默认行为。直接在实现里再次调用同一个 `func(*args, **kwargs)`，若参数仍含该子类，可能无限递归。

### 3. `NotImplemented` 是协作信号

```python
if func not in HANDLED:
    return NotImplemented
```

它不等同于抛 `NotImplementedError`。返回 `NotImplemented` 表示“当前类型不处理，让其他参与类型尝试”；若所有实现都返回它，PyTorch 最终抛 `TypeError`。

多个类型共同参与时，子类通常比父类优先，其他顺序规则应以当前官方文档为准。实现只覆盖少量函数时，要对未覆盖行为有明确策略和测试。

### 4. wrapper 类型的 unwrap → call → wrap

概念模式：

```python
class MetadataTensor:
    def __init__(self, tensor, metadata):
        self.tensor = tensor
        self.metadata = metadata

    @classmethod
    def __torch_function__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}

        unwrapped = tuple(
            x.tensor if isinstance(x, cls) else x
            for x in args
        )
        result = func(*unwrapped, **kwargs)
        return cls(result, metadata="propagated")
```

真实实现还要递归处理 list/tuple/dict、多个输出、标量返回、`out=`、原地操作和 alias。只处理平铺位置参数的 demo 不能证明完整正确性。

### 5. `TorchFunctionMode`

mode 是上下文管理器，不要求把每个 tensor 都替换成特定子类，并且能拦截某些没有 Tensor 输入的函数：

```python
from torch.overrides import TorchFunctionMode


class LogMode(TorchFunctionMode):
    def __torch_function__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}
        print(func)
        return func(*args, **kwargs)


with LogMode():
    y = torch.add(x, x)
```

mode 的嵌套顺序和 subclass 的交互较复杂；需要覆盖全局 API 时阅读 [Extending all `torch` API with Modes](https://docs.pytorch.org/docs/stable/notes/extending.html#extending-all-torch-api-with-modes)。

### 6. `__torch_function__` 与 `__torch_dispatch__`

实用区分：

```text
Python public API
    ↓ __torch_function__
Python API decomposition / binding
    ↓ ATen operator
__torch_dispatch__
    ↓ C++ dispatcher / backend kernel
```

- `__torch_function__` 更高层，`func` 可能是 `torch`、`Tensor` method、`nn.functional` 等公共 API。
- `__torch_dispatch__` 更靠近 ATen 运算，覆盖面与语义更底层，适合 tensor subclass、proxy、functionalization 等高级用途。
- 二者都不是设备后端的 C++ dispatch key 注册。
- 高层函数可能分解成多个 ATen op，所以两层看到的操作粒度不同。

`__torch_dispatch__` 是高级接口，内部细节变化较快；应对照当前版本 [Tensor subclass extension points](https://docs.pytorch.org/docs/stable/notes/extending.html#tensor-subclasses) 和源码测试。

### 7. 协议实现的典型 bug

- 递归：实现内再次调用同一个协议入口。
- alias 丢失：每个输出都新建 wrapper，破坏 view/identity。
- 原地语义错误：返回了新对象，但 API 承诺修改原对象。
- `out=` 未处理。
- pytree 中嵌套 tensor 没有 unwrap/wrap。
- 标量、`None`、多返回值处理错误。
- dtype/device/stride 被无意改变。
- 日志中打印 tensor，再次触发协议。

---

## 六、其他常见 PyTorch hook

### 1. Module forward hook

```python
handle = module.register_forward_hook(hook)
try:
    output = module(input)
finally:
    handle.remove()
```

hook 应尽量轻量，并用 `try/finally` 移除。大量 dump 会同步设备、占满磁盘或保留 tensor 引用。接口见 [`Module.register_forward_hook`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)。

### 2. Tensor gradient hook

```python
handle = tensor.register_hook(
    lambda grad: torch.nan_to_num(grad)
)
```

返回新 gradient 会改变反向结果；只做观测时不要无意修改。接口和触发时机见 [`Tensor.register_hook`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html) 与 [Autograd mechanics: hooks](https://docs.pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution)。

### 3. saved tensor hooks

autograd 可在保存/读取反向中间量时调用 pack/unpack hook，用于 offload、压缩或调试。它影响生命周期和性能，错误实现会破坏反向。对应文档：[`saved_tensors_hooks`](https://docs.pytorch.org/docs/stable/autograd.html#torch.autograd.graph.saved_tensors_hooks)。

---

## 七、Python 如何调用 C/C++ 扩展

### 1. 从 `import` 到机器码

```text
import my_extension
  → import system 在 sys.path 中查找模块
  → 找到平台相关的 .so
  → 动态链接器加载该 .so 及依赖
  → 调用模块初始化函数（CPython C API 通常为 PyInit_*）
  → 初始化函数创建 module，并注册 Python 可见方法/类型
my_extension.op(x)
  → Python callable/binding
  → 参数类型检查与转换
  → C/C++ 函数
  → ATen / dispatcher / 设备 API
  → C/C++ 返回值转为 Python 对象
```

CPython 扩展的基础流程见 [Extending Python with C or C++](https://docs.python.org/3/extending/extending.html)；C API 总览见 [Python/C API Introduction](https://docs.python.org/3/c-api/intro.html)。

### 2. 一个最小 CPython C API 轮廓

```c
static PyObject* add_one(
    PyObject* self,
    PyObject* args
) {
    long value;
    if (!PyArg_ParseTuple(args, "l", &value)) {
        return NULL;
    }
    return PyLong_FromLong(value + 1);
}

static PyMethodDef methods[] = {
    {"add_one", add_one, METH_VARARGS, "Add one."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "example",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_example(void) {
    return PyModule_Create(&module);
}
```

关键约定：

- C 函数返回 `PyObject*`。
- Python 异常状态通常通过“设置异常 + 返回 `NULL`”传播。
- 参数解析失败后不能继续使用未初始化值。
- reference ownership 必须遵守 new/strong/borrowed reference 规则。

对应文档：[Defining extension modules](https://docs.python.org/3/extending/extending.html#the-module-s-method-table-and-initialization-function) · [Reference counting](https://docs.python.org/3/c-api/refcounting.html)。

### 3. pybind11 / PyTorch binding 做了什么

现代 C++ 扩展常用 pybind11 或 PyTorch 提供的注册 API隐藏原始 C API 样板：

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("op", &op);
}
```

或通过 `TORCH_LIBRARY` 把 operator 注册到 dispatcher，再从 `torch.ops.namespace.op` 调用。两条路径的差别是：

- pybind 暴露普通 Python callable。
- dispatcher operator 有明确 schema 和 dispatch key，可与 autograd、compile、fake tensor 等 PyTorch 子系统组合。

PyTorch 当前推荐的自定义 C++ operator 路径见 [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)；构建辅助 API 见 [`torch.utils.cpp_extension`](https://docs.pytorch.org/docs/stable/cpp_extension.html)。

### 4. GIL

传统 CPython 构建中，执行 Python bytecode和大多数 C API 前需要持有 GIL。C/C++ 扩展可在长时间、不访问 Python 对象的计算期间释放 GIL，让其他 Python 线程运行；重新访问 Python API 前必须恢复。

不要简单理解成“有 GIL 就不会并发”：

- C 扩展可能释放 GIL。
- 设备工作本身异步并行。
- 多进程训练不共享同一个 GIL。
- 新版本 CPython 还存在 free-threaded build，扩展兼容性要求不同。

以当前 Python 版本的 [Thread State and the Global Interpreter Lock](https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock) 为准。

### 5. 引用计数与生命周期

CPython C API 会区分：

- **strong/new reference**：调用者拥有，需要适时 `Py_DECREF` 或转移所有权。
- **borrowed reference**：不拥有；原 owner 销毁后指针可能悬空。

典型原生崩溃来源：

- borrowed reference 跨越 owner 生命周期。
- 少 `INCREF` 造成 use-after-free。
- 少 `DECREF` 造成泄漏。
- 多 `DECREF` 造成 double free/use-after-free。
- 后台线程在没有正确 thread state 的情况下调用 Python API。

引用语义见 [Reference Counting](https://docs.python.org/3/c-api/refcounting.html)。

### 6. C++ 异常如何回到 Python

binding 层应把 C++ 异常转换为 Python 异常。若异常越过不允许的 C ABI 边界、原生代码触发未定义行为或直接收到信号，Python 不一定有机会生成 traceback，进程可能直接崩溃。

因此：

```text
Python traceback
  → binding 正常捕获并转换了错误

Segmentation fault / abort
  → 进入 core + GDB，不能只靠 pdb
```

与原生崩溃的联动流程见 [[linux-debugging-for-npu-adaptation]]。

---

## 八、定位“这个 Python API 最终调用了哪个 C++”

### 1. 先做 Python introspection

```python
import inspect

print(type(obj))
print(obj)
print(obj.__module__)
print(getattr(obj, "__qualname__", None))
print(inspect.signature(obj))
print(inspect.getsourcefile(obj))
print(inspect.getsource(obj))
```

内建函数、pybind function 或 C extension method 往往没有 Python source；这本身就是“已经跨到原生绑定”的证据。

### 2. 看模块文件

```python
import my_extension

print(my_extension.__file__)
print(my_extension.op)
```

若 `__file__` 指向 `.so`，用：

```bash
ldd path/to/extension.so
nm -C path/to/extension.so
readelf -Ws path/to/extension.so
```

确认依赖和符号。若 import 报 undefined symbol，先检查实际加载的 `.so` 和 ABI，不要在 Python 逻辑里兜圈。

### 3. 找 dispatcher operator

若 API 最终变成 `torch.ops.ns.op` 或 `aten::op`：

- 查 operator schema。
- 查 `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`。
- 查对应 dispatch key 的 kernel。
- 区分 Python wrapper 与真正设备实现。

这部分与 [[cpp-reading-for-pytorch-backends]] 和 [[torch-dispatcher]] 相连。

### 4. 用最小 hook 观察，不要让观测改变行为

日志只记录：

- op 名。
- shape/dtype/device。
- 调用计数。
- 必要时小型标量统计。

避免：

- 打印完整设备 tensor。
- 在每个 op 中 `.cpu()` / `.item()`。
- 保存所有输入的强引用。
- hook 内再次执行大量 torch op，造成递归。

---

## 九、一个组合示例：装饰器 + context + hook

```python
from contextlib import contextmanager
from functools import wraps


@contextmanager
def registered_hook(module, hook):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def capture_first_call(module, hook):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with registered_hook(module, hook):
                return fn(*args, **kwargs)

        return wrapper

    return decorate
```

阅读这段代码应能说清：

1. import/定义时，装饰器工厂保存 `module` 和 `hook`。
2. 被装饰函数被替换成 `wrapper`。
3. 每次调用 wrapper 才注册 hook。
4. 无论原函数成功还是抛异常，`finally` 都移除 hook。
5. hook 中若执行 torch op，仍可能触发 `__torch_function__`/dispatcher。

---

## 十、常见误区

1. **把装饰器当作调用时注释。** 它会在定义完成后替换对象。
2. **忘记多装饰器顺序。** `@outer @inner` 是 `outer(inner(f))`。
3. **上下文管理器只处理正常路径。** 它最重要的价值恰恰是异常路径恢复。
4. **`__exit__` 返回值无所谓。** truthy 会抑制异常。
5. **`NotImplemented` 等于报错。** 它是多类型协议的协作返回值。
6. **`__torch_function__` 就是 dispatcher。** 它位于更高的 Python API 层。
7. **C 扩展报错一定有 traceback。** 原生未定义行为可能直接导致信号和 core。
8. **import 到的 `.so` 就是刚编译的那份。** 必须检查 `module.__file__` 和动态依赖。
9. **hook 只观察不影响行为。** 打印、保存引用、执行 tensor op 都可能改变同步、内存和递归。

---

## 十一、两周学习安排

### 第 1–3 天：函数对象、闭包、descriptor

- 自己写 callable 类、闭包和方法装饰器。
- 用 `inspect` 找原函数、签名和 closure。

### 第 4–5 天：装饰器

- 写无参数、有参数、类形式三种装饰器。
- 验证多装饰器顺序和异常 traceback。

### 第 6–7 天：context manager

- 分别用类和 `@contextmanager` 实现临时状态切换。
- 故意在 body 抛异常，确认状态恢复。

### 第 8–10 天：PyTorch override/hook

- 写一个记录 `torch.add` 的 `Tensor` wrapper。
- 正确处理 `NotImplemented` 和递归。
- 写 forward hook，确保 `finally` 移除。

### 第 11–12 天：C 扩展调用链

- 读一个最小 CPython C API 或 pybind11 模块。
- 能指出模块初始化、参数转换、异常转换和返回值构造。

### 第 13–14 天：追一个真实 PyTorch API

- 从 Python wrapper 追到 `torch.ops`/ATen schema。
- 再追到 C++ 注册和后端 kernel。
- 画出高层 override 与低层 dispatcher 的边界。

---

## 十二、自测题

- `@a @b def f` 的定义和调用顺序分别是什么？
- 为什么装饰器应使用 `functools.wraps`？
- `__exit__` 返回 `True` 会发生什么？
- generator context 为什么必须把 `yield` 放在 `try/finally` 中？
- `__torch_function__` 里直接调用 `func(*args)` 为什么可能递归？
- 返回 `NotImplemented` 与抛 `NotImplementedError` 有什么区别？
- `__torch_function__` 与 `__torch_dispatch__` 分别位于哪一层？
- `import extension` 后如何确认实际加载了哪一个 `.so`？
- C 扩展为什么可能绕过 Python traceback 直接 segfault？
- hook 中打印完整 NPU tensor 为什么会扰动性能和报错位置？

---

## 相关

[[npu-training-adaptation-learning-path]] · [[cpp-reading-for-pytorch-backends]] · [[linux-debugging-for-npu-adaptation]] · [[deep-learning-training-numerics]] · [[torch-dispatcher]]
