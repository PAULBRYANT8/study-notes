# Makefile 常用语法与 Linux Kbuild 笔记

Makefile 描述目标、依赖及生成目标所需的命令；GNU Make 根据依赖关系和文件时间决定需要执行哪些命令。本文以 GNU Make 为准，最后结合 VIPROC 驱动介绍 Linux 内核 Kbuild 的写法。



`CMakeLists.txt` 解决的是启动之前的问题：**这个程序由哪些源文件和库组成，要用什么选项把它构建出来？**

## 1. 基本规则

```makefile
目标: 依赖1 依赖2
	命令
```

例如：

```makefile
app: main.o util.o
	$(CC) -o app main.o util.o

main.o: main.c util.h
	$(CC) -c main.c -o main.o

util.o: util.c util.h
	$(CC) -c util.c -o util.o
```

- `app` 依赖 `main.o` 和 `util.o`；Make 会先更新依赖，再判断 `app` 是否需要更新。
- 普通文件目标不存在，或某个普通依赖比目标新时，通常需要重新执行目标的命令。头文件列为依赖后，头文件变化也会触发对应目标更新。
- 命令行默认必须以真正的 **Tab** 开头，而不是几个空格；GNU Make 允许用 `.RECIPEPREFIX` 修改这个前缀。
- 默认构建目标通常是第一个非特殊目标；也可用 `.DEFAULT_GOAL := app` 指定。
- Make 根据依赖判断，不能仅靠时间戳可靠识别所有环境变化，例如编译选项变化或时间戳异常。

## 2. 变量和赋值

```makefile
NAME = hello
OUT := $(NAME)
CFLAGS += -Wall
PREFIX ?= /usr/local
```

引用变量常写为 `$(NAME)`，也可写 `${NAME}`。变量名区分大小写。常见赋值方式：

- `=`：递归展开；右侧变量引用在使用该变量时才展开。
- `:=`：立即展开右侧并保存结果；适合想在赋值时固定结果的情况。
- `?=`：仅当变量**未定义**时赋值；已定义但值为空时也不会赋值。
- `+=`：追加内容；右侧何时展开取决于原变量类型：原变量是 `:=` 类型时立即展开，是 `=` 类型时延迟展开；变量尚未定义时按递归展开变量处理。

```makefile
A = $(B)
C := $(B)
B = world
# 此时 $(A) 是 world；$(C) 在赋值时 B 未定义，所以为空。
```

`CC`、`CFLAGS`、`CPPFLAGS`、`LDFLAGS`、`LDLIBS` 常用于指定编译器、C 编译选项、预处理选项、链接选项和链接库。它们是惯用变量和内置规则使用的变量；自定义命令也需要显式引用它们，才会生效。

命令行可以传入变量：

```bash
make CC=clang CFLAGS='-O2 -Wall'
```

命令行变量赋值一般优先于 Makefile 中普通赋值；Makefile 可用 `override` 显式改变这一行为。环境变量也可以传入 Make，但通常会被 Makefile 中的普通赋值覆盖。

## 3. 伪目标和常用命令

```makefile
.PHONY: all clean

all: app

clean:
	rm -f app main.o util.o
```

`all` 和 `clean` 是动作名称，不一定对应同名文件。`.PHONY` 告诉 Make 它们是伪目标，避免同名文件妨碍执行；执行 `make clean` 时也不需要先生成 `app`。

```bash
make                 # 构建默认目标
make app             # 构建 app
make clean           # 执行 clean 规则
make -j4             # 最多并行运行 4 个任务；前提是依赖关系写对
make -n              # 打印将执行的命令，通常不实际执行
make -B              # 强制认为目标需要更新
make -f Build.mk     # 使用指定的 Makefile
make -C /path/to/project  # 先切换目录再执行
make V=1             # 仅对项目支持 V=1 的构建系统才有显示详细命令的效果
```

`make -n` 并非无条件的安全沙箱：某些特殊命令（如含 `$(MAKE)` 的递归调用）仍可能执行；Make 在解析或展开时也可能运行 `$(shell ...)` 等操作。不要对不可信的 Makefile 假定它完全无副作用。

## 4. 自动变量和模式规则

```makefile
%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

app: main.o util.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
```

- `%`：模式中的占位部分；例如 `main.o` 可匹配 `%.o`，对应依赖为 `main.c`。
- `$@`：当前规则的目标名；上例中可能是 `main.o` 或 `app`。
- `$<`：第一个依赖；编译 `main.o` 时为 `main.c`。
- `$^`：所有普通依赖，去除重复项，以空格分隔。
- `$+`：所有普通依赖，保留重复项和顺序，链接库需要重复时可能有用。
- `$?`：比目标新的普通依赖。
- `$*`：模式规则中的匹配部分，例如 `main`。

这些自动变量主要在规则的命令中使用。实际项目还需要为头文件建立依赖，否则只写 `%.o: %.c` 时头文件变化不一定触发重新编译。

## 5. 常用字符串和文件名函数

函数通常写作 `$(函数名 参数)`：

```makefile
SRC := $(wildcard *.c)
OBJ := $(patsubst %.c,%.o,$(SRC))
# 简写：OBJ := $(SRC:.c=.o)

HEADERS := $(addprefix include/,config.h types.h)
NAMES := $(notdir src/main.c lib/util.c)
DIRS := $(dir src/main.c lib/util.c)
TEXT := $(strip   a   b  )
```

- `wildcard`：列出当前匹配到的文件；未匹配的模式会被省略。配合 `:=` 使用时，匹配发生在赋值时。
- `patsubst`：按模式替换每个单词；上例将 `.c` 文件名转为 `.o`。
- `addprefix` / `addsuffix`：给每个单词添加前缀 / 后缀。
- `notdir` / `dir`：提取文件名部分 / 目录部分；`dir` 的结果保留结尾斜杠。
- `strip`：去掉首尾空白，并把内部连续空白压缩成单个空格。
- `filter` / `filter-out`：按照模式保留 / 排除单词。
- `shell`：执行 shell 命令并取其标准输出；它在函数被展开时执行，换行会转换为空格。需要注意执行时机和额外开销。

```makefile
CFILES := $(filter %.c,$(SRC))
VERSION := $(shell uname -r)
```

## 6. 条件判断、包含文件和注释

```makefile
ifeq ($(MODE),debug)
CFLAGS += -g -O0
else
CFLAGS += -O2
endif

ifneq ($(strip $(FEATURE)),)
CFLAGS += -DFEATURE_ENABLED
endif
```

- `ifeq` / `ifneq`：比较两个展开后的字符串是否相等 / 不相等。
- `ifdef` / `ifndef`：判断变量是否具有非空值 / 非该情况；判断的是变量本身的值，复杂的间接展开判断可明确使用 `$(strip ...)` 配合 `ifeq`。
- Make 的条件语句在**读取 Makefile 时**生效，不是 shell 的运行时 `if`。
- `#` 通常用于 Makefile 注释；命令行中则由 shell 解析，细节取决于上下文和引用方式。

```makefile
include config.mk
-include optional.mk
```

`include` 读取另一个 Makefile；缺失文件会在 Make 尝试重建它后导致错误。`-include` 对无法读取或重建的包含文件不报致命错误，常用于可选配置和自动生成的依赖文件。两者都先展开文件名。

## 7. 命令、shell 与转义

```makefile
show:
	@echo "target=$@"
	@echo "shell HOME=$$HOME"
```

- 命令前的 `@` 让 Make 不打印该命令本身，命令输出仍会显示。
- 命令前的 `-` 让 Make 忽略该命令的失败退出状态；不要随意用于可能隐藏真实错误的步骤。
- `$(NAME)` 由 Make 展开；要把 `$HOME` 等美元符号交给 shell，需要在 Makefile 中写 `$$HOME`。
- 默认情况下，每条独立的命令行在单独的 shell 中执行。`cd`、shell 变量赋值不会自动延续到下一行。

```makefile
# 保证 cd 和后续命令位于同一个 shell：
build-subdir:
	cd subdir && $(MAKE)
```

使用 `$(MAKE)` 递归调用 Make，可正确传递某些 Make 的选项和作业控制信息。`.ONESHELL` 等特殊设置可以改变每行 shell 的默认行为。

## 8. 自动生成头文件依赖

```makefile
CC := gcc
CPPFLAGS += -MMD -MP
CFLAGS += -Wall -O2
SRC := main.c util.c
OBJ := $(SRC:.c=.o)
DEP := $(OBJ:.o=.d)

.PHONY: all clean
all: app
app: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

-include $(DEP)
clean:
	rm -f app $(OBJ) $(DEP)
```

GCC 的 `-MMD` 在编译时生成用户头文件依赖，`-MP` 为依赖中的头文件添加伪规则，减少头文件删除后旧依赖文件导致的问题；`-include` 允许首次构建时 `.d` 文件尚不存在。示例假设这些源文件都存在且提供可链接的程序。

## 9. Linux 内核 Kbuild 常用写法

以下变量由 Linux 内核 Kbuild 解释，不是普通 GNU Make 本身内建的模块规则：

```makefile
obj-m += drv_viproc.o
drv_viproc-objs := viproc_probe.o viproc_dev.o viproc_proc.o viproc_irq.o viproc_func.o
ccflags-y += -I$(src)/include
```

- `obj-m`：列出按可加载模块构建的目标；上例通常生成 `drv_viproc.ko`。
- `drv_viproc-objs`：列出组成该模块的对象文件；模块基础名必须与 `drv_viproc.o` 对应。Kbuild 也支持常见的 `drv_viproc-y` 写法。
- `obj-y`：列出构建进内核的对象；`obj-$(CONFIG_FOO)` 可随配置值 `y` 或 `m` 选择内建或模块构建，其他值不会作为这两类目标处理。
- `ccflags-y`：向当前 Kbuild 文件所在目录的 C 编译追加选项；`-I` 添加头文件搜索目录。
- `$(src)`：Kbuild 中当前源目录的路径，适合从 Kbuild 文件引用该目录下的头文件。
- `-Wno-error=date-time`：使 `date-time` 这一类警告不因对应的 `-Werror` 而升级为错误；不等于禁止该警告，也不保证启用该警告。

外部模块的常见构建形式：

```bash
make -C /path/to/kernel/build M="$PWD" modules
```

`-C` 指定已准备好的目标内核构建目录，`M` 指向外部模块的绝对路径。模块需要匹配目标内核的配置及构建环境；交叉编译还须使用项目要求的架构和工具链。这个通用示例不能替代特定项目的构建脚本。

本机 VIPROC 文件中的 `include $(M)/modules.mk`、`MODULES_PATH`、`PRODUCT`、`CHIP_NAME` 和 `CHIP_VER` 涉及项目外部配置；当前 `monalisa` 工作区未找到 `modules.mk` 及这些变量的定义，无法仅凭该文件确定它们的实际值或完整构建命令。第 4 行和第 6 行重复添加了产品头文件目录，通常属于冗余搜索路径。

## 10. 常见问题速查

- `missing separator`：先检查规则命令是否以 Tab 开头，以及行是否误写成命令或规则。
- `No rule to make target ...`：检查文件名、工作目录、依赖是否存在，或是否缺少生成该依赖的规则。
- 头文件修改后没有重新编译：检查头文件是否出现在依赖中，考虑使用自动依赖文件。
- 并行编译偶发失败：检查目标之间是否遗漏依赖，不能只依靠规则书写顺序。
- 找不到头文件：检查实际编译命令中的 `-I` 路径及顺序；带双引号的 `#include` 还涉及包含文件所在目录等搜索规则。
- 改了编译选项但没有重新编译：Make 通常只根据依赖和时间戳判断；应采用项目提供的重新构建方式，或将配置变化纳入依赖管理。

## 11. CMakeLists.txt、Makefile 与 CMake 生成的文件

C 项目可以选用 GNU Make、CMake、Meson 等构建系统。看到一个文件时，先辨认它是项目维护的输入，还是工具在配置或构建时生成的产物。

- `CMakeLists.txt` 是 CMake 的构建描述。顶层文件是配置入口，子目录可以通过 `add_subdirectory()` 加入构建。常见命令有 `add_executable()`（定义可执行程序）、`add_library()`（定义库）、`target_include_directories()`（设置目标的头文件搜索路径）和 `target_link_libraries()`（设置目标的链接依赖）。修改源文件列表或目标依赖时，通常从这些命令入手。
- `Makefile` 是 Make 读取的规则文件。它既可以是项目维护的文件，也可以是 CMake 在构建目录生成的文件。如项目按 CMake 配置，编译规则的改动通常写回 `CMakeLists.txt` 或相关的 `.cmake` 文件，重新配置时便会反映到生成的规则里。
- `CMakefile` 这个拼法没有约定的 CMake 入口含义。实际遇到同名文件时，查看文件内容和项目中引用它的位置。容易混淆的 `CMakeFiles/` 带复数 `s`，是 CMake 生成的目录，保存内部构建信息、中间文件等。
- `CMakeCache.txt` 是构建目录的配置缓存，记录选中的生成器、编译器路径、依赖探测结果和部分项目选项。排查选项为何仍使用旧值时可以查看它。切换工具链或生成器时，通常为新配置使用单独的构建目录。
- `build.ninja` 是使用 Ninja 生成器时产生的规则文件，对应 Unix Makefiles 生成器产生的 `Makefile`。`cmake --build` 会调用已选的后端。
- `compile_commands.json` 是编译数据库，记录每个编译单元的源文件、工作目录和编译命令。clangd、clang-tidy 等工具可用它取得项目实际使用的头文件路径和宏。CMake 的 Makefile 或 Ninja 生成器可用 `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 导出它。

相关文档见 [CMake 命令行手册](https://cmake.org/cmake/help/latest/manual/cmake.1.html)、[CMake 语言](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)、[目标和依赖](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html)、[Unix Makefiles 生成器](https://cmake.org/cmake/help/latest/generator/Unix%20Makefiles.html)、[Ninja 生成器](https://cmake.org/cmake/help/latest/generator/Ninja.html)、[导出编译数据库](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html)。

### 11.1 一个最小的 CMake 项目

假设 `main.c` 中有 `main()`，并且它和以下 `CMakeLists.txt` 放在同一个目录。

```cmake
cmake_minimum_required(VERSION 3.16)
project(demo LANGUAGES C)
add_executable(demo main.c)
```

在源码目录运行以下命令。

```bash
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build
```

第一条命令从当前目录读取配置，在 `build/` 中写入 `Makefile`、`CMakeCache.txt`、`CMakeFiles/` 和编译数据库。第二条命令调用 Make，执行编译与链接。在常见的 Linux 环境下，产物可用 `./build/demo` 运行。若选择 Ninja，可在已安装 Ninja 的环境中用 `-B build-ninja -G Ninja` 配置，再用 `cmake --build build-ninja` 构建。

可按 [[work/example/cmake-build-vs-run/readme|CMake 构建与直接运行示例]] 操作一遍。示例用同一份 C 源码说明 `CMakeLists.txt`、`CMakeFiles/` 等文件在构建时的作用，以及构建程序和直接执行已有程序的区别。

### 11.2 为什么 VIPROC 目录里两种文件都有

在本机 `monalisa/dirvers/device/cv/viproc/` 中，`CMakeLists.txt` 只有下面这一行。

```cmake
add_mdrv_target("device" drv_viproc "${CMAKE_CURRENT_SOURCE_DIR}")
```

这行把类别字符串 `"device"`、目标名 `drv_viproc` 和当前源码目录传给项目的 `add_mdrv_target()`。同目录的手写 `Makefile` 则用 `obj-m`、`drv_viproc-objs` 和 `ccflags-y` 描述内核模块、组成对象和编译选项（见第 9 节）。当前工作区只找到 `add_mdrv_target()` 的调用，未找到定义，实际的调用链应在上层构建工程中确认。

## 12. 源码仓库常见的其他文件

### 12.1 构建配置、依赖与脚本

- `*.cmake` 可存放 CMake 公共函数、依赖查找模块、工具链设置或软件包信息。例如 `cmake/toolchain-aarch64.cmake` 可能通过 `-DCMAKE_TOOLCHAIN_FILE=路径` 指定，用于设置目标系统、编译器或 sysroot（查找目标系统头文件和库的根目录）。`FooConfig.cmake` 则可能由库提供，供其他项目通过 `find_package(Foo CONFIG)` 获取库的配置。
- `CMakePresets.json` 保存项目共享的配置、构建或测试参数，例如生成器、构建目录和缓存选项。`CMakeUserPresets.json` 保存开发者本机的预设，通常排除在版本控制之外。
- `*.mk` 是常见的 Makefile 片段，存放公共变量、选项或规则，通常通过 `include` 引入。VIPROC 的 `modules.mk` 就是这个命名形式，文件内容取决于上层工程。
- `config.h.in`、`*.in` 常用作模板。比如 CMake 的 `configure_file()` 可以把版本号或功能开关写入生成的 `config.h`。改模板中的内容，下次配置时会生成对应结果。
- `*.pc` 是 `pkg-config` 使用的库元数据，通常描述版本、依赖、头文件参数和链接参数，可用 `pkg-config --cflags --libs 库名` 查询。
- `meson.build` 是 Meson 的构建入口之一。在 GNU Autotools 项目中，常见 `configure.ac`、`Makefile.am` 和 `configure`。Autoconf 从 `configure.ac` 生成 `configure` 脚本，Automake 根据 `Makefile.am` 和 `configure.ac` 生成 `Makefile.in`，运行 `./configure` 通常会按当前环境生成 `Makefile`。
- `build.sh`、`configure.sh` 或 `scripts/` 下的 `.sh`、`.py` 脚本常用于设置工具链环境、调用构建工具、生成代码、运行测试或打包。具体入口通常写在 `README.md` 中。

相关文档见 [CMake 工具链](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)、[软件包查找](https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html)、[构建预设](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)、[文件模板](https://cmake.org/cmake/help/latest/command/configure_file.html)、[pkg-config 使用指南](https://www.chiark.greenend.org.uk/doc/pkg-config/pkg-config-guide.html)、[Meson 快速入门](https://mesonbuild.com/Quick-guide.html)、[GNU Automake 示例](https://www.gnu.org/software/automake/manual/html_node/Creating-amhello.html)。

### 12.2 文档、版本控制和开发工具

- `README.md` 介绍项目、依赖、构建和使用方法。`LICENSE` 或 `COPYING` 保存许可条款，`CHANGELOG.md` 记录版本变更。
- `.gitignore` 指定 Git 要忽略的未跟踪文件，常见条目有构建目录和本地预设。`.gitmodules` 记录 Git 子模块的路径、仓库地址等配置。
- `.clang-format` 配置代码格式，`.clang-tidy` 配置静态检查，`.editorconfig` 配置缩进、换行符和编码等编辑器行为。
- `.vscode/tasks.json` 可定义编辑器里运行的构建任务，`.vscode/launch.json` 定义调试启动参数，`.vscode/c_cpp_properties.json` 可配置 C/C++ 扩展的 IntelliSense。编辑器的头文件提示与实际构建选项来自各自的配置，排查编译报错时以构建命令为准。
- `.github/workflows/*.yml`、`.gitlab-ci.yml` 常用来定义自动构建、测试和打包流程。`*.ini`、`*.json`、`*.yaml` 也常用作应用配置或测试数据，具体用途看由哪个程序读取。

### 12.3 嵌入式和 Linux 驱动中常见的文件

- `Kconfig` 定义内核选项、类型和依赖，内核 `.config` 保存一次配置选出的值，`Kbuild` 或内核 `Makefile` 再据此决定构建哪些对象。例如第 9 节的 `obj-$(CONFIG_FOO)`。内核树中的 `*_defconfig` 通常是板卡或平台的默认配置模板。
- `*.S`、`*.s` 是汇编文件，常见于启动代码、中断入口和架构相关代码。按 GCC 约定，`.S` 会先预处理，可使用宏和 `#include`。`.s` 通常直接交给汇编器，也可能是编译器用 `-S` 输出的文件。
- `*.ld`、`*.lds` 常用作链接脚本，描述代码段、数据段等在 Flash、RAM 或其他地址空间里的布局。
- `*.dts`、`*.dtsi` 通常是设备树源码和可复用的包含片段，用来描述硬件资源，例如寄存器地址和中断。设备树编译器根据源码生成 `*.dtb`。Linux 的设备树绑定常用 `*.yaml` 描述节点的属性要求，例如 `compatible`、`reg` 和 `interrupts`。

相关文档见 [Linux Kconfig](https://docs.kernel.org/kbuild/kconfig.html)、[Linux Kbuild](https://docs.kernel.org/kbuild/makefiles.html)、[Linux 设备树](https://docs.kernel.org/devicetree/usage-model.html)、[设备树绑定](https://docs.kernel.org/devicetree/bindings/writing-schema.html)、[GCC 输入输出选项](https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html)、[GNU ld 链接脚本](https://sourceware.org/binutils/docs/ld.html#Scripts)。

## 13. 常见中间文件和构建产物

- `*.i` 是 C 预处理输出，可用来确认宏展开或实际包含的头文件。`*.s` 可作为查看编译器生成的汇编的中间文件。
- `*.o`（Windows 工具链常见 `*.obj`）是目标文件，包含编译后的代码、数据和链接所需的信息。`*.d` 通常是编译器生成的依赖列表，用于增量构建（见第 8 节）。
- `*.a` 一般是静态库，链接器从归档中选取需要的对象。`*.so` 是 Linux 下常见的共享库。Windows 的 `*.dll` 是动态链接库，`*.lib` 可能是静态库，也可能是 DLL 对应的导入库。
- `app`、`*.elf`、`*.exe` 常见于链接后的程序。Linux 用户态程序通常采用 ELF 格式，文件名可以没有扩展名。固件工程的 `*.elf` 常用来调试，`*.bin` 和 `*.hex` 可能是从它转换出的烧录文件，具体格式和加载地址取决于工程。
- `*.map` 常指链接器生成的映射文件，可查看段地址、符号和贡献这些内容的目标文件。排查内存布局或固件体积时可以结合链接脚本阅读。
- `*.ko` 是最终的 Linux 可加载内核模块。`Module.symvers` 列出构建导出的符号，并在启用模块版本校验时记录相应 CRC。`modules.order` 记录模块顺序。`*.cmd` 可保存 Kbuild 的命令及依赖信息，供增量构建使用。

相关文档见 [GCC 输入输出选项](https://gcc.gnu.org/onlinedocs/gcc/Overall-Options.html)、[GNU ld 选项](https://sourceware.org/binutils/docs/ld/Options.html)、[GNU objcopy](https://sourceware.org/binutils/docs/binutils/objcopy.html)、[内核外部模块](https://docs.kernel.org/kbuild/modules.html)、[Kbuild 输出文件](https://docs.kernel.org/kbuild/kbuild.html)。

## 参考资料

- [GNU Make 官方手册](https://www.gnu.org/software/make/manual/)
- [Linux 内核文档：Linux Kernel Makefiles](https://docs.kernel.org/kbuild/makefiles.html)
- [Linux 内核文档：Building External Modules](https://docs.kernel.org/kbuild/modules.html)
- [GCC 文档：预处理器选项](https://gcc.gnu.org/onlinedocs/gcc/Preprocessor-Options.html)
