# Verilog 零基础入门与菜鸟教程知识梳理

Verilog 是一种描述数字电路结构和行为的硬件描述语言：可以用它设计计数器、图像处理流水线等电路，先仿真验证，再将可综合的描述转换为 FPGA 或 ASIC 电路。

本文面向没有数字电路和 Verilog 基础的读者，按 [菜鸟教程 Verilog 基础篇目录](https://www.runoob.com/w3cnote/verilog-tutorial.html)梳理 1.1～8.1 的主题，并补充阅读实例所需的背景。文末给出各章节链接。目录中的“高级篇”是独立的后续教程，本文介绍其入口，不逐章展开。教程个别表述有歧义或实例参数可能有问题，下文会特别说明。

## 1. 先建立整体认识

### 1.1 Verilog 用来做什么

普通软件通常描述“处理器按什么步骤执行程序”；Verilog 描述“有哪些电路、它们如何连接、信号何时变化”。同一块芯片中的电路通常同时工作，因此不能把整个 Verilog 文件理解为从上到下执行一遍的程序。

例如，一个 ISP（图像信号处理器）可能持续接收传感器的像素数据，由多个硬件模块分别完成坏点校正、滤波、颜色处理等。Verilog 可以描述这些模块的接口和处理逻辑；实际 ISP 的算法、带宽和电路结构因产品而异，这里只是帮助理解应用场景。

常见用途：

- **仿真**：在电脑上模拟信号随时间的变化，检查设计功能。仿真是模型上的验证，不等于实物一定正确。
- **FPGA 开发**：将设计综合并配置到可重新编程的器件中，便于验证与迭代。
- **ASIC 开发**：将设计经过综合、物理设计等流程实现为专用芯片。Verilog 本身不是芯片，也不能直接决定全部物理实现细节。

Verilog 源于 20 世纪 80 年代，1995 年成为 IEEE 1364 标准，后来有 Verilog-2001 等版本。SystemVerilog 是在 Verilog 基础上扩展的语言，增加了设计和验证功能；本文示例主要使用 Verilog-2001。

### 1.2 先认识几个数字电路概念

- **bit**：一个二进制位，通常表示 0 或 1。8 bit 可以表示 256 种不同的位组合。
- **信号与位宽**：一根信号可以是 1 bit，也可以是多 bit 的一组线，例如 `data[7:0]` 是 8 bit。
- **组合逻辑**：稳定后的输出由当前输入决定，如与门、加法器、选择器。现实电路中输入变化到输出稳定有传播延迟。
- **时序逻辑**：输出还与电路保存的状态有关，如寄存器、计数器。同步设计通常在时钟边沿更新状态。
- **时钟 `clk`**：周期性变化的信号。上升沿 `posedge` 是从低到高的变化，下降沿 `negedge` 是从高到低的变化。一个时钟周期不是一条软件指令。
- **复位 `reset`**：把状态置为约定初值。名字带 `n`、`_n` 或 `rstn` 常表示低电平有效；名字只是约定，要以实际代码和接口定义为准。
- **使能 `en`**：允许某项动作发生的控制信号。未使能时，寄存器通常保持原值。
- **周期与频率**：例如时钟周期为 20 ns 时，频率是 50 MHz。

### 1.3 从需求到电路的流程

1. 明确功能、输入输出、时钟、复位、吞吐量等需求。
2. 自顶向下拆成多个模块，编写 RTL（寄存器传输级）描述：寄存器保存什么，寄存器之间如何计算和传输。
3. 写 testbench（测试平台）产生输入和预期结果，运行仿真，检查输出、波形与边界条件。
4. **综合**：工具将支持综合的 RTL 转换成门、寄存器等器件组成的逻辑网表。
5. FPGA 还需器件适配、布局布线、时序分析和生成配置文件；ASIC 还需工艺库、物理设计、时序检查、制造等步骤。
6. 上板或芯片测试。不同工具与项目流程会有差异。

描述方式可分为**行为级**（描述行为，例如 `always`）、**数据流**（例如 `assign` 表达组合关系）和**结构化**（连接门或子模块）。一个模块中可以混合使用；能否综合取决于语句、工具和目标器件。

### 1.4 与软件思维最不同的三件事

1. **模块和独立过程并行**：两个 `always` 块可以同时响应同一个时钟边沿；源码先后位置不代表执行先后。
2. **硬件有位宽和状态**：4 bit 不可能保存任意大的整数；寄存器会保持状态直到下一次更新。
3. **仿真语句不一定能变成电路**：例如测试平台中的 `#10`、打印和结束仿真通常不是可综合的硬件延迟或电路。

## 2. 从一个完整电路读懂基本写法

下面是模 10 计数器：每个时钟上升沿加 1，数到 9 后下一个上升沿回到 0。

```verilog
module counter10 (
    input wire       clk,
    input wire       rst_n,
    output reg [3:0] count,
    output wire      at_nine
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 4'd0;
        else if (count == 4'd9)
            count <= 4'd0;
        else
            count <= count + 4'd1;
    end

    assign at_nine = (count == 4'd9);
endmodule
```

逐行理解：

- `module` / `endmodule` 定义一个电路模块；括号中列出端口。
- `input` / `output` 表示从当前模块视角看的输入和输出。
- `[3:0]` 是 4 位向量；`4'd9` 是 4 位十进制数 9。
- `always @(posedge clk or negedge rst_n)` 表示当时钟上升沿或复位下降沿发生时运行这个过程。
- `if (!rst_n)` 是低有效复位；因为复位下降沿也在事件列表中，这是**低有效异步复位**：拉低复位不必等时钟边沿。复位释放如何满足时序要求需要在实际设计中考虑。
- `<=` 是非阻塞赋值，常用来描述时钟触发的寄存器更新。
- `assign` 表示持续的组合逻辑关系：只要 `count` 等于 9，`at_nine` 就为 1。它是状态比较结果，不是由硬件自动产生的独立时钟，也不是仅在某个边沿出现的瞬时脉冲。

若复位释放后从 0 开始，在连续时钟边沿后的值依次为 `1, 2, ..., 9, 0, 1`。`count` 在等于 9 的整个状态期间使 `at_nine` 为高；注意这是数值 0～9 的模 10 计数，存储位仍然是二进制。

## 3. 基础语法与数据

### 3.1 书写规则

Verilog 区分大小写，标识符用来命名模块、信号等，不可与关键字混淆。空白通常用于分隔标记，不能随意拆开标识符或数字。语句通常以分号结束，但 `begin`、`end`、`module` 等结构关键字不是都需要分号。

```verilog
// 单行注释
/* 多行
   注释 */
wire valid;
wire [7:0] pixel_data;
```

### 3.2 四态逻辑和数字字面量

仿真中的基本状态：`0`、`1`、`x`（未知）和 `z`（高阻、无有效驱动）。未初始化的 `reg` 常为 `x`；无驱动的普通 `wire` 通常为 `z`。`x` 是仿真模型中的未知，不是额外的一种稳定电压；`z` 也不等于逻辑 0。实际接口读到什么还取决于驱动和上下拉。

常用形式为 `位宽'进制数值`，进制 `b`、`o`、`d`、`h` 分别是二、八、十、十六进制；`s` 可表示有符号字面量。

```verilog
4'b1010    // 4 位二进制，十进制 10（按无符号解释）
8'hFF      // 8 位十六进制，全为 1
8'd15      // 8 位十进制 15
8'sd5      // 8 位有符号数 5
-8'sd5     // 8 位有符号数 -5
```

位宽不足时高位可能被截掉；扩展时应关注符号。未指定大小的常量和混合位宽表达式有较复杂的宽度规则，初学时尽量明确写宽度，并单独检查中间结果。实数、字符串也可用于仿真和参数等场景，但并不意味着都能综合成硬件。

### 3.3 `wire`、`reg`、向量和数组

```verilog
wire       ready;          // 线网，通常由 assign 或模块输出驱动
reg        flag;           // 过程变量，通常在 always / initial 中赋值
wire [7:0] pixel;          // 8 位，pixel[7] 是最高位
reg  [7:0] samples [0:15]; // 16 个元素，每个元素 8 位
```

`wire` 用来表示由驱动源决定的线网值。`reg` 在经典 Verilog 中表示可在过程块内赋值的变量，**不保证一定生成寄存器**：用完整的组合逻辑赋值可能综合成纯组合电路；用时钟边沿更新则通常推导成触发器。物理电路由完整的行为决定，而非只看类型名。

`integer` 常用于整数循环变量；`real` 和 `time` 等适合相应仿真用途，综合支持应查具体工具。向量切片如 `pixel[7:4]`；数组如 `samples[3]` 取一个 8 位元素。内存能否映射成 FPGA RAM 取决于描述方式及工具。

### 3.4 表达式和常见运算符

| 类别 | 运算符与含义 | 示例 |
| --- | --- | --- |
| 算术 | `+ - * / %` | `a + b`；除法的资源和时序成本与实现有关 |
| 比较 | `< <= > >=` | `a < b`，结果一般为 1 bit |
| 相等 | `== !=`；四态严格比较 `=== !==` | `a == b` 遇到未知位可能得到 `x`；`===` 把 `x/z` 当作状态比较 |
| 按位 | `~ & \| ^` | 每一位分别运算；`~a` 是逐位取反 |
| 逻辑 | `! && \|\|` | 将操作数作为真假值判断，结果为 1 bit |
| 归约 | `&a`、`\|a`、`^a` | 对一个向量的所有位分别做与、或、异或 |
| 移位 | `<< >> <<< >>>` | `>>>` 在左操作数为有符号时才按算术右移进行符号扩展 |
| 选择 | `cond ? a : b` | 二选一，常对应多路选择器 |
| 拼接、重复 | `{a, b}`、`{4{a}}` | 拼接比特，或将一个片段重复 4 次 |

`&` 与 `&&`、`|` 与 `||` 的含义不同。运算符优先级不确定时加括号。`===` / `!==` 很适合检查仿真中的未知值，不应简单假定四态比较在所有综合场景都具有相同硬件含义。

```verilog
wire [7:0] word = {4'b1010, 4'b0011}; // 8'b10100011
wire [7:0] fill = {8{1'b1}};           // 8'b11111111
wire any_bit = |word;                   // 任意位为 1
```

### 3.5 有符号数、补码与位宽

$N$ 位无符号数范围为 $0$ 到 $2^N-1$；$N$ 位二进制补码有符号数范围为 $-2^{N-1}$ 到 $2^{N-1}-1$。例如 4 位下：`0110` 可解释为 6；`1010` 按无符号解释为 10，按有符号补码解释为 -6。**位本身不变，解释方式可以不同。**

计算 -6 的 4 位补码：`0110` 全部按位取反得到 `1001`，再加 1 得到 `1010`。负数按有符号补码扩宽时复制符号位：`4'sb1010` 直接赋给 8 位有符号变量后为 `8'b11111010`，仍表示 -6。注意 `4'b1010` 是无符号字面量，直接赋给 8 位变量时会补零得到 `8'b00001010`，不能把两种扩展混为一谈。截断、混合有符号与无符号运算、移位和表达式结果位宽都可能改变结果，要明确检查。符号扩展与补零的区别参见 [Xilinx HDL 算术实现说明](https://docs.amd.com/api/khub/documents/IlS_WnPxkLVOpr6TRmaFQg/content)。

绝对值特别注意最小负数：4 位有符号数的 -8 没有对应的 4 位有符号正数 +8，需要先扩宽或制定饱和策略。[原教程数值转换](https://www.runoob.com/w3cnote/verilog-numerical-conversion.html)中给出的正数也加 1 的表达式是偏移映射，并不是通常意义上的绝对值；翻转最高位也是把范围整体平移的编码变换，并不是保持数值不变的通用类型转换。

### 3.6 编译指令

编译指令以反引号开头，主要在编译/预处理阶段发挥作用：

```verilog
`timescale 1ns/1ps       // 时间单位 / 时间精度
`define DATA_WIDTH 8    // 文本宏
`include "defs.vh"      // 包含其他文本文件
`ifdef SIMULATION
    // 仅在相应宏已定义时参与编译
`endif
```

还包括 `undef`、`default_nettype`、`resetall`、`celldefine` / `endcelldefine`、`unconnected_drive` / `nounconnected_drive` 等。`default_nettype none` 可帮助发现拼错后被隐式声明为线网的问题，但应考虑其对后续文件的影响；宏是文本替换，模块参数通常更适合配置某一个实例。`timescale` 影响仿真时间单位与舍入精度，并不创建时钟。

## 4. 连续赋值、过程块和时序控制

### 4.1 连续赋值与全加器

`assign` 在右侧信号变化时持续驱动左侧线网，可以用来描述组合逻辑。以下全加器把三个 1 bit 输入相加，得到本位和 `sum` 与进位 `carry`：

```verilog
assign sum   = a ^ b ^ cin;
assign carry = (a & b) | (a & cin) | (b & cin);
```

例如 `a=1, b=1, cin=0`，二进制结果为 `10`，因此 `sum=0, carry=1`。

### 4.2 `initial` 和 `always`

- `initial` 从仿真开始执行一次，常用于测试平台初始化、产生激励、检查结果和结束仿真。部分 FPGA 工具支持特定初始化方式，但不能把所有 `initial` 都当作通用可综合 RTL。
- `always` 反复执行，通常结合事件控制描述组合或时序逻辑。多个过程彼此独立、并发运行。
- `always @(*)` 常用于组合逻辑；`always @(posedge clk)` 常用于上升沿触发的时序逻辑。

```verilog
always @(*) begin
    if (sel)
        y = a;
    else
        y = b;
end
```

在经典 Verilog 中，过程块中赋值的 `y` 应声明为 `reg`，但这里电路仍是组合选择器。`@(*)` 自动收集敏感信号，比手写列表更不易遗漏。

### 4.3 阻塞与非阻塞赋值

- `=` **阻塞赋值**：执行时立即更新左值，再执行后续语句；通常用于组合过程。
- `<=` **非阻塞赋值**：执行该语句时计算右值，并安排在当前仿真时刻后续更新阶段更新左值；通常用于时钟驱动的时序过程。

```verilog
always @(posedge clk) begin
    a <= in;
    b <= a;
end
```

同一个边沿，`b` 得到的是边沿前的旧 `a`，不是刚送入的 `in`。这通常对应两个串联寄存器。`<=` 不是“整个语句块并行执行”的意思，而是右值采样与左值更新的调度方式不同。

时序逻辑用 `<=`、组合逻辑用 `=` 是常用且实用的约定；不要让多个过程同时驱动同一个变量，也不要依赖不同过程在同一时刻的执行先后顺序。

### 4.4 时间、事件和语句块

`#5` 表示仿真中等待 5 个时间单位；`@(posedge clk)` 等待一个上升沿；`@(a or b)` 等待列出的信号发生事件；`wait(condition)` 等待条件为真，如果条件已经为真则可立即继续。时间精度由 `timescale` 等设置决定。

```verilog
initial begin
    clk = 1'b0;
    forever #5 clk = ~clk; // 测试平台时钟，周期 10 个时间单位
end
```

`begin ... end` 是顺序语句块，语句依次执行；`fork ... join` 是并发语句块，常用于仿真，等待各分支完成后继续。块可以嵌套、命名，并可用 `disable` 终止指定命名块。延迟、并行块和 `force` 一般用于验证或建模，不应直接假设可综合。综合设计的真实延迟由电路、约束和实现决定，而不是靠 RTL 中的 `#` 指定。

连续赋值可写模拟时延 `assign #5 y = a;`。它具有惯性延迟特性：足够短的输入变化可能不出现在输出中。它是仿真建模用途，不应作为实现真实硬件延迟的通用方法。

### 4.5 `if`、`case` 和循环

`if/else` 按条件选择，组合逻辑中的分支不完整可能产生锁存器；`case` 根据表达式选择分支，通常写 `default`。

```verilog
always @(*) begin
    case (sel)
        2'b00: y = a;
        2'b01: y = b;
        2'b10: y = c;
        default: y = 8'b0;
    endcase
end
```

普通 `case` 做四态精确匹配；`casez` 将 `z` 和 `?` 用作通配，`casex` 连 `x` 也视为通配，可能掩盖未知值，实际 RTL 应谨慎使用。

`for`、`while`、`repeat`、`forever` 用于循环。综合器对固定边界的过程 `for` 往往展开成多个硬件操作，不等于 CPU 循环逐周期执行；需要逐周期复用同一个计算单元时，必须显式设计状态、计数器和寄存器。无界循环通常仅适合仿真。

### 4.6 过程连续赋值与强制信号

教程还介绍了过程中的 `assign/deassign` 和 `force/release`。前者可临时对变量进行过程连续赋值；后者可强制信号取值，再释放控制。这些机制与模块级普通 `assign` 不同，主要用于仿真、调试或特殊建模；初学 RTL 时优先使用普通组合逻辑和时序逻辑，不把它们当作通用可综合设计手段。释放后实际值取决于目标类型及其其他驱动/后续赋值。

## 5. 模块、端口、参数和复用

### 5.1 层次化与例化

模块像一个可复用的电路部件；例化就是在上层创建一个具体实例并接好线。多个实例可同时工作。

```verilog
module mux2 #(parameter WIDTH = 8) (
    input wire [WIDTH-1:0] a, b,
    input wire sel,
    output wire [WIDTH-1:0] y
);
    assign y = sel ? b : a;
endmodule

module top (
    input wire [7:0] p0, p1,
    input wire choose,
    output wire [7:0] pixel_out
);
    mux2 #(.WIDTH(8)) u_pixel_mux (
        .a(p0), .b(p1), .sel(choose), .y(pixel_out)
    );
endmodule
```

`mux2` 是模块类型，`u_pixel_mux` 是实例名，`.a(p0)` 把实例端口 `a` 接到上层信号 `p0`。按名称连接比按位置连接更不容易接错。`input`、`output`、`inout` 的方向相对于被定义的模块而言；`inout` 多用于需要双向驱动的接口，具体实现依器件而定。

`parameter` 用于配置位宽等常量，上例通过 `#(.WIDTH(8))` 覆盖默认值。教程也讲了 `defparam`，但它通过层次路径修改参数，可读性和维护性较差，通常更推荐例化时显式传参。参数在展开设计时确定，不是运行时可随意变化的输入。

`generate`、`genvar` 可以根据常量参数生成重复结构，常用于例化多级处理单元。它发生在设计展开阶段；生成循环变量不是运行时计数器。

### 5.2 函数与任务

**函数 `function`** 适合封装不消耗仿真时间的计算，使用输入和一个函数返回值。例如：

```verilog
function [7:0] invert_byte;
    input [7:0] value;
    begin
        invert_byte = ~value;
    end
endfunction
```

函数可用于组合计算、位操作、译码等；是否可综合取决于函数内部内容和工具。`automatic` 可让每次调用拥有独立局部存储，有助于避免并发调用互相干扰。

**任务 `task`** 可有多个输入、输出或双向参数，也可包含延迟和事件控制，常用来组织 testbench 激励。函数和任务都是复用描述的方法，不自动意味着只有一份共享硬件；综合结果取决于调用方式和工具。

## 6. 从语言走向电路设计

### 6.1 状态机 FSM

有限状态机用有限个状态及状态之间的转移描述控制过程。例如像素处理模块可能有 `IDLE`（空闲）、`WORK`（处理）、`DONE`（完成）三个状态：收到开始信号后进入处理，处理结束后报告完成，再返回空闲。

- **Moore**：输出由当前状态决定。
- **Mealy**：输出由当前状态和输入共同决定，可能更快响应输入，但输入变化也可能直接影响输出。
- 常见三段式：时钟过程更新当前状态；组合过程计算下一状态；另一个过程或连续赋值计算输出。也可采用两段式；关键是复位、所有分支、非法状态恢复和输出时序要明确。

简化示例，`done_in` 表示外部处理完成：

```verilog
localparam IDLE = 2'd0, WORK = 2'd1, DONE = 2'd2;
reg [1:0] state, next_state;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else        state <= next_state;
end

always @(*) begin
    next_state = state;
    case (state)
        IDLE: if (start)   next_state = WORK;
        WORK: if (done_in) next_state = DONE;
        DONE:              next_state = IDLE;
        default:           next_state = IDLE;
    endcase
end

assign done = (state == DONE);
```

这段代码省略了外围模块和端口声明，是用于说明状态机的片段；`done` 是 Moore 输出，在 `DONE` 状态有效。

### 6.2 锁存器、触发器和不完整赋值

锁存器（latch）在使能有效期间可以随输入变化，失能后保持；触发器（flip-flop）通常在指定时钟边沿采样。不要误以为 `reg` 一定对应触发器。

```verilog
always @(*) begin
    if (en)
        y = a; // en 为 0 时没有赋值，要求保留原值，可能推导锁存器
end
```

如果想要纯组合逻辑，应给每条路径赋值，例如先写 `y = 0;` 再根据 `en` 覆盖，或者补充 `else`。时钟过程中的不赋值通常表示触发器保持，不是同一类问题。教程的锁存器章节前后有相反的电平描述；应以器件规定的有效电平为准：**有效时透明，失效时保持**。

### 6.3 竞争、冒险和毛刺

仿真**竞争**指多个并发过程在同一时刻读写相同信号，结果依赖调度顺序。时钟寄存器使用非阻塞赋值、单一驱动和避免 testbench 在采样边沿同时改输入，有助于减少这类问题。

电路中的**冒险/毛刺**则是不同组合路径传播延迟不一致造成短暂错误跳变。即使零延迟 RTL 仿真看起来正确，真实门电路也可能有毛刺。可根据具体电路采用调整逻辑、寄存器采样、约束时序等方法；寄存器并不自动消除所有毛刺问题，跨时钟域和异步输入还需要专门的同步与验证。不要把仿真调度竞争与电路传播延迟混为一谈。

### 6.4 流水线：延迟与吞吐量

将长计算拆成若干级，中间插入寄存器：

```text
输入 → 第 1 级组合计算 → 寄存器 → 第 2 级组合计算 → 寄存器 → 输出
```

不同数据可以同时处在不同级。假设每级都能每周期接收一个输入，那么流水线填满后可每周期输出一个结果；单个输入仍需经过多个周期，称为**延迟**。这两个量不能混淆。真实最大频率还取决于各级组合延迟、寄存器和时序约束。

有效信号 `valid`、坐标、帧标记等附带信息也要和数据经过匹配的延迟；处理暂停、复位和背压时还要明确各级如何保持和移动。这对 ISP 中逐像素处理尤其重要。

## 7. 仿真和第一个可运行练习

### 7.1 Testbench 做什么

testbench 通常包含：待测模块（DUT）实例、时钟和复位、输入激励、输出检查、仿真结束。建议不仅看波形，也编写自动检查，并覆盖复位、正常运行、边界值和非法输入等情况。

教程介绍 Quartus II + ModelSim 的旧版环境；其截图、安装路径和版本可能已过时，实际应以工具当前官方文档为准。简单入门可使用 Icarus Verilog 编译/仿真，GTKWave 查看波形；这只是示例选择，并非 FPGA 烧录完整流程。

### 7.2 计数器自检示例

将第 2 节模块保存为 `counter10.v`，下面内容保存为 `tb_counter10.v`。测试等待下降沿后释放复位，避开 DUT 的采样上升沿，并在上升沿后短暂等待非阻塞赋值生效。

```verilog
`timescale 1ns/1ps
module tb_counter10;
    reg clk = 0;
    reg rst_n = 1;
    wire [3:0] count;
    wire at_nine;
    integer i;
    reg [3:0] expected;

    counter10 dut (.clk(clk), .rst_n(rst_n),
                   .count(count), .at_nine(at_nine));

    always #5 clk = ~clk;

    initial begin
        $dumpfile("counter10.vcd");
        $dumpvars(0, tb_counter10);
        expected = 0;
        #1 rst_n = 0; // 显式产生复位下降沿
        #1;
        if (count !== 4'd0) begin
            $display("FAIL: reset count=%d", count);
            $finish;
        end
        @(negedge clk);
        rst_n = 1;
        for (i = 0; i < 22; i = i + 1) begin
            @(posedge clk);
            if (expected == 4'd9) expected = 4'd0;
            else                  expected = expected + 4'd1;
            #1;
            if (count !== expected || at_nine !== (expected == 4'd9)) begin
                $display("FAIL: step=%0d count=%d expected=%d", i, count, expected);
                $finish;
            end
        end
        $display("PASS: counter10");
        $finish;
    end
endmodule
```

安装相应工具后，在文件所在目录可运行：

```bash
iverilog -g2001 -s tb_counter10 -o sim_counter counter10.v tb_counter10.v
vvp sim_counter
gtkwave counter10.vcd
```

预期终端打印 `PASS: counter10`。`$display` 打印，`$finish` 结束仿真，`$dumpfile` / `$dumpvars` 生成波形文件。实际测试应继续增加复位中途拉低等场景；仿真通过只覆盖已测试的情形。

教程还涉及文件读写与载入测试数据，例如 `$fopen`、`$fdisplay`、`$fclose`、`$readmemh`；这些常用于自动化测试和初始化，能否用于综合取决于具体结构和工具。

## 8. 教程后半部分的数字信号处理实例

这些实例不仅需要 Verilog 语法，还需要一点信号处理知识。下面先讲“做什么、怎么拆硬件、该注意什么”，不要求第一次阅读就掌握全部算法推导。

### 8.1 固定点除法器

例如 $27 \div 5 = 5$ 余 $2$。二进制逐位除法和手算竖式类似：从被除数高位向低位依次引入一位，将当前部分余数与除数比较；够减就减并产生商位 1，否则商位为 0；最后得到商和余数。

硬件可以把一个步骤反复执行，以少量资源换取多个周期；也可以将多个步骤展开并插入寄存器形成流水线，以更多资源换吞吐量。比较时要保留足够的余数位，处理除数为 0、有符号数、溢出和输入输出有效信号。教程举例是特定设计，不代表所有除法器都必须有相同延迟。

来源：[7.1 除法器设计](https://www.runoob.com/w3cnote/verilog-dividend.html)。该页本次直接访问失败，以上对教程实例的描述根据搜索引擎抓取摘要，未逐行核验完整代码。

### 8.2 FIR：有限冲激响应滤波器

滤波可以理解为把当前和历史样本按不同权重相加，让某些频率成分通过、另一些受到抑制。一个 $N$ 抽头 FIR 的一般表达式为：

$$
y[n] = \sum_{k=0}^{N-1} h[k]x[n-k]
$$

其中 $x[n-k]$ 是当前或过去的输入样本，$h[k]$ 是系数，$y[n]$ 是输出。直接型 FIR 通常可以理解为延迟线、乘法器和加法器的组合；有限长本身不保证严格线性相位，线性相位还需要系数满足相应的对称或反对称条件。

- **并行 FIR**：多个乘法和加法单元同时工作，便于获得高吞吐量，但通常消耗更多资源。
- **串行 FIR**：复用乘加单元，分多个时钟周期计算一个输出，节省硬件资源，但要求足够的内部时钟和数据缓存。若系数对称，可以先将相应样本相加，从而减少乘法次数。
- 系数量化、乘积位宽、累加位宽、截断/舍入和溢出处理都会影响精度；流水化还需对齐输出有效信号。

教程的串行 FIR 页面列出了 50 MHz 采样、7.5 MHz 输入和 1～6 MHz 阻带等参数，同时声称滤除 7.5 MHz；由于 7.5 MHz 不在所列阻带内，不能仅凭这些参数认定设计达到目标，应查看实际滤波器频率响应。页面中的 `250KMHz` 也与上下文的 `250 KHz` 不一致。

### 8.3 CIC：积分梳状滤波器

CIC 常用于采样率转换，例如数字下变频后的抽取。它用积分器（累加器）和梳状差分器实现滤波，典型结构不需要通用乘法器。**抽取**是按一定倍率降低输出采样率，但必须同时满足抗混叠要求；不能理解为随意丢样本就一定得到正确结果。

数字下变频（DDC）可概括为：输入与本振信号混频，把目标频段搬移，再滤波和抽取。CIC 实现简单，但通带可能有幅度下垂，实际系统可能需要后续补偿滤波器；累加器位宽也需依据级数、倍率和数据范围设计。

### 8.4 FFT：快速傅里叶变换

DFT（离散傅里叶变换）把一段离散时域样本映射到不同频率的复数分量；FFT 是高效计算 DFT 的算法，不是另一种独立的物理变换。教程的 8 点实例使用多级**蝶形运算**，通过复数加减和旋转因子乘法逐级处理；某些基 2 实现需要位反转顺序，具体输入或输出排列取决于算法结构。

硬件实现需考虑复数实部/虚部、旋转因子查表或常量乘法、定点精度、每级位宽增长、截断/舍入、缓存和流水线延迟。不要直接把某一种 FFT 的排序方式或缩放策略推广到全部实现。

### 8.5 DDS：直接数字频率合成

DDS 用时钟驱动相位累加器，每个周期加上相位增量，再根据相位查表生成数字波形；如果需要模拟输出，可接 DAC 和重建滤波器。假设相位累加器为 $N$ 位、每周期增量为 $K$、时钟为 $f_{clk}$，其离散相位推进对应的频率为：

$$
f_{out} = \frac{K}{2^N} f_{clk}
$$

这是理想离散频率关系；实际输出频谱还受采样混叠、相位截断、幅度量化、DAC 和滤波器影响。例如 $N=8$、$K=16$、时钟 1 MHz 时，对应频率为 62.5 kHz。查表地址可以只取累加器高位，降低表大小，但可能增加杂散。

## 9. 推荐的学习顺序和自查问题

1. 先认识 0/1、位宽、组合逻辑、寄存器、时钟和复位；手算与门、二选一和计数器。
2. 阅读第 2～4 节，重点区分 `wire/reg`、`assign/always`、`=/<=`，再自己画出可能对应的电路。
3. 跑通计数器 testbench，查看复位、从 9 回 0 的边沿和 `at_nine`；尝试修改成模 6，并同步更新预期结果。
4. 学会模块例化、参数、状态机、避免锁存器和流水线，再阅读 FIFO、握手等实际接口资料。
5. 有数字信号处理基础后再读 FIR、CIC、FFT、DDS，重点追踪位宽、有效信号和周期，而不是只照抄代码。

自查：

- 这段代码描述的是组合逻辑还是需要保存状态？
- 哪个时钟边沿更新？复位是同步还是异步、何时有效？
- 每个信号有多少位、是否有符号、运算会不会溢出？
- 数据和 `valid` 是否在同一周期对应？每周期可以接收多少数据？
- 每种输入和状态下都有明确输出吗？是否意外推导 latch？
- 这句话只是仿真辅助，还是能由目标工具综合？

## 10. 原教程章节索引与延伸阅读

- 总体：[1.1 教程](https://www.runoob.com/w3cnote/verilog-tutorial.html)、[1.2 简介](https://www.runoob.com/w3cnote/verilog-intro.html)、[1.3 环境](https://www.runoob.com/w3cnote/verilog-install.html)、[1.4 设计方法](https://www.runoob.com/w3cnote/verilog-design-method.html)。
- 基础：[2.1 语法](https://www.runoob.com/w3cnote/verilog-basic-syntax.html)、[2.2 数值](https://www.runoob.com/w3cnote/verilog-number.html)、[2.3 数据类型](https://www.runoob.com/w3cnote/verilog-data-type.html)、[2.4 表达式](https://www.runoob.com/w3cnote/verilog-expression.html)、[2.5 编译指令](https://www.runoob.com/w3cnote/verilog-compile-instruction.html)。
- 描述与过程：[3.1 连续赋值](https://www.runoob.com/w3cnote/verilog-assign.html)、[3.2 时延](https://www.runoob.com/w3cnote/verilog-time-delay.html)、[4.1 过程结构](https://www.runoob.com/w3cnote/verilog-process-structure.html)、[4.2 过程赋值](https://www.runoob.com/w3cnote/verilog-process-assign.html)、[4.3 时序控制](https://www.runoob.com/w3cnote/verilog-timing-control.html)、[4.4 语句块](https://www.runoob.com/w3cnote/verilog-statements-block.html)、[4.5 条件](https://www.runoob.com/w3cnote/verilog-condition-statement.html)、[4.6 多路分支](https://www.runoob.com/w3cnote/verilog-case.html)、[4.7 循环](https://www.runoob.com/w3cnote/verilog-loop.html)、[4.8 过程连续赋值](https://www.runoob.com/w3cnote/verilog-deassign.html)。
- 模块和进阶：[5.1 模块与端口](https://www.runoob.com/w3cnote/verilog-module-port.html)、[5.2 模块例化](https://www.runoob.com/w3cnote/verilog-generate.html)、[5.3 参数例化](https://www.runoob.com/w3cnote/verilog-defparam.html)；[6.1 函数](https://www.runoob.com/w3cnote/verilog-function.html)、[6.2 任务](https://www.runoob.com/w3cnote/verilog-task.html)、[6.3 状态机](https://www.runoob.com/w3cnote/verilog-fsm.html)、[6.4 竞争与冒险](https://www.runoob.com/w3cnote/verilog-competition-hazard.html)、[6.5 避免 latch](https://www.runoob.com/w3cnote/verilog-latch.html)、[6.6 仿真激励](https://www.runoob.com/w3cnote/verilog-testbench.html)、[6.7 流水线](https://www.runoob.com/w3cnote/verilog-pipeline-design.html)。
- 实例与数值：[7.1 除法器](https://www.runoob.com/w3cnote/verilog-dividend.html)、[7.2 并行 FIR](https://www.runoob.com/w3cnote/verilog-fir.html)、[7.3 串行 FIR](https://www.runoob.com/w3cnote/verilog-serial-fir.html)、[7.4 CIC](https://www.runoob.com/w3cnote/verilog-cic.html)、[7.5 FFT](https://www.runoob.com/w3cnote/verilog-fft.html)、[7.6 DDS](https://www.runoob.com/w3cnote/verilog-dds.html)、[8.1 数值转换](https://www.runoob.com/w3cnote/verilog-numerical-conversion.html)。
- 另有独立的 [Verilog 高级篇](https://www.runoob.com/w3cnote/verilog2-tutorial.html)，涉及门级建模、用户自定义原语、时序分析、同步与异步、复位与时钟、低功耗、系统任务、PLI 和逻辑综合。适合掌握本文基础后按需继续学习，不属于本文逐章总结范围。
