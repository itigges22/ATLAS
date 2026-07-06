> **[English](../../ARCHITECTURE.md)** | **简体中文** | **[日本語](../ja/ARCHITECTURE.md)** | **[한국어](../ko/ARCHITECTURE.md)**

# ATLAS 架构

ATLAS V3.1.3 的系统架构。采用双层设计：外层 agent 循环负责工具调用的编排，内层 V3 pipeline 则生成多样化的代码候选，并配合构建验证与基于能量的选择。

---

## 1. 系统概览

```mermaid
graph LR
    User["User"] --> TUI["atlas-tui\n(Bubbletea)"]
    TUI --> Proxy["atlas-proxy\n:8090"]

    subgraph outer["Outer Layer"]
        Proxy -->|"grammar JSON"| LLM["llama-server\n:8080"]
        Proxy -->|"T2 files"| V3Service["v3-service\n:8070"]
    end

    subgraph inner["Inner Layer"]
        V3Service --> LLM
        V3Service --> Lens["geometric-lens\n:8099"]
        V3Service --> Sandbox["sandbox\n:30820"]
        Lens --> LLM
    end

    style User fill:#333,color:#fff
    style TUI fill:#1a3a5c,color:#fff
    style Proxy fill:#1a3a5c,color:#fff
    style LLM fill:#5c1a1a,color:#fff
    style V3Service fill:#2d5016,color:#fff
    style Lens fill:#2d5016,color:#fff
    style Sandbox fill:#2d5016,color:#fff
```

各服务既可以通过 Docker Compose（推荐）作为容器运行，也可以通过 `atlas` 启动器作为本地进程运行。只有 llama-server 使用 GPU，其余所有组件都跑在 CPU 上。

聊天前端是 **atlas-tui**（Bubbletea）：一个原生 Go 终端 UI，消费 `/v1/agent`（按轮次的聊天 SSE）和 `/events`（面向 pipeline 窗格的全局类型化信封事件流）。用 `atlas`（默认交互模式）或 `atlas tui`（显式指定）启动。Pipeline 窗格实时展示 V3 各阶段；聊天窗格通过 glamour 渲染助手的 markdown；斜杠命令 `/add /diff /commit /run` 等负责处理本地文件上下文与 shell 调用。输入是模式感知的（chat / `!bash` / `/slash`），并带有提示下拉。

希望使用工具调用 + V3 pipeline 的第三方客户端应直接对接 `/v1/agent`；`/v1/chat/completions` 是对 llama-server 的透传（见 §3）。该契约记录在 [API.md](../../API.md) 中。

### 1.1 支持的加速器

llama-server 是唯一使用 GPU 的服务；其余每个 ATLAS 服务都跑在 CPU 上（代理是 Go，v3-service / geometric-lens / sandbox 是 Python）。这让多后端的表面积保持很小 —— 添加一个新加速器意味着一个新的 Dockerfile 加上一个入口点环境变量分支，而不是改动整个 pipeline。

| 后端 | 状态 (V3.1.x) | 镜像 / 构建路径 | Compose override | 已测试显卡 |
|---|---|---|---|---|
| **CUDA** (NVIDIA) | 自 V3.1.0 起发布 | `inference/Dockerfile.v31` → `atlas-llama` | （默认） | RTX 5060 Ti 16GB（基准），RTX 30xx/40xx/50xx |
| **ROCm / HIP** (AMD) | V3.1.1 发布 | `inference/Dockerfile.rocm` → `atlas-llama-rocm` | `docker-compose.rocm.yml` | RX 7900 XTX（社区冒烟测试，GH #26） |
| **Metal** (Apple Silicon) | 已发布 ([#32](https://github.com/itigges22/ATLAS/issues/32)) | 混合方案：原生 llama-server (Metal) + 其余组件用 Docker（macOS 无法将 GPU 直通给容器） | `docker-compose.macos.yml` | M 系列；≤16 GB 用 Q4_K_M，≥24 GB 统一内存用 Q6_K |
| **SYCL** (Intel Arc) | 路线图 | 待定 | 待定 | Arc A770 16 GB（目标） |

**后端选择发生在安装时，而非运行时。** `atlas init` 运行 `tier.detect_gpu()`（见 `atlas/cli/commands/tier.py`），在所有检测到的厂商中挑选显存最大的 GPU（可用 `ATLAS_GPU_VENDOR` / `ATLAS_GPU_INDEX` 覆盖），并把 `ATLAS_BACKEND={cuda|rocm|metal|sycl}` 写入 `.env`。每个后端都有自己预构建的镜像；用户不会运行一个打包了所有后端库的臃肿镜像。在不受支持的后端主机上，向导会拒绝运行，而不是写出一个无法启动的 `.env`。

**自带模型的表面（V3.1.1）。** `atlas lens check` 是针对运行中的 llama-server 的一次廉价预检，用于报告当前加载的模型是否与 Lens 兼容。`atlas lens build --samples <path>` 封装了 `geometric-lens/geometric_lens/training.py`，按模型原生的嵌入维度训练全新的 `cost_field.pt` 工件。二者结合让用户无需 fork lens 代码即可换入非默认的 GGUF —— C(x) 构造函数接受任意 `input_dim`，因此逐模型变化的只有训练出来的权重。面向用户的流程见 [CLI.md § atlas lens](../../CLI.md#atlas-lens)；注册表写回与 HuggingFace 分发是闭环这条链路的后续工作。

**与厂商无关的部分**（在每个后端上都可用）：语法约束的 JSON、自嵌入（`/embedding`）、逐层隐藏状态、ASA 控制向量（由 llama.cpp 的 `control_vector_load` 加载，与后端无关）、KV 缓存量化、整个外层 agent 循环、V3 pipeline、Geometric Lens 以及 sandbox。

**逐后端有差异的部分：**
- **Flash attention。** CUDA + ROCm：完整支持。Metal：受限（llama.cpp 的 Metal 后端对部分 head 尺寸支持 flash-attn；不支持时默认关闭）。SYCL：待定。
- **固定（pinned）主机内存。** `GGML_CUDA_NO_PINNED` 适用于 CUDA + ROCm（HIP 在 GGML 兼容层镜像了 CUDA 的路径）。Metal/SYCL 不使用固定内存。
- **多 GPU + 张量并行。** V1 在每个后端上都只支持单 GPU；多 GPU 是 GH #34，不绑定到特定厂商。
- **Apple 统一内存。** macOS 共享 GPU+系统内存；"VRAM" 的算法实际上是"总共 16 GB 减去操作系统 + 应用"。见 §7。

K3s 部署路径（`scripts/install.sh`，清单在 `templates/` 中）截至 V3.1.1 仅支持 CUDA —— ROCm K8s 方案已推迟到 V3.2 的基础设施清单（需要 `/dev/kfd` + `/dev/dri` hostPath 挂载以及 `render`/`video` 组成员身份，相当于集群级别的 `docker-compose.rocm.yml`）。

---

## 2. 服务

| 服务 | 端口 | 语言 | 用途 |
|---------|------|----------|---------|
| **llama-server** | 8080 | C++ (llama.cpp) | LLM 推理（CUDA / ROCm / Metal / Vulkan；SYCL 在路线图上 —— 见 §1.1）、语法约束的 JSON、自嵌入、逐层残差隐藏状态 |
| **atlas-proxy** | 8090 | Go | agent 循环、工具调用路由、tier 分类、`/v1/agent` SSE、`/events` 类型化 SSE、`/cancel`。`/v1/chat/completions` 原样透传给 llama-server。 |
| **atlas-tui** | （客户端） | Go | Bubbletea TUI；消费 `/events` 和 `/v1/agent` SSE 流。 |
| **v3-service** | 8070 | Python | V3 pipeline 的 HTTP 封装（PlanSearch、DivSampling、PR-CoT 等） |
| **geometric-lens** | 8099 | Python (FastAPI) | C(x) 能量打分、G(x) XGBoost 质量预测、RAG/项目索引；拥有 SQLite 状态存储（`lens-state` 卷上的 `SQLITE_DB_PATH`），支撑模式缓存、共现图、任务队列和路由器状态 |
| **sandbox** | 30820（主机）/ 8020（容器） | Python (FastAPI) | 隔离的代码执行、编译、检查、测试运行 |

---

## 3. atlas-proxy（外层）

代理是聊天前端的入口点。它在 `/v1/agent` 上接收用户消息（类型化事件流 —— TUI 使用的就是它），并运行一个内部 agent 循环：调用 llama-server、解析工具调用、执行它们，然后把事件流式回传。`/v1/chat/completions` 端点是对 llama-server 的透明透传；保留它是为了 SDK 兼容性，它并不运行 agent 循环。完整的事件类型目录见 [API.md](../../API.md)。

```mermaid
graph LR
    subgraph core["Core Loop"]
        Grammar["Grammar"] --> AgentLoop["Agent Loop"] --> TierClass["Tier Classifier"]
    end
    subgraph tools["Tools"]
        ReadF["read_file"] ~~~ WriteF["write_file"] ~~~ EditF["edit_file"] ~~~ RunCmd["run_command"]
    end
    subgraph pipeline["Verify-Repair"]
        VR["Verify-Repair"] --> BOK["Best-of-K"] --> BV["Build Verifier"]
    end
    subgraph format["I/O"]
        SSE["SSE / Events"] --> V3Bridge["V3 Bridge"] --> ProjDet["Project Detector"]
    end

    core --> tools --> pipeline --> format

    style core fill:#1a3a5c,color:#fff
    style tools fill:#333,color:#fff
    style pipeline fill:#2d5016,color:#fff
    style format fill:#555,color:#fff
```

### Agent 循环流程

```mermaid
flowchart LR
    Start["User msg"] --> Build["Build prompt"] --> Call["llama-server"] --> Parse["Parse JSON"]
    Parse --> Route{Type?}

    Route -->|"tool_call"| Tier{"T2?"}
    Tier -->|"Yes"| V3["V3 Pipeline"] --> Result["Append result"]
    Tier -->|"No"| Exec["Execute tool"] --> Result
    Result --> Budget{"Budget?"}
    Budget -->|"< 4"| Call
    Budget -->|"4"| Warn["Nudge: write now"] --> Call
    Budget -->|"5+"| Skip["Skip read"] --> Call

    Route -->|"text"| Stream["Stream"] --> Call
    Route -->|"done"| Done["End"]

    style Start fill:#1a3a5c,color:#fff
    style Done fill:#333,color:#fff
    style V3 fill:#2d5016,color:#fff
```

### 语法强制执行

llama-server 的 `response_format: {"type": "json_object"}` 强制每一次模型输出恰好是三种有效 JSON 形态之一：

```json
{"type": "tool_call", "name": "<tool_name>", "args": {...}}
{"type": "text", "content": "<message>"}
{"type": "done", "summary": "<summary>"}
```

该 JSON schema 使用带 `additionalProperties: false` 的 `oneOf`，并从注册表中枚举工具名。模型无法产生无效的 JSON —— token 生成在 llama-server 层面就受语法约束。

### 工具

`proxy/tools.go` 中注册了 14 个工具：

| 工具 | 用途 | 只读 |
|------|---------|-----------|
| `read_file` | 读取文件内容（可选 offset/limit） | 是 |
| `outline_file` | 列出文件的顶层函数/类及其行号范围，不含函数体（`.py` 使用 tree-sitter，其余为尽力而为的扫描）。外科式读取的入口点：先 outline，再用带 offset/limit 的 `read_file` | 是 |
| `write_file` | 创建一个新文件（对超过 5 行的已有文件会被拒绝 —— 见安全限制） | 否 |
| `edit_file` | 针对 ≤10 行改动的外科式内联字符串替换（old_str/new_str） | 否 |
| `ast_edit` | 通过 tree-sitter 选择器（`function:NAME`、`class:NAME`、`<tag>`）对整个函数/类/HTML 元素进行重写；对整节点替换而言，必须优先于 edit_file 使用。GH #39，v1 中仅支持 .py/.html/.htm | 否 |
| `delete_file` | 删除文件或空目录（之后强制退出循环） | 否 |
| `move_file` | 在工作区内移动或重命名文件（例如 `index.html` → `templates/`）。纯粹的重定位 —— 绕过 V3/外科式编辑门控，拒绝覆盖已存在的目标。由于 shell `mv`/`cp` 会被拒绝，这是"重新组织文件"的受支持路径 | 否 |
| `find_file` | 按文件**名**/路径做正则搜索（廉价的存在性检查 + 定位）。区别于在文件内容中 grep 的 `search_files`。 | 是 |
| `search_files` | 跨文件内容做正则搜索（最多 200 个匹配，跳过 .git/node_modules） | 是 |
| `list_directory` | 列出目录内容及其类型和大小 | 是 |
| `run_command` | 通过 sandbox 容器执行 shell 命令；5 分钟超时上限 | 否 |
| `run_background` | 在 sandbox 中启动一个长时间运行的进程（例如 `python app.py`）；立即返回一个 `job_id` | 否 |
| `tail_background` | 通过 `job_id` 获取某个后台任务新增的 stdout/stderr | 是 |
| `stop_background` | 通过 `job_id` 对某个后台任务发送 SIGTERM/SIGKILL | 否 |

### 工具选择偏差缓解

一次实测的参考部署显示出一种偏差：即便 `ast_edit` 才是正确选择，模型也倾向于用 `edit_file`（BiasBusters arxiv 2510.00307 —— 相邻工具名的嵌入会相互竞争；描述比名称更重要）。代理中组合了四道与模型无关的防线：

1. **描述重写**（`proxy/tools.go`）。edit_file 的描述
   警告不要用于整文件/整函数；ast_edit 的描述
   声明对 >10 行 / 整节点替换是必需的；write_file 的描述
   声明仅用于新文件。
2. **条件式 GBNF 语法**（`proxy/grammar.go`，
   `proxy/agent.go:stepExclusions`）。当一个 write_file 对
   一个 >5 行的已有 .py/.html/.htm 文件被拒绝时，下一次 LLM 调用会
   被一个 GBNF 语法约束，该语法从工具名产生式中禁掉
   edit_file 和 write_file。模型在物理上无法发出
   它们。该限制在一次决策后失效。
3. **逐步工具列表过滤**（同一触发条件）。注入一条临时的
   `[system note]` 用户消息，提醒模型在这一步
   ast_edit 是唯一的结构性编辑工具。
4. **ASA 操控向量**（`geometric-lens/asa_calibration/`）。
   激活操控在上游移动残差流分布，因此即使在任何拒绝
   触发之前的首次尝试决策中，也会偏好 ast_edit。仅当
   `/models/ast_edit_steering.gguf` 的 `.model` 侧车标记与所选模型
   匹配时，才由 `inference/entrypoint-v3.1.sh` 自动加载 —— 一旦通过
   `geometric-lens/asa_calibration/README.md` 中的工作流构建出兼容的
   向量，它就始终生效。可通过 `ATLAS_CONTROL_VECTOR*` 环境变量
   覆盖路径/缩放/层范围。

   **逐模型耦合。** 每个 ASA 向量都是针对某个特定模型的
   残差流几何结构训练出来的。任何跨模型回退都不安全。
   `atlas asa check` 验证 `.model` 侧车标记，探测已加载的嵌入维度，
   解析 GGUF 层元数据，并报告 `compat` / `needs-build` /
   `incompatible`。`atlas asa build` 从已加载的模型推导提取层，
   写出向量和标记，并运行在 lens 容器内部。`atlas asa publish`
   在上传前会拒绝缺失或不匹配的标记。见 [CLI.md § atlas asa](../../CLI.md#atlas-asa)。

### 逐文件 Tier 分类

每一次 `write_file`/`edit_file` 调用都被独立分类：

| Tier | 最大轮次 | 动作 |
|------|-----------|--------|
| T0（对话型） | 5 | 仅文本回复 |
| T1（简单） | 0（无上限） | 直接写入 —— 无 V3 开销 |
| T2（功能） | 0（无上限） | 触发 V3 pipeline |
| T3（困难） | 0（无上限） | 触发 V3 pipeline |

tier 上限为 0（无上限）；由循环内部的检测器栈决定何时中断：lens 回退（`agent_lens_intervention`）、推理重复（`agent_reasoning_intervention`）、工具调用重复（`agent_repeat_intervention`）、路径感知的错误熔断器、无动作即 done 门控、claim-check 门控、计划遵循阈值，以及空回复回退。对于一次性的"修复整个应用"提示，运维人员可用 `ATLAS_MAX_TURNS=<n>` 覆盖 —— 见 `proxy/types.go::envOverrideMaxTurns`。

分类器在 `proxy/tools.go`（`classifyFileTier`）；逻辑模式匹配器在同一文件中（`hasLogicIndicators`）。

**始终为 T1（直接写入）：**
- 按名称匹配的配置文件（如 `package.json`、`go.mod`、`pyproject.toml`、`dockerfile`、`docker-compose.*`）
- 按扩展名匹配的数据文件（`.json`、`.yaml`、`.yml`、`.toml`、`.csv`、`.xml`、`.env`）
- 样式文件（`.css`、`.scss`、`.less`）
- 文档（`.md`、`.txt`、`.rst`）和 shell 脚本（`.sh`、`.bash`）
- 少于 **10 行**的极小文件（在那种体量下 V3 没有任何可以有意义地多样化的东西）
- 没有逻辑指标的未知扩展名

完整的配置文件列表和扩展名集合位于 `proxy/tools.go:classifyFileTier`。

**T2（V3 pipeline）** —— 当文件 ≥10 行且满足以下任一条件时合格：
- `hasLogicIndicators(content)` 返回 true —— 在覆盖函数/方法定义、控制流、错误处理、Flask/FastAPI/Django 路由、Express/Node API、React 状态/数据、校验、数据库调用、JSX/React 组件模式和导入的模式家族中出现 **2 次以上匹配**（字面 token 列表见 `proxy/tools.go:hasLogicIndicators`）
- 或者该文件具有可识别的源代码 / 标记语言扩展名（`.py`、`.go`、`.rs`、`.ts`、`.tsx`、`.js`、`.jsx`、`.html`、`.htm` 等）且没有触发逻辑指标 —— 在 T2 给予它疑点利益（覆盖诸如 12 行组件骨架这类极简但真实的文件）

**T3（困难）** —— 目前分类器自身从不直接发出 T3；圈复杂度精炼器（`refineTierWithCC`，经由 GH #39 第 2 点的 `/internal/cyclomatic_complexity`）可以在 McCabe CC 表明存在真实的分支密度时将 T2 *升级* 为 T3。从不降级。

### Plan 模式（按轮次预检）

Plan 模式是一个预检式的规划步骤，在每个 agent 轮次、第一次工具调用之前运行一次：规划器采样候选计划，用启发式打分，并把获胜者渲染进系统提示；当模型偏离计划胡乱折腾时，一个遵循门控会自动修订计划。它减少了探索折腾，并通过守住计划的验证步骤来阻止无证据的 `done`。

完整的流程、组件、可调项、跳过条件、成本和测试矩阵见 [PLAN_MODE.md](../../PLAN_MODE.md)。

### 安全限制

面向运维的限制及其调优旋钮。内部操控守卫（回溯定位、缺失模块/大小写不匹配操控、符号接地、空操作/空内容/语法门控、doctype 剥离）位于 `proxy/guardrails.go` 和 `proxy/agent.go`。

| 限制 | 取值 | 用途 |
|-------|-------|---------|
| 对话裁剪 | 按 slot 调整大小的滑动窗口：保留系统消息 + 最近的用户指令 + 当前活动文件的内容 + 尽可能多的尾部消息以塞满 `per-slot context − ATLAS_MAX_TOKENS − 2048`（下限：保留 8；硬上限通过 `ATLAS_AGENT_HISTORY_BUDGET`） | 在不丢掉正在编辑的文件的前提下防止上下文溢出 |
| 冗余读取短路 | 对一个未改动文件的整文件重读仅在其内容仍然在场时返回"已在上下文中"指针；否则重新提供完整文件（`ATLAS_DEDUP_READS=0` 禁用） | 避免每轮重新编码一个未改动的文件，同时不让模型盲编辑 |
| V3 交互式墙钟上限 | 单次 V3 pipeline 调用被限制在 `ATLAS_V3_TIMEOUT`（默认 180s）；超时时代理回退到模型自身的语法门控内容（`0` 禁用） | 在长时间修复停滞下保持交互式会话的响应 |
| 逐轮推理预算 | 在约 6144 个推理 token 后切断流（`ATLAS_REASONING_BUDGET`，0 禁用）；恢复时从推理中提取一个内嵌的 tool_call 或重新提示 | 限定推理螺旋 |
| 对已有文件的 write_file | 文件 > 5 行时拒绝；在 .py/.html/.htm 上，逐步语法门控操控转向 `ast_edit` | 强制外科式编辑（`edit_file`）或整节点编辑（`ast_edit`） |
| 可疑收缩守卫 | 当 `oldSize >= 100B` 且 `newSize < 64B` 时拒绝 `ast_edit`/`edit_file`（`proxy/guardrails.go::validateNotSuspiciouslyShrunk`） | 在破坏性的桩重写落盘之前抓住它们 |
| ast_edit 失控内容守卫 | 当 `content` > 8 KB 且 > 4× 文件大小时拒绝 | 抓住作为替换节点发出的推理泄漏块 |
| 错误循环熔断器 | 连续 3 次失败 | 停止失控的失败循环 |
| 探索预算 | 连续 4 次只读调用时警告；5 次以上时跳过该读取 | 推动模型去写，而不是无限探索 |
| 命令输出截断 | stdout 8,000 字符，stderr 4,000 字符 | 防止上下文泛滥 |
| 搜索结果 | 最多 200 个匹配；文件搜索跳过 > 1 MB 的文件 | 限定搜索成本 |
| 截断检测 | 对工具参数做 JSON 解析检查 | 抓住被截断的模型输出 |

---

## 4. V3 Pipeline（内层）

在 T2+ 文件的 `write_file`/`edit_file` 执行器内部激活。该 pipeline 有四个阶段，且在每个阶段都设有提前退出。

### Pipeline 流程

```mermaid
flowchart LR
    Entry["T2 detected"] --> Probe["Probe"] --> Score1["C(x)/G(x)"] --> SB1["Sandbox"]
    SB1 --> Pass1{"Pass?"}
    Pass1 -->|"Yes"| Done["Done"]

    Pass1 -->|"No"| PS["PlanSearch"] --> DS["DivSampling"] --> BF["BudgetForcing"] --> Build["Build Check"] --> Score2["Score K"] --> SB2["Test K"]

    SB2 --> AnyPass{"Passed?"}
    AnyPass -->|"2+"| SStar["S* Tiebreak"] --> Done
    AnyPass -->|"1"| Select["Lens Select"] --> Done

    AnyPass -->|"0"| FA["Failure Analysis"] --> PRCOT["PR-CoT"]
    PRCOT --> PRPass{"Pass?"}
    PRPass -->|"Yes"| Done
    PRPass -->|"No"| Refine["Refinement"]
    Refine --> RefPass{"Pass?"}
    RefPass -->|"Yes"| Done
    RefPass -->|"No"| Derive["Derivation"] --> Done

    style Entry fill:#1a3a5c,color:#fff
    style Done fill:#333,color:#fff
    style Probe fill:#1a3a5c,color:#fff
    style PS fill:#1a3a5c,color:#fff
    style DS fill:#1a3a5c,color:#fff
    style BF fill:#1a3a5c,color:#fff
    style SStar fill:#2d5016,color:#fff
    style Select fill:#2d5016,color:#fff
    style Score1 fill:#2d5016,color:#fff
    style Score2 fill:#2d5016,color:#fff
    style SB1 fill:#2d5016,color:#fff
    style SB2 fill:#2d5016,color:#fff
    style Build fill:#2d5016,color:#fff
    style PRCOT fill:#5c3a1a,color:#fff
    style Refine fill:#5c3a1a,color:#fff
    style Derive fill:#5c3a1a,color:#fff
    style FA fill:#5c3a1a,color:#fff
```

图例：蓝色 = 生成，绿色 = 验证/选择，棕色 = 修复。

### 各阶段细节

**Phase 0: Probe** 以渐进式预算重试（light → standard → 直接回复）生成单个基线候选。它用所选模型的 C(x)/G(x) 工件打分，并在 sandbox 中测试。如果通过，pipeline 立即退出。

**Phase 1: 约束驱动的生成**

- **PlanSearch** 通过提取不同的约束集合，生成 3 个结构上不同的实现计划
- **DivSampling** 施加扰动多样性：4 个角色（competitive_programmer、systems_engineer、mathematician、pragmatist）+ 4 条指令（step_by_step、edge_case_first、complexity_aware、constraint_driven）+ 4 种风格（functional、pythonic、optimize_iteratively、structured）
- **Budget Forcing** 控制思考 token 的分配：

| Tier | 思考 token | Wait 注入 |
|------|----------------|----------------|
| nothink | 0 | 模板级禁用思考 |
| light | 1,024 | 无 |
| standard | 2,048 | 若思考结束时 < 512 token |
| hard | 4,096 | 若思考结束时 < 1,024 token |
| extreme | 8,192 | 若思考结束时 < 2,048 token |

Wait 注入会追加 "Wait, let me reconsider.\n" 以请求更长的一轮推理。Tier 选择使用所选模型经过校准的 C(x) 能量；没有校准时，ATLAS 使用配置的默认预算，而不是套用另一个模型的常量。

**Phase 2: 验证与选择**

- **构建验证**：Python（`py_compile`）、TypeScript（`tsc --noEmit`）、JavaScript（`node --check`）、Go（`go build`）、Rust（`cargo check`）、C/C++（`gcc/g++ -fsyntax-only`）、Shell（`bash -n`）。针对 Next.js、React、Flask、Django、Express 有框架级覆盖。
- **S* 决胜**（2 个以上通过）：生成边界情形输入，运行两个候选，多数获胜
- **Lens 选择**（1 个通过或回退）：按 C(x) 能量排序，最低者获胜

**Phase 3: 修复**（若 0/K 通过）—— 三种策略，顺序执行并带提前退出：

- **失败分析**：对失败分类（wrong_algorithm、implementation_bug、edge_case_miss、time_limit、format_error、partial_correct）
- **元认知评估**：从观察到的失败类别推导并注入补偿性约束
- **PR-CoT**：4 个视角（logical_consistency、information_completeness、biases、alternative_solutions）×（分析 + 修复）= 约 8 次 LLM 调用，最多 3 轮
- **Refinement Loop**：失败分析 → 约束精炼 → 代码生成 → 测试 → 学习。2 次迭代，120s 预算，每次约 5+ 次 LLM 调用。余弦距离过滤（>= 0.15）防止假设重复
- **Derivation Chains**：分解为至多 5 个子问题，逐个用 sandbox 验证，组合出最终结果。约 7+ 次 LLM 调用

### 模块图

`benchmark/v3/` 中的 18 个 Python 模块。`v3-service/main.py` 编排其中的 13 个；`reasc`、`ace_pipeline`、`lens_feedback` 和 `embedding_store` 只在离线 bench 运行器（`benchmark/v3_runner.py`）下运行，而 `ablation_analysis` 是一个独立的分析脚本（未在图中显示）：

```mermaid
graph LR
    Main["main.py"] --> PS["PlanSearch 1A"]
    Main --> DS["DivSampling 1B"]
    Main --> BF["BudgetForcing 1C"]
    Main --> BASC["BlendASC 2A"]
    Bench["v3_runner.py\n(bench only)"] --> REASC["ReASC 2B"]
    Main --> SSTAR["S* 2C"]
    Main --> CS["CandidateSelection"]
    Main --> FA["FailureAnalysis 3A"]
    Main --> CR["ConstraintRefiner 3B"]
    Main --> PRCOT["PR-CoT 3C"]
    Main --> DC["DerivationChains 3D"]
    Main --> RL["RefinementLoop 3E"]
    Main --> MC["Metacognitive 3F"]
    Bench --> ACE["ACE 3G"]
    Main --> STG["SelfTestGen"]
    Bench --> LF["LensFeedback"]
    Bench --> ES["EmbeddingStore"]

    RL --> FA
    RL --> CR
    RL --> DC
    BASC --> BF
    REASC --> BF
    LF --> BASC
    LF --> BF

    style Main fill:#333,color:#fff
    style Bench fill:#333,color:#fff
    style PS fill:#1a3a5c,color:#fff
    style DS fill:#1a3a5c,color:#fff
    style BF fill:#1a3a5c,color:#fff
    style BASC fill:#2d5016,color:#fff
    style REASC fill:#2d5016,color:#fff
    style SSTAR fill:#2d5016,color:#fff
    style CS fill:#2d5016,color:#fff
    style FA fill:#5c3a1a,color:#fff
    style CR fill:#5c3a1a,color:#fff
    style PRCOT fill:#5c3a1a,color:#fff
    style DC fill:#5c3a1a,color:#fff
    style RL fill:#5c3a1a,color:#fff
    style MC fill:#5c3a1a,color:#fff
    style ACE fill:#5c3a1a,color:#fff
    style STG fill:#333,color:#fff
    style LF fill:#333,color:#fff
    style ES fill:#333,color:#fff
```

图例：蓝色 = Phase 1（生成），绿色 = Phase 2（选择），棕色 = Phase 3（修复），灰色 = 工具。由 `v3_runner.py` 供给的模块仅用于 bench 运行器；服务不会调用它们。

---

## 5. Geometric Lens

一个神经打分系统，通过分析模型嵌入的几何结构，在不执行代码的情况下评估代码质量。完全运行在 CPU 上。同时也作为项目索引、检索、置信度路由和模式缓存的 RAG API。

#### 为什么叫 "Geometric Lens"？

Geometric Lens 背后的核心理念源自一个简单的前提：停止扩大模型，转而用支撑性的基础设施把它们包裹起来。Jose Crespo 的 ["Everyone's Wrong About AI Programming"](https://www.josecrespophd.org/p/everyones-wrong-about-ai-programming) 论证了 AI 生成的代码会漂向错误，因为当前的 LLM 工作在扁平的嵌入空间中，正确与错误的代码路径代价相同。解决方案是在模型周围构建一个能量景观，让正确的代码处于"下坡"、错误的代码处于"上坡"。

Anthropic 的 [Manipulating Manifolds](https://transformer-circuits.pub/2025/linebreaks/index.html) 研究提供了证据，表明 transformer 已经在其嵌入空间中创造出可操纵的几何结构 —— 原材料早已存在。Bar 等人的 [Geometric Unification of Generative AI](https://arxiv.org/html/2510.00666v1) 形式化了如何在数据流形上学习并使用距离函数来打分。

ATLAS 用两个互补的模型实现这一点。C(x) 是建立在所选模型自身嵌入之上的一个习得的能量函数（`hidden_dim`→512→128→1 的 MLP）。每个代码候选都由 llama-server 嵌入，C(x) 给它在那个几何结构中所处的位置打分。低能量意味着该候选与已知正确的代码聚成一类。高能量意味着它与已知错误的代码聚成一类。无需外部预言机，无需执行 —— 仅仅是所选模型表示的几何结构。

G(x) 是质量预测器 —— 一个建立在 PCA 降维嵌入之上的 XGBoost 分类器，根据候选在降维空间中所处的位置预测通过/失败。当 C(x) 回答"这个候选有多好？"时，G(x) 回答"这个候选可能通过吗？"它是唯一的 G(x) 实现：早先的度量张量形式及其可修正性端点在 XGBoost 成为部署路径后已被移除（几何感知变体见 git 历史）。

### 打分模型

```mermaid
graph LR
    EE["Embedding Extractor\nllama-server /embedding\nmodel hidden dim"] --> CX["C(x) Cost Field\nd→512→128→1\nSiLU + Softplus"]
    EE --> GX["G(x) XGBoost\nPCA(128) + classifier"]
    CX --> SVC["Service Layer\nevaluate_combined()"]
    GX --> SVC
    SVC --> V{"Verdict"}
    V -->|"at/above artifact low"| LC["likely_correct"]
    V -->|"between severe and low"| UN["uncertain"]
    V -->|"below artifact severe"| LI["likely_incorrect"]

    TR["Training Pipeline\ncontrastive ranking loss"] --> CX
    EWC["EWC\nFisher information\nprevents catastrophic forgetting"] --> TR
    RB["Replay Buffer\ndomain-stratified\n30% old / 70% new"] --> TR

    MT["Metric Tensor\ndiagonal G(x) in PCA space\n(code exists, not deployed)"] -.-> CORR["Correction Engine\n-α · G⁻¹ · ∇C"]

    style EE fill:#333,color:#fff
    style CX fill:#2d5016,color:#fff
    style GX fill:#2d5016,color:#fff
    style SVC fill:#333,color:#fff
    style TR fill:#1a3a5c,color:#fff
    style EWC fill:#1a3a5c,color:#fff
    style RB fill:#1a3a5c,color:#fff
    style MT fill:#555,color:#ccc
    style CORR fill:#555,color:#ccc
```

以下数字描述的是已发表的 V3 研究所用的冻结参考工件；它们是出处记录，不是运行时的维度或默认值：

| 模型 | 参考架构 | 训练数据 | 性能 |
|-------|-------------|---------------|-------------|
| **C(x)** | 4096→512→128→1 MLP (SiLU, Softplus) | 597 个 LCB 嵌入（504 PASS，93 FAIL） | Val AUC 0.9467，分离度 2.04x |
| **G(x)** | PCA(4096→128) + XGBoost | 13,398 个嵌入（4,835 PASS，8,563 FAIL） | PCA 80.8% 方差 |

C(x) 的归一化是 `sigmoid(steepness × (energy - midpoint))`。所选模型的 `cx_normalization.json` 提供这两个值；`atlas lens build` 会从该模型带标签的 PASS/FAIL 候选中推导它们。G(x) 的判定阈值同样来自 `gx_thresholds.json`。缺少任一校准时，归一化的判定保持中性/未校准状态，而不是借用参考工件的标度。

每个当前的 Lens 工件包还包含 `model_identity.json`。服务要求其中的模型名与 `ATLAS_MODEL_NAME` 匹配；仅凭嵌入宽度相等无法确立两个不同模型之间的兼容性。

> **注意：** 模型权重（.pt、.pkl 文件）未提交到仓库 —— 它们在训练期间构建，并烘焙进容器镜像或在运行时挂载。当模型文件缺失时，服务会优雅降级：C(x) 返回中性能量，G(x) 返回 `gx_score: 0.5` 和 `verdict: "unavailable"`。训练数据与权重可在 [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS) 获取。

### RAG / PageIndex V2

```mermaid
graph LR
    subgraph indexing["Indexing Pipeline"]
        AST["AST Parser\ntree-sitter Python"] --> TB["Tree Builder\nhierarchical index"]
        TB --> BM25I["BM25 Index\ninverted index, k1=1.5"]
        TB --> SUM["Summarizer\nLLM-generated summaries"]
        BM25I --> PERS["Persistence\nJSON to disk"]
        SUM --> PERS
    end

    subgraph retrieval["Retrieval"]
        BM25S["BM25 Searcher\nmin_score=0.1, top_k=20"]
        TreeS["Tree Searcher\nLLM-guided traversal\nmax_depth=6, max_calls=40"]
        HYB["Hybrid Retriever\nroutes: bm25_first / tree_only / both"]
        BM25S --> HYB
        TreeS --> HYB
    end

    style indexing fill:#1a3a5c,color:#fff
    style retrieval fill:#2d5016,color:#fff
```

### 置信度路由器与模式缓存

```mermaid
graph LR
    subgraph router["Confidence Router"]
        SIG["Signal Collector\npattern_cache, retrieval_confidence\nquery_complexity, geometric_energy"]
        DIFF["Difficulty Estimator\nweighted fusion → D(x)"]
        TS["Thompson Sampling\nBeta(α,β) posteriors\nper-route cost weighting"]
        FB["Feedback Recorder\nSQLite-backed"]
        FC["Fallback Chain\nCACHE_HIT → FAST_PATH\n→ STANDARD → HARD_PATH"]
        SIG --> DIFF --> TS --> FC
        FB --> TS
    end

    subgraph cache["Pattern Cache"]
        PS["Pattern Store\nSQLite: STM (100) / LTM / PERSISTENT"]
        PM["Pattern Matcher\nBM25 over summaries"]
        PE["Pattern Extractor\nLLM-driven"]
        PSC["Pattern Scorer\nEbbinghaus decay"]
        COO["Co-occurrence Graph\nlinked pattern retrieval"]
        PE --> PS
        PS --> PM
        PM --> PSC
        PS --> COO
    end

    style router fill:#5c3a1a,color:#fff
    style cache fill:#5c3a1a,color:#fff
```

4 条路由采用代价加权的 Thompson Sampling：CACHE_HIT (cost=1, k=0) → FAST_PATH (cost=50, k=1) → STANDARD (cost=300, k=5) → HARD_PATH (cost=1500, k=20)。

---

## 6. Sandbox

带编译、测试和检查的隔离代码执行。

```mermaid
graph LR
    subgraph executors["Language Executors"]
        Py["Python\npylint (0-10) + pytest"]
        JS["JavaScript\nNode.js 20"]
        TS["TypeScript\ntsc --noEmit + tsx"]
        Go["Go 1.22\ngo build + run"]
        Rust["Rust stable\nrustc + run"]
        C["C / C++\ngcc/g++ -Wall"]
        Bash["Bash\nbash -n + run"]
    end

    subgraph support["Support"]
        Syn["Syntax Checker\nper-language AST validation"]
        Err["Error Classifier\n15 types: SyntaxError, NameError\nTypeError, CompileError, Timeout..."]
        Trunc["Output Truncation\nstdout: 4000 chars\nstderr: 2000 chars"]
    end

    style executors fill:#2d5016,color:#fff
    style support fill:#333,color:#fff
```

接受的语言别名：`py`/`python3`（Python）、`js`/`node`（JavaScript）、`ts`（TypeScript）、`golang`（Go）、`rs`（Rust）、`c++`（C++）、`sh`/`shell`（Bash）。最大执行时间：Docker 部署中为 300s（compose 设置 `MAX_EXECUTION_TIME=${ATLAS_SANDBOX_MAX_EXECUTION_TIME:-300}` 以匹配代理 5 分钟的 `run_command` 上限；裸代码默认值为 60s）。最大内存：512 MB。两个工作区路径：**`/execute`**（V3 候选测试路径）使用 `/tmp/sandbox`（tmpfs）下的一个临时草稿目录；**`/shell`**（agent 的 `run_command` 路由，外加用于后台进程的 `/jobs/*`）针对 `/workspace` 运行 —— 即来自 `ATLAS_PROJECT_DIR`（Docker）或 hostPath `${ATLAS_PROJECTS_DIR}`（K3s）绑定挂载的项目根，与代理看到的是同一路径。

---

## 7. VRAM 预算示例

一次实测的 RTX 5060 Ti 16GB 部署，使用 9B Q6 模型和 32K 上下文：

| 组件 | VRAM |
|-----------|------|
| Qwen3.5-9B-Q6_K 模型权重 | ~6.9 GB |
| KV 缓存（32K 上下文） | ~1.3 GB |
| **llama-server 合计** | **~8.2 GB** |
| Geometric Lens | 0（仅 CPU，模型约 12 MB RAM，PyTorch 运行时约 128 MB） |
| v3-service | 0（仅 CPU） |
| sandbox | 0（仅 CPU） |
| atlas-proxy | 0（Go 二进制，约 30 MB RAM） |
| **空闲 VRAM** | **~7.8 GB** |

llama-server 之外的所有计算都跑在 CPU 上。GPU 仅用于 LLM 推理和嵌入提取。

### 7.1 逐后端的 VRAM 预算

上面 8.2 GB / 7.8 GB 空闲的拆分是一个示例，不是 ATLAS 的模型默认值。实际用量取决于 `atlas init` 所选的模型、量化、上下文和并行 slot 设置。其他后端在结构上有所不同：

| 后端 | 报告的 "VRAM" | 负载下的现实预算 | 备注 |
|---|---|---|---|
| **CUDA**（专用 VRAM） | 硬件规格（基准 5060 Ti 上为 16 GB） | 约规格的 95%（驱动保留约 500 MB） | 上表中的数字直接适用。 |
| **ROCm**（专用 VRAM） | 硬件规格 | 约规格的 90–95%（HIP 运行时比 CUDA 的略重） | RX 7900 XTX (24 GB) → 可以从容运行 14B Q5 + 32K 上下文，带 2 个并行 slot。 |
| **Metal**（Apple 统一内存） | 系统总 RAM | 系统 RAM 的 **约 70%** | 操作系统 + 浏览器 + IDE 吃掉约 30%。一台 16 GB 的 MBP 有约 11 GB 的*现实*预算 —— 对 Qwen3.5-9B Q6_K（7.5 GB + 2-4 GB KV 缓存）来说太紧。≤16 GB 用 Q4_K_M（5 GB）；Q6_K 想要 ≥24 GB 统一内存。 |
| **SYCL**（Intel Arc） | 硬件规格 | 未知 —— 发布时待定 | A770 (16 GB) 目标在保守意义上等价于 NVIDIA 16 GB。 |

---

## 8. 部署

服务依赖图（各部署模式完全一致）：

```mermaid
graph LR
    LLM["llama-server"] -->|"healthy"| GL["geometric-lens"] -->|"healthy"| AP["atlas-proxy"]
    LLM -->|"healthy"| V3["v3-service"] -->|"healthy"| AP
    GL -->|"healthy"| V3
    SB["sandbox"] -->|"healthy"| AP

    style LLM fill:#5c1a1a,color:#fff
    style GL fill:#2d5016,color:#fff
    style V3 fill:#2d5016,color:#fff
    style SB fill:#2d5016,color:#fff
    style AP fill:#1a3a5c,color:#fff
```

`llama-server` 和 `sandbox` 独立启动。`geometric-lens` 等待 `llama-server` 变为健康；`v3-service` 等待 `llama-server` 和 `geometric-lens`；`atlas-proxy` 等待 `llama-server`、`geometric-lens`、`v3-service` 和 `sandbox`。所有模式都由同一个 `inference/entrypoint-v3.1.sh` 驱动，因此上下文大小、KV 缓存量化、flash attention 和 mlock 都由环境变量控制，行为在 Docker Compose、裸机、macOS 混合 Metal 和 K3s 之间完全一致。

安装及各模式的拉起步骤（NVIDIA / ROCm override、裸机、macOS 混合 Metal、K3s 清单）见 [SETUP.md](../../SETUP.md)；macOS 原生路径见 [SETUP_MACOS.md](../../SETUP_MACOS.md)。

---

## 9. 数据流

### T1：简单文件写入

```mermaid
sequenceDiagram
    participant U as User
    participant A as Client
    participant P as atlas-proxy :8090
    participant L as llama-server :8080

    U->>A: "Create a config file"
    A->>P: POST /v1/agent (SSE)
    P->>L: POST /v1/chat/completions<br/>response_format: json_object
    L-->>P: {"type":"tool_call","name":"write_file","args":{...}}
    Note over P: Tier = T1 (config file)<br/>Direct write, no V3
    P-->>P: Write file to disk
    P-->>A: SSE stream: file content
    A-->>U: File created
```

一次 LLM 调用。无 V3 开销。

### T2：功能文件写入

```mermaid
sequenceDiagram
    participant U as User
    participant A as Client
    participant P as atlas-proxy :8090
    participant L as llama-server :8080
    participant V as v3-service :8070
    participant G as geometric-lens :8099
    participant S as sandbox :30820

    U->>A: "Create a REST API handler"
    A->>P: POST /v1/agent (SSE)
    P->>L: POST /v1/chat/completions<br/>response_format: json_object
    L-->>P: {"type":"tool_call","name":"write_file","args":{...}}
    Note over P: Tier = T2 (≥10 lines, logic)<br/>Route to V3

    P->>V: POST /v3/generate (SSE)
    Note over V: Phase 0: Probe
    V->>L: POST /v1/chat/completions (generate code)
    L-->>V: probe candidate
    V->>L: POST /v1/embeddings (model hidden dim)
    L-->>V: embedding vector
    V->>G: POST /internal/lens/gx-score
    G-->>V: {cx_energy, gx_score, verdict}
    V->>S: POST /execute (test probe)
    S-->>V: {success: false}

    Note over V: Phase 1: PlanSearch + DivSampling
    V->>L: POST /v1/chat/completions (x K candidates)
    L-->>V: K candidates
    V->>S: POST /execute (test each)
    S-->>V: {success: true} for candidate 2

    Note over V: Phase 2: Lens select winner
    V->>G: POST /internal/lens/gx-score
    G-->>V: scores

    V-->>P: SSE result: winning code
    P-->>P: Write file to disk
    P-->>A: SSE stream: file content
    A-->>U: File created
```

最少 3 次 llama-server 调用（1 次 probe 生成 + 1 次自测生成 + 1 次嵌入提取）。如果 Phase 3 修复启用了所有策略，最多 30+ 次。

### 编辑已有代码

```mermaid
sequenceDiagram
    participant U as User
    participant A as Client
    participant P as atlas-proxy :8090
    participant L as llama-server :8080

    U->>A: "Fix the bug in auth.py"
    A->>P: POST /v1/agent (SSE)
    P->>L: POST /v1/chat/completions<br/>response_format: json_object
    L-->>P: {"type":"tool_call","name":"read_file","args":{"path":"auth.py"}}
    P-->>P: Read file from disk
    P->>L: POST /v1/chat/completions (with file content)
    L-->>P: {"type":"tool_call","name":"edit_file","args":{"old_str":"...","new_str":"..."}}
    P-->>P: Apply old_str→new_str replacement
    P->>L: POST /v1/chat/completions (with edit result)
    L-->>P: {"type":"done","summary":"Fixed auth bug"}
    P-->>A: SSE stream: edited content
    A-->>U: File updated
```

超过 5 行的已有文件对 `write_file` 会被拒绝 —— 模型必须使用 `edit_file`（外科式，≤10 行）或 `ast_edit`（整节点重写，仅 .py/.html/.htm）。在 `.py`/`.html`/`.htm` 文件上，逐步语法门控（BiasBusters #2）会在下一次决策中主动从工具名产生式里禁掉 `edit_file`/`write_file`，使模型无法退回到错误的捷径。
