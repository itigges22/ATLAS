> **[English](../../TROUBLESHOOTING.md)** | **简体中文** | **[日本語](../ja/TROUBLESHOOTING.md)** | **[한국어](../ko/TROUBLESHOOTING.md)**

# ATLAS 故障排除指南

ATLAS V3.0.1 的常见问题与解决方案，按服务分类。

---

## 快速诊断

遇到问题时，请先运行以下命令确定问题所在：

```bash
# Docker Compose - 一次检查所有服务
docker compose ps

# 逐个健康检查
curl -s http://localhost:8080/health | python3 -m json.tool   # llama-server
curl -s http://localhost:8099/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy（显示所有服务状态）

# GPU 状态
nvidia-smi

# Docker Compose 日志（每个服务最近 50 行）
docker compose logs --tail 50
```

atlas-proxy 的健康检查端点会报告所有上游服务的状态：
```json
{
  "status": "ok",
  "inference": true,
  "lens": true,
  "sandbox": true,
  "port": "8090",
  "stats": { "requests": 0, "repairs": 0, "sandbox_passes": 0, "sandbox_fails": 0 }
}
```

如果任何字段为 `false`，则该服务存在问题。

---

## Docker / Podman 问题

### 容器中未检测到 GPU

**现象：** llama-server 容器启动但模型在 CPU 上加载（非常慢，约 2 tok/s）。主机上 `nvidia-smi` 能看到 GPU 但容器内无法访问。

**解决方法：** 安装 NVIDIA Container Toolkit：

```bash
# RHEL/Fedora
sudo dnf install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=podman
sudo systemctl restart podman

# Ubuntu/Debian
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证容器内 GPU 是否可见：
```bash
# Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Podman
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0-base nvidia-smi
```

### 首次构建失败（找不到 CUDA）

**现象：** `docker compose build` 在 llama-server 编译过程中出现 CUDA 相关错误。

**解决方法：** llama-server 的 Dockerfile 在 `nvidia/cuda:12.8.0-devel` 基础镜像中构建 llama.cpp，因此构建时不需要主机 GPU 访问即可使用 CUDA 头文件。常见的构建失败原因：
1. 磁盘空间不足（构建产物需要约 5GB）
2. 下载 CUDA 基础镜像或克隆 llama.cpp 时的网络问题
3. Podman 非 root 构建可能因权限问题失败 - 尝试使用 `podman-compose build` 加上 `--podman-build-args="--format docker"`

### SELinux 阻止容器访问（Fedora/RHEL）

**现象：** 容器无法读取挂载的卷，模型文件权限被拒绝。

**解决方法：**
```bash
# 允许容器访问模型目录
chcon -Rt svirt_sandbox_file_t ~/models/

# 或为卷挂载添加 :Z 标志（Docker Compose 会自动处理）
```

### Sandbox 不可达

**现象：** 代理健康检查显示 `"sandbox": false`。V3 构建验证失败。

**解决方法：** 确保所有服务在同一个 Docker 网络上。Docker Compose 会自动创建 `atlas` 网络。如果手动运行容器：
```bash
docker network create atlas
# 使用 --network atlas 启动所有容器
```

### 端口冲突

**现象：** `docker compose up` 因某端口"address already in use"而失败。

**解决方法：** 检查占用端口的进程，然后停止它或更改 `.env` 中的 ATLAS 端口：
```bash
# 查找占用 8080 端口的进程
lsof -i :8080

# 在 .env 中更改端口
ATLAS_LLAMA_PORT=8081    # llama-server 使用不同端口
```

所有端口均可通过 `.env` 配置。参见 [CONFIGURATION.md](../../CONFIGURATION.md)。

---

## llama-server 问题

### 模型在 CPU 而非 GPU 上加载

**现象：** 生成速度约 2 tok/s 而非约 50 tok/s。`nvidia-smi` 未显示 llama-server 使用 GPU。

**解决方法：** 确保设置了 `--n-gpu-layers 99`（将所有层卸载到 GPU）。Docker Compose 中这是默认设置。裸机部署时，请检查启动命令：
```bash
ps aux | grep llama-server | grep 'n-gpu-layers'
```

如果使用 Docker，请确保已配置 NVIDIA 容器运行时（参见上方 GPU 章节）。

### 模型文件未找到

**现象：** llama-server 立即退出，报错 "failed to load model" 或类似信息。

**解决方法：** 检查模型路径：
```bash
# Docker Compose - 模型必须在 ATLAS_MODELS_DIR 中（默认：./models/）
ls -la models/Qwen3.5-9B-Q6_K.gguf

# 裸机 - 检查 ATLAS_MODEL_PATH
ls -la ~/models/Qwen3.5-9B-Q6_K.gguf
```

文件名必须与 `.env` 中的 `ATLAS_MODEL_FILE` 匹配（默认：`Qwen3.5-9B-Q6_K.gguf`）。

### 显存不足

**现象：** llama-server 启动后不久崩溃或被 OOMKill。`nvidia-smi` 显示显存接近 100%。

**解决方法：** 9B Q6_K 模型需要约 8.2 GB 显存（模型 + KV 缓存）。请确保：
1. 没有其他 GPU 进程在运行（`nvidia-smi` - 检查其他 CUDA 进程）
2. 你有 16GB+ 显存
3. 上下文大小未设置过高（默认 32K 即可，增大前请检查显存）

```bash
# 如有需要，终止其他 GPU 进程
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill {}
```

### 语法未强制执行（模型输出思维块）

**现象：** 模型输出 `<think>` 标签或原始文本，而非 JSON 工具调用。

**解决方法：** 当 `ATLAS_AGENT_LOOP=1` 时，代理会自动设置 `response_format: {"type": "json_object"}`。如果直接使用 llama-server，请在请求中包含该参数：
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q6_K",
    "messages": [{"role":"user","content":"Say hi"}],
    "max_tokens": 50,
    "response_format": {"type": "json_object"}
  }'
```

如果返回的是原始文本而非 JSON，说明你的 llama.cpp 构建不支持 `response_format`。请从最新源码重新构建。

### 上下文窗口过小

**现象：** 工具调用参数被截断。`write_file` 因 "unexpected end of JSON" 失败，或代理日志显示 "truncation detected"。

**解决方法：** 上下文大小应为 32768（Docker Compose 中的默认值）。请检查：
```bash
# Docker Compose
grep CTX_SIZE .env

# 裸机
ps aux | grep llama-server | grep ctx-size
```

---

## 代理问题

### 代理循环未激活

**现象：** 请求直接发送到 llama-server。没有工具调用、没有流式状态图标、没有 V3 Pipeline。

**解决方法：** 设置 `ATLAS_AGENT_LOOP=1`。`atlas` 启动器会自动执行此操作。如果手动运行代理：
```bash
ATLAS_AGENT_LOOP=1 atlas-proxy-v2
```

在 Docker Compose 中，此项已在 `docker-compose.yml` 中设置，无需手动配置。

### V3 Pipeline 未对功能文件触发

**现象：** 所有 `write_file` 调用都是 T1（直接写入）。输出中没有 V3 Pipeline 阶段。

V3 仅在**同时满足以下三个条件**时触发：
1. 文件内容**超过 50 行**
2. 文件有**3 个以上逻辑指标**（函数定义、控制流、API 模式）
3. V3 服务可通过 `ATLAS_V3_URL` 访问

**诊断方法：**
```bash
# 检查 V3 服务健康状态
curl -s http://localhost:8070/health

# 检查代理日志中的层级分类
docker compose logs atlas-proxy | grep "write_file"
# 查找：T1（直接写入）vs T2（V3 Pipeline）
```

如果 V3 不可达，代理会静默回退到直接写入。

### 截断错误（write_file 反复失败）

**现象：** 反复出现类似 "Your output was truncated - the content is too long for a single tool call." 的错误。

**原因：** 模型试图在一次调用中写入过多内容。代理检测到截断的 JSON 并拒绝了该工具调用。

**自动处理机制：**
- 对于超过 100 行的现有文件：代理拒绝 `write_file` 并告知模型改用 `edit_file`
- 连续 3 次失败后：错误循环断路器会停止代理并返回摘要

**你可以做的：** 重新表述请求，要求进行针对性修改而非完整文件重写。例如，使用"为登录函数添加输入验证"而非"重写 auth.py"。

### 编辑前未读取文件

**现象：** `edit_file` 失败并报错 "file not read yet - use read_file first before editing."

**原因：** 代理会跟踪已读取的文件。如果模型试图编辑本次会话中未读取的文件，编辑会因过时保护而被拒绝。

**解决方法：** 这是正常行为 - 模型应先读取文件。如果持续失败，模型可能混淆了已查看的文件。尝试在 Aider 中使用 `/clear` 并重新表述请求。

### 文件被外部修改

**现象：** `edit_file` 失败并报错 "file modified since last read - read it again before editing."

**原因：** 文件在模型读取后被磁盘上的其他操作（你或其他进程）修改。代理会比较修改时间戳。

**解决方法：** 模型需要重新读取文件。这通常在下一轮会自动解决。

### 探索预算警告

**现象：** 输出显示 "You have full project context in the system prompt. Do not read more files." 或读取操作被跳过。

**原因：** 模型连续进行了 4 次以上的只读调用（read_file、search_files、list_directory）而没有写入任何内容。4 次读取后代理会发出警告。5 次以上则直接跳过读取并告知模型进行写入。

**解决方法：** 这是保护性行为。如果模型确实在探索中卡住了，请更具体地说明你想要修改的内容。

---

## Geometric Lens 问题

### Lens 未加载/不可用

**现象：** 代理健康检查显示 `"lens": false`。或启动时显示 "Lens unavailable - verification disabled."

**影响：** ATLAS 仍可工作，但没有 C(x)/G(x) 评分。V3 候选选择回退到仅沙箱验证。

**解决方法：** 检查 Lens 健康状态和日志：
```bash
curl -s http://localhost:8099/health
docker compose logs geometric-lens
```

常见原因：
- Lens 无法连接到 llama-server（检查 `LLAMA_URL` 环境变量）
- 模型权重文件缺失（服务会优雅降级 - 如果你尚未训练自定义模型，这是预期行为）

### 所有分数接近 0.5

**现象：** 无论代码质量如何，每个候选都得到 `cx_energy: 0.0` 和 `gx_score: 0.5`。

**原因：** 模型权重未加载。模型缺失时，服务返回中性默认值。

**验证方法：**
```bash
curl -s http://localhost:8099/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(1)"}' | python3 -m json.tool
```

如果返回 `enabled: false` 或 `cx_energy: 0.0`，则模型未加载。对于全新安装来说这是预期行为 - 模型权重不包含在仓库中，需要训练或从 [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS) 下载。

### 嵌入向量提取失败

**现象：** Lens 日志显示 "embedding extraction failed" 或超时等错误。

**原因：** Lens 调用 llama-server 的 `/v1/embeddings` 端点。如果 llama-server 过载或该端点未启用，则会失败。

**解决方法：**
```bash
# 直接测试嵌入端点
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}' | python3 -m json.tool
```

对于生成模型的自嵌入，`/v1/embeddings` 端点在 llama.cpp 中无需特殊标志即可使用。在 K3s 中，入口脚本显式设置了 `--embeddings` 标志以获得完整的嵌入支持。

---

## Sandbox 问题

### Sandbox 不可达

**现象：** 代码从未被测试。代理健康检查显示 `"sandbox": false`。

**解决方法：** 检查 Sandbox 健康状态：
```bash
# Docker Compose（主机端口 30820 映射到容器端口 8020）
curl -s http://localhost:30820/health

# 裸机（直接端口 8020）
curl -s http://localhost:8020/health
```

如果 Sandbox 容器正在运行但不健康，请检查日志：
```bash
docker compose logs sandbox
```

### 代码执行超时

**现象：** Sandbox 返回 `"error_type": "Timeout"`。代码执行时间过长。

**默认超时：** 每个请求 30 秒，最大 60 秒（可通过 `MAX_EXECUTION_TIME` 环境变量配置）。

**解决方法：** 如果你的代码确实需要更多时间，请在请求中设置更高的超时值。如果代码存在无限循环，这是预期行为。

### 语言不受支持

**现象：** Sandbox 对特定语言返回错误。

**支持的语言：** Python、JavaScript、TypeScript、Go、Rust、C、C++、Bash。

检查可用的运行时：
```bash
curl -s http://localhost:30820/languages | python3 -m json.tool
```

---

## Aider 问题

### `atlas` 显示 REPL 而非 Aider（无法读写文件）

**现象：** 运行 `atlas` 后显示内置 REPL，带有 `Model`、`Speed`、`Lens`、`Sandbox` 状态块和 `◆` 提示符。输入请求可以工作但不会创建或修改文件。`--message` 标志被忽略。

**原因：** `atlas` 命令会自动检测代理和 Aider。如果任一缺失，则回退到内置 REPL，该 REPL 支持 `/solve` 和 `/bench` 但不支持文件操作。

**解决方法：**
1. 确保代理正在运行：`curl -s http://localhost:8090/health`
2. 确保已安装 Aider：`pip install aider-chat`
3. 确保服务已启动：`docker compose ps`（所有服务应显示 "healthy"）

如果代理健康且已安装 Aider，`atlas` 将自动启动 Aider 并运行完整的代理循环（工具调用、文件读写、V3 Pipeline）。

如果安装了 Go 1.24+，`atlas` 还可以自动构建并启动代理 - 无需手动启动。

### 代理列出错误的目录或 `/tmp`

**现象：** 模型列出的是 `/tmp` 或 ATLAS 仓库的文件而非你的项目文件。`write_file` 在错误的位置创建文件。

**原因：** Docker Compose 代理在容器内运行，只能看到启动时挂载的目录。如果你在其他目录中工作，代理无法访问。

**解决方法（推荐）：** 安装 Go 1.24+（[https://go.dev/dl/](https://go.dev/dl/)）。`atlas` CLI 会自动在当前目录本地构建并启动代理，获得完整的文件访问权限。无需 Docker 挂载。

**解决方法（无 Go）：** 在 `.env` 中将 `ATLAS_PROJECT_DIR` 设置为你的项目路径，然后重启代理：
```bash
# 在 .env 中：
ATLAS_PROJECT_DIR=/path/to/your/project

# 重启代理以应用新的挂载：
docker compose up -d atlas-proxy
```

每次切换项目目录都需要更新此配置并重启。这是代理在 Docker 中运行的限制。

### 克隆后缺少 `.env.example`

**现象：** `cp .env.example .env` 失败并报错 "No such file or directory"。

**解决方法：** 此问题已在 V3.0.1 中修复。如果你在修复前克隆的，请拉取最新代码：
```bash
git pull
cp .env.example .env
```

### Aider 在长任务中断开连接

**现象：** Aider 在代理循环完成前超时或断开连接，特别是在 V3 Pipeline 阶段。

**解决方法：** Aider 的 HTTP 请求超时需要足够长以支持 V3 Pipeline 执行（可能需要几分钟）。仓库中的 `.aider.model.settings.yml` 配置了流式模式以保持连接活跃。如果仍然遇到超时：

1. 确保使用了仓库的配置文件（`.aider.model.settings.yml` 和 `.aider.model.metadata.json`）
2. 检查设置文件中 `streaming: true` 是否已设置

### 空响应

**现象：** Aider 显示了完成摘要但未生成任何文件内容。

**原因：** 模型发出了 `done` 信号但没有进行任何文件更改。以下情况可能触发：
- 非常短的对话提示（"hi"、"thanks"）
- 模糊的请求，模型不知道该创建哪个文件

**解决方法：** 请更具体。明确告诉模型要创建或编辑哪个文件。

### 工作目录错误

**现象：** 文件创建在错误的位置。`list_directory` 显示意外内容。

**原因：** 代理通过查找最近修改的 `.aider.chat.history.md` 文件来检测项目目录。如果你有多个 Aider 会话打开，最新的那个会生效。

**解决方法：** 关闭其他 Aider 会话，或在运行 `atlas` 前先 `cd` 到正确的项目目录。

### "Model not found" 错误

**现象：** Aider 启动失败并报模型相关错误。

**解决方法：** 确保 ATLAS 根目录中存在这两个 Aider 配置文件：
```bash
ls -la .aider.model.settings.yml .aider.model.metadata.json
```

这些文件包含在仓库中。如果缺失，请重新克隆或从备份恢复。它们告诉 Aider 使用指向代理的 `openai/atlas` 模型。

---

## 性能问题

### 生成速度慢（约 2 tok/s）

模型正在 CPU 而非 GPU 上运行。请检查：
1. `nvidia-smi` - llama-server 是否列为 GPU 进程？
2. `--n-gpu-layers 99` - 所有层是否已卸载到 GPU？
3. NVIDIA Container Toolkit - 容器运行时是否已配置 GPU 访问？

**预期性能：** 在 RTX 5060 Ti 16GB 上启用语法强制执行时约 51 tok/s。

### V3 Pipeline 需要几分钟

对于 T2 文件来说这是正常的。V3 Pipeline 会进行多次 LLM 调用：
- **仅探测（最佳情况）：** 约 10-15 秒（1 次生成 + 1 次评分 + 1 次测试）
- **第一阶段生成：** 约 1-2 分钟（PlanSearch + DivSampling + 评分）
- **第三阶段修复：** 约 2-5 分钟（PR-CoT + Refinement + Derivation，如果需要）

如需更快（但质量较低）的结果：
- 保持文件在 50 行以下（维持 T1，不触发 V3）
- 降低逻辑复杂度（减少函数、控制流）
- V3 仅在确实需要时才触发 - 简单文件会立即写入

### 内存使用过高

**现象：** 系统变得卡顿或服务被 OOMKill。

**预期内存使用：**
- llama-server：约 8 GB（模型在显存中，仅占少量系统内存）
- geometric-lens：约 200 MB（PyTorch 运行时 + 模型）
- v3-service：约 150 MB（PyTorch 运行时）
- sandbox：约 100 MB（基础值，编译时会有峰值）
- atlas-proxy：约 30 MB（Go 二进制文件）

**总计：** 约 500 MB 系统内存 + 8.2 GB 显存。如果你的系统内存不足 14 GB，其他服务可能会争夺内存。

---

## 获取帮助

如果你的问题未在此列出：
1. 查看服务日志：`docker compose logs <service-name>`
2. 检查代理健康检查端点：`curl http://localhost:8090/health`
3. 参见 [CONFIGURATION.md](../../CONFIGURATION.md) 了解所有环境变量
4. 在 [GitHub](https://github.com/itigges22/ATLAS/issues) 上提交 issue
