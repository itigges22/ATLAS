> **[English](../../SETUP.md)** | **简体中文** | **[日本語](../ja/SETUP.md)** | **[한국어](../ko/SETUP.md)**

# ATLAS 安装指南

当前出货的部署方式有两种：Docker Compose（推荐且经过测试）和裸机部署。K3s/Kubernetes 路径曾用于 V3.0 的 llama.cpp 栈，但尚未移植到 V3.0.1 的双实例 vLLM 架构。请参见底部的 K3s 章节。

---

## 前置要求（所有方式通用）

| 要求 | 详情 |
|------|------|
| **NVIDIA GPU** | 16GB+ 显存（在 RTX 5060 Ti 16GB 上测试通过） |
| **NVIDIA 驱动** | 已安装专有驱动（`nvidia-smi` 应能显示你的 GPU） |
| **Python 3.9+** | 含 pip |
| **HuggingFace CLI**（或 wget） | 用于下载模型权重 |
| **模型权重** | 来自 HuggingFace 的 QuantTrio/Qwen3.5-9B-AWQ（约 12GB AWQ-Q4 safetensors 分片目录） |

### 验证 GPU

```bash
nvidia-smi
# 应显示你的 GPU 及驱动版本和显存信息
# 如果此命令失败，请先安装 NVIDIA 专有驱动
```

---

## 方式一：Docker Compose（推荐）

这是 V3.0.1 经过测试的部署方式。

### 额外前置要求

- **Docker** 配合 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，**或 Podman**
- 约 20GB 磁盘空间（模型权重 + 容器镜像）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS

# 2. 下载模型权重（约 12 GiB AWQ-Q4 safetensors 分片目录）
#    vLLM 不原生加载 GGUF，因此直接使用 AWQ 构建。
make model
# 或直接：
#   pip install -q huggingface_hub
#   huggingface-cli download QuantTrio/Qwen3.5-9B-AWQ \
#       --local-dir models/Qwen3.5-9B-AWQ --local-dir-use-symlinks False

# 3. 安装 ATLAS CLI + Aider
pip install -e . aider-chat

# 4.（推荐）安装 Go 1.24+ 以获得任意目录的完整文件访问权限
#    https://go.dev/dl/ - 代理会在首次运行时自动构建
#    未安装 Go 时，代理在 Docker 中运行，文件访问仅限于 ATLAS_PROJECT_DIR

# 5. 配置环境变量
cp .env.example .env
# 如果模型在 ./models/ 目录下，默认配置即可 - 仅在更改了路径时才需编辑 .env

# 6. 启动所有服务（首次运行会构建容器镜像 - 需要几分钟）
docker compose up -d         # 或：podman-compose up -d

# 7. 验证所有服务是否健康（等待所有服务显示 "healthy"）
docker compose ps

# 8. 开始编码（在你的项目目录中）
cd /path/to/your/project
atlas
```

### 首次运行说明

1. Docker 从源码构建 5 个容器镜像：
   - **vLLM** - 编译 vLLM 并启用 CUDA（最慢，约 5-10 分钟）
   - **geometric-lens** - 安装 PyTorch CPU + FastAPI
   - **v3-service** - 安装 PyTorch CPU + benchmark 模块
   - **sandbox** - 安装 Node.js、Go、Rust、gcc
   - **atlas-proxy** - 编译 Go 二进制文件
2. vLLM 将 7GB 模型加载到 GPU 显存中（约 1-2 分钟）
3. 所有服务开始健康检查
4. 当全部 5 个服务报告健康后，`atlas` 连接并启动 Aider

后续执行 `docker compose up -d` 启动速度很快（几秒），因为镜像已被缓存。

### 验证安装

```bash
# 逐个检查每个服务
curl -s http://localhost:8000/health | python3 -m json.tool   # vLLM
curl -s http://localhost:31144/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy

# 快速功能测试（需要 aider：pip install aider-chat）
atlas --message "Create hello.py that prints hello world"
```

所有健康检查端点应返回 `{"status": "ok"}` 或 `{"status": "healthy"}`。

> **注意：** `atlas` 命令会自动检测代理并启动 Aider 以运行完整的代理循环（工具调用、V3 Pipeline、文件读写）。如果未安装 Aider，则回退到内置 REPL，该 REPL 支持 `/solve` 和 `/bench` 但不支持文件操作。安装 Aider 以获得完整体验：`pip install aider-chat`

### 停止服务

```bash
docker compose down          # 停止所有服务（保留镜像）
docker compose down --rmi all  # 停止并删除镜像（下次启动时重新构建）
```

### 查看日志

```bash
docker compose logs -f vLLM    # 跟踪 vLLM 日志
docker compose logs -f geometric-lens  # 跟踪 Lens 日志
docker compose logs -f v3-service      # 跟踪 V3 Pipeline 日志
docker compose logs -f atlas-proxy     # 跟踪代理日志
docker compose logs -f sandbox         # 跟踪沙箱日志
docker compose logs --tail 50          # 所有服务的最近 50 行日志
```

### 更新

```bash
git pull
docker compose down
docker compose build         # 重新构建已更改的镜像
docker compose up -d
```

---

## 方式二：裸机部署

将所有服务作为本地进程运行，无需容器。适用于开发环境或无法使用 Docker 的系统。

### 额外前置要求

| 要求 | 详情 |
|------|------|
| **Go 1.24+** | 用于构建 atlas-proxy |
| **vLLM** | 从源码编译并启用 CUDA（参见 [vLLM 构建说明](https://github.com/ggml-org/vLLM?tab=readme-ov-file#build)） |
| **Aider** | `pip install aider-chat` |
| **Node.js 20+** | 沙箱执行 JavaScript/TypeScript 所需 |
| **Rust** | 沙箱执行 Rust 所需 |

### 构建

```bash
# 1. 克隆仓库并安装 Python CLI
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS
pip install -e .

# 2. 下载模型权重
mkdir -p models
make model
# 或: huggingface-cli download QuantTrio/Qwen3.5-9B-AWQ \
#     --local-dir models/Qwen3.5-9B-AWQ --local-dir-use-symlinks False

# 3. 构建 atlas-proxy
cd atlas-proxy
go build -o ~/.local/bin/atlas-proxy-v2 .
cd ..

# 4. 安装 geometric-lens Python 依赖
pip install -r geometric-lens/requirements.txt

# 5. 安装 V3 服务 PyTorch（仅 CPU）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 6. 安装沙箱依赖
pip install fastapi uvicorn pylint pytest pydantic
```

### 启动服务

在不同的终端中分别启动每个服务（或使用 `&` 并重定向到日志文件）：

```bash
# 终端 1a：vLLM gen 实例（GPU，端口 8000）
vllm serve models/Qwen3.5-9B-AWQ \
  --served-model-name qwen3.5-9b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.55 \
  --enable-prefix-caching --reasoning-parser qwen3 --trust-remote-code

# 终端 1b：vLLM embed 实例（GPU，端口 8001）
vllm serve models/Qwen3.5-9B-AWQ \
  --served-model-name qwen3.5-9b-embed \
  --runner pooling --convert embed \
  --host 0.0.0.0 --port 8001 \
  --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.20 --trust-remote-code

# 终端 2：Geometric Lens
cd geometric-lens
LLAMA_GEN_URL=http://localhost:8000 \
LLAMA_EMBED_URL=http://localhost:8001 \
GEOMETRIC_LENS_ENABLED=true \
PROJECT_DATA_DIR=/tmp/atlas-projects \
python -m uvicorn main:app --host 0.0.0.0 --port 31144

# 终端 3：V3 Pipeline
cd v3-service
ATLAS_INFERENCE_URL=http://localhost:8000 \
ATLAS_LENS_URL=http://localhost:31144 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
python main.py

# 终端 4：Sandbox
cd sandbox
python executor_server.py

# 终端 5：atlas-proxy
ATLAS_PROXY_PORT=8090 \
ATLAS_INFERENCE_URL=http://localhost:8000 \
ATLAS_LLAMA_URL=http://localhost:8000 \
ATLAS_LENS_URL=http://localhost:31144 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
ATLAS_V3_URL=http://localhost:8070 \
ATLAS_AGENT_LOOP=1 \
LLAMA_GEN_MODEL=qwen3.5-9b \
atlas-proxy-v2
```

> **注意：** 裸机模式下沙箱监听端口为 **8020**（没有 Docker 端口映射）。代理的 `ATLAS_SANDBOX_URL` 必须使用端口 8020，而非 30820。

### 使用启动脚本

你也可以将启动脚本复制到 PATH 中：

```bash
cp /path/to/atlas-launcher ~/.local/bin/atlas
chmod +x ~/.local/bin/atlas
atlas    # 启动所有缺失的服务并运行 Aider
```

启动脚本会自动检测哪些服务已在运行，只启动缺失的服务。如果检测到 Docker Compose 栈，则直接连接。

---

## 方式三：K3s -- 当前不支持

V3.0（Qwen3-14B + llama.cpp + spec-decode）出货了完整的 K3s 部署。`scripts/install.sh` 会安装 K3s + NVIDIA GPU Operator，构建容器镜像，通过 `envsubst` 从 `atlas.conf` 生成清单文件，并 `kubectl apply` 到 `atlas` 命名空间。比较项是 llama.cpp 特有的（每 slot 上下文、Flash attention、q8_0/q4_0 KV 量化、mlock、`--embeddings` 标志）。

**对于 V3.0.1（Qwen3.5-9B-AWQ + vLLM 双实例），K3s 路径尚未重新移植。** 当前代码仓库不附带 `manifests/` 目录，也不附带 `scripts/generate-manifests.sh` 所消费的模板集。`scripts/install.sh` 会预先检测目录缺失，并明确指引到 docker-compose 后退出，因此运行它无害但也无意义。

如果今天需要 Kubernetes 部署，实际可行的路径：

- **每节点上的 Docker Compose** -- 在 kubelet 旁安装了 docker 的 K3s/k8s 节点上能干净运行（最常见的单节点配置）。
- **从 `docker-compose.yml` 手写 Deployment + Service 清单** -- 拷贝 `vllm-gen`、`vllm-embed`、`geometric-lens`、`v3-service`、`sandbox`、`atlas-proxy` 各服务，在两个 vLLM 服务上加 `nvidia.com/gpu` 资源请求。compose 中的 `command:` 块就是 vLLM CLI 参数的事实标准。
- **等待 K3s 移植** -- 已记录但未排期。欢迎 PR。

仍然有效的 K3s 工具链（`atlas.conf` 解析器、端口校验、GPU 检测助手）请参见 [CONFIGURATION.md](../../CONFIGURATION.md)。

---

## 硬件配置

| 资源 | 最低要求 | 推荐配置 | 备注 |
|------|----------|----------|------|
| GPU 显存 | 16 GB | 16 GB | 模型（约 7GB）+ KV 缓存（约 1.3GB）+ 开销 |
| 系统内存 | 14 GB | 16 GB+ | PyTorch 运行时 + 容器开销 |
| 磁盘 | 15 GB | 25 GB | 模型（7GB）+ 容器镜像（5-8GB）+ 工作空间 |
| CPU | 4 核 | 8+ 核 | V3 Pipeline 在修复阶段对 CPU 要求较高 |

### 支持的 GPU

任何具有 16GB+ 显存和 CUDA 支持的 NVIDIA GPU。已测试：
- RTX 5060 Ti 16GB（主要开发用 GPU）

AMD 和 Intel GPU 尚未测试。vLLM 支持 ROCm 和其他后端 - ROCm 支持是 V3.1 的优先事项。

---

## Geometric Lens 权重（可选）

ATLAS 在没有 Geometric Lens 权重的情况下也能正常工作 - 服务会优雅降级，返回中性分数。V3 Pipeline 回退到仅沙箱验证。

要启用 C(x)/G(x) 评分，你需要训练好的模型权重。预训练权重和训练数据可在 HuggingFace 上获取：

**[ATLAS 数据集（HuggingFace）](https://huggingface.co/datasets/itigges22/ATLAS)** - 包含嵌入向量、训练数据和权重文件。

将权重文件放在 `geometric-lens/geometric_lens/models/` 目录中（或通过 Docker Compose 中的 `ATLAS_LENS_MODELS` 进行挂载）。服务启动时会自动加载。

如果你希望使用自己的基准测试数据进行训练，`scripts/` 目录中提供了训练脚本：
- `scripts/retrain_cx_phase0.py` - 从收集的嵌入向量进行初始 C(x) 训练
- `scripts/retrain_cx.py` - 带类别权重的生产 C(x) 重训练
- `scripts/collect_lens_training_data.py` - 从基准测试运行中收集通过/失败的嵌入向量
- `scripts/prepare_lens_training.py` - 准备和验证训练数据格式

---

## 后续步骤

- [CLI.md](../../CLI.md) - ATLAS 运行后的使用指南
- [CONFIGURATION.md](../../CONFIGURATION.md) - 所有环境变量和调优选项
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md) - 常见问题与解决方案
- [ARCHITECTURE.md](../../ARCHITECTURE.md) - 系统内部工作原理
