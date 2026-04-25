> **[English](../../SETUP.md)** | **[简体中文](../zh-CN/SETUP.md)** | **日本語** | **[한국어](../ko/SETUP.md)**

# ATLAS セットアップガイド

現在出荷されているデプロイ方法は 2 つです: Docker Compose (推奨・テスト済み) とベアメタル。K3s/Kubernetes パスは V3.0 の llama.cpp スタック向けに存在しましたが、V3.0.1 のデュアルインスタンス vLLM アーキテクチャにはまだ移植されていません。下部の K3s セクションをご覧ください。

---

## 前提条件 (全方法共通)

| 要件 | 詳細 |
|------|------|
| **NVIDIA GPU** | 16GB 以上の VRAM (RTX 5060 Ti 16GB でテスト済み) |
| **NVIDIA ドライバー** | プロプライエタリドライバーがインストール済みであること (`nvidia-smi` で GPU が表示されること) |
| **Python 3.9+** | pip 付き |
| **HuggingFace CLI** (または wget) | モデルウェイトのダウンロード用 |
| **モデルウェイト** | HuggingFace から QuantTrio/Qwen3.5-9B-AWQ (~12GB の AWQ-Q4 safetensors シャードのディレクトリ) |

### GPU の確認

```bash
nvidia-smi
# GPU がドライバーバージョンと VRAM と共に表示されるはずです
# 失敗する場合は、先に NVIDIA プロプライエタリドライバーをインストールしてください
```

---

## 方法 1: Docker Compose (推奨)

V3.0.1 でテスト済みのデプロイ方法です。

### 追加の前提条件

- **Docker** ([nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 付き)、**または Podman**
- 約 20GB のディスク容量 (モデルウェイト + コンテナイメージ)

### セットアップ

```bash
# 1. クローン
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS

# 2. モデルウェイトのダウンロード (~12 GiB の AWQ-Q4 safetensors シャードのディレクトリ)。
#    vLLM は GGUF をネイティブにロードしないため、AWQ ビルドを直接使用します。
make model
# または直接：
#   pip install -q huggingface_hub
#   huggingface-cli download QuantTrio/Qwen3.5-9B-AWQ \
#       --local-dir models/Qwen3.5-9B-AWQ --local-dir-use-symlinks False

# 3. ATLAS CLI + Aider のインストール
pip install -e . aider-chat

# 4. (推奨) 任意のディレクトリからの完全なファイルアクセスのために Go 1.24+ をインストール
#    https://go.dev/dl/ -- プロキシは初回実行時に自動的にビルドされます
#    Go なしの場合、プロキシは Docker 内で実行され、ファイルアクセスは ATLAS_PROJECT_DIR に制限されます

# 5. 環境設定
cp .env.example .env
# モデルが ./models/ にある場合はデフォルト設定のままで動作します -- パスを変更した場合のみ .env を編集してください

# 6. 全サービスの起動 (初回実行時はコンテナイメージのビルドのため数分かかります)
docker compose up -d         # または: podman-compose up -d

# 7. 全サービスが正常であることを確認 (全サービスが "healthy" と表示されるまで待機)
docker compose ps

# 8. コーディング開始 (プロジェクトディレクトリから)
cd /path/to/your/project
atlas
```

### 初回実行時の動作

1. Docker が 5 つのコンテナイメージをソースからビルドします:
   - **vLLM** -- vLLM を CUDA でコンパイル (最も遅い、約 5-10 分)
   - **geometric-lens** -- PyTorch CPU + FastAPI をインストール
   - **v3-service** -- PyTorch CPU + ベンチマークモジュールをインストール
   - **sandbox** -- Node.js、Go、Rust、gcc をインストール
   - **atlas-proxy** -- Go バイナリをコンパイル
2. vLLM が 7GB のモデルを GPU VRAM にロード (約 1-2 分)
3. 全サービスがヘルスチェックを開始
4. 5 つのサービスすべてが正常と報告されると、`atlas` が接続して Aider を起動

2 回目以降の `docker compose up -d` はイメージがキャッシュされているため高速 (数秒) で起動します。

### インストールの確認

```bash
# 各サービスを個別に確認
curl -s http://localhost:8000/health | python3 -m json.tool   # vLLM
curl -s http://localhost:31144/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy

# 簡単な機能テスト (aider が必要: pip install aider-chat)
atlas --message "Create hello.py that prints hello world"
```

すべてのヘルスエンドポイントが `{"status": "ok"}` または `{"status": "healthy"}` を返すはずです。

> **注意:** `atlas` コマンドはプロキシを自動検出し、完全なエージェントループ (ツールコール、V3 パイプライン、ファイル読み書き) のために Aider を起動します。Aider がインストールされていない場合は、`/solve` と `/bench` をサポートするがファイル操作はできない組み込み REPL にフォールバックします。完全な体験のために Aider をインストールしてください: `pip install aider-chat`

### 停止

```bash
docker compose down          # 全サービスを停止 (イメージは保持)
docker compose down --rmi all  # 停止してイメージも削除 (次回起動時に再ビルド)
```

### ログの確認

```bash
docker compose logs -f vLLM    # vLLM のログをフォロー
docker compose logs -f geometric-lens  # Lens のログをフォロー
docker compose logs -f v3-service      # V3 パイプラインのログをフォロー
docker compose logs -f atlas-proxy     # プロキシのログをフォロー
docker compose logs -f sandbox         # サンドボックスのログをフォロー
docker compose logs --tail 50          # 全サービスの直近 50 行
```

### アップデート

```bash
git pull
docker compose down
docker compose build         # 変更されたイメージを再ビルド
docker compose up -d
```

---

## 方法 2: ベアメタル

コンテナを使用せず、すべてのサービスをローカルプロセスとして実行します。開発用途や Docker が利用できないシステムに適しています。

### 追加の前提条件

| 要件 | 詳細 |
|------|------|
| **Go 1.24+** | atlas-proxy のビルド用 |
| **vLLM** | CUDA 付きでソースからビルド ([vLLM ビルド手順](https://github.com/ggml-org/vLLM?tab=readme-ov-file#build) を参照) |
| **Aider** | `pip install aider-chat` |
| **Node.js 20+** | サンドボックスの JavaScript/TypeScript 実行に必要 |
| **Rust** | サンドボックスの Rust 実行に必要 |

### ビルド

```bash
# 1. クローンと Python CLI のインストール
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS
pip install -e .

# 2. モデルウェイトのダウンロード (~12 GiB)
make model
# または: huggingface-cli download QuantTrio/Qwen3.5-9B-AWQ \
#         --local-dir models/Qwen3.5-9B-AWQ --local-dir-use-symlinks False

# 3. atlas-proxy のビルド
cd atlas-proxy
go build -o ~/.local/bin/atlas-proxy-v2 .
cd ..

# 4. geometric-lens の Python 依存関係をインストール
pip install -r geometric-lens/requirements.txt

# 5. V3 サービスの PyTorch (CPU のみ) をインストール
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 6. サンドボックスの依存関係をインストール
pip install fastapi uvicorn pylint pytest pydantic
```

### サービスの起動

各サービスを別々のターミナルで起動します (または `&` を使ってログファイルにリダイレクトします):

```bash
# ターミナル 1a: vLLM gen インスタンス (GPU、ポート 8000)
vllm serve models/Qwen3.5-9B-AWQ \
  --served-model-name qwen3.5-9b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.55 \
  --enable-prefix-caching --reasoning-parser qwen3 --trust-remote-code

# ターミナル 1b: vLLM embed インスタンス (GPU、ポート 8001)
vllm serve models/Qwen3.5-9B-AWQ \
  --served-model-name qwen3.5-9b-embed \
  --runner pooling --convert embed \
  --host 0.0.0.0 --port 8001 \
  --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.20 --trust-remote-code

# ターミナル 2: Geometric Lens
cd geometric-lens
LLAMA_GEN_URL=http://localhost:8000 \
LLAMA_EMBED_URL=http://localhost:8001 \
GEOMETRIC_LENS_ENABLED=true \
PROJECT_DATA_DIR=/tmp/atlas-projects \
python -m uvicorn main:app --host 0.0.0.0 --port 31144

# ターミナル 3: V3 パイプライン
cd v3-service
ATLAS_INFERENCE_URL=http://localhost:8000 \
ATLAS_LENS_URL=http://localhost:31144 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
python main.py

# ターミナル 4: Sandbox
cd sandbox
python executor_server.py

# ターミナル 5: atlas-proxy
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

> **注意:** サンドボックスはベアメタルモードではポート **8020** でリッスンします (Docker のポートリマッピングなし)。プロキシの `ATLAS_SANDBOX_URL` には 30820 ではなくポート 8020 を使用してください。

### `atlas` コマンドの使用

`atlas` コマンドはビルドのステップ 1 (`pip install -e .`) で自動的にインストールされます -- `pyproject.toml` で宣言された `atlas.cli.repl:run` の Python エントリポイントです。任意のプロジェクトディレクトリから:

```bash
atlas    # 起動中のサービスに接続し、Aider を立ち上げます
```

上記のベアメタルサービスが既に実行中であることを前提としています。標準ポートで Docker Compose スタックを検出した場合は、そちらに接続します。

---

## 方法 3: K3s -- 現在サポート対象外

V3.0 (Qwen3-14B + llama.cpp + spec-decode) は完全な K3s デプロイを出荷していました。`scripts/install.sh` が K3s + NVIDIA GPU Operator をインストールし、コンテナイメージをビルドし、`atlas.conf` から `envsubst` 経由でマニフェストを生成し、`atlas` 名前空間に `kubectl apply` していました。比較項目は llama.cpp 固有のもの (スロットあたりコンテキスト、Flash attention、q8_0/q4_0 KV 量子化、mlock、`--embeddings` フラグ) でした。

**V3.0.1 (Qwen3.5-9B-AWQ + vLLM デュアルインスタンス) では、K3s パスはまだ移植されていません。** リポジトリには現在 `manifests/` ディレクトリも `scripts/generate-manifests.sh` が消費するテンプレートセットも含まれていません。`scripts/install.sh` はディレクトリの不在を最初に検出し、docker-compose への明確な誘導とともに終了するため、実行しても害はありませんが意味もありません。

今日 Kubernetes デプロイが必要な場合の現実的な選択肢:

- **各ノードで Docker Compose** -- kubelet と並んで docker がインストールされている K3s/k8s ノードでクリーンに動作します (最も一般的なシングルノード構成)。
- **`docker-compose.yml` から Deployment + Service マニフェストを手書き** -- `vllm-gen`、`vllm-embed`、`geometric-lens`、`v3-service`、`sandbox`、`atlas-proxy` の各サービスをコピーし、2 つの vLLM サービスに `nvidia.com/gpu` リソースリクエストを付けます。compose の `command:` ブロックが vLLM CLI 引数の正解です。
- **K3s 移植を待つ** -- 追跡されていますが未スケジュールです。PR を歓迎します。

引き続き動作する K3s ツール (`atlas.conf` パーサー、ポート検証、GPU 検出ヘルパー) については [CONFIGURATION.md](../../CONFIGURATION.md) をご覧ください。

---

## ハードウェアサイジング

| リソース | 最小 | 推奨 | 備考 |
|----------|------|------|------|
| GPU VRAM | 16 GB | 16 GB | モデル (~7GB) + KV キャッシュ (~1.3GB) + オーバーヘッド |
| システム RAM | 14 GB | 16 GB+ | PyTorch ランタイム + コンテナオーバーヘッド |
| ディスク | 15 GB | 25 GB | モデル (7GB) + コンテナイメージ (5-8GB) + 作業スペース |
| CPU | 4 コア | 8 コア以上 | V3 パイプラインは修復フェーズで CPU 負荷が高い |

### 対応 GPU

16GB 以上の VRAM と CUDA サポートを持つ任意の NVIDIA GPU。テスト済み:
- RTX 5060 Ti 16GB (主要開発 GPU)

AMD および Intel GPU は未テストです。vLLM は ROCm やその他のバックエンドをサポートしています -- ROCm サポートは V3.1 の優先事項です。

---

## Geometric Lens ウェイト (オプション)

ATLAS は Geometric Lens ウェイトなしでも動作します -- サービスはグレースフルにデグレードし、ニュートラルスコアを返します。V3 パイプラインはサンドボックスのみの検証にフォールバックします。

C(x)/G(x) スコアリングを有効にするには、トレーニング済みのモデルウェイトが必要です。事前トレーニング済みウェイトとトレーニングデータは HuggingFace で入手できます:

**[ATLAS Dataset on HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS)** -- エンベディング、トレーニングデータ、ウェイトファイルが含まれています。

ウェイトファイルを `geometric-lens/geometric_lens/models/` に配置してください (または Docker Compose で `ATLAS_LENS_MODELS` 経由でマウント)。サービスは起動時に自動的にロードします。

独自のベンチマークデータでトレーニングしたい場合は、`scripts/` にトレーニングスクリプトが用意されています:
- `scripts/retrain_cx_phase0.py` -- 収集したエンベディングからの初期 C(x) トレーニング
- `scripts/retrain_cx.py` -- クラスウェイト付き本番 C(x) リトレーニング
- `scripts/collect_lens_training_data.py` -- ベンチマーク実行から合格/不合格エンベディングを収集
- `scripts/prepare_lens_training.py` -- トレーニングデータフォーマットの準備と検証

---

## 次のステップ

- [CLI.md](../../CLI.md) -- ATLAS 起動後の使い方
- [CONFIGURATION.md](../../CONFIGURATION.md) -- すべての環境変数とチューニングオプション
- [TROUBLESHOOTING.md](../ja/TROUBLESHOOTING.md) -- よくある問題と解決方法
- [ARCHITECTURE.md](../../ARCHITECTURE.md) -- システム内部の仕組み
