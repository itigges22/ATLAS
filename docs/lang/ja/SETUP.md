> **[English](../../SETUP.md)** | **[简体中文](../zh-CN/SETUP.md)** | **日本語** | **[한국어](../ko/SETUP.md)**

# ATLAS セットアップガイド

3 つのデプロイ方法があります: Docker Compose (推奨・テスト済み)、ベアメタル、K3s。

---

## 前提条件 (全方法共通)

| 要件 | 詳細 |
|------|------|
| **NVIDIA GPU** | 16GB 以上の VRAM (RTX 5060 Ti 16GB でテスト済み) |
| **NVIDIA ドライバー** | プロプライエタリドライバーがインストール済みであること (`nvidia-smi` で GPU が表示されること) |
| **Python 3.9+** | pip 付き |
| **wget** | モデルウェイトのダウンロード用 |
| **モデルウェイト** | HuggingFace から Qwen3.5-9B-Q6_K.gguf (~7GB) |

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

# 2. モデルウェイトのダウンロード (~7GB)
mkdir -p models
wget https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q6_K.gguf \
     -O models/Qwen3.5-9B-Q6_K.gguf

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
   - **llama-server** -- llama.cpp を CUDA でコンパイル (最も遅い、約 5-10 分)
   - **geometric-lens** -- PyTorch CPU + FastAPI をインストール
   - **v3-service** -- PyTorch CPU + ベンチマークモジュールをインストール
   - **sandbox** -- Node.js、Go、Rust、gcc をインストール
   - **atlas-proxy** -- Go バイナリをコンパイル
2. llama-server が 7GB のモデルを GPU VRAM にロード (約 1-2 分)
3. 全サービスがヘルスチェックを開始
4. 5 つのサービスすべてが正常と報告されると、`atlas` が接続して Aider を起動

2 回目以降の `docker compose up -d` はイメージがキャッシュされているため高速 (数秒) で起動します。

### インストールの確認

```bash
# 各サービスを個別に確認
curl -s http://localhost:8080/health | python3 -m json.tool   # llama-server
curl -s http://localhost:8099/health | python3 -m json.tool   # geometric-lens
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
docker compose logs -f llama-server    # llama-server のログをフォロー
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
| **llama.cpp** | CUDA 付きでソースからビルド ([llama.cpp ビルド手順](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#build) を参照) |
| **Aider** | `pip install aider-chat` |
| **Node.js 20+** | サンドボックスの JavaScript/TypeScript 実行に必要 |
| **Rust** | サンドボックスの Rust 実行に必要 |

### ビルド

```bash
# 1. クローンと Python CLI のインストール
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS
pip install -e .

# 2. モデルウェイトのダウンロード
mkdir -p models
wget https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q6_K.gguf \
     -O models/Qwen3.5-9B-Q6_K.gguf

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
# ターミナル 1: llama-server (GPU)
llama-server \
  --model models/Qwen3.5-9B-Q6_K.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 --n-gpu-layers 99 --no-mmap

# ターミナル 2: Geometric Lens
cd geometric-lens
LLAMA_URL=http://localhost:8080 \
LLAMA_EMBED_URL=http://localhost:8080 \
GEOMETRIC_LENS_ENABLED=true \
PROJECT_DATA_DIR=/tmp/atlas-projects \
python -m uvicorn main:app --host 0.0.0.0 --port 8099

# ターミナル 3: V3 パイプライン
cd v3-service
ATLAS_INFERENCE_URL=http://localhost:8080 \
ATLAS_LENS_URL=http://localhost:8099 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
python main.py

# ターミナル 4: Sandbox
cd sandbox
python executor_server.py

# ターミナル 5: atlas-proxy
ATLAS_PROXY_PORT=8090 \
ATLAS_INFERENCE_URL=http://localhost:8080 \
ATLAS_LLAMA_URL=http://localhost:8080 \
ATLAS_LENS_URL=http://localhost:8099 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
ATLAS_V3_URL=http://localhost:8070 \
ATLAS_AGENT_LOOP=1 \
ATLAS_MODEL_NAME=Qwen3.5-9B-Q6_K \
atlas-proxy-v2
```

> **注意:** サンドボックスはベアメタルモードではポート **8020** でリッスンします (Docker のポートリマッピングなし)。プロキシの `ATLAS_SANDBOX_URL` には 30820 ではなくポート 8020 を使用してください。

### ランチャースクリプトでの起動

代替として、ランチャースクリプトを PATH にコピーすることもできます:

```bash
cp /path/to/atlas-launcher ~/.local/bin/atlas
chmod +x ~/.local/bin/atlas
atlas    # 未起動のサービスをすべて起動し、Aider を立ち上げます
```

ランチャーはどのサービスが既に実行中かを自動検出し、不足しているものだけを起動します。Docker Compose スタックを検出した場合は、そちらに接続します。

---

## 方法 3: K3s

GPU スケジューリング、ヘルスプローブ、リソース制限を備えた本番 Kubernetes デプロイ用です。

### 追加の前提条件

| 要件 | 詳細 |
|------|------|
| **K3s** | シングルノードまたはマルチノードクラスター |
| **NVIDIA GPU Operator** または **device plugin** | GPU が `nvidia.com/gpu` リソースとして認識される必要があります |
| **Helm** | GPU Operator のインストール用 |
| **Podman または Docker** | コンテナイメージのビルド用 |

### 自動インストール

インストールスクリプトが完全なセットアップを処理します -- K3s のインストール、GPU Operator、コンテナビルド、デプロイ:

```bash
# 1. 設定
cp atlas.conf.example atlas.conf
# atlas.conf を編集: モデルパス、GPU レイヤー数、コンテキストサイズ、NodePorts

# 2. インストーラーの実行 (root 権限が必要)
sudo scripts/install.sh
```

インストーラーは以下を実行します:
1. 前提条件の確認 (NVIDIA ドライバー、GPU VRAM、システム RAM)
2. K3s が未実行の場合はインストール
3. GPU がクラスターに認識されていない場合、Helm 経由で NVIDIA GPU Operator をインストール
4. コンテナイメージをビルドし、K3s containerd にインポート
5. `atlas.conf` から envsubst 経由でマニフェストを生成
6. `atlas` 名前空間にデプロイ
7. すべてのサービスが正常になるまで待機

### 手動デプロイ

K3s が既に GPU サポート付きで実行されている場合:

```bash
# 1. 設定
cp atlas.conf.example atlas.conf
# atlas.conf を編集

# 2. イメージのビルドとインポート
scripts/build-containers.sh

# 3. atlas.conf からマニフェストを生成
scripts/generate-manifests.sh

# 4. デプロイ
kubectl apply -n atlas -f manifests/

# 5. 確認
scripts/verify-install.sh
```

### K3s 固有の設定

K3s は設定に `.env` ではなく `atlas.conf` を使用します。Docker Compose との主な違い:

| 設定項目 | Docker Compose | K3s |
|----------|---------------|-----|
| 設定ファイル | `.env` | `atlas.conf` |
| コンテキストサイズ | 32K | スロットあたり 40K (x 4 スロット = 合計 160K) |
| 並列スロット | 1 (暗黙的) | 4 |
| Flash attention | オフ | オン |
| KV キャッシュ量子化 | なし | q8_0 (キー) + q4_0 (バリュー) |
| メモリロック | なし | mlock 有効 |
| エンベディングエンドポイント | 非公開 | `--embeddings` フラグ |
| サービス公開 | ホストポート | NodePorts |

全 `atlas.conf` リファレンスは [CONFIGURATION.md](../../CONFIGURATION.md) をご覧ください。

### K3s デプロイの確認

```bash
# Pod の確認
kubectl get pods -n atlas

# GPU 割り当ての確認
kubectl describe nodes | grep nvidia.com/gpu

# 検証スイートの実行
scripts/verify-install.sh
```

> **注意:** Docker Compose は V3.0.1 の検証済みデプロイ方法です。K3s マニフェストはデプロイ時にテンプレートから生成されます。K3s デプロイは Qwen3-14B での V3.0 ベンチマークに使用され本番テスト済みですが、テンプレートファイルはお使いのクラスター構成に合わせて調整が必要な場合があります。

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

AMD および Intel GPU は未テストです。llama.cpp は ROCm やその他のバックエンドをサポートしています -- ROCm サポートは V3.1 の優先事項です。

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
