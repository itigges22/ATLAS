> **[English](../../TROUBLESHOOTING.md)** | **[简体中文](../zh-CN/TROUBLESHOOTING.md)** | **日本語** | **[한국어](../ko/TROUBLESHOOTING.md)**

# ATLAS トラブルシューティングガイド

ATLAS V3.0.1 のよくある問題と解決方法を、サービスごとにまとめています。

---

## クイック診断

まず以下を実行して、問題の箇所を特定してください:

```bash
# Docker Compose -- 全サービスを一括確認
docker compose ps

# 個別のヘルスチェック
curl -s http://localhost:8000/health | python3 -m json.tool   # vLLM
curl -s http://localhost:31144/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy (全サービスのステータスを表示)

# GPU ステータス
nvidia-smi

# Docker Compose ログ (サービスごとに直近 50 行)
docker compose logs --tail 50
```

atlas-proxy のヘルスエンドポイントは、すべての上流サービスのステータスを報告します:
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

いずれかのフィールドが `false` の場合、そのサービスに問題があります。

---

## Docker / Podman の問題

### コンテナで GPU が検出されない

**症状:** vLLM コンテナが起動するが、モデルが CPU で読み込まれる (非常に遅い、約 2 tok/s)。ホストからは `nvidia-smi` で GPU が見えるが、コンテナからは見えない。

**修正:** NVIDIA Container Toolkit をインストールしてください:

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

コンテナ内から GPU が見えることを確認します:
```bash
# Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Podman
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0-base nvidia-smi
```

### 初回ビルドの失敗 (CUDA が見つからない)

**症状:** `docker compose build` が vLLM のコンパイル中に CUDA 関連のエラーで失敗する。

**修正:** vLLM の Dockerfile は `nvidia/cuda:12.8.0-devel` ベースイメージ内で vLLM をビルドするため、ビルド中はホスト GPU アクセスなしで CUDA ヘッダーが利用可能です。ビルド失敗の一般的な原因:
1. ディスク容量不足 (ビルド成果物に約 5GB 必要)
2. CUDA ベースイメージのダウンロードまたは vLLM のクローン時のネットワーク問題
3. Podman のルートレスビルドはパーミッションの問題で失敗する場合がある -- `podman-compose build` で `--podman-build-args="--format docker"` を試してください

### SELinux がコンテナアクセスをブロック (Fedora/RHEL)

**症状:** コンテナがマウントされたボリュームを読めない、モデルファイルへのパーミッション拒否。

**修正:**
```bash
# モデルディレクトリへのコンテナアクセスを許可
chcon -Rt svirt_sandbox_file_t ~/models/

# または、ボリュームマウントに :Z フラグを追加 (Docker Compose は自動的に処理します)
```

### サンドボックスに到達できない

**症状:** プロキシのヘルスが `"sandbox": false` を表示する。V3 のビルド検証が失敗する。

**修正:** すべてのサービスが同じ Docker ネットワーク上にあることを確認してください。Docker Compose は `atlas` ネットワークを自動的に作成します。コンテナを手動で実行している場合:
```bash
docker network create atlas
# すべてのコンテナを --network atlas で起動
```

### ポート競合

**症状:** `docker compose up` がポートの "address already in use" で失敗する。

**修正:** ポートを使用しているプロセスを確認し、停止するか `.env` で ATLAS のポートを変更してください:
```bash
# vLLM の gen と embed のポートを使用しているプロセスを確認
lsof -i :8000 -i :8001

# .env で各ポートを変更
ATLAS_GEN_PORT=8000    # vLLM gen インスタンス (デフォルト 8000)
ATLAS_EMBED_PORT=8001  # vLLM embed インスタンス (デフォルト 8001)
```

すべてのポートは `.env` で設定可能です。[CONFIGURATION.md](../../CONFIGURATION.md) をご覧ください。

---

## vLLM の問題

### モデルが GPU ではなく CPU で読み込まれている

**症状:** 1 桁の tok/s しか出ない、`nvidia-smi` で vLLM が GPU プロセスとして表示されない。

**修正:** vLLM は GPU を自動的に拾います -- `--n-gpu-layers` というつまみは存在しません。`--gpu-memory-utilization` で実用的なスライスを確保すれば、すべてのレイヤーが GPU に乗ります。それでも遅い場合は (順番に) 確認:

1. `nvidia-smi` で `python3` (vllm) プロセスが GPU プロセスリストに表示されるか? 表示されない場合、コンテナが GPU を見えていません。`nvidia-container-toolkit` がインストールされ、ランタイムが `nvidia` になっているか確認してください。
2. `docker logs vllm-gen | grep -i 'cpu\|device'` で起動時のデバイス選択を確認。「Falling back to CPU」が出ていれば GPU は見えていたが拒否された (ドライバーが古い、compute capability が低い、またはこのアーキテクチャで量子化が非サポート)。
3. 並列度: vLLM の PagedAttention は同時リクエストでスケールします。`ATLAS_LLM_PARALLEL=0` または `ATLAS_GEN_MAX_NUM_SEQS=1` だと事実上シングルスロット -- 両方を上げてください。
4. KV キャッシュページング: `nvidia-smi` の VRAM が 100% でもスループットが低い → KV キャッシュがシステム RAM にページング中。`ATLAS_GEN_CTX_SIZE` (デフォルト 32768) または `ATLAS_GEN_MAX_NUM_SEQS` を下げて VRAM の余裕を確保。

### モデルファイルが見つからない

**症状:** vLLM が "failed to load model" などのメッセージで即座に終了する。

**修正:** モデルディレクトリを確認してください (vLLM は単一の GGUF ファイルではなく、AWQ-Q4 safetensors シャードのディレクトリを直接使用します):
```bash
# Docker Compose -- モデルは ATLAS_MODELS_DIR (デフォルト: ./models/) にある必要があります
ls -la models/Qwen3.5-9B-AWQ/

# ベアメタル -- ATLAS_MODEL_PATH を確認
ls -la ~/models/Qwen3.5-9B-AWQ/
```

パスは `.env` の `ATLAS_MODEL_PATH` (デフォルト: `/models/Qwen3.5-9B-AWQ`) と一致する必要があります。`make model` を実行して HuggingFace から取得できます。

### VRAM 不足

**症状:** vLLM が起動直後にクラッシュまたは OOMKilled される。`nvidia-smi` で VRAM がほぼ 100% を表示。

**修正:** 9B AWQ-Q4 ビルドは gen で約 12 GB、embed サイドカーで約 3 GB の VRAM が必要です (両方とも同じ GPU を共有)。以下を確認してください:
1. 他の GPU プロセスが実行されていない (`nvidia-smi` -- 他の CUDA プロセスを確認)
2. 16GB 以上の VRAM がある
3. コンテキストサイズが大きすぎない (デフォルトの 32K で問題なし、VRAM を確認せずに増やさないでください)

```bash
# 必要に応じて他の GPU プロセスを終了
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill {}
```

### 文法が強制されない (モデルが思考ブロックを出力する)

**症状:** モデルが JSON ツールコールの代わりに `<think>` タグや生テキストを出力する。

**修正:** プロキシは `ATLAS_AGENT_LOOP=1` のとき自動的に `response_format: {"type": "json_object"}` を設定します。vLLM を直接使用する場合は、リクエストに含めてください:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role":"user","content":"Say hi"}],
    "max_tokens": 50,
    "chat_template_kwargs": {"enable_thinking": false},
    "response_format": {"type": "json_object"}
  }'
```

JSON ではなく生テキストが返される場合、お使いの vLLM ビルドが `response_format` をサポートしていません。最新のソースからリビルドしてください。

### コンテキストウィンドウが小さすぎる

**症状:** ツールコールの引数が切り詰められる。`write_file` が "unexpected end of JSON" で失敗するか、プロキシログに "truncation detected" と表示される。

**修正:** コンテキストサイズは 32768 であるべきです (Docker Compose のデフォルト)。確認方法:
```bash
# Docker Compose
grep CTX_SIZE .env

# ベアメタル
ps aux | grep vLLM | grep ctx-size
```

---

## プロキシの問題

### エージェントループが起動しない

**症状:** リクエストが vLLM に直接送られる。ツールコールなし、ストリーミングステータスアイコンなし、V3 パイプラインなし。

**修正:** `ATLAS_AGENT_LOOP=1` を設定してください。`atlas` ランチャーはこれを自動的に行います。プロキシを手動で実行する場合:
```bash
ATLAS_AGENT_LOOP=1 atlas-proxy-v2
```

Docker Compose では `docker-compose.yml` で設定済みのため、手動設定は不要です。

### 機能ファイルで V3 パイプラインが起動しない

**症状:** すべての `write_file` コールが T1 (直接書き込み) になる。出力に V3 パイプラインのステージがない。

V3 は以下の **3 つの条件すべて** が満たされた場合にのみ起動します:
1. ファイルのコンテンツが **50 行以上**
2. ファイルに **3 つ以上のロジックインジケーター** がある (関数定義、制御フロー、API パターン)
3. V3 サービスが `ATLAS_V3_URL` で到達可能

**診断:**
```bash
# V3 サービスのヘルスを確認
curl -s http://localhost:8070/health

# プロキシログでティア分類を確認
docker compose logs atlas-proxy | grep "write_file"
# 確認項目: T1 (直接) vs T2 (V3 パイプライン)
```

V3 に到達できない場合、プロキシはサイレントに直接書き込みにフォールバックします。

### 切り詰めエラー (write_file が繰り返し失敗)

**症状:** "Your output was truncated -- the content is too long for a single tool call." のようなエラーが繰り返される。

**原因:** モデルが 1 回のコールで多すぎるコンテンツを書き込もうとしています。プロキシが切り詰められた JSON を検出し、ツールコールを拒否します。

**自動的に行われる処理:**
- 100 行を超える既存ファイルの場合: プロキシが `write_file` を拒否し、代わりに `edit_file` を使用するようモデルに指示
- 3 回連続で失敗した場合: エラーループブレーカーがエージェントを停止し、サマリーを返す

**ユーザーができること:** ファイル全体の書き換えではなく、ターゲットを絞った変更を依頼するようリクエストを言い換えてください。例えば、"auth.py を書き直して" の代わりに "ログイン関数に入力バリデーションを追加して" と依頼してください。

### 編集前にファイルが読み込まれていない

**症状:** `edit_file` が "file not read yet -- use read_file first before editing." で失敗する。

**原因:** プロキシはエージェントがどのファイルを読んだかを追跡しています。モデルがこのセッションで読んでいないファイルを編集しようとすると、古さ防止のため編集が拒否されます。

**修正:** これは正常な動作です -- モデルはまずファイルを読む必要があります。失敗が続く場合、モデルがどのファイルを見たか混乱している可能性があります。Aider で `/clear` を実行し、リクエストを言い換えてください。

### 外部でファイルが変更された

**症状:** `edit_file` が "file modified since last read -- read it again before editing." で失敗する。

**原因:** モデルがファイルを読んだ後に、ディスク上のファイルが (ユーザーまたは別のプロセスによって) 変更されました。プロキシは変更タイムスタンプを比較します。

**修正:** モデルがファイルを再読み込みする必要があります。通常、次のターンで自動的に解決されます。

### 探索バジェット警告

**症状:** 出力に "You have full project context in the system prompt. Do not read more files." と表示されるか、読み取りがスキップされる。

**原因:** モデルが何も書き込まずに 4 回以上連続で読み取り専用コール (read_file、search_files、list_directory) を行いました。4 回の読み取り後にプロキシが警告します。5 回以上になると、読み取りを完全にスキップし、モデルに書き込みを指示します。

**修正:** これは保護的な動作です。モデルが本当に探索で行き詰まっている場合は、変更したい内容をより具体的に指示してください。

---

## Geometric Lens の問題

### Lens が読み込まれない / 利用不可

**症状:** プロキシのヘルスが `"lens": false` を表示する。または起動時に "Lens unavailable -- verification disabled." と表示される。

**影響:** ATLAS は C(x)/G(x) スコアリングなしでも動作します。V3 の候補選択はサンドボックスのみの検証にフォールバックします。

**修正:** Lens のヘルスとログを確認してください:
```bash
curl -s http://localhost:31144/health
docker compose logs geometric-lens
```

一般的な原因:
- Lens が vLLM に接続できない (`LLAMA_URL` 環境変数を確認)
- モデルウェイトファイルが見つからない (サービスはグレースフルにデグレードします -- カスタムモデルをトレーニングしていない場合はこれが想定される動作です)

### すべてのスコアが 0.5 付近

**症状:** コード品質に関係なく、すべての候補が `cx_energy: 0.0` および `gx_score: 0.5` になる。

**原因:** モデルウェイトが読み込まれていません。モデルが存在しない場合、サービスはニュートラルなデフォルト値を返します。

**確認:**
```bash
curl -s http://localhost:31144/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(1)"}' | python3 -m json.tool
```

`enabled: false` または `cx_energy: 0.0` の場合、モデルが読み込まれていません。新規インストールではこれが想定される動作です -- モデルウェイトはリポジトリに含まれておらず、トレーニングするか [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS) からダウンロードする必要があります。

### エンベディング抽出の失敗

**症状:** Lens のログに "embedding extraction failed" やタイムアウトなどのエラーが表示される。

**原因:** Lens は vLLM の `/v1/embeddings` エンドポイントを呼び出します。vLLM が過負荷状態であるか、エンドポイントが有効になっていない場合、これが失敗します。

**修正:**
```bash
# エンベディングエンドポイントを直接テスト
curl -s http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}' | python3 -m json.tool
```

`/v1/embeddings` エンドポイントは、生成モデルからのセルフエンベディング用に特別なフラグなしで vLLM で利用可能です。K3s では、完全なエンベディングサポートのためにエントリーポイントで `--embeddings` フラグが明示的に設定されています。

---

## サンドボックスの問題

### サンドボックスに到達できない

**症状:** コードがテストされない。プロキシのヘルスが `"sandbox": false` を表示する。

**修正:** サンドボックスのヘルスを確認してください:
```bash
# Docker Compose (ホストポート 30820 がコンテナポート 8020 にマッピング)
curl -s http://localhost:30820/health

# ベアメタル (直接ポート 8020)
curl -s http://localhost:8020/health
```

サンドボックスコンテナが実行中だが正常でない場合、ログを確認してください:
```bash
docker compose logs sandbox
```

### コード実行のタイムアウト

**症状:** サンドボックスが `"error_type": "Timeout"` を返す。コードの実行に時間がかかりすぎている。

**デフォルトタイムアウト:** リクエストあたり 30 秒、最大 60 秒 (`MAX_EXECUTION_TIME` 環境変数で設定可能)。

**修正:** コードが正当により多くの時間を必要とする場合は、リクエストでより長いタイムアウトを設定してください。コードに無限ループがある場合、これは想定された動作です。

### 言語がサポートされていない

**症状:** 特定の言語でサンドボックスがエラーを返す。

**サポート対象言語:** Python、JavaScript、TypeScript、Go、Rust、C、C++、Bash。

利用可能なランタイムを確認:
```bash
curl -s http://localhost:30820/languages | python3 -m json.tool
```

---

## Aider の問題

### `atlas` が Aider ではなく REPL を表示する (ファイル読み書きなし)

**症状:** `atlas` を実行すると `Model`、`Speed`、`Lens`、`Sandbox` のステータスブロックと `◆` プロンプトを持つ組み込み REPL が表示される。リクエストの入力は動作するが、ファイルの作成や変更は行われない。`--message` フラグは無視される。

**原因:** `atlas` コマンドはプロキシと Aider を自動検出します。どちらかがない場合、`/solve` と `/bench` をサポートするがファイル操作はできない組み込み REPL にフォールバックします。

**修正:**
1. プロキシが実行中であることを確認: `curl -s http://localhost:8090/health`
2. Aider がインストール済みであることを確認: `pip install aider-chat`
3. サービスが起動していることを確認: `docker compose ps` (すべて "healthy" と表示されるべき)

プロキシが正常で Aider がインストール済みであれば、`atlas` は完全なエージェントループ (ツールコール、ファイル読み書き、V3 パイプライン) で Aider を自動的に起動します。

Go 1.24+ がインストールされている場合、`atlas` はプロキシを自動的にビルドして起動することもできます -- 手動で起動する必要はありません。

### プロキシが間違ったディレクトリまたは `/tmp` を表示する

**症状:** モデルが自分のプロジェクトではなく `/tmp` や ATLAS リポジトリのファイルを表示する。`write_file` がファイルを間違った場所に作成する。

**原因:** Docker Compose のプロキシはコンテナ内で実行され、起動時にマウントされたディレクトリしか見えません。別のディレクトリで作業している場合、プロキシはそれを見ることができません。

**修正 (推奨):** Go 1.24+ をインストールしてください ([https://go.dev/dl/](https://go.dev/dl/))。`atlas` CLI がプロキシを現在のディレクトリで完全なファイルアクセス付きで自動的にビルドして起動します。Docker マウントは不要です。

**修正 (Go なしの場合):** `.env` で `ATLAS_PROJECT_DIR` をプロジェクトパスに設定し、プロキシを再起動してください:
```bash
# .env 内:
ATLAS_PROJECT_DIR=/path/to/your/project

# プロキシを再起動して新しいマウントを反映:
docker compose up -d atlas-proxy
```

プロジェクトディレクトリを切り替えるたびにこの設定を更新して再起動する必要があります。これはプロキシを Docker 内で実行する場合の制限です。

### クローン後に `.env.example` がない

**症状:** `cp .env.example .env` が "No such file or directory" で失敗する。

**修正:** これは V3.0.1 で修正されました。修正前にクローンした場合は、最新版をプルしてください:
```bash
git pull
cp .env.example .env
```

### 長いタスクで Aider が切断される

**症状:** 特に V3 パイプラインフェーズ中に、エージェントループの完了前に Aider がタイムアウトまたは切断される。

**修正:** Aider の HTTP リクエストタイムアウトは、V3 パイプラインの実行 (数分かかることがある) に十分な長さである必要があります。リポジトリの `.aider.model.settings.yml` はストリーミングモードを設定しており、接続を維持します。それでもタイムアウトが発生する場合:

1. リポジトリの設定ファイル (`.aider.model.settings.yml` と `.aider.model.metadata.json`) を使用していることを確認
2. 設定ファイルで `streaming: true` が設定されていることを確認

### 空のレスポンス

**症状:** Aider が完了サマリーを表示するが、ファイルコンテンツが生成されていない。

**原因:** モデルがファイル変更を行わずに `done` シグナルを発行しました。以下の場合に発生することがあります:
- 非常に短い会話的なプロンプト ("hi"、"thanks")
- モデルがどのファイルを作成すべきかわからない曖昧なリクエスト

**修正:** より具体的に指示してください。作成または編集するファイルを正確に指定してください。

### 作業ディレクトリが間違っている

**症状:** ファイルが間違った場所に作成される。`list_directory` が予期しないコンテンツを表示する。

**原因:** プロキシは最も最近変更された `.aider.chat.history.md` ファイルを見つけることでプロジェクトディレクトリを検出します。複数の Aider セッションが開いている場合、最新のセッションが優先されます。

**修正:** 他の Aider セッションを閉じるか、`atlas` を実行する前に正しいプロジェクトディレクトリに `cd` してください。

### "Model not found" エラー

**症状:** Aider がモデル関連のエラーで起動に失敗する。

**修正:** ATLAS ルートに両方の Aider 設定ファイルが存在することを確認してください:
```bash
ls -la .aider.model.settings.yml .aider.model.metadata.json
```

これらはリポジトリに含まれています。見つからない場合は、再クローンするかバックアップから復元してください。これらのファイルは Aider にプロキシを指す `openai/atlas` モデルを使用するよう指示します。

---

## パフォーマンス

### 生成が遅い (1 桁の tok/s)

vLLM は GPU を自動的に拾います -- `--n-gpu-layers` というつまみは存在せず、`--gpu-memory-utilization` で実用的なスライスを確保すれば全レイヤーが GPU に乗ります。それでも遅い場合は (順番に) 確認してください:

1. `nvidia-smi` -- `python3` (vllm) プロセスが GPU プロセスリストに表示されているか? 表示されない場合、コンテナが GPU を見えていません (`nvidia-container-toolkit` がインストール済みでランタイムが `nvidia` か確認)。
2. `docker logs vllm-gen | grep -i 'cpu\|device'` -- 起動時に vLLM が使用するデバイスをログ出力します。「Falling back to CPU」が出ていれば GPU は見えていたが拒否された (ドライバーが古い、compute capability が低い、量子化が非サポート)。
3. 並列度: vLLM の PagedAttention は同時リクエストでスケールします。`ATLAS_LLM_PARALLEL=0` または `ATLAS_GEN_MAX_NUM_SEQS=1` だと事実上シングルスロット -- 両方を上げてください。
4. KV キャッシュページング: `nvidia-smi` の VRAM が 100% でもスループットが低い → KV キャッシュがシステム RAM にページング中。`ATLAS_GEN_CTX_SIZE` (デフォルト 32768) または `ATLAS_GEN_MAX_NUM_SEQS` を下げて VRAM の余裕を確保。

**想定パフォーマンス:** vLLM AWQ-Q4 のスループットは並列度・プロンプト長・量子化アーキ (Qwen3.5 DeltaNet ハイブリッドカーネルは初回リクエスト時に Triton コンパイルされる) で大きく変動するため、信頼できる単一の数値はありません。`benchmark/measure_bok_latency.sh` でご自身のワークロードを計測してください。

### V3 パイプラインに数分かかる

これは T2 ファイルでは正常な動作です。V3 パイプラインは複数の LLM コールを行います:
- **プローブのみ (最良ケース):** 約 10-15 秒 (1 回の生成 + 1 回のスコアリング + 1 回のテスト)
- **Phase 1 生成:** 約 1-2 分 (PlanSearch + DivSampling + スコアリング)
- **Phase 3 修復:** 約 2-5 分 (PR-CoT + Refinement + Derivation、必要な場合)

より高速な (ただし品質は低い) 結果を得るには:
- ファイルを 50 行未満に保つ (T1 のまま、V3 なし)
- ロジックの複雑さを減らす (関数や制御フローを少なく)
- V3 は本当に必要な場合にのみ起動します -- シンプルなファイルは即座に書き込まれます

### RAM 使用量が高い

**症状:** システムが遅くなるか、サービスが OOMKilled される。

**想定 RAM 使用量:**
- vLLM: 約 8 GB (モデルは VRAM 内、RAM 使用は最小)
- geometric-lens: 約 200 MB (PyTorch ランタイム + モデル)
- v3-service: 約 150 MB (PyTorch ランタイム)
- sandbox: 約 100 MB (ベース、コンパイル中にスパイク)
- atlas-proxy: 約 30 MB (Go バイナリ)

**合計:** 約 500 MB RAM + 8.2 GB VRAM。システム RAM が 14 GB 未満の場合、他のサービスとメモリが競合する可能性があります。

---

## ヘルプを得る

ここに問題が記載されていない場合:
1. サービスログを確認: `docker compose logs <service-name>`
2. プロキシのヘルスエンドポイントを確認: `curl http://localhost:8090/health`
3. すべての環境変数については [CONFIGURATION.md](../../CONFIGURATION.md) をご覧ください
4. [GitHub](https://github.com/itigges22/ATLAS/issues) で Issue を作成してください
