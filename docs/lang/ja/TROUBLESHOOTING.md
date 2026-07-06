> **[English](../../TROUBLESHOOTING.md)** | **[简体中文](../zh-CN/TROUBLESHOOTING.md)** | **日本語** | **[한국어](../ko/TROUBLESHOOTING.md)**

# ATLAS トラブルシューティングガイド

よくある問題と解決方法を、サービスごとにまとめています。

---

## クイック診断

まず以下を実行して、問題の箇所を特定してください:

```bash
# Docker Compose — check all services at once
docker compose ps

# GPU status
nvidia-smi

# Docker Compose logs (last 50 lines per service)
docker compose logs --tail 50
```

サービスごとのヘルスチェック curl は [SETUP.md § インストールの確認](./SETUP.md#インストールの確認) を参照してください。トリアージには atlas-proxy のヘルスエンドポイントがもっとも有用です — すべての上流サービスのステータスを報告します:
```json
{
  "status": "ok",
  "inference": true,
  "lens": true,
  "lens_ready": true,
  "sandbox": true,
  "port": "8090",
  "stats": { "requests": 0, "repairs": 0, "sandbox_passes": 0, "sandbox_fails": 0 }
}
```

いずれかのフィールドが `false` の場合、そのサービスに問題があります。`inference`、`lens`、`lens_ready`、`sandbox` のいずれかが false になると、`status` は `"degraded"` に切り替わります。`lens` と `lens_ready` が分かれていることで、「Lens プロセスは起動しているが `/ready` ゲートが失敗している — 通常はウェイト欠落か埋め込み次元の不一致」と「Lens の HTTP に到達できない」を区別できます。

---

## Docker / Podman の問題

### コンテナで GPU が検出されない

**症状:** llama-server コンテナが起動するが、モデルが CPU で読み込まれる（非常に遅い、約 2 tok/s）。ホストからは `nvidia-smi` で GPU が見えるが、コンテナからは見えない。

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

### `libnvidia-ml.so.1: cannot open shared object file`

**症状:** `docker compose up` 中に llama-server が次のエラーで失敗する:

```
nvidia-container-cli: initialization error: load library failed:
libnvidia-ml.so.1: cannot open shared object file: no such file or directory
```

**意味:** ホストには NVIDIA の*カーネルモジュール*がある（そのため `nvidia-smi` は動く）が、*ユーザー空間のドライバライブラリ*がコンテナツールキットの期待する場所にありません。RHEL/Rocky/Alma のミニマルインストールでは `nvidia-driver-cuda-libs` パッケージがデフォルトでは入りません。Debian/Ubuntu では通常、ドライバ更新後の古い `ldconfig` キャッシュが原因です。

**修正手順** — 順番に試し、`docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` が動いたら止めてください:

1. **ldconfig の更新 + docker の再起動:**
   ```bash
   sudo ldconfig
   sudo systemctl restart docker
   ```

2. **RHEL 9 — CUDA リポジトリの追加 + open-dkms モジュールのインストール**（RTX 5060 Ti を搭載した RHEL 9.7 で動作確認済み）:
   ```bash
   # Add NVIDIA's CUDA repo
   sudo dnf config-manager --add-repo \
     https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

   # Enable CodeReady Builder (provides dkms / kernel-devel)
   sudo subscription-manager repos --enable=codeready-builder-for-rhel-9-x86_64-rpms

   # Make sure EPEL is present
   sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

   # Install the open driver module (REQUIRED for Blackwell — RTX 50xx)
   sudo dnf module install -y nvidia-driver:open-dkms

   sudo ldconfig && sudo systemctl restart docker
   ```

   **Rocky/Alma/CentOS Stream 9** — 上と同じですが、`subscription-manager` の行を次で置き換えてください:
   ```bash
   sudo dnf config-manager --set-enabled crb
   ```

   > 注: `nvidia-driver-cuda-libs` パッケージは NVIDIA の CUDA リポジトリを追加して初めて存在します。RHEL 9 の標準の `BaseOS`/`AppStream` リポジトリは NVIDIA パッケージを出荷していません。`nvidia-driver:open-dkms` モジュールは Blackwell GPU（RTX 5060/70/80/90）には**必須**です。より古い GPU は open とプロプライエタリのどちらでも受け付けます。

3. **Ubuntu/Debian — 対応するユーザー空間ライブラリのインストール:**
   ```bash
   DRV_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
   sudo apt install -y libnvidia-compute-${DRV_MAJOR}
   sudo ldconfig && sudo systemctl restart docker
   ```

4. **CDI スペックの生成:**
   ```bash
   sudo mkdir -p /etc/cdi
   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
   docker run --rm --device=nvidia.com/gpu=all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

`atlas-bootstrap.sh` スクリプトは現在、ステップ 1、2（RHEL/Rocky/Alma とサブスクリプションパスを自動判別）、4 を自動的に実行します。ステップ 3 は Debian/Ubuntu 上で、実行中のドライババージョンに合わせた `libnvidia-compute-NN` により自動処理されます。

### AMD GPU が検出されない (ROCm)

**症状:** 明らかに AMD GPU を積んでいるホストで `atlas tier` が「no GPU detected」と言う。または `docker compose up` が `/dev/kfd: no such file or directory` で失敗する。

**意味:** `amdgpu` カーネルドライバがコンピュートサポート（`kfd` — Kernel Fusion Driver — サブモジュール）付きでロードされていません。ディスプレイ専用でロードされた `amdgpu` は `/dev/kfd` を公開しません。

**修正手順:**

1. **ドライバがロードされ `/dev/kfd` が存在することを確認する:**
   ```bash
   lsmod | grep amdgpu       # should print amdgpu + amdkfd
   ls -l /dev/kfd            # should print a character-device entry
   ls -l /dev/dri/render*    # should print one or more render nodes
   ```

2. **ROCm + カーネルドライバをインストールする（/dev/kfd がない場合）:**
   - **RHEL 9 / Rocky / Alma:**
     ```bash
     sudo dnf install -y https://repo.radeon.com/amdgpu-install/6.2/rhel/9.4/amdgpu-install-6.2.60200-1.el9.noarch.rpm
     sudo amdgpu-install --usecase=dkms,rocm
     sudo reboot   # required — the kernel module needs a fresh boot
     ```
   - **Ubuntu/Debian:** お使いのディストロ向けの[公式 AMD インストールガイド](https://rocm.docs.amd.com/projects/install-on-linux/)に従ってください。典型的な手順は、AMDGPU リポジトリの追加後に `amdgpu-install --usecase=dkms,rocm` です。

3. **再起動後、`rocm-smi` が GPU を認識することを確認する:**
   ```bash
   rocm-smi --showproductname --showmeminfo vram
   ```

### AMD GPU は検出されるが Docker から到達できない

**症状:** `atlas doctor` が「AMD GPU detected but Docker can't reach `/dev/kfd`」と報告する。または ROCm コンテナが `/dev/kfd` の `Permission denied` で失敗する。

**意味:** Docker を実行しているユーザーが `render` および/または `video` グループに属していません。ROCm はこれらのグループで `/dev/kfd` と `/dev/dri/render*` へのアクセスを制限しています。

**修正:**

```bash
# 1. Confirm which groups you're currently in
id -nG | tr ' ' '\n' | grep -E '^(render|video)$'
# Expect both. If either is missing:

# 2. Create the groups if they don't exist (rare; default on most distros)
sudo groupadd -f render
sudo groupadd -f video

# 3. Add your user to both
sudo usermod -aG video,render $USER

# 4. Re-login (or use newgrp for the current shell)
newgrp render
newgrp video

# 5. Re-verify, then re-run `atlas doctor`
id -nG | grep -E 'render.*video|video.*render'
atlas doctor
```

### AMD GPU が ROCm の「非対応」だが、それでも試したい

**症状:** `rocm-smi` は GPU を報告するが `rocminfo` は報告しない。あるいは HIP カーネルが「no kernel image is available for execution on the device」で失敗する。

**意味:** llama.cpp の HIP カーネルが、お使いの GPU を含まない `gfx` ターゲット向けにコンパイルされています。ROCm には、古いコンシューマ GPU を公式サポートから外しつつ、適切なオーバーライドがあれば動作させ続けるという長年のパターンがあります。

**修正:** `ATLAS_HSA_OVERRIDE_GFX_VERSION` で互換性のある gfx バージョンを実行時に強制します。一般的なオーバーライド（カード → gfx の正規の対応表は [SETUP.md § AMD GPU ターゲット](./SETUP.md#amd-gpu-ターゲット-dockerfilerocm) を参照）:

| GPU | 設定する `ATLAS_HSA_OVERRIDE_GFX_VERSION=` |
|---|---|
| RDNA1 (RX 5700 XT / 5500 XT) | `10.3.0`（RDNA2 / gfx1030 のように見せる） |
| Vega 56/64 (gfx900) | `9.0.0`（通常はすでに対応済みで、オーバーライドはほぼ不要） |
| Polaris (RX 580/590, gfx803) | `8.0.3`（深いオーバーライド。動作は保証されない） |

compose オーバーライド経由でコンテナ環境に伝播するよう、変数は `.env` に設定してください:

```bash
echo "ATLAS_HSA_OVERRIDE_GFX_VERSION=10.3.0" >> .env
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --force-recreate llama-server
```

以前は非対応だったカードでこれが動いた場合は、ぜひ [GH #26](https://github.com/itigges22/ATLAS/issues/26) にメモを残してください — コミュニティ検証済みのオーバーライドは次のリリースのドキュメントに反映されます。

### RDNA4 (RX 9070 / 9070 XT, gfx1200 / gfx1201) — ROCm 7.x が必要

**症状:** `docker compose ... build llama-server` 中に `error: AMDGPU target 'gfx1201' is not supported` のようなエラーでビルドが失敗する。またはコンテナは起動するが HIP の初期化エラーで即座に終了する。

**意味:** デフォルトの ROCm ベースイメージ（`rocm/dev-ubuntu-22.04:6.2-complete`）は RDNA4 より前のものです。gfx1200 と gfx1201 のコンパイラターゲットは ROCm 7.0 で追加されました — 対応ハードウェアの完全なリストは [ROCm 互換性マトリクス](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)を参照してください。

**修正:** ビルド前に `ATLAS_ROCM_TAG` を ROCm 7.x のタグに設定してください:

```env
# Add to your .env
ATLAS_ROCM_TAG=7.2.3-complete
ATLAS_GFX_TARGET=gfx1201   # gfx1200 for RX 9070, gfx1201 for RX 9070 XT
```

その後、再ビルドしてスタックを立ち上げます:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml build llama-server
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d
```

**重要: gfx1200/gfx1201 では `ATLAS_HSA_OVERRIDE_GFX_VERSION` を設定しないでください。** ROCm 7.0+ はこれらのターゲットをネイティブにサポートします。Docker 内で GFX バージョンをオーバーライドすると、コンパイル済みカーネルとランタイムターゲットの不一致が生じ、クラッシュにつながります。`ATLAS_HSA_OVERRIDE_GFX_VERSION` は未設定（デフォルト）のままにしてください。

> AMD Radeon AI PRO R9700 (gfx1201)、ROCm 7.2、`ATLAS_ROCM_TAG=7.2.3-complete` でテスト済み。hidden-states パッチは固定された llama.cpp SHA にクリーンに適用されます。テキスト生成と埋め込み生成の両方で、追加フラグなしに推論が正しく動作します。

### ROCm コンテナが `rocm/rocm-terminal` を取得できない

**症状:** `atlas doctor` の ROCm チェックがイメージ取得でタイムアウトする。または `docker compose -f ... -f docker-compose.rocm.yml pull` が `llama-server` のビルドで失敗する。

**意味:** ROCm イメージは大きく（約 2 GB）、Docker Hub は匿名の取得をレート制限しています。

**修正:** 認証する（無料の Docker Hub アカウントでレート制限が緩和されます）、オフピーク時間帯に取得する、または `.env` で特定のタグに固定してください:

```bash
docker login
ATLAS_ROCM_TAG=6.2-complete docker compose -f docker-compose.yml -f docker-compose.rocm.yml pull
```

### 初回ビルドの失敗 (CUDA が見つからない)

**症状:** `docker compose build` が llama-server のコンパイル中に CUDA 関連のエラーで失敗する。

**修正:** llama-server の Dockerfile は `nvidia/cuda:12.8.0-devel` ベースイメージ内で llama.cpp をビルドするため、ビルド中はホスト GPU アクセスなしで CUDA ヘッダーが利用可能です。ビルド失敗の一般的な原因:
1. ディスク容量不足（ビルド成果物に約 5GB 必要）
2. CUDA ベースイメージのダウンロードまたは llama.cpp のクローン時のネットワーク問題
3. Podman のルートレスビルドはパーミッションの問題で失敗する場合がある — `podman-compose build` で `--podman-build-args="--format docker"` を試してください

### llama.cpp のクローンがタイムアウトする

**症状:** ビルドが `llama-server builder 3/3` ステージでハングし、最終的に次のエラーで失敗する:

```
error: RPC failed; curl 56 OpenSSL SSL_read: Connection timed out, errno 110
fatal: early EOF
fatal: fetch-pack: invalid index-pack output
```

**原因:** llama.cpp の完全な git 履歴は大きく（約 1 GB）、クローンは不安定/低速な接続に敏感です。一瞬の停滞で SSL 読み取りがタイムアウトし、転送全体が中断されます。

**修正:** `inference/Dockerfile.v31` は `git clone --depth 1 --single-branch` を `http.postBuffer=524288000` と `http.lowSpeedLimit/Time` 付きで使い、死んだ接続では素早く失敗するようになっています。古い Dockerfile を使っている、または問題が再発する場合:

1. ビルドを再試行する — 特に家庭用回線では、一時的なネットワークの途切れは起こり得ます。
2. 再試行しても失敗し続ける場合は、ホスト上でリポジトリを事前取得し、ビルドコンテキストにバインドマウントしてください。手早いレシピ:
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
   # then edit Dockerfile.v31 to COPY from /tmp/llama.cpp instead of cloning
   ```
3. GHCR のプレビルド llama-server イメージはこのステップを完全にスキップします — ビルドの代わりに pull してください。

### llama.cpp の再ビルド（新しいモデルアーキテクチャ、またはパッチのドリフト）

開発者向けのメンテナンスタスクです。ここに至るトリガーは2つ:

- **投入したモデルのロードが失敗する** — `error loading model: unknown (model) architecture 'gemma4'` — 固定された llama.cpp がそのアーキテクチャより古い。
- **ビルドが失敗する** — `error: patch failed: tools/server/server-context.cpp:NN` / `patch does not apply` — 上流が固定 SHA を追い越してドリフトした。

`atlas-llama` イメージは 4 つの Dockerfile すべて（`Dockerfile`、`Dockerfile.v31`、`Dockerfile.rocm`、`Dockerfile.vulkan`）で `LLAMA_CPP_REV` により llama.cpp を固定しており、compose ファイルがビルドに使う 3 つ — `Dockerfile.v31`、`Dockerfile.rocm`、`Dockerfile.vulkan` — はビルド中に `inference/patches/expose-hidden-states.patch`（Geometric Lens が依存するレイヤーごとの `hidden_states` 拡張）を適用します。素の `Dockerfile` はリビジョンを固定しますがパッチは適用しないため、そこからビルドされたサーバーには lens の配管がありません。新しいアーキテクチャを認識させるには、それを含む llama.cpp の SHA に固定を移動してください。プレビルドの GHCR イメージはローカルビルドをスキップします。公開イメージより新しいアーキテクチャが必要な場合にのみ再ビルドしてください。

**hidden-states パッチは維持してください — 削除ではなくリベースを。** `git apply` ステップを削除すると、lens の配管を静かに失ったサーバーがビルドされます（`/embedding` が `layers:` パラメータを無視します）。バンプの手順書:

1. **対象の SHA に対してパッチを検証する**（高速、Docker 不要）:
   ```bash
   mkdir -p /tmp/llama-check && cd /tmp/llama-check
   git init -q llama.cpp && cd llama.cpp
   git remote add origin https://github.com/ggml-org/llama.cpp
   git fetch --depth 1 origin <NEW_SHA> && git checkout -q FETCH_HEAD
   git apply --check $REPO/inference/patches/expose-hidden-states.patch
   ```
   （`git apply` されるのはこのパッチだけです。spec-decode の embeddings 修正は Dockerfile 内の `sed` で、対象行がなければ no-op です。）
2. **クリーンに適用できた場合:** 4 つの Dockerfile すべての `LLAMA_CPP_REV` を新しい SHA にバンプします。CI のスモークテストが一致を検証します。
3. **失敗した場合:** `git apply --reject …` でクリーンなハンクを適用し、各 `*.rej` ハンクを移動先のアンカーに再挿入し（周辺コードの上流でのリネーム、例えば `model` → `model_tgt` に注意し、パッチの追加行を更新する）、`git diff > $REPO/inference/patches/expose-hidden-states.patch` を実行します。ステップ 1 を再実行してください。長い CUDA ビルドの前にメンバー/型エラーを捕捉するため、触ったファイルだけを CPU のみでコンパイルします: `cmake -B build-cpu -DGGML_CUDA=OFF && make -C build-cpu server-context`。
4. 再ビルドして立ち上げる:
   ```bash
   docker compose build --build-arg LLAMA_CPP_REV=<sha> llama-server
   docker compose up -d llama-server --no-deps
   ```

古い SHA への固定よりも、パッチの再生成を優先してください — 後ろ向きに固定すると上流の修正を取りこぼします。

再ビルドでモデルがロードされた後も、新しいモデルには Geometric Lens の再トレーニングが必要です — [CONFIGURATION.md § Adding your own model](../../CONFIGURATION.md#adding-your-own-model-drop-in--unregistered) を参照。

### SELinux がコンテナアクセスをブロック (Fedora/RHEL)

**症状:** コンテナがマウントされたボリュームを読めない、モデルファイルへのパーミッション拒否。

**修正:**
```bash
# Allow container access to model directory
chcon -Rt svirt_sandbox_file_t ~/models/

# Or add :Z flag to volume mounts (Docker Compose handles this)
```

### サンドボックスに到達できない

**症状:** プロキシのヘルスが `"sandbox": false` を表示する。V3 のビルド検証が失敗する。

**修正:** すべてのサービスが同じ Docker ネットワーク上にあることを確認してください。Docker Compose は `atlas` ネットワークを自動的に作成します。コンテナを手動で実行している場合:
```bash
docker network create atlas
# Start all containers with --network atlas
```

### ポート競合

**症状:** `docker compose up` がポートの "address already in use" で失敗する。

**修正:** ポートを使用しているプロセスを確認し、停止するか `.env` で ATLAS のポートを変更してください:
```bash
# Find what's using port 8080
lsof -i :8080

# Change port in .env
ATLAS_LLAMA_PORT=8081    # Different port for llama-server
```

すべてのポートは `.env` で設定可能です。[CONFIGURATION.md](../../CONFIGURATION.md) をご覧ください。

---

## llama-server の問題

### モデルが GPU ではなく CPU で読み込まれている

**症状:** 約 50 tok/s ではなく約 2 tok/s で生成される。`nvidia-smi` で llama-server が GPU を使用していない。

**修正:** `--n-gpu-layers 99` が設定されていることを確認してください（全レイヤーを GPU にオフロード）。Docker Compose ではこれがデフォルトです。ベアメタルの場合、コマンドを確認してください:
```bash
ps aux | grep llama-server | grep 'n-gpu-layers'
```

Docker を使用している場合、NVIDIA コンテナランタイムが設定されていることを確認してください（上記の GPU セクションを参照）。

### モデル + KV キャッシュが GPU に収まらない（起動失敗、または生成が 5 倍遅い）

**症状（現行エントリポイント）:** llama-server が起動時に「fitting params to device memory」の直後の CUDA アロケーションエラーで終了する。

**症状（`--fit off` のない旧エントリポイント）:** サーバーは*起動*し、`nvidia-smi` ではモデルがロードされて見えるが、生成速度が期待値の数分の一になり、llama-server プロセスが複数の CPU コアを消費し（`top` で 400–800%）、ホスト側 RSS がモデル重みの数ギガバイトを保持する — llama.cpp のメモリ自動フィッタがレイヤーを CPU に黙って移したためです。

**原因:** モデルの重み + KV キャッシュ（`ATLAS_CTX_SIZE` × `PARALLEL` スロット × レイヤーごとの KV 次元）+ コンピュートバッファ（約 `ATLAS_UBATCH` × 隠れ次元 × 280 バイト）が VRAM を超えています。この予算はモデルごとに異なります — あるモデル向けの設定が、KV ジオメトリの異なる別のモデルでは溢れることがあります。

**修正:** このモデルと GPU に合わせてランタイムをサイズ調整し、コンテナを再作成してください:
```bash
atlas tier fit --write
docker compose up -d llama-server --no-deps --force-recreate
```
`atlas tier fit` は GGUF ヘッダーと GPU の VRAM を読み取り、完全に GPU 上で動く最大の構成を求めます（[CLI.md § atlas tier fit](../../CLI.md#atlas-tier-fit) を参照）。ATLAS は llama-server を `--fit off` で実行するため、収まらない構成は CPU で部分的に黙って動く代わりに、起動時に明確に失敗します。

`atlas tier fit` が **DOES NOT FIT** と報告する場合、モデル自体がこのカードには大きすぎます — 出力には収まる最大の量子化ファイルサイズが表示されます。優先順:

1. **同じモデルのより小さい量子化を使う**（例: Q6_K の代わりに Q4_K_M — 16 GB VRAM 未満では通常これが品質/GiB の最良トレード）。
2. **並列スロットを減らす**: `atlas tier fit --slots 1 --write` でスロットごとの KV 最小値が解放されます（`/demo` の分割ペインと V3 並列候補は使えなくなりますが、シングルストリームでの利用は可能）。
3. **より小さいモデルを選ぶ。** 下のサイズ表を参照。

### この GPU には何が収まる?

ダウンロード前の概算ルール: デフォルトの 4 スロットでは、GGUF は次の条件で余裕を持って収まります

```
file size  ≤  VRAM − ~4.5 GiB
```

（約 4.5 GiB は 4 × 8k コンテキストでの最小 KV キャッシュ、コンピュートバッファ、約 1.9 GiB の CUDA 固定オーバーヘッドをカバーします）。`--slots 1` ではマージンはおよそ `VRAM − 3 GiB` まで縮みます。スライディングウィンドウモデル（Gemma 系）はこれより少なくて済みます; このルールはフルアテンションモデル向けです。

| VRAM | GGUF ファイルサイズ (4 スロット) | GGUF ファイルサイズ (1 スロット) | 代表的なモデル |
|------|--------------------------|--------------------------|----------------|
| 8 GB | ≤ 約 3 GiB | ≤ 約 4.5 GiB | 3–4B Q4–Q6, 7–8B Q2–Q3 |
| 12 GB | ≤ 約 7 GiB | ≤ 約 8.5 GiB | 7–9B Q4–Q6, 12B Q3–Q4 |
| 16 GB | ≤ 約 11 GiB | ≤ 約 12.5 GiB | 9B Q6–Q8, 12–14B Q4–Q6 |
| 24 GB | ≤ 約 19 GiB | ≤ 約 20.5 GiB | 14B Q8, 27–32B Q4 |

HuggingFace のモデルページには量子化ごとのファイルサイズが記載されています — ダウンロード前にこの表と照らし合わせてください。この表はダウンロード前の概算にすぎません; ファイルがディスクに置かれたら `atlas tier fit /path/to/model.gguf` が正確な答えです（モデルの実際の KV ジオメトリを読み取るため、予算はどちらの方向にも数ギガバイト変わり得ます）。`atlas onboard` も同じフィットを自動で表示します。

### モデルファイルが見つからない

**症状:** llama-server が "failed to load model" などのメッセージで即座に終了する。

**修正:** モデルパスを確認してください:
```bash
# Docker Compose — model must be in ATLAS_MODELS_DIR (default: ./models/)
ls -la "models/$ATLAS_MODEL_FILE"

# Bare metal — check ATLAS_MODEL_PATH
ls -la "$ATLAS_MODELS_DIR/$ATLAS_MODEL_FILE"
```

ファイル名は `.env` の必須項目 `ATLAS_MODEL_FILE` の選択と一致する必要があります。

### VRAM 不足

**症状:** llama-server が起動直後にクラッシュまたは OOMKilled される。`nvidia-smi` で VRAM がほぼ 100% を表示。

**修正:** 以下を確認してください:
1. 他の GPU プロセスが実行されていない（`nvidia-smi` — 他の CUDA プロセスを確認）
2. 16GB 以上の VRAM がある
3. ランタイムがモデルと GPU に合わせてサイズ調整されている: `atlas tier fit --write`（推奨値を超えて `ATLAS_CTX_SIZE` を上げないでください）

```bash
# Kill other GPU processes if needed
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill {}
```

### 文法が強制されない（モデルが思考ブロックを出力する）

**症状:** モデルが JSON ツールコールの代わりに `<think>` タグや生テキストを出力する。

**修正:** プロキシは `/v1/agent` エージェントループハンドラ内で自動的に `response_format: {"type": "json_object"}` を設定します — これは無条件です（環境変数による切り替えはありません）。llama-server を `/v1/chat/completions` または `/v1/completions` で直接呼び出す場合は、パラメータを自分で含める必要があります:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role":"user","content":"Say hi"}],
    "max_tokens": 50,
    "response_format": {"type": "json_object"}
  }'
```

JSON ではなく生テキストが返される場合、お使いの llama.cpp ビルドが `response_format` をサポートしていません。最新のソースからリビルドしてください。

### コンテキストウィンドウが小さすぎる

**症状:** ツールコールの引数が切り詰められる。`write_file` が "unexpected end of JSON" で失敗するか、プロキシログに "truncation detected" と表示される。

**修正:** スロットあたりのコンテキスト（`ATLAS_CTX_SIZE` ÷ `ATLAS_PARALLEL_SLOTS`、compose のデフォルトは 131072 ÷ 4 = スロットあたり 32k）がタスクに対して小さすぎる可能性があります。`atlas tier fit` で GPU が対応できる最大予算を確認できます。確認方法:
```bash
# Docker Compose
grep CTX_SIZE .env

# Bare metal
ps aux | grep llama-server | grep ctx-size
```

---

## プロキシの問題

### エージェントループが起動しない

**症状:** リクエストが llama-server に直接送られる。ツールコールなし、ストリーミングステータスアイコンなし、V3 パイプラインなし。

**原因:** エンドポイントが間違っています。エージェントループは `POST /v1/agent` 上でのみ動作します。`POST /v1/chat/completions`（および `/v1/` 配下の他のパス）は llama-server への透過的なパススルーで、ツールも V3 もストリーミングチャットイベントもありません。

**修正:** クライアントを `POST http://localhost:8090/v1/agent` に向けてください。Bubbletea TUI（`atlas` / `atlas tui`）と内蔵の `/solve` REPL はこれを自動で行います。サードパーティクライアントを書く場合は、[docs/API.md](../../API.md) の `/v1/agent` SSE イベントプロトコルを参照してください。`ATLAS_AGENT_LOOP` 環境変数によるトグルはもうありません — 分岐はエンドポイントベースで、設定ベースではありません。

### 機能ファイルで V3 パイプラインが起動しない

**症状:** すべての `write_file` *または* `edit_file` コールが T1（直接書き込み）になる。出力に V3 パイプラインのステージがない。

V3 は**すべての条件**が満たされた場合に起動します:
1. ファイルのコンテンツが **10 行以上**（10 行未満は常に T1）
2. ファイルに **2 つ以上のロジックインジケーター**がある（関数定義、制御フロー、API パターン）— **または**認識されるコード/マークアップ拡張子（`.py`、`.go`、`.js`、`.html` など）を持つ場合、インジケーターがゼロでも 10 行以上で T2 になります
3. V3 サービスが `ATLAS_V3_URL` で到達可能
4. `edit_file` のみ: 結果のファイルがファイル全体の再実行に値する — サイクロマティック複雑度が 8 以上、または複雑度を測定できないときは 80 行以上

設定・データ・スタイル・Markdown・シェルファイル（`package.json`、`.yaml`、`.css`、`.md`、`.sh` など）はサイズに関わらず常に T1 です。リクエストのティアは V3 に転送されますが、起動をゲートするのはファイル自身のティアです。

`write_file` と `edit_file` の両方が V3 を経由します。

**診断:**
```bash
# Check V3 service health
curl -s http://localhost:8070/health

# Check proxy logs for tier classification + V3 activation
docker compose logs atlas-proxy | grep -E "write_file|edit_file|tier="
# Look for:
#   "tier=T2:medium" or higher in classifier output
#   "[edit_file] V3 pipeline activating for X (file_tier=2, req_tier=2)"
#   "[write_file] V3 pipeline activating for X"
# T1 means direct write — no V3.
```

V3 に到達できない場合、プロキシは `V3 failed: ...` をログに出し、編集を壊さずに直接書き込みにフォールバックします。

### 切り詰めエラー（write_file が繰り返し失敗）

**症状:** "Your output was truncated — the content is too long for a single tool call." のようなエラーが繰り返される。

**原因:** モデルが 1 回のコールで多すぎるコンテンツを書き込もうとしています。プロキシが切り詰められた JSON を検出し、ツールコールを拒否します。

プロキシは既存パスに対するすべての `write_file` を拒否し — `write_file` はファイルの新規作成用です — 代わりに `edit_file` を使うようモデルに指示します。例外は、5 行以下のファイル、ディスク上で破損して見えるファイル（散文の前置き、はぐれた Markdown フェンス）、このセッション自身が書き込んだファイルだけです。3 回連続で失敗すると、エラーループブレーカーがエージェントを停止し、サマリーを返します。

**修正:** ファイル全体の書き換えではなく、ターゲットを絞った変更を依頼するようリクエストを言い換えてください — "auth.py を書き直して" の代わりに "ログイン関数に入力バリデーションを追加して"。

プロキシは、本物の切り詰め（args ペイロードが 200 バイト超）と、`args` が空または欠落したまま送られたツールコールを区別します — 後者には切り詰めの再マップではなく、`read_file: no arguments provided. Call with {"path":"<file>"}` のようなツールごとのヒントが与えられます。また、OpenAI 形式（`arguments`）、Anthropic 形式（`parameters`）、トップレベルにインライン化された引数の形状を、正規の `args` エンベロープに正規化します。正規化後もツールコールが空のまま届く場合、プロキシは `[agent] turn=N EMPTY ARGS — raw model output: "..."` をログに出すので、正確な形状を確認して言い換えられます。

### ツール結果と次のアクションの間の長い停止

**症状:** ツールが成功した後、次のターンが発火するまでエージェントループが約 30 秒アイドル状態になる。エラーも出力もなく、やがて次のツールコールが現れる。

**何が起きているか:** 制約付き JSON 文法の下では、一部のローカルモデルがツール結果の後の最初のトークンとして EOS を発行し、空のコンテンツを返します。パースエラーのリトライパスがそこから回復する必要があり、それが失われる約 30 秒です。

**どうするか:** プロキシは `callLLMConstrained` 内で空のターンを捕捉し、`temperature=0.7` と継続を促すナッジ付きでインラインに 1 回リトライします。一貫して再発する場合は、llama.cpp のスロットキャッシュをクリアするためにプロキシを再起動してください:
```bash
docker compose restart atlas-proxy llama-server
```
`docker compose logs atlas-proxy | grep -E "empty LLM|raw_len=0"` を確認してください — 最初の呼び出しとリトライの両方で `raw_len=0` の場合、モデルはリトライで対処できる以上に悪い状態です。

### V3 が修正を確認済みなのにモデルが編集を続ける

**症状:** エージェントが V3 検証済みの編集に成功する（TUI に `Probe passed` で終わる V3 進捗イベントが表示される）のに、同じファイルを再読み込みして無関係な関数の編集を始める。後続の編集はそれぞれ、完全な V3 サイクル（約 110 秒）を再度トリガーする。

**何が起きているか:** コンパクトなローカルモデルは「ユーザーの元の問題は解決したか?」の自己評価が苦手なことがあり、検証済みの編集の後もさらに作業を計画し続けます。

**どうするか:** エージェントループは V3 検証済みの書き込みの後に、`{"type":"done"}` の発行へ向かわせる強いユーザーロールのナッジを追加します。モデルがそれを無視する場合は、その 1 つの変更だけが望みであることをプロンプトでより明示してください。より強い停止手段（ファイルごとの編集上限、自動 done）はフォローアップの選択肢として追跡されています。

### モデルが以前のセッションのファイル名を幻覚する

**症状:** 真新しいセッション、新しいプロンプトなのに、モデルの最初のツールコールが、このワークスペースに存在しないファイル名への `read_file` になる — 通常は最近作業した別の場所に存在するものです。

**何が起きているか:** llama.cpp の KV スロットは、キャッシュを温かく保つためにチャット補完間で持続します。セッションをまたぐと、前のセッションのトークンからの残留アテンションバイアスが、捏造されたファイル名のような低エントロピーの出力に漏れることがあります。

**どうするか:** すべてのユーザーターンは llama のスロット 0 を消去することから始まるため、次の補完はシステムプロンプトを新規に再エンコードします（温まった GPU で約 1〜2 秒）。キャッシュを完全に温かく保ちたい場合にセッションごとの消去を無効化するには:
```bash
# .env
ATLAS_FRESH_SLOT_PER_SESSION=0
```
変更後はプロキシを再起動してください。消去を無効にした状態で幻覚が見られる場合は、`llama-server` を再起動して全スロットをクリアしてください。

### 複数ファイルのプロジェクト: サンドボックスの `ModuleNotFoundError`

**症状:** 同じプロジェクト内の別モジュールをインポートするファイルへの編集。自分のマシンではインポートが動くのに、V3 が `ModuleNotFoundError: No module named 'utils'` で検証失敗を報告する。

**何が起きているか:** V3 の `SandboxAdapter` は、エージェントが読んだすべてのファイルを `solution.py` と一緒にサンドボックスのワークスペースに搬入します。読み取りセット（`ctx.FilesRead`）にないファイルはそこに存在しないため、そのインポートは失敗します。

**どうするか:** 欠けているファイルを `read_file` で読み、プロジェクトコンテキストに載せてください。サンドボックスの `/execute` API を直接呼んでいる場合は、リクエストボディで補助ファイルを渡してください:
```bash
curl -X POST http://localhost:30820/execute -d '{
  "code": "from utils import greet\nprint(greet(\"x\"))",
  "language": "python",
  "files": {"utils.py": "def greet(n): return f\"hi {n}\""}
}'
```

### Curses の最下行 `addwstr() returned ERR`

**症状:** curses プログラムが実行時に `_curses.error: addwstr() returned ERR` でクラッシュするが、ATLAS は編集が V3 検証に合格したと報告している。

**何が起きているか:** curses ウィンドウの最後のセル（row=LINES-1 または column=COLS-1）への書き込みは、curses の文書化された挙動として ERR を返します。`interactive_lint` は `try/except curses.error` のラップなしにそこへ書き込む候補を拒否するため、V3 は認証の前にラップされたバリアントを見つける必要があります。慣用的な修正:
```python
try:
    stdscr.addstr(curses.LINES - 1, 0, border)
except curses.error:
    pass  # writing the bottom-right cell errors; benign
```

**どうするか:** V3 が自力でラップを合成できない場合は、モデルに明示的に伝えてください: *「N 行目の addstr 呼び出しを `try: ... except curses.error: pass` で包め」*。`docker compose logs v3-service | grep interactive_lint` でリントゲートが発火したことを確認できます。

### 非 Python ファイルで V3 が数分ハングする

**症状:** ATLAS に HTML/CSS/JSON ファイルの作成を頼むと、PR-CoT の修復試行と LLM タイムアウトを伴う約 5 分の停止が起きる。ファイルは最終的に直接書き込みのフォールバック経由で着地する。

**何が起きているか:** V3 のスモークチェックは言語対応です — 対象ファイルの拡張子から言語を導出し、適切なチェッカーにルーティングします（`.py` → Python コンパイル、`.js` → `node --check`、`.ts` → `tsc --noEmit`、`.go` → `gofmt -e`、`.rs` → `rustc`、`.sh` → `bash -n`、`.html` → `html.parser`、`.xml` → `ElementTree`、`.json` → `json.loads`、`.yaml` → `yaml.safe_load`）。認識されない拡張子は Python にフォールバックして失敗し、修復へと連鎖します。`.c`/`.cpp`/`.h` は拡張子マップ（`v3-service/main.py` の `_ext_to_lang`）に含まれていないため、サンドボックス自体には C/C++ チェッカーがあるにもかかわらず、C/C++ ファイルは Python フォールバックに当たることに注意してください。

`/v3/generate` が承認済みのプロジェクトビルドコマンドを受け取った場合、V3 は構文/セルフテスト検証の後に `build_verify` イベントを発行します。コマンドは、候補をプロジェクトにオーバーレイした一時的なサンドボックスワークスペースで実行されるため、失敗したビルドの証拠は、候補を実際のチェックアウトに書き込むことなく `passed=true` をブロックします。オーバーレイのスナップショットは依存キャッシュ、シークレット、モデル/データアーティファクト、シンボリックリンク、大きなファイルをスキップし、ファイル数とバイト数の制限を課します。プロジェクトのビルドに重い依存が必要な場合は、明示的な検証ワークフローの一部としてサンドボックスワークスペース内にインストールしてください。

**どうするか:** 認識されない拡張子については、`v3-service/main.py` の `_ext_to_lang` に追加して `v3-service` イメージを再ビルドしてください。V3 がエラーになるとプロキシは直接書き込みにフォールバックするため、ファイルはいずれにせよ着地します — 遅いだけです。`docker compose logs v3-service | grep smoke_check` で正しい言語がルーティングされたことを確認できます。

### 「もう一度直して」のプロンプトで V3 パイプラインが発火しない

**症状:** 最初のリクエストはファイルを作成し V3 が動く。"ok" や "yes" のような素っ気ないフォローアップには会話的な返答が返る — ツールコールも V3 イベントもなし。

**何が起きているか:** エージェントループのティア分類器は、些細でないメッセージにはすべて T2 を下限として扱います — "still doesn't work, try again" は T2 に分類されパイプラインが動きます。5 文字未満のメッセージ、または小さな些細チャットのリスト（`hi`、`thanks`、`ok`、`yes` など）への完全一致だけが T0 に留まります: 会話的、パイプラインなし。

**どうするか:** 短くてもいいので、望むことを言ってください — "yes, fix it" は T0 のゲートを通過します。フォローアップがエージェントループを実行するのに V3 が沈黙している場合、ゲートはリクエストのティアではなく、ファイル自身のティアです。[機能ファイルで V3 パイプラインが起動しない](#機能ファイルで-v3-パイプラインが起動しない) を参照し、`docker compose logs atlas-proxy | grep -E "write_file|edit_file"` でファイルティアの行（例: `[write_file] app.py → T1:simple (8 lines)`）を確認してください。

### 編集前にファイルが読み込まれていない

**症状:** `edit_file` が "file not read yet — use read_file first before editing." で失敗する。

**原因:** プロキシはエージェントがどのファイルを読んだかを追跡しています。モデルがこのセッションで読んでいないファイルを編集しようとすると、古さ防止のため編集が拒否されます。

**修正:** モデルはまずファイルを読む必要があります。失敗が続く場合は、TUI で `/clear` と入力してチャット履歴をリセットし、リクエストを言い換えてください。

### 外部でファイルが変更された

**症状:** `edit_file` が "file modified since last read — read it again before editing." で失敗する。

**原因:** モデルがファイルを読んだ後に、ディスク上のファイルが（ユーザーまたは別のプロセスによって）変更されました。プロキシは変更タイムスタンプを比較します。

**修正:** モデルがファイルを再読み込みする必要があります。通常、次のターンで自動的に解決されます。

### 探索予算の警告

**症状:** 出力に "You have full project context in the system prompt. Do not read more files." と表示される。

**原因:** モデルが何も書き込まずに 4 回以上連続で読み取り専用コール（read_file、search_files、list_directory）を行いました。4 回の読み取りでプロキシが書き込みを促すナッジを注入し、5 回以上ではより強いナッジになります。読み取りは常に実行されます — ナッジは次のターンを誘導するだけで、読み取りがスキップされることはありません。

**修正:** モデルが本当に探索で行き詰まっている場合は、変更したい内容をより具体的に指示してください。

---

## Geometric Lens の問題

### Lens が読み込まれない / 利用不可

**症状:** プロキシのヘルスが `"lens": false` を表示する。または起動時に "Lens unavailable — verification disabled." と表示される。

**影響:** ATLAS は C(x)/G(x) スコアリングなしでも動作します。V3 の候補選択はサンドボックスのみの検証にフォールバックします。

**修正:** Lens のヘルスとログを確認してください:
```bash
curl -s http://localhost:8099/health
docker compose logs geometric-lens
```

一般的な原因:
- Lens が llama-server に接続できない（`LLAMA_URL` 環境変数を確認）
- モデルウェイトファイルが見つからない（サービスはグレースフルにデグレードします — カスタムモデルをトレーニングしていない場合はこれが想定される動作です）

### すべてのスコアが 0.5 付近

**症状:** コード品質に関係なく、すべての候補が `cx_energy: 0.0` および `gx_score: 0.5` になる。

**原因:** モデルウェイトが読み込まれていません。モデルが存在しない場合、サービスはニュートラルなデフォルト値を返します。

**確認:**
```bash
curl -s http://localhost:8099/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(1)"}' | python3 -m json.tool
```

`enabled: false` または `cx_energy: 0.0` の場合、モデルが読み込まれていません。新規インストールではこれが想定される動作です — モデルウェイトはリポジトリに含まれておらず、トレーニングするか [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS) からダウンロードする必要があります。

### エンベディング抽出の失敗

**症状:** Lens のログに "embedding extraction failed" やタイムアウトなどのエラーが表示される。

**原因:** Lens は llama-server の `/v1/embeddings` エンドポイントを呼び出します。llama-server が過負荷状態であるか、エンドポイントが有効になっていない場合、これが失敗します。

**修正:**
```bash
# Test embedding endpoint directly
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}' | python3 -m json.tool
```

`/v1/embeddings` エンドポイントは、生成モデルからのセルフエンベディング用に特別なフラグなしで llama.cpp で利用可能です。K3s では、完全なエンベディングサポートのためにエントリーポイントで `--embeddings` フラグが明示的に設定されています。

### `/internal/lens/retrain` が 503 "models directory is mounted read-only" を返す

**症状:** lens サービスに `/internal/lens/retrain` を POST すると、`"reason": "models directory is mounted read-only; run host-side retrain via atlas lens retrain"` 付きの HTTP 503 が返る。

**原因:** 標準の Compose デプロイは lens のモデルディレクトリをコンテナに読み取り専用（`:ro`）でマウントするため、サービス内の retrain エンドポイントは新しいウェイトを書き込めません。エンドポイントはトレーニング前に書き込み可能性をプローブし、トレーニング実行を無駄にする代わりに最初から拒否します。

**修正:** リトレーニングはホスト側で実行してください — `atlas lens retrain`（フィードバックコーパス）または `atlas lens build`（ベンチ候補）がホスト上にアーティファクトを書き込み、その後サービスが `/internal/lens/reload`（または `docker compose restart geometric-lens`）経由でリロードします。ベンチマーク駆動のオンライン再キャリブレーション（`lens_feedback`）は拒否をログに記録してサンプルバッファを保持するため、何も失われません。

---

## サンドボックスの問題

### サンドボックスに到達できない

**症状:** コードがテストされない。プロキシのヘルスが `"sandbox": false` を表示する。

**修正:** サンドボックスのヘルスを確認してください:
```bash
# Docker Compose (host port 30820 maps to container port 8020)
curl -s http://localhost:30820/health

# Bare metal (direct port 8020)
curl -s http://localhost:8020/health
```

サンドボックスコンテナが実行中だが正常でない場合、ログを確認してください:
```bash
docker compose logs sandbox
```

### コード実行のタイムアウト

**症状:** サンドボックスが `"error_type": "Timeout"` を返す。コードの実行に時間がかかりすぎている。

**デフォルトタイムアウト:** リクエストあたり 30 秒、上限は `MAX_EXECUTION_TIME`。Compose スタックはその上限を 300 秒に設定しており（`.env` の `ATLAS_SANDBOX_MAX_EXECUTION_TIME` 経由）、長いビルドやテストスイートが完走できるよう、プロキシの `run_command` の上限に合わせています。compose の外では、エグゼキュータのコード内上限は 60 秒です。

**修正:** コードが正当により多くの時間を必要とする場合は、リクエストでより長いタイムアウトを設定する（上限まで）か、`ATLAS_SANDBOX_MAX_EXECUTION_TIME` を引き上げてください。コードに無限ループがある場合、これは想定された動作です。タイムアウト時はプロセスグループ全体が kill されるため、コマンドが生成した子プロセスが残留することはありません。

### 言語がサポートされていない

**症状:** 特定の言語でサンドボックスがエラーを返す。

**サポート対象言語:** Python、JavaScript、TypeScript、Go、Rust、C、C++、Bash。

利用可能なランタイムを確認:
```bash
curl -s http://localhost:30820/languages | python3 -m json.tool
```

---

## ベンチマークの問題

### bench が要求より少ないタスクしか実行しない（`LIMITED MODE: running N tasks` の N が `--tasks` より小さい）

**症状:** `atlas bench --tasks 200` が `LIMITED MODE: running 100 tasks`（または要求より少ない数）と表示する。あるいは再開した実行が `Resuming: N/N complete, 0 remaining` と出力して即座に終了する。

**原因:** LiveCodeBench のデータセットキャッシュ（`benchmark/datasets/.cache/livecodebench_v5.jsonl`）が部分ダウンロードです。HuggingFace rows API はページネーション途中で失敗することがあり、旧バージョンは取得済み分をキャッシュしてそのファイルを永久に信頼していました。release_v5 の完全なセットは約 880 タスクです。

**修正:** キャッシュを partial としてフラグ付けして再実行してください — ローダーが完全な再取得を試みます（全ソースが失敗した場合のみ既存コピーにフォールバック）:
```bash
touch benchmark/datasets/.cache/livecodebench_v5.jsonl.partial
atlas bench --run-id <your-run-id> --tasks 200
```
完了済みタスクが失われることはありません: 結果は `benchmark/results/<run-id>/v3_lcb/per_task/` にタスクごとの JSON として保存され、ランナーは結果ファイルが存在するタスクをスキップして再開します。何らかの理由（OOM、再起動、セッションのクローズ）で中断された実行も同じ方法で再開します — 同一の `atlas bench` コマンドを再実行するだけです。

## パフォーマンス

### 生成が遅い（約 2 tok/s）

モデルが GPU ではなく CPU で実行されています。以下を確認してください:
1. `nvidia-smi` — llama-server が GPU プロセスとして表示されているか?
2. `--n-gpu-layers 99` — すべてのレイヤーがオフロードされているか?
3. NVIDIA Container Toolkit — コンテナランタイムが GPU アクセス用に設定されているか?

**想定パフォーマンス:** RTX 5060 Ti 16GB で文法強制時に約 51 tok/s。

### V3 パイプラインに数分かかる

これは T2 ファイルでは正常な動作です。V3 パイプラインは複数の LLM コールを行います:
- **プローブのみ（最良ケース）:** 約 10-15 秒（1 回の生成 + 1 回のスコアリング + 1 回のテスト）
- **Phase 1 生成:** 約 1-2 分（PlanSearch + DivSampling + スコアリング）
- **Phase 3 修復:** 約 2-5 分（PR-CoT + Refinement + Derivation、必要な場合）

より高速な（ただし品質は低い）結果を得るには:
- ファイルを 10 行未満に保つ（T1 のまま、V3 なし）— 認識されるコード拡張子は 10 行以上で複雑さに関わらず T2 になります
- ロジックの複雑さを減らす（関数や制御フローを少なく）
- V3 は本当に必要な場合にのみ起動します — シンプルなファイルは即座に書き込まれます

### RAM 使用量が高い

**症状:** システムが遅くなるか、サービスが OOMKilled される。

**想定 RAM 使用量:**
- llama-server: 約 8 GB（モデルは VRAM 内、RAM 使用は最小）
- geometric-lens: 約 200 MB（PyTorch ランタイム + モデル）
- v3-service: 約 150 MB（PyTorch ランタイム）
- sandbox: 約 100 MB（ベース、コンパイル中にスパイク）
- atlas-proxy: 約 30 MB（Go バイナリ）

**合計:** 約 500 MB RAM + 8.2 GB VRAM。システム RAM が 14 GB 未満の場合、他のサービスとメモリが競合する可能性があります。

---

## ヘルプを得る

ここに問題が記載されていない場合:
1. サービスログを確認: `docker compose logs <service-name>`
2. プロキシのヘルスエンドポイントを確認: `curl http://localhost:8090/health`
3. すべての環境変数については [CONFIGURATION.md](../../CONFIGURATION.md) をご覧ください
4. [GitHub](https://github.com/itigges22/ATLAS/issues) で Issue を作成してください
