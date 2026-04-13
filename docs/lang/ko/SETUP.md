> **[English](../../SETUP.md)** | **[简体中文](../zh-CN/SETUP.md)** | **[日本語](../ja/SETUP.md)** | **한국어**

# ATLAS 설정 가이드

세 가지 배포 방법을 제공합니다: Docker Compose(권장 및 테스트 완료), 베어메탈, K3s.

---

## 사전 요구 사항 (모든 방법 공통)

| 요구 사항 | 세부 내용 |
|-----------|----------|
| **NVIDIA GPU** | 16GB 이상 VRAM (RTX 5060 Ti 16GB에서 테스트됨) |
| **NVIDIA 드라이버** | 전용 드라이버 설치 필요 (`nvidia-smi`에서 GPU가 표시되어야 합니다) |
| **Python 3.9+** | pip 포함 |
| **wget** | 모델 가중치 다운로드용 |
| **모델 가중치** | HuggingFace의 Qwen3.5-9B-Q6_K.gguf (~7GB) |

### GPU 확인

```bash
nvidia-smi
# GPU가 드라이버 버전 및 VRAM과 함께 표시되어야 합니다
# 실패할 경우, 먼저 NVIDIA 전용 드라이버를 설치하십시오
```

---

## 방법 1: Docker Compose (권장)

V3.0.1에서 테스트된 배포 방법입니다.

### 추가 사전 요구 사항

- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)이 설치된 **Docker**, **또는 Podman**
- 약 20GB 디스크 공간 (모델 가중치 + 컨테이너 이미지)

### 설정

```bash
# 1. 클론
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS

# 2. 모델 가중치 다운로드 (~7GB)
mkdir -p models
wget https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q6_K.gguf \
     -O models/Qwen3.5-9B-Q6_K.gguf

# 3. ATLAS CLI + Aider 설치
pip install -e . aider-chat

# 4. (권장) 모든 디렉토리에서 전체 파일 접근을 위해 Go 1.24+ 설치
#    https://go.dev/dl/ - 프록시는 첫 실행 시 자동으로 빌드됩니다
#    Go가 없으면 프록시는 Docker 내에서 실행되며 파일 접근이 ATLAS_PROJECT_DIR로 제한됩니다

# 5. 환경 설정
cp .env.example .env
# 모델이 ./models/에 있으면 기본값으로 동작합니다 - 경로를 변경한 경우에만 .env를 수정하십시오

# 6. 모든 서비스 시작 (첫 실행 시 컨테이너 이미지를 빌드합니다 - 수 분이 소요됩니다)
docker compose up -d         # 또는: podman-compose up -d

# 7. 모든 서비스가 정상인지 확인 (모든 서비스가 "healthy"로 표시될 때까지 대기)
docker compose ps

# 8. 코딩 시작 (프로젝트 디렉토리에서)
cd /path/to/your/project
atlas
```

### 첫 실행 시 동작

1. Docker가 소스에서 5개의 컨테이너 이미지를 빌드합니다:
   - **llama-server** - CUDA로 llama.cpp를 컴파일합니다 (가장 느림, 약 5-10분)
   - **geometric-lens** - PyTorch CPU + FastAPI를 설치합니다
   - **v3-service** - PyTorch CPU + 벤치마크 모듈을 설치합니다
   - **sandbox** - Node.js, Go, Rust, gcc를 설치합니다
   - **atlas-proxy** - Go 바이너리를 컴파일합니다
2. llama-server가 7GB 모델을 GPU VRAM에 로드합니다 (약 1-2분)
3. 모든 서비스가 헬스 체크를 시작합니다
4. 5개 서비스가 모두 정상으로 보고되면, `atlas`가 연결되어 Aider를 실행합니다

이후 `docker compose up -d` 실행은 이미지가 캐시되어 있으므로 빠르게(수 초) 시작됩니다.

### 설치 확인

```bash
# 각 서비스를 개별적으로 확인
curl -s http://localhost:8080/health | python3 -m json.tool   # llama-server
curl -s http://localhost:8099/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy

# 간단한 기능 테스트 (aider 필요: pip install aider-chat)
atlas --message "Create hello.py that prints hello world"
```

모든 헬스 엔드포인트는 `{"status": "ok"}` 또는 `{"status": "healthy"}`를 반환해야 합니다.

> **참고:** `atlas` 명령은 프록시를 자동 감지하고 전체 에이전트 루프(도구 호출, V3 파이프라인, 파일 읽기/쓰기)를 위해 Aider를 실행합니다. Aider가 설치되어 있지 않으면 `/solve`와 `/bench`를 지원하지만 파일 작업은 불가능한 내장 REPL로 폴백합니다. 전체 기능을 사용하려면 Aider를 설치하십시오: `pip install aider-chat`

### 중지

```bash
docker compose down          # 모든 서비스 중지 (이미지는 유지)
docker compose down --rmi all  # 중지 및 이미지 삭제 (다음 시작 시 재빌드)
```

### 로그 확인

```bash
docker compose logs -f llama-server    # llama-server 로그 실시간 확인
docker compose logs -f geometric-lens  # Lens 로그 실시간 확인
docker compose logs -f v3-service      # V3 파이프라인 로그 실시간 확인
docker compose logs -f atlas-proxy     # 프록시 로그 실시간 확인
docker compose logs -f sandbox         # 샌드박스 로그 실시간 확인
docker compose logs --tail 50          # 모든 서비스의 최근 50줄
```

### 업데이트

```bash
git pull
docker compose down
docker compose build         # 변경된 이미지 재빌드
docker compose up -d
```

---

## 방법 2: 베어메탈

컨테이너 없이 모든 서비스를 로컬 프로세스로 실행합니다. 개발 환경이나 Docker를 사용할 수 없는 시스템에 적합합니다.

### 추가 사전 요구 사항

| 요구 사항 | 세부 내용 |
|-----------|----------|
| **Go 1.24+** | atlas-proxy 빌드용 |
| **llama.cpp** | CUDA로 소스 빌드 필요 ([llama.cpp 빌드 안내](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#build) 참조) |
| **Aider** | `pip install aider-chat` |
| **Node.js 20+** | 샌드박스의 JavaScript/TypeScript 실행에 필요 |
| **Rust** | 샌드박스의 Rust 실행에 필요 |

### 빌드

```bash
# 1. 클론 및 Python CLI 설치
git clone https://github.com/itigges22/ATLAS.git
cd ATLAS
pip install -e .

# 2. 모델 가중치 다운로드
mkdir -p models
wget https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q6_K.gguf \
     -O models/Qwen3.5-9B-Q6_K.gguf

# 3. atlas-proxy 빌드
cd atlas-proxy
go build -o ~/.local/bin/atlas-proxy-v2 .
cd ..

# 4. geometric-lens Python 의존성 설치
pip install -r geometric-lens/requirements.txt

# 5. V3 서비스 PyTorch 설치 (CPU 전용)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 6. 샌드박스 의존성 설치
pip install fastapi uvicorn pylint pytest pydantic
```

### 서비스 시작

각 서비스를 별도의 터미널에서 시작합니다 (또는 `&`와 로그 파일 리다이렉션 사용):

```bash
# 터미널 1: llama-server (GPU)
llama-server \
  --model models/Qwen3.5-9B-Q6_K.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 --n-gpu-layers 99 --no-mmap

# 터미널 2: Geometric Lens
cd geometric-lens
LLAMA_URL=http://localhost:8080 \
LLAMA_EMBED_URL=http://localhost:8080 \
GEOMETRIC_LENS_ENABLED=true \
PROJECT_DATA_DIR=/tmp/atlas-projects \
python -m uvicorn main:app --host 0.0.0.0 --port 8099

# 터미널 3: V3 파이프라인
cd v3-service
ATLAS_INFERENCE_URL=http://localhost:8080 \
ATLAS_LENS_URL=http://localhost:8099 \
ATLAS_SANDBOX_URL=http://localhost:8020 \
python main.py

# 터미널 4: Sandbox
cd sandbox
python executor_server.py

# 터미널 5: atlas-proxy
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

> **참고:** 샌드박스는 베어메탈 모드에서 포트 **8020**에서 수신합니다 (Docker 포트 리매핑 없음). 프록시의 `ATLAS_SANDBOX_URL`은 30820이 아닌 8020 포트를 사용해야 합니다.

### 런처 스크립트로 시작

대안으로 런처 스크립트를 PATH에 복사할 수 있습니다:

```bash
cp /path/to/atlas-launcher ~/.local/bin/atlas
chmod +x ~/.local/bin/atlas
atlas    # 누락된 서비스를 시작하고 Aider를 실행합니다
```

런처는 이미 실행 중인 서비스를 자동 감지하고 누락된 것만 시작합니다. Docker Compose 스택이 감지되면 해당 스택에 연결합니다.

---

## 방법 3: K3s

GPU 스케줄링, 헬스 프로브, 리소스 제한을 갖춘 프로덕션 Kubernetes 배포입니다.

### 추가 사전 요구 사항

| 요구 사항 | 세부 내용 |
|-----------|----------|
| **K3s** | 단일 노드 또는 다중 노드 클러스터 |
| **NVIDIA GPU Operator** 또는 **device plugin** | GPU가 `nvidia.com/gpu` 리소스로 표시되어야 합니다 |
| **Helm** | GPU Operator 설치용 |
| **Podman 또는 Docker** | 컨테이너 이미지 빌드용 |

### 자동 설치

설치 스크립트가 K3s 설치, GPU Operator, 컨테이너 빌드, 배포까지 전체 설정을 처리합니다:

```bash
# 1. 설정
cp atlas.conf.example atlas.conf
# atlas.conf 수정: 모델 경로, GPU 레이어, 컨텍스트 크기, NodePorts

# 2. 설치 프로그램 실행 (root 권한 필요)
sudo scripts/install.sh
```

설치 프로그램은 다음을 수행합니다:
1. 사전 요구 사항 확인 (NVIDIA 드라이버, GPU VRAM, 시스템 RAM)
2. K3s가 실행 중이 아닌 경우 설치
3. GPU가 클러스터에 보이지 않으면 Helm을 통해 NVIDIA GPU Operator 설치
4. 컨테이너 이미지를 빌드하고 K3s containerd에 가져오기
5. `atlas.conf`에서 envsubst를 통해 매니페스트 생성
6. `atlas` 네임스페이스에 배포
7. 모든 서비스가 정상이 될 때까지 대기

### 수동 배포

K3s가 이미 GPU 지원과 함께 실행 중인 경우:

```bash
# 1. 설정
cp atlas.conf.example atlas.conf
# atlas.conf 수정

# 2. 이미지 빌드 및 가져오기
scripts/build-containers.sh

# 3. atlas.conf에서 매니페스트 생성
scripts/generate-manifests.sh

# 4. 배포
kubectl apply -n atlas -f manifests/

# 5. 확인
scripts/verify-install.sh
```

### K3s 전용 설정

K3s는 설정에 `.env`가 아닌 `atlas.conf`를 사용합니다. Docker Compose와의 주요 차이점:

| 설정 항목 | Docker Compose | K3s |
|-----------|---------------|-----|
| 설정 파일 | `.env` | `atlas.conf` |
| 컨텍스트 크기 | 32K | 슬롯당 40K (x 4 슬롯 = 총 160K) |
| 병렬 슬롯 | 1 (암묵적) | 4 |
| Flash attention | 꺼짐 | 켜짐 |
| KV 캐시 양자화 | 없음 | q8_0 (키) + q4_0 (값) |
| 메모리 잠금 | 아니오 | mlock 활성화 |
| 임베딩 엔드포인트 | 미노출 | `--embeddings` 플래그 |
| 서비스 노출 | 호스트 포트 | NodePorts |

전체 `atlas.conf` 레퍼런스는 [CONFIGURATION.md](../../CONFIGURATION.md)를 참조하십시오.

### K3s 배포 확인

```bash
# 파드 확인
kubectl get pods -n atlas

# GPU 할당 확인
kubectl describe nodes | grep nvidia.com/gpu

# 검증 스위트 실행
scripts/verify-install.sh
```

> **참고:** Docker Compose는 V3.0.1에서 검증된 배포 방법입니다. K3s 매니페스트는 배포 시점에 템플릿에서 생성됩니다. K3s 배포는 V3.0 벤치마크에서 Qwen3-14B로 사용되었으며 프로덕션에서 테스트되었지만, 클러스터 설정에 따라 템플릿 파일 조정이 필요할 수 있습니다.

---

## 하드웨어 사양

| 리소스 | 최소 | 권장 | 비고 |
|--------|------|------|------|
| GPU VRAM | 16 GB | 16 GB | 모델 (~7GB) + KV 캐시 (~1.3GB) + 오버헤드 |
| 시스템 RAM | 14 GB | 16 GB+ | PyTorch 런타임 + 컨테이너 오버헤드 |
| 디스크 | 15 GB | 25 GB | 모델 (7GB) + 컨테이너 이미지 (5-8GB) + 작업 공간 |
| CPU | 4 코어 | 8+ 코어 | V3 파이프라인은 수리 단계에서 CPU 집약적입니다 |

### 지원 GPU

16GB 이상의 VRAM과 CUDA를 지원하는 모든 NVIDIA GPU에서 사용 가능합니다. 테스트 완료:
- RTX 5060 Ti 16GB (주 개발 GPU)

AMD 및 Intel GPU는 아직 테스트되지 않았습니다. llama.cpp는 ROCm 및 기타 백엔드를 지원하며, ROCm 지원은 V3.1의 우선 과제입니다.

---

## Geometric Lens 가중치 (선택 사항)

ATLAS는 Geometric Lens 가중치 없이도 동작합니다. 서비스는 중립 점수를 반환하며 정상적으로 성능을 저하시킵니다. V3 파이프라인은 샌드박스 전용 검증으로 폴백합니다.

C(x)/G(x) 점수 산출을 활성화하려면 학습된 모델 가중치가 필요합니다. 사전 학습된 가중치와 학습 데이터는 HuggingFace에서 제공됩니다:

**[HuggingFace의 ATLAS 데이터셋](https://huggingface.co/datasets/itigges22/ATLAS)** - 임베딩, 학습 데이터, 가중치 파일이 포함되어 있습니다.

가중치 파일을 `geometric-lens/geometric_lens/models/`에 배치하거나 Docker Compose에서 `ATLAS_LENS_MODELS`를 통해 마운트하십시오. 서비스가 시작 시 자동으로 로드합니다.

자체 벤치마크 데이터로 학습하려는 경우 `scripts/`에 학습 스크립트가 제공됩니다:
- `scripts/retrain_cx_phase0.py` - 수집된 임베딩에서 초기 C(x) 학습
- `scripts/retrain_cx.py` - 클래스 가중치를 적용한 프로덕션 C(x) 재학습
- `scripts/collect_lens_training_data.py` - 벤치마크 실행에서 통과/실패 임베딩 수집
- `scripts/prepare_lens_training.py` - 학습 데이터 형식 준비 및 검증

---

## 다음 단계

- [CLI.md](../../CLI.md) - 실행 후 ATLAS 사용 방법
- [CONFIGURATION.md](../../CONFIGURATION.md) - 모든 환경 변수 및 튜닝 옵션
- [TROUBLESHOOTING.md](../ko/TROUBLESHOOTING.md) - 일반적인 문제 및 해결 방법
- [ARCHITECTURE.md](../../ARCHITECTURE.md) - 시스템 내부 동작 원리
