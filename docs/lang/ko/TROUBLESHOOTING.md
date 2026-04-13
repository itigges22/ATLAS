> **[English](../../TROUBLESHOOTING.md)** | **[简体中文](../zh-CN/TROUBLESHOOTING.md)** | **[日本語](../ja/TROUBLESHOOTING.md)** | **한국어**

# ATLAS 문제 해결 가이드

ATLAS V3.0.1의 일반적인 문제와 해결 방법을 서비스별로 정리했습니다.

---

## 빠른 진단

문제의 원인을 파악하기 위해 다음을 먼저 실행하십시오:

```bash
# Docker Compose - 모든 서비스를 한 번에 확인
docker compose ps

# 개별 헬스 체크
curl -s http://localhost:8080/health | python3 -m json.tool   # llama-server
curl -s http://localhost:8099/health | python3 -m json.tool   # geometric-lens
curl -s http://localhost:8070/health | python3 -m json.tool   # v3-service
curl -s http://localhost:30820/health | python3 -m json.tool  # sandbox
curl -s http://localhost:8090/health | python3 -m json.tool   # atlas-proxy (모든 서비스 상태 표시)

# GPU 상태
nvidia-smi

# Docker Compose 로그 (서비스당 마지막 50줄)
docker compose logs --tail 50
```

atlas-proxy 헬스 엔드포인트는 모든 업스트림 서비스의 상태를 보고합니다:
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

어떤 필드라도 `false`이면 해당 서비스에 문제가 있습니다.

---

## Docker / Podman 문제

### 컨테이너에서 GPU가 감지되지 않음

**증상:** llama-server 컨테이너가 시작되지만 모델이 CPU에서 로드됩니다 (매우 느림, 약 2 tok/s). 호스트에서 `nvidia-smi`로 GPU가 보이지만 컨테이너에서는 접근할 수 없습니다.

**해결:** NVIDIA Container Toolkit을 설치하십시오:

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

컨테이너 내에서 GPU가 보이는지 확인합니다:
```bash
# Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Podman
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.0-base nvidia-smi
```

### 첫 빌드 실패 (CUDA를 찾을 수 없음)

**증상:** llama-server 컴파일 중 `docker compose build`가 CUDA 관련 오류로 실패합니다.

**해결:** llama-server Dockerfile은 `nvidia/cuda:12.8.0-devel` 기본 이미지 내에서 llama.cpp를 빌드하므로, 호스트 GPU 접근 없이도 빌드 시 CUDA 헤더를 사용할 수 있습니다. 빌드 실패의 일반적인 원인:
1. 디스크 공간 부족 (빌드 아티팩트에 약 5GB 필요)
2. CUDA 기본 이미지 다운로드 또는 llama.cpp 클론 시 네트워크 문제
3. Podman rootless 빌드가 권한 문제로 실패할 수 있음 - `podman-compose build`에 `--podman-build-args="--format docker"`를 추가해 보십시오

### SELinux가 컨테이너 접근을 차단함 (Fedora/RHEL)

**증상:** 컨테이너가 마운트된 볼륨을 읽을 수 없으며, 모델 파일에 대해 권한 거부가 발생합니다.

**해결:**
```bash
# 모델 디렉토리에 컨테이너 접근 허용
chcon -Rt svirt_sandbox_file_t ~/models/

# 또는 볼륨 마운트에 :Z 플래그 추가 (Docker Compose에서 자동 처리됨)
```

### Sandbox에 연결할 수 없음

**증상:** 프록시 헬스에서 `"sandbox": false`가 표시됩니다. V3 빌드 검증이 실패합니다.

**해결:** 모든 서비스가 동일한 Docker 네트워크에 있는지 확인하십시오. Docker Compose는 `atlas` 네트워크를 자동으로 생성합니다. 컨테이너를 수동으로 실행하는 경우:
```bash
docker network create atlas
# 모든 컨테이너를 --network atlas로 시작
```

### 포트 충돌

**증상:** `docker compose up`이 포트에서 "address already in use" 오류로 실패합니다.

**해결:** 해당 포트를 사용하는 프로세스를 확인하고, 중지하거나 `.env`에서 ATLAS 포트를 변경하십시오:
```bash
# 포트 8080을 사용하는 프로세스 확인
lsof -i :8080

# .env에서 포트 변경
ATLAS_LLAMA_PORT=8081    # llama-server에 다른 포트 사용
```

모든 포트는 `.env`를 통해 설정 가능합니다. [CONFIGURATION.md](../../CONFIGURATION.md)를 참조하십시오.

---

## llama-server 문제

### 모델이 GPU 대신 CPU에서 로드됨

**증상:** 약 50 tok/s 대신 약 2 tok/s로 생성됩니다. `nvidia-smi`에서 llama-server가 GPU를 사용하지 않는 것으로 표시됩니다.

**해결:** `--n-gpu-layers 99`가 설정되어 있는지 확인하십시오 (모든 레이어를 GPU로 오프로드합니다). Docker Compose에서는 기본값입니다. 베어메탈의 경우 명령을 확인합니다:
```bash
ps aux | grep llama-server | grep 'n-gpu-layers'
```

Docker를 사용하는 경우 NVIDIA 컨테이너 런타임이 설정되어 있는지 확인하십시오 (위의 GPU 섹션 참조).

### 모델 파일을 찾을 수 없음

**증상:** llama-server가 "failed to load model" 또는 유사한 메시지와 함께 즉시 종료됩니다.

**해결:** 모델 경로를 확인하십시오:
```bash
# Docker Compose - 모델이 ATLAS_MODELS_DIR (기본값: ./models/)에 있어야 합니다
ls -la models/Qwen3.5-9B-Q6_K.gguf

# 베어메탈 - ATLAS_MODEL_PATH 확인
ls -la ~/models/Qwen3.5-9B-Q6_K.gguf
```

파일명은 `.env`의 `ATLAS_MODEL_FILE`과 일치해야 합니다 (기본값: `Qwen3.5-9B-Q6_K.gguf`).

### VRAM 부족

**증상:** llama-server가 시작 직후 충돌하거나 OOMKilled됩니다. `nvidia-smi`에서 VRAM 사용량이 거의 100%로 표시됩니다.

**해결:** 9B Q6_K 모델은 약 8.2 GB VRAM이 필요합니다 (모델 + KV 캐시). 다음을 확인하십시오:
1. 다른 GPU 프로세스가 실행 중이지 않은지 (`nvidia-smi` - 다른 CUDA 프로세스 확인)
2. 16GB 이상의 VRAM이 있는지
3. 컨텍스트 크기가 너무 크게 설정되지 않았는지 (기본값 32K가 적절하며, VRAM 확인 없이 늘리지 마십시오)

```bash
# 필요한 경우 다른 GPU 프로세스 종료
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I{} kill {}
```

### 문법이 적용되지 않음 (모델이 사고 블록을 출력함)

**증상:** 모델이 JSON 도구 호출 대신 `<think>` 태그나 일반 텍스트를 출력합니다.

**해결:** 프록시는 `ATLAS_AGENT_LOOP=1`일 때 자동으로 `response_format: {"type": "json_object"}`를 설정합니다. llama-server를 직접 사용하는 경우 요청에 포함하십시오:
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

JSON 대신 일반 텍스트가 반환되면 llama.cpp 빌드가 `response_format`을 지원하지 않는 것입니다. 최신 소스에서 다시 빌드하십시오.

### 컨텍스트 윈도우가 너무 작음

**증상:** 도구 호출 인수가 잘립니다. `write_file`이 "unexpected end of JSON"으로 실패하거나 프록시 로그에 "truncation detected"가 표시됩니다.

**해결:** 컨텍스트 크기는 32768이어야 합니다 (Docker Compose의 기본값). 확인:
```bash
# Docker Compose
grep CTX_SIZE .env

# 베어메탈
ps aux | grep llama-server | grep ctx-size
```

---

## 프록시 문제

### 에이전트 루프가 활성화되지 않음

**증상:** 요청이 llama-server로 직접 전달됩니다. 도구 호출, 스트리밍 상태 아이콘, V3 파이프라인이 없습니다.

**해결:** `ATLAS_AGENT_LOOP=1`을 설정하십시오. `atlas` 런처가 자동으로 설정합니다. 프록시를 수동으로 실행하는 경우:
```bash
ATLAS_AGENT_LOOP=1 atlas-proxy-v2
```

Docker Compose에서는 `docker-compose.yml`에 설정되어 있으므로 수동 설정이 필요하지 않습니다.

### V3 파이프라인이 기능 파일에서 실행되지 않음

**증상:** 모든 `write_file` 호출이 T1 (직접 쓰기)입니다. 출력에 V3 파이프라인 단계가 표시되지 않습니다.

V3는 **세 가지 조건이 모두** 충족될 때만 실행됩니다:
1. 파일에 **50줄 이상**의 콘텐츠가 있을 것
2. 파일에 **3개 이상의 로직 지표**가 있을 것 (함수 정의, 제어 흐름, API 패턴)
3. V3 서비스가 `ATLAS_V3_URL`에서 접근 가능할 것

**진단:**
```bash
# V3 서비스 헬스 확인
curl -s http://localhost:8070/health

# 프록시 로그에서 등급 분류 확인
docker compose logs atlas-proxy | grep "write_file"
# 확인: T1 (직접) vs T2 (V3 파이프라인)
```

V3에 연결할 수 없으면 프록시는 조용히 직접 쓰기로 폴백합니다.

### 잘림 오류 (write_file이 반복적으로 실패)

**증상:** "Your output was truncated - the content is too long for a single tool call."과 같은 오류가 반복됩니다.

**원인:** 모델이 한 번의 호출에 너무 많은 콘텐츠를 작성하려고 합니다. 프록시가 잘린 JSON을 감지하고 도구 호출을 거부합니다.

**자동으로 수행되는 동작:**
- 100줄이 넘는 기존 파일의 경우: 프록시가 `write_file`을 거부하고 모델에게 `edit_file`을 대신 사용하도록 안내합니다
- 3회 연속 실패 후: 오류 루프 차단기가 에이전트를 중지하고 요약을 반환합니다

**사용자가 할 수 있는 조치:** 전체 파일 재작성 대신 대상을 지정한 변경을 요청하도록 문구를 수정하십시오. 예를 들어, "auth.py를 다시 작성해줘" 대신 "로그인 함수에 입력 유효성 검사를 추가해줘"로 요청하십시오.

### 편집 전에 파일을 읽지 않음

**증상:** `edit_file`이 "file not read yet - use read_file first before editing."으로 실패합니다.

**원인:** 프록시가 에이전트가 읽은 파일을 추적합니다. 모델이 이 세션에서 읽지 않은 파일을 편집하려 하면 비활성 보호를 위해 편집이 거부됩니다.

**해결:** 이는 정상적인 동작입니다. 모델이 먼저 파일을 읽어야 합니다. 계속 실패하면 모델이 어떤 파일을 확인했는지 혼동하고 있을 수 있습니다. Aider에서 `/clear`를 입력하고 다시 요청하십시오.

### 외부에서 파일이 수정됨

**증상:** `edit_file`이 "file modified since last read - read it again before editing."으로 실패합니다.

**원인:** 모델이 파일을 읽은 후 디스크에서 파일이 변경되었습니다 (사용자 또는 다른 프로세스에 의해). 프록시가 수정 타임스탬프를 비교합니다.

**해결:** 모델이 파일을 다시 읽어야 합니다. 일반적으로 다음 턴에서 자동으로 해결됩니다.

### 탐색 예산 경고

**증상:** 출력에 "You have full project context in the system prompt. Do not read more files."가 표시되거나 읽기가 건너뛰어집니다.

**원인:** 모델이 아무것도 쓰지 않고 4회 이상 연속으로 읽기 전용 호출(read_file, search_files, list_directory)을 했습니다. 4회 읽기 후 프록시가 경고합니다. 5회 이상이면 읽기를 완전히 건너뛰고 모델에게 쓰기를 지시합니다.

**해결:** 이는 보호 동작입니다. 모델이 진정으로 탐색에 막혀있다면 변경하고 싶은 내용을 더 구체적으로 요청하십시오.

---

## Geometric Lens 문제

### Lens가 로드되지 않음 / 사용 불가

**증상:** 프록시 헬스에서 `"lens": false`가 표시됩니다. 또는 시작 시 "Lens unavailable - verification disabled."가 표시됩니다.

**영향:** ATLAS는 C(x)/G(x) 점수 산출 없이도 동작합니다. V3 후보 선택이 샌드박스 전용 검증으로 폴백합니다.

**해결:** Lens 헬스 및 로그를 확인하십시오:
```bash
curl -s http://localhost:8099/health
docker compose logs geometric-lens
```

일반적인 원인:
- Lens가 llama-server에 연결할 수 없음 (`LLAMA_URL` 환경 변수 확인)
- 모델 가중치 파일 누락 (서비스가 정상적으로 성능 저하됨 - 사용자 정의 모델을 학습하지 않은 경우 이는 예상된 동작입니다)

### 모든 점수가 0.5 부근

**증상:** 코드 품질에 관계없이 모든 후보가 `cx_energy: 0.0` 및 `gx_score: 0.5`를 받습니다.

**원인:** 모델 가중치가 로드되지 않았습니다. 모델이 없을 때 서비스는 중립 기본값을 반환합니다.

**확인:**
```bash
curl -s http://localhost:8099/internal/lens/gx-score \
  -H "Content-Type: application/json" \
  -d '{"text": "print(1)"}' | python3 -m json.tool
```

`enabled: false` 또는 `cx_energy: 0.0`이면 모델이 로드되지 않은 것입니다. 새로 설치한 경우 이는 예상된 동작입니다. 모델 가중치는 저장소에 포함되어 있지 않으며 학습하거나 [HuggingFace](https://huggingface.co/datasets/itigges22/ATLAS)에서 다운로드해야 합니다.

### 임베딩 추출 실패

**증상:** Lens 로그에 "embedding extraction failed" 또는 타임아웃 오류가 표시됩니다.

**원인:** Lens가 llama-server의 `/v1/embeddings` 엔드포인트를 호출합니다. llama-server에 과부하가 걸리거나 해당 엔드포인트가 활성화되지 않으면 실패합니다.

**해결:**
```bash
# 임베딩 엔드포인트 직접 테스트
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test"}' | python3 -m json.tool
```

`/v1/embeddings` 엔드포인트는 생성 모델의 셀프 임베딩을 위해 특별한 플래그 없이 llama.cpp에서 사용 가능합니다. K3s에서는 전체 임베딩 지원을 위해 엔트리포인트에 `--embeddings` 플래그가 명시적으로 설정됩니다.

---

## Sandbox 문제

### Sandbox에 연결할 수 없음

**증상:** 코드가 테스트되지 않습니다. 프록시 헬스에서 `"sandbox": false`가 표시됩니다.

**해결:** 샌드박스 헬스를 확인하십시오:
```bash
# Docker Compose (호스트 포트 30820이 컨테이너 포트 8020에 매핑됨)
curl -s http://localhost:30820/health

# 베어메탈 (포트 8020 직접)
curl -s http://localhost:8020/health
```

샌드박스 컨테이너가 실행 중이지만 비정상인 경우 로그를 확인하십시오:
```bash
docker compose logs sandbox
```

### 코드 실행 타임아웃

**증상:** 샌드박스가 `"error_type": "Timeout"`을 반환합니다. 코드 실행에 너무 오래 걸립니다.

**기본 타임아웃:** 요청당 30초, 최대 60초 (`MAX_EXECUTION_TIME` 환경 변수로 설정 가능).

**해결:** 코드가 정당하게 더 많은 시간이 필요하다면 요청에서 더 높은 타임아웃을 설정하십시오. 코드에 무한 루프가 있는 경우 이는 예상된 동작입니다.

### 지원되지 않는 언어

**증상:** 특정 언어에 대해 샌드박스가 오류를 반환합니다.

**지원 언어:** Python, JavaScript, TypeScript, Go, Rust, C, C++, Bash.

사용 가능한 런타임을 확인합니다:
```bash
curl -s http://localhost:30820/languages | python3 -m json.tool
```

---

## Aider 문제

### `atlas`가 Aider 대신 REPL을 표시함 (파일 읽기/쓰기 불가)

**증상:** `atlas` 실행 시 `Model`, `Speed`, `Lens`, `Sandbox` 상태 블록과 `◆` 프롬프트가 표시되는 내장 REPL이 나타납니다. 요청 입력은 동작하지만 파일이 생성되거나 수정되지 않습니다. `--message` 플래그가 무시됩니다.

**원인:** `atlas` 명령이 프록시와 Aider를 자동 감지합니다. 둘 중 하나라도 없으면 `/solve`와 `/bench`를 지원하지만 파일 작업은 불가능한 내장 REPL로 폴백합니다.

**해결:**
1. 프록시가 실행 중인지 확인: `curl -s http://localhost:8090/health`
2. Aider가 설치되어 있는지 확인: `pip install aider-chat`
3. 서비스가 정상인지 확인: `docker compose ps` (모두 "healthy"로 표시되어야 합니다)

프록시가 정상이고 Aider가 설치되어 있으면 `atlas`가 전체 에이전트 루프(도구 호출, 파일 읽기/쓰기, V3 파이프라인)와 함께 Aider를 자동으로 실행합니다.

Go 1.24 이상이 설치되어 있으면 `atlas`가 프록시를 자동으로 빌드하고 실행할 수 있으므로 수동으로 시작할 필요가 없습니다.

### 프록시가 잘못된 디렉토리 또는 `/tmp`을 표시함

**증상:** 모델이 프로젝트 대신 `/tmp` 또는 ATLAS 저장소의 파일을 나열합니다. `write_file`이 잘못된 위치에 파일을 생성합니다.

**원인:** Docker Compose 프록시는 컨테이너 내부에서 실행되며 시작 시 마운트된 디렉토리만 볼 수 있습니다. 다른 디렉토리에서 작업하는 경우 프록시가 접근할 수 없습니다.

**해결 (권장):** Go 1.24+ ([https://go.dev/dl/](https://go.dev/dl/))를 설치하십시오. `atlas` CLI가 자동으로 현재 디렉토리에서 전체 파일 접근이 가능한 로컬 프록시를 빌드하고 실행합니다. Docker 마운트가 필요하지 않습니다.

**해결 (Go 없이):** `.env`에서 `ATLAS_PROJECT_DIR`을 프로젝트 경로로 설정한 후 프록시를 다시 시작하십시오:
```bash
# .env에서:
ATLAS_PROJECT_DIR=/path/to/your/project

# 새 마운트를 적용하기 위해 프록시 재시작:
docker compose up -d atlas-proxy
```

프로젝트 디렉토리를 전환할 때마다 이를 업데이트하고 다시 시작해야 합니다. 이는 프록시를 Docker 내부에서 실행하는 데 따른 제한 사항입니다.

### 클론 후 `.env.example`이 없음

**증상:** `cp .env.example .env`가 "No such file or directory"로 실패합니다.

**해결:** V3.0.1에서 수정되었습니다. 수정 전에 클론한 경우 최신 버전을 가져오십시오:
```bash
git pull
cp .env.example .env
```

### 긴 작업 중 Aider 연결이 끊어짐

**증상:** 에이전트 루프가 완료되기 전에 Aider가 타임아웃되거나 연결이 끊어집니다. 특히 V3 파이프라인 단계에서 발생합니다.

**해결:** Aider의 HTTP 요청 타임아웃이 V3 파이프라인 실행(수 분이 걸릴 수 있음)에 충분히 길어야 합니다. 저장소의 `.aider.model.settings.yml`이 연결을 유지하는 스트리밍 모드를 설정합니다. 여전히 타임아웃이 발생하는 경우:

1. 저장소의 설정 파일(`.aider.model.settings.yml` 및 `.aider.model.metadata.json`)을 사용하고 있는지 확인하십시오
2. 설정 파일에서 `streaming: true`가 설정되어 있는지 확인하십시오

### 빈 응답

**증상:** Aider가 완료 요약을 표시하지만 파일 콘텐츠가 생성되지 않습니다.

**원인:** 모델이 파일 변경 없이 `done` 신호를 보냈습니다. 다음과 같은 경우에 발생할 수 있습니다:
- 매우 짧은 대화형 프롬프트 ("hi", "thanks")
- 모델이 어떤 파일을 생성해야 할지 모르는 모호한 요청

**해결:** 더 구체적으로 요청하십시오. 모델에게 정확히 어떤 파일을 생성하거나 편집할지 알려주십시오.

### 잘못된 작업 디렉토리

**증상:** 파일이 잘못된 위치에 생성됩니다. `list_directory`가 예상치 못한 내용을 표시합니다.

**원인:** 프록시가 가장 최근에 수정된 `.aider.chat.history.md` 파일을 찾아 프로젝트 디렉토리를 감지합니다. 여러 Aider 세션이 열려 있으면 가장 최신 세션이 우선합니다.

**해결:** 다른 Aider 세션을 닫거나, `atlas`를 실행하기 전에 올바른 프로젝트 디렉토리에서 `cd`를 실행하십시오.

### "Model not found" 오류

**증상:** Aider가 모델 관련 오류로 시작에 실패합니다.

**해결:** ATLAS 루트에 두 Aider 설정 파일이 모두 있는지 확인하십시오:
```bash
ls -la .aider.model.settings.yml .aider.model.metadata.json
```

이 파일들은 저장소에 포함되어 있습니다. 누락된 경우 다시 클론하거나 백업에서 복원하십시오. Aider에게 프록시를 가리키는 `openai/atlas` 모델을 사용하도록 지시하는 파일입니다.

---

## 성능

### 느린 생성 속도 (~2 tok/s)

모델이 GPU 대신 CPU에서 실행되고 있습니다. 다음을 확인하십시오:
1. `nvidia-smi` - llama-server가 GPU 프로세스로 표시되는지
2. `--n-gpu-layers 99` - 모든 레이어가 오프로드되었는지
3. NVIDIA Container Toolkit - 컨테이너 런타임이 GPU 접근용으로 설정되었는지

**예상 성능:** RTX 5060 Ti 16GB에서 문법 적용 시 약 51 tok/s.

### V3 파이프라인이 수 분 소요됨

이는 T2 파일에 대해 정상적인 동작입니다. V3 파이프라인은 여러 번의 LLM 호출을 수행합니다:
- **프로브만 (최상의 경우):** 약 10-15초 (1회 생성 + 1회 점수 산출 + 1회 테스트)
- **Phase 1 생성:** 약 1-2분 (PlanSearch + DivSampling + 점수 산출)
- **Phase 3 수리:** 약 2-5분 (PR-CoT + Refinement + Derivation, 필요한 경우)

더 빠른 (그러나 품질이 낮은) 결과를 원하는 경우:
- 파일을 50줄 미만으로 유지 (T1 유지, V3 미실행)
- 로직 복잡도 줄이기 (함수, 제어 흐름 감소)
- V3는 진정으로 필요한 경우에만 실행됩니다 - 단순 파일은 즉시 작성됩니다

### 높은 RAM 사용량

**증상:** 시스템이 느려지거나 서비스가 OOMKilled됩니다.

**예상 RAM 사용량:**
- llama-server: 약 8 GB (모델은 VRAM에, 최소 RAM 사용)
- geometric-lens: 약 200 MB (PyTorch 런타임 + 모델)
- v3-service: 약 150 MB (PyTorch 런타임)
- sandbox: 약 100 MB (기본, 컴파일 중 급증)
- atlas-proxy: 약 30 MB (Go 바이너리)

**합계:** 약 500 MB RAM + 8.2 GB VRAM. 시스템 RAM이 14 GB 미만이면 다른 서비스와 메모리를 경합할 수 있습니다.

---

## 도움 받기

여기에 나열되지 않은 문제가 있는 경우:
1. 서비스 로그를 확인하십시오: `docker compose logs <service-name>`
2. 프록시 헬스 엔드포인트를 확인하십시오: `curl http://localhost:8090/health`
3. 모든 환경 변수는 [CONFIGURATION.md](../../CONFIGURATION.md)를 참조하십시오
4. [GitHub](https://github.com/itigges22/ATLAS/issues)에서 이슈를 등록하십시오
