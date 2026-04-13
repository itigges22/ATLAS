> **[English](../../../README.md)** | **[简体中文](../zh-CN/README.md)** | **[日本語](../ja/README.md)** | **한국어**

<p align="center">
  <img src="../../images/banner.png" alt="ATLAS 배너"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-V3.0.1-blue" alt="버전"/>
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="라이선스"/>
  <img src="https://img.shields.io/badge/model-Qwen3.5--9B-green" alt="모델"/>
  <img src="https://img.shields.io/badge/GPU-RTX_5060_Ti_16GB-red" alt="GPU"/>
</p>

<h1 align="center">A.T.L.A.S.</h1>
<p align="center"><b>Adaptive Test-time Learning and Autonomous Specialization</b></p>

ATLAS는 지능형 추론 인프라를 기반으로 한 완전 자체 호스팅 코딩 어시스턴트입니다. 동결된 로컬 모델에 제약 기반 생성, 에너지 기반 검증, 자기 검증 수리를 적용합니다. 파인튜닝, API 호출, 클라우드가 필요하지 않습니다. ATLAS는 의미 있는 AI 도구가 프론티어 모델, 클라우드 API, 또는 대규모 예산을 필요로 하지 않는다는 것을 증명하기 위해 존재합니다. 뛰어난 오픈 웨이트 모델을 감싸는 스마트 인프라만 있으면 됩니다.

---

## 최신 소식

- **2026-04-05** - **[V3.0.1 출시](../../../CHANGELOG.md)** - 대화형 CLI, Docker Compose 배포, 95.8% 안정성
- **2026-04-03** - ["$500 GPU Beats Claude: Local AI Revolution for Web Devs"](https://ownet.it/blog/500-gpu-beats-claude-local-ai-revolution-for-web-devs) - ownet.it
- **2026-03-29** - ["A $500 GPU Just Outscored Claude Sonnet on Coding Benchmarks"](https://aivy.com.au/news/atlas-500-gpu-outperforms-claude-sonnet-coding/) - Aivy
- **2026-03-28** - ["Why a $500 GPU Can Beat Claude Sonnet on Coding Benchmarks"](https://medium.com/data-science-collective/why-a-500-gpu-can-beat-claude-sonnet-on-coding-benchmarks-6c8169ffe4fe) - Data Science Collective
- **2026-03-27** - ["ATLAS: A $500 GPU Outperforms Claude Sonnet"](https://clauday.com/article/b92c5551-b490-4d76-ae3d-d8dedf10d88b) - Clauday
- **2026-03-26** - [Hacker News 첫 페이지](https://news.ycombinator.com/item?id=47533297) - 489 포인트, 285 댓글
- **2026-03-05** - **[V3.0 출시](../../reports/V3_ABLATION_STUDY.md)** - 동결된 Qwen3-14B에서 LiveCodeBench pass@1-v(k=3) 74.6% 달성
- **2026-02-18** - **[V2.0 출시](../../../CHANGELOG.md)** - 벤치마크 인프라, HumanEval/MBPP/LiveCodeBench/GPQA/SciCode 평가 모음

---

## ATLAS의 기능

1. **[atlas-proxy](../../ARCHITECTURE.md#3-atlas-proxy-outer-layer)** - 전체 시스템을 조율하는 Go 기반 에이전트 루프입니다.
  - a. [도구 호출 라우팅](../../ARCHITECTURE.md#tools) - 파일 작업을 복잡도 등급별로 분류합니다
  - b. [문법 강제](../../ARCHITECTURE.md#grammar-enforcement) - GBNF 스키마가 100% 유효한 JSON 출력을 보장합니다
  - c. [안전 제한](../../ARCHITECTURE.md#safety-limits) - 턴 제한, 토큰 예산, 타임아웃 적용

2. **[V3 파이프라인](../../ARCHITECTURE.md#4-v3-pipeline-inner-layer)** - 단일 프롬프트를 검증된 고품질 출력으로 변환하는 다단계 코드 생성입니다.
  - a. [PlanSearch](../../reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - 제약 기반 구조화된 계획 수립
  - b. [DivSampling](../../reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - 온도와 전략을 달리한 다양한 후보 생성
  - c. [Budget Forcing](../../reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - 단계별 사고 토큰 할당 제어
  - d. [PR-CoT Repair](../../reports/V3_ABLATION_STUDY.md#pr-cot-repair-36-rescues) - 자체 생성 테스트 케이스를 활용한 반복 수정 사이클
  - e. [Refinement Loops](../../reports/V3_ABLATION_STUDY.md#refinement-loop-6-rescues) - 반복적인 샌드박스 검증 및 수정
  - f. [Derivation Chains](../../reports/V3_ABLATION_STUDY.md#derivation-chains-0-rescues) - 복잡한 문제를 위한 다단계 추론

3. **[Geometric Lens](../../ARCHITECTURE.md#5-geometric-lens)** - 외부 오라클 없이 에너지 기반 점수 산출 및 검색을 수행합니다. (["Geometric Lens"란?](../../ARCHITECTURE.md#why-geometric-lens))
  - a. [C(x) Cost Field](../../ARCHITECTURE.md#scoring-models) - 임베딩에서 후보 품질을 평가하는 MLP
  - b. [G(x) Quality Prediction](../../ARCHITECTURE.md#scoring-models) - 선택 결정을 위한 XGBoost 모델
  - c. [RAG / PageIndex V2](../../ARCHITECTURE.md#rag--pageindex-v2) - AST 인식 코드 검색 및 프로젝트 인덱싱
  - d. [Confidence Router](../../ARCHITECTURE.md#confidence-router--pattern-cache) - Thompson Sampling으로 중요한 곳에 연산을 집중합니다

4. **[Sandbox](../../ARCHITECTURE.md#6-sandbox)** - 빌드 검증을 위한 격리된 실행 환경입니다.
  - a. 다중 언어 실행 - Python, Rust, Go, C, Shell 등
  - b. 컴파일 및 린팅 - 점수 산출 전 구문 검증
  - c. 테스트 실행 - 생성된 테스트 및 기존 테스트 스위트 실행

5. **[llama-server](../../CONFIGURATION.md#6-llama-server)** - 단일 소비자용 GPU에서의 로컬 LLM 추론입니다.
  - a. CUDA 가속 - 양자화된 모델 추론 (Q6_K / Q4_K_M)
  - b. 문법 제약 디코딩 - 토큰 수준의 구조화된 출력
  - c. 셀프 임베딩 - 별도 모델 없이 임베딩 추출

6. **[대화형 CLI](../../CLI.md)** - 프로젝트 디렉토리에서 `atlas`를 입력하면 바로 개발을 시작할 수 있습니다.
  - a. [도구 호출 에이전트 루프](../../CLI.md#streaming-output) - 읽기, 쓰기, 편집, 삭제, 명령 실행
  - b. [스트리밍 출력](../../CLI.md#how-streaming-works) - SSE를 통한 실시간 응답
  - c. [프로젝트 인식 컨텍스트](../../CLI.md#proxy-file-access) - 자동 파일 탐색 및 주입

전체 문서 - 설정 가이드, 아키텍처, 설정, 문제 해결, 벤치마크 보고서 - 는 [docs/](../../) 디렉토리에 있습니다.

---

## 시작하기

ATLAS는 16GB 이상의 VRAM을 갖춘 GPU, Docker(nvidia-container-toolkit 포함) 또는 Podman, 그리고 Python 3.9 이상이 필요합니다. 현재 NVIDIA GPU에서 테스트되었습니다. ATLAS는 NVIDIA에 종속되지 않으며, AMD GPU를 위한 ROCm 지원이 로드맵에 포함되어 있습니다. Docker Compose, 베어메탈, K3s 배포를 다루는 전체 설치 안내는 **[SETUP.md](../ko/SETUP.md)**를 참조하십시오. 실행 후 프로젝트 디렉토리에서 `atlas`를 입력하면 바로 개발을 시작할 수 있습니다.

---

## 알려진 제한 사항

- **NVIDIA에서만 테스트됨** - ATLAS는 추론에 llama.cpp를 사용하며, 다양한 가속기 백엔드를 지원합니다. ROCm 지원은 V3.1의 우선 과제입니다.
- **9B 모델 공식 벤치마크 미실시** - CLI는 전체 V3 파이프라인과 함께 Qwen3.5-9B를 제공하지만, 공식 LiveCodeBench 점수는 14B 모델 기준입니다. 9B 벤치마크는 V3.1 작업 항목입니다.
- **복잡한 기능 추가 시 실패 가능** - 기존 프로젝트에 기능을 추가하는 작업은 약 67%의 성공률을 보입니다. 모델이 코드를 작성하는 대신 탐색에 과도하게 시간을 소비하는 경우가 있습니다.
- **문법 제약 추론 속도** - llama-server에서 약 51 tok/s입니다. 더 빠른 문법 통합이 V3.1에서 계획되어 있습니다.

---

## 로드맵

**V3.0.1** - 현재 릴리스. 대화형 CLI, Docker Compose 배포, V3 파이프라인 통합.

**V3.1** - 진행 중.
- ROCm 지원 - llama.cpp ROCm 백엔드를 통한 AMD GPU 추론
- 공식 9B 벤치마크 - Qwen3.5-9B에서 LiveCodeBench, GPQA Diamond, SciCode 평가
- CLI 안정성 - 확장된 테스트, L6 90% 이상 목표
- 문법 속도 - 더 빠른 제약 디코딩을 위한 C 사이드 샘플러 체인

---

## 기여하기

ATLAS는 오픈으로 개발되고 있으며, 기여자와 핵심 메인테이너를 적극적으로 찾고 있습니다. 버그 수정, 가속기 지원 추가, 하위 시스템 전면 재설계 등 어떤 형태의 기여든 환영합니다. 오픈 모델이 더 나은 인프라를 갖추어야 한다고 생각하신다면, 함께 만들어 가시기 바랍니다.

가이드라인은 **[CONTRIBUTING.md](../../../CONTRIBUTING.md)**를 참조하십시오.

---

## 라이선스

[GNU Affero General Public License v3.0 (AGPL-3.0)](../../../LICENSE)에 따라 라이선스가 부여됩니다.
