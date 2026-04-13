> **[English](../../../README.md)** | **[简体中文](../zh-CN/README.md)** | **日本語** | **[한국어](../ko/README.md)**

<p align="center">
  <img src="../../images/banner.png" alt="ATLAS バナー"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-V3.0.1-blue" alt="Version"/>
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License"/>
  <img src="https://img.shields.io/badge/model-Qwen3.5--9B-green" alt="Model"/>
  <img src="https://img.shields.io/badge/GPU-RTX_5060_Ti_16GB-red" alt="GPU"/>
</p>

<h1 align="center">A.T.L.A.S.</h1>
<p align="center"><b>Adaptive Test-time Learning and Autonomous Specialization</b></p>

ATLAS は、インテリジェントな推論インフラストラクチャを基盤とした、完全セルフホスト型のコーディングアシスタントです。凍結されたローカルモデルを、制約駆動生成、エネルギーベース検証、自己検証修復で包み込みます -- ファインチューニングなし、API 呼び出しなし、クラウドなし。ATLAS は、意味のある AI ツールにフロンティアモデル、クラウド API、巨額の予算は不要であることを証明するために存在します -- 優れたオープンウェイトモデルを取り巻くスマートなインフラストラクチャがあれば十分です。

---

## 最新ニュース

- **2026-04-05** - **[V3.0.1 リリース](../../../CHANGELOG.md)** - インタラクティブ CLI、Docker Compose デプロイ、95.8% の信頼性
- **2026-04-03** - ["$500 GPU Beats Claude: Local AI Revolution for Web Devs"](https://ownet.it/blog/500-gpu-beats-claude-local-ai-revolution-for-web-devs) - ownet.it
- **2026-03-29** - ["A $500 GPU Just Outscored Claude Sonnet on Coding Benchmarks"](https://aivy.com.au/news/atlas-500-gpu-outperforms-claude-sonnet-coding/) - Aivy
- **2026-03-28** - ["Why a $500 GPU Can Beat Claude Sonnet on Coding Benchmarks"](https://medium.com/data-science-collective/why-a-500-gpu-can-beat-claude-sonnet-on-coding-benchmarks-6c8169ffe4fe) - Data Science Collective
- **2026-03-27** - ["ATLAS: A $500 GPU Outperforms Claude Sonnet"](https://clauday.com/article/b92c5551-b490-4d76-ae3d-d8dedf10d88b) - Clauday
- **2026-03-26** - [Hacker News フロントページ](https://news.ycombinator.com/item?id=47533297) - 489 ポイント、285 コメント
- **2026-03-05** - **[V3.0 リリース](../../reports/V3_ABLATION_STUDY.md)** - 凍結された Qwen3-14B で LiveCodeBench pass@1-v(k=3) 74.6%
- **2026-02-18** - **[V2.0 リリース](../../../CHANGELOG.md)** - ベンチマークインフラ、HumanEval/MBPP/LiveCodeBench/GPQA/SciCode 評価スイート

---

## ATLAS の機能

1. **[atlas-proxy](../../ARCHITECTURE.md#3-atlas-proxy-outer-layer)** - システム全体をオーケストレーションする Go ベースのエージェントループ。
  - a. [ツールコールルーティング](../../ARCHITECTURE.md#tools) - ファイル操作を複雑度ティアで分類します
  - b. [文法強制](../../ARCHITECTURE.md#grammar-enforcement) - GBNF スキーマにより 100% 有効な JSON 出力を保証します
  - c. [安全制限](../../ARCHITECTURE.md#safety-limits) - ターン上限、トークン予算、タイムアウト強制

2. **[V3 パイプライン](../../ARCHITECTURE.md#4-v3-pipeline-inner-layer)** - 単一のプロンプトを検証済みの高品質な出力に変換するマルチフェーズコード生成。
  - a. [PlanSearch](../../reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - 制約駆動の構造化プランニング
  - b. [DivSampling](../../reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - 温度と戦略をまたぐ多様な候補生成
  - c. [Budget Forcing](../../reports/V3_ABLATION_STUDY.md#phase-1-constraint-driven-generation-124pp) - フェーズごとの思考トークン割り当てを制御
  - d. [PR-CoT Repair](../../reports/V3_ABLATION_STUDY.md#pr-cot-repair-36-rescues) - 反復修正サイクルのための自己生成テストケース
  - e. [Refinement Loops](../../reports/V3_ABLATION_STUDY.md#refinement-loop-6-rescues) - サンドボックス検証と修正の繰り返し
  - f. [Derivation Chains](../../reports/V3_ABLATION_STUDY.md#derivation-chains-0-rescues) - 複雑な問題に対するマルチステップ推論

3. **[Geometric Lens](../../ARCHITECTURE.md#5-geometric-lens)** - 外部オラクルを使わないエネルギーベースのスコアリングと検索。([「Geometric Lens」とは?](../../ARCHITECTURE.md#why-geometric-lens))
  - a. [C(x) Cost Field](../../ARCHITECTURE.md#scoring-models) - エンベディングから候補品質をスコアリングする MLP
  - b. [G(x) Quality Prediction](../../ARCHITECTURE.md#scoring-models) - 選択判断のための XGBoost モデル
  - c. [RAG / PageIndex V2](../../ARCHITECTURE.md#rag--pageindex-v2) - AST 対応のコード検索とプロジェクトインデキシング
  - d. [Confidence Router](../../ARCHITECTURE.md#confidence-router--pattern-cache) - Thompson Sampling により重要な箇所に計算リソースを配分

4. **[Sandbox](../../ARCHITECTURE.md#6-sandbox)** - ビルド検証のための分離された実行環境。
  - a. マルチ言語実行 - Python、Rust、Go、C、Shell など
  - b. コンパイルとリント - スコアリング前の構文検証
  - c. テスト実行 - 生成されたテストスイートと既存のテストスイートを実行

5. **[llama-server](../../CONFIGURATION.md#6-llama-server)** - 単一のコンシューマ GPU でのローカル LLM 推論。
  - a. CUDA アクセラレーション - 量子化モデル推論 (Q6_K / Q4_K_M)
  - b. 文法制約デコーディング - トークンレベルでの構造化出力
  - c. セルフエンベディング - 別途モデルを用意せずにエンベディング抽出

6. **[インタラクティブ CLI](../../CLI.md)** - 任意のプロジェクトディレクトリで `atlas` と入力すれば開発を始められます。
  - a. [ツールコールエージェントループ](../../CLI.md#streaming-output) - 読み取り、書き込み、編集、削除、コマンド実行
  - b. [ストリーミング出力](../../CLI.md#how-streaming-works) - SSE によるリアルタイムレスポンス
  - c. [プロジェクト対応コンテキスト](../../CLI.md#proxy-file-access) - 自動ファイル検出とインジェクション

完全なドキュメント -- セットアップガイド、アーキテクチャ、設定、トラブルシューティング、ベンチマークレポート -- は [docs/](../../) ディレクトリにあります。

---

## はじめに

ATLAS には、16GB 以上の VRAM を持つ GPU、Docker (nvidia-container-toolkit 付き) または Podman、Python 3.9 以上が必要です。現在 NVIDIA GPU でテスト済みです -- ATLAS は NVIDIA 専用ではなく、AMD GPU 向けの ROCm サポートはロードマップに含まれています。Docker Compose、ベアメタル、K3s デプロイを網羅した完全なインストール手順は **[SETUP.md](../ja/SETUP.md)** をご覧ください。起動後、任意のプロジェクトディレクトリで `atlas` と入力すれば開発を始められます。

---

## 既知の制限事項

- **NVIDIA のみでテスト済み** - ATLAS は推論に llama.cpp を使用しており、複数のアクセラレータバックエンドをサポートしています。ROCm サポートは V3.1 の優先事項です。
- **9B モデルは正式にベンチマーク未実施** - CLI は完全な V3 パイプラインを備えた Qwen3.5-9B を同梱していますが、LiveCodeBench の正式スコアは 14B モデルのものです。9B ベンチマークは V3.1 の作業です。
- **複雑な機能追加は失敗する場合がある** - 既存プロジェクトへの機能追加は約 67% の確率で成功します。モデルがコードを書く代わりに探索しすぎることがあります。
- **文法制約推論の速度** - llama-server で約 51 tok/s です。より高速な文法統合は V3.1 で予定されています。

---

## ロードマップ

**V3.0.1** - 現在のリリース。インタラクティブ CLI、Docker Compose デプロイ、V3 パイプライン統合。

**V3.1** - 開発中。
- ROCm サポート - llama.cpp ROCm バックエンドによる AMD GPU 推論
- 正式な 9B ベンチマーク - Qwen3.5-9B での LiveCodeBench、GPQA Diamond、SciCode
- CLI の信頼性 - テスト拡充、L6 >= 90% を目標
- 文法速度 - より高速な制約デコーディングのための C サイドサンプラーチェーン

---

## コントリビュート

ATLAS はオープンに開発されており、コントリビューターとコアメンテナーを積極的に募集しています。バグ修正、アクセラレータサポートの追加、サブシステム全体の再設計 -- どのような貢献でも歓迎します。オープンモデルにはより良いインフラストラクチャが必要だと考える方は、ぜひ一緒に開発しましょう。

ガイドラインは **[CONTRIBUTING.md](../../../CONTRIBUTING.md)** をご覧ください。

---

## ライセンス

[GNU Affero General Public License v3.0 (AGPL-3.0)](../../../LICENSE) の下でライセンスされています。
