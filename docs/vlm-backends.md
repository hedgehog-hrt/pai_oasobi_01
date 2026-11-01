> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# S2・Safety・Monitorの実モデルbackend接続

> v0.7で[Pod再開後のモデル検証・プロファイル取得スクリプト](postpod-model-profiling.md)を追加した。本文の検証件数と結果はv0.6時点の履歴。

2026年10月28日、v0.6.0.dev1。Qwen2.5-VL用のTransformers参照backendとvLLM非同期backendを追加した。公式processor、ランダム初期化した小型Qwenモデル、実RayのCPU workerで接続を検証した。Podへの接続、学習済み7B/3B重みのロード、GPU推論は今回行っていない。

## 論文との対応と固定値

[ROSA v1 §4.1とFigure 3](https://arxiv.org/html/2607.01088v1)のモデル構成に対応させる。下表の公開checkpoint revisionは本プロジェクトで選んだ固定値であり、論文の実験checkpointと同一であることは未確認。プロンプト、カメラ選択、画像サイズ、生成設定、JSON出力形式も補完仕様である。

| 成分 | 公開モデル | 固定revision |
| --- | --- | --- |
| S2 | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Safety | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Monitor | S2と同じ7Bモデル | 同上。成分・worker・プロンプトは別 |

設定は[Transformers用](../examples/vlm_transformers_configs.json)と[vLLM用](../examples/vlm_vllm_configs.json)に保存した。`vlm_contract.py`の設定検査でrole・model・revision・生成条件を照合し、全設定のハッシュをruntime fingerprintに含める。fingerprint単体はCUDA/kernel/依存関係全体の同一性を保証しない。

## 入出力とログ

`video.image`と`video.wrist_image`の2視点を、記載順で受け取る。各入力はuint8 RGBの`[1,T,H,W,3]`とし、最後のフレームをコピーして使う。float画像、CHW画像、曖昧なbatch、欠落した視点は拒否する。RGB/BGR変換、正規化、クロップを暗黙に追加しない。

その後のリサイズ・画像token化は[公式Qwen2.5-VL processor](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)に委譲する。`use_fast=False`、最小224²・最大448²画素を明示した。画像入力の上限と文脈長も検査し、無言の切捨ては行わない。既定の最大文脈は4096 tokens、出力は128 tokens、float16、seed 42。生成温度0相当のgreedy出力を用いる。

タスク指示、現在のS2 plan、実行済みチャンク・ステップ数を明示的な文脈として渡す。仮想待機の実行履歴は`virtual_timed_chunks`と記す。モデルの報告をLIBEROの真値へ置き換えない。

| 成分 | モデルに要求するJSON | ランナーへの出力 |
| --- | --- | --- |
| S2 | `{"plan":"次のsubgoal"}` | plan本文。既存のH回再利用・S1言語入力注入へ接続 |
| Safety | `{"decision":"safe/unsafe/unknown","reason":"理由"}` | safeだけ`stop=false`、unsafe/unknownは`stop=true` |
| Monitor | `{"status":"ongoing/done/failed/unknown","reason":"理由"}` | `text`と`task_status`。終了ラベルだけではepisodeを終えない |

JSON以外の文章、コードフェンス、追加キー、重複キー、誤型、空文字、未知ラベルを拒否する。終了理由が`length`や中断なら、有効そうなJSONでも受理しない。モデルの自然文を検索して「safeを含むから許可」と判定する経路はない。

`request_details.jsonl`の`backend_evidence`へ、モデル・revision・設定ハッシュ、各画像の選択フレームとハッシュ、プロンプト・入力ハッシュ、生成原文、終了理由、token数、parser結果を保存する。不正出力も原文付きの失敗要求として残す。vLLM timeoutはengine要求をabortし、部分出力を行動に使わない。入力検査やロード段階の例外は一般の失敗記録となるため、生成原文が存在しない場合がある。

Safetyは既存ランナーの**次チャンク開始ゲート**へ接続する。未取得・失敗・期限切れも開始を抑止する。実行中チャンクへの割込みや実機の非常停止はこの実装の範囲外。

## 2種類の実行backend

| 項目 | Transformers | vLLM |
| --- | --- | --- |
| factory | `rosa_repro.vlm_backends:transformers_vlm_backend` | `rosa_repro.vlm_backends:vllm_vlm_backend` |
| 用途 | 直列の出力照合用参照実装 | autoregressiveモデルの継続的な要求処理 |
| 処理 | `Qwen2_5_VLForConditionalGeneration.generate` | vLLM V1 `AsyncLLM.generate` |
| 同時要求 | 1 | 既定16。workerとbackendで上限を適用 |
| 入場batch | 1、待機0ms | 1、待機0ms。token単位batchはengineが構成 |
| GPU | 1 workerにつきRayが1 GPUを割当 | 同左。TP=1、PP=1 |
| timeout | Transformersの`max_time`。硬い中断保証ではない | `asyncio.timeout`とengine abort |

論文のautoregressiveモデルのcontinuous batchingへ向け、vLLMを直接非同期接続する経路を用意した。Transformers直列経路をcontinuous batchingの再現結果として扱わない。vLLMの外側で静的Serveバッチを作らず、個別要求をengineへ渡す。S1の既存テンソルバッチ経路は維持した。

vLLMは0.11.2 APIを対象に、prefix cacheとmultimodal processor cacheを無効化し、chunked prefillを無効、eager実行を指定した。CUDA kernelはvLLMの選択に委ねるので、FlashAttention等が有効だという宣言は行わない。`actual_batch_size=1`や`model_calls`は入場・backend呼出しの計数で、vLLM内部のtoken batch数ではない。内部batchの計測は未接続である。

[固定vLLMソース](https://github.com/vllm-project/vllm/tree/v0.11.2)を参照し、19個のEngineArgs名、AsyncLLM呼出し形式、Qwen processorの`use_fast`指定を静的確認した。これは実engineのロード成功を保証する試験ではない。確認記録（非同梱の履歴資料: `../results/prepod-vlm-20261028/official-source-api-check.json`）に参照URLとSHA256を記録した。

## モデルと環境の検査

[モデルpin一覧](../src/rosa_repro/model_pins.json)にはprocessor/configと全safetensors shardのサイズ・SHA256を保存した。GPU factory起動時に全対象ファイルを読み、固定hashと照合してからロードする。ディレクトリ名だけをrevisionの証拠にしない。今回実ファイルを検証したのはprocessor/configのみで、重みhashはHugging Face公式メタデータから取得した期待値である。

ロードはlocal-only、`trust_remote_code=False`。モデル不足や別revisionの場合は起動を拒否する。Transformers 4.57.6、Torch 2.9.0、Pillow 12.0.0、vLLM経路ではvLLM 0.11.2を検査する。CUDA build suffix、GPU名、TorchのCUDA版、設定、checkpoint検査結果もworker inventoryへ残す。HF経路では実パラメータのdtypeも記録する。

- [requirements-vlm-cpu.lock](../requirements-vlm-cpu.lock): Mac ARM64/Python 3.12.14で検証した全依存のfreeze。
- [requirements-vlm-gpu.in](../requirements-vlm-gpu.in): Linux/CUDA環境用の主要依存の候補。解決済みlockではなく、対象GPU imageで依存解決・起動確認・全依存freezeが必要。
- GR00Tの既存環境には混在インストールしない。bindingに`runtime_env.py_executable`を指定してworkerごとの環境を分けられる。RayのこのAPIは[experimental機能](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html)で、head/workerのRay版とPython版を揃え、全割当nodeに実行環境とソースを配置する。

今回のRay試験は実在するPythonパスがworkerに反映されることを確認した。異なるGR00T/VLM依存環境を実GPU上で同時運転した確認ではない。GPU向け依存候補と既存GR00T環境との通信互換性も再開後に確認する。

## CPU検証結果

配布検証記録（非同梱の履歴資料: `../results/prepod-vlm-20261028/release_verification.json`）にソース・依存・証拠ファイルのhashをまとめた。

| 試験 | 結果と限界 |
| --- | --- |
| pytest（非同梱の履歴資料: `../results/prepod-vlm-20261028/pytest-release.xml`） | **243件合格、失敗0、skip 0**。前回194件に49件追加 |
| 公式processorと小型CPUモデル（非同梱の履歴資料: `../results/prepod-vlm-20261028/processor-and-tiny-cpu-final/verification.json`） | S2/Safety/Monitorの2視点入力、chat template、画像token化が成功。224×224の2画像は128画像tokens。4,940,224パラメータのランダム小型Qwenで実際にgenerateを通し、不正出力を拒否 |
| vLLM protocol模擬試験 | engineを模擬し、同時要求上限、ID分離、timeout abort、誤ID・打切り出力拒否を確認。実vLLM kernelは未実行 |
| 実Ray非同期試験（非同梱の履歴資料: `../results/prepod-vlm-20261028/async-ray-cpu-final/verification.json`） | CPU 4 worker。3要求を最大2件同時処理し、ID・生成ログ・個別batch・Python指定を確認。出力は明示的な合成データ |
| 既存経路の実Ray回帰（非同梱の履歴資料: `../results/prepod-vlm-20261028/fleet-regression-cpu/verification.json`） | 4別プロセス、S1 52要求を13回のbatch 4で実行。S2 2回。P1〜P4各8適格仮想チャンク。生ログ再集計が一致 |

CPUの小型モデルは学習済み7B/3Bの代用品として精度を評価したものではない。実モデルのJSON遵守率、Safety/Monitorの判断品質、S2 planの有効性、p99/SLO、VRAM、複数GPU性能は未測定。

開発履歴の`async-ray-cpu`は当初のServe API指定エラーによる失敗記録であり、その後修正した。`async-ray-cpu-v2`と`processor-and-tiny-cpu`は途中版の成功記録。今回の最終証拠は上表の`*-final`と`fleet-regression-cpu`で、配布ソースとのhash一致を検査している。試験用Rayは終了済み。

## 再実行とPod再開後の手順

以下はproject rootから実行する。`<vlm-cpu-python>`はCPU用lockを入れた専用Python、`<serve-python>`は既存の`requirements-serve-cpu.lock`環境を指す。出力先は未使用の名前にする。

```bash
# CPU用環境を作る例
uv venv --python 3.12 .venv-vlm-cpu
uv pip install --python .venv-vlm-cpu/bin/python -r requirements-vlm-cpu.lock

# 公開processor/configだけを取得。学習済み重みはダウンロードしない
PYTHONPATH=src <vlm-cpu-python> scripts/vlm_backend.py download-processors \
  --model 3b --output work/processors-3b
PYTHONPATH=src <vlm-cpu-python> scripts/vlm_backend.py download-processors \
  --model 7b --output work/processors-7b
PYTHONPATH=src <vlm-cpu-python> scripts/verify_vlm_cpu.py \
  --snapshot-3b work/processors-3b --snapshot-7b work/processors-7b \
  --output results/vlm-processor-new
PYTHONPATH=src <vlm-cpu-python> -m pytest -q

# CPUの実Ray。2つのRay検証スクリプトは順番に実行する
PYTHONPATH=src <serve-python> scripts/verify_async_vlm_serve_cpu.py \
  --plan examples/runtime_mix4_plan.json --output results/vlm-async-new
PYTHONPATH=src <serve-python> scripts/verify_fleet_cpu.py \
  --plan examples/runtime_mix4_plan.json --output results/vlm-regression-new
```

Pod上での準備・実行は次の順に行う。

1. 既存GR00TのR02/T01基準を保ち、新しいVLM専用環境を構築する。GPU/driver/CUDAとLinux依存解決結果を保存する。
2. 上表のrevisionでモデル全ファイルを用意し、`vlm_backend.py verify-snapshot --model 3b/7b --directory ... --output ...`で全hashを確認する。`--processors-only`は重み検証の代わりにはならない。
3. Transformers参照経路から保存観測で実モデルの生成・parser受理・Safety未知判定・Monitorラベル・S2注入を確認する。次に同じ観測でvLLM単独要求と同時要求を確認する。両engineの生成が完全一致するとは仮定しない。
4. 固定backend・画像処理・生成長・同時数ごとに実profileを取る。R07のcomponentのmodel_revision/runtime_fingerprint、hardware、profileを実測条件に更新して配置を再計算する。VLMは入場batch 1・待機0msのみ。人工profileの名称だけを実測へ付け替えない。
5. 実測に対応したbindingsでGPU計画をcompileし、inventory確認後にP2→P3→P4の短いreplayを行う。ログからSLO/Qを再集計する。その後にLIBEROのepisode lifecycle・成功率評価を接続する。

bindings生成例（パスは対象Podに実際に配置したものへ変更）：

```bash
PYTHONPATH=src <vlm-python> scripts/vlm_backend.py bindings --engine vllm \
  --snapshot-3b /workspace/models/qwen25vl-3b \
  --snapshot-7b /workspace/models/qwen25vl-7b \
  --python-executable /opt/rosa-vlm/bin/python \
  --output results/vlm-bindings-new.json
```

[生成済みtemplate](../examples/vlm_vllm_bindings_template.json)は上記3成分のみを含む。実fleetのcomponent IDへ対応付け、既存S1 bindingを加えた全成分のbindingsを`pipeline_fleet.py compile --mode gpu`へ渡す。モデルサイズが同じでもS2とMonitorの設定・profileを混同しない。`runtime_fingerprint(config)`の値がcomponent宣言と異なる場合は起動が失敗する。

現行R07は実行可能な下限を返すヒューリスティック。今回のbackend追加も、正式なROSA性能再現、最適性、実機の安全性を証明するものではない。
