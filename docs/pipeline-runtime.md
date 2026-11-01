> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# P2〜P4ログ収集とR07実行接続

> 本文はv0.5時点の記録。v0.6で[S2・Safety・Monitor backend](vlm-backends.md)、非同期生成要求、生成ログ、worker別Python指定を追加した。最新の検証範囲はそちらを参照。

2026年10月28日、v0.5.0.dev1。P2〜P4評価器への実行時ログ接続と、R07の配置・頻度を複数Ray Serve workerへ反映する経路を実装した。Mac上のCPUダミーモデルで実際に4つのworkerプロセスを起動し、要求送信から保存ログの再集計まで検証した。Podへは接続していない。

## 検証結果

最終Ray結合試験（非同梱の履歴資料: `../results/prepod-runtime-20261028/ray-verified-001/verification.json`）とJUnit（非同梱の履歴資料: `../results/prepod-runtime-20261028/pytest-junit.xml`）を保存した。回帰試験は**194件合格**、既存165件＋追加29件、失敗・skipともに0。

| 項目 | 実測したCPU接続試験の結果 |
| --- | --- |
| 実行基盤 | Ray 2.59.0、4つの別workerプロセス、GPU割当0 |
| ワークロード | P1〜P4各1 robot、S1上限各2Hz、行動時間0.2秒の仮想待機 |
| ログ | S1 52要求、S2 2要求、Safety 26要求、Monitor 12要求。合計92要求を保存 |
| S1バッチ | 52要求 → 実バッチ4件×13回のモデル呼出し |
| 測定区間 | warmup 2.1秒、測定4.2秒。P1〜P4それぞれ8適格チャンク、合計32 |
| S2の依存 | P4で13回のS1に対しS2を2回実行。plan本文とrequest IDをS1へ渡した |
| 頻度制御 | 全robotでS1送信間隔が1/f以上であることを生ログから確認 |
| 再集計 | JSONL生ログとfinal auditの一致を検査し、metricsを同一に再生成 |
| 異常系 | 応答の値・ID対応、端数バッチ、重複・誤成分の拒否、推論中reset、別robotの継続、timeout後のbusy保持と回復 |
| 終了 | 検証用のServeアプリとローカルRayを終了 |

この32チャンクはCPUモデルと仮想待機による接続確認で、LIBEROや実ロボットの成功数ではない。Q=32/4.2=7.619は短い人工試行の集計値であり、ROSA論文の性能再現値ではない。`gpu-N`というworker IDもR07が付けた論理名で、この試験にGPUはない。

## ログ収集と制御

`pipeline_runtime.py`は1つのdriverの単調時計で全要求・行動を記録する。周期・S2・S1を別workerへ送るが、時刻の比較にサーバー時計を混ぜない。サーバー内のqueue待ち・モデル時間は別ファイルへ残す。

- S1はクラス別fと「前の要求・行動の完了」の両方を守る。遅れを取り戻す一括送信はしない。f=0でも必要な周期モデルは動く。
- Safety/Monitorはrun epochからの固定周期で送る。前の応答待ち等で周期が飛んだ場合は飛ばしたslotをlifecycleへ記録し、評価器が未送信枠を`missing`として数える。遅く送った要求の予定時刻を現在時刻へ付け替えない。
- S2は最初のS1前とH回消費後に呼ぶ。失敗時はplanなしのS1を送らず、新しい要求IDで再試行する。S1失敗や未適用でもplan使用数を消費する。S2が有効出力を返しても期限を外れた場合、機能上は利用できるが評価器では不適格になる。
- S2のplanはS1観測の言語欄に、元の指示＋改行＋`Plan: 本文`として渡す。元の観測は変更しない。このプロンプト結合は本プロジェクトの補完仕様であり、公式ROSAの未公開コードとの一致は未確認。
- Safetyの停止判断、未取得・失敗・予定時刻基準の期限切れは、**次のチャンク開始を抑止**する。停止判断は`executed_steps=0`のaction記録として残す。実行中チャンクへの割込みや実機の非常停止機能ではない。
- client timeoutは`timeout`として完了記録を残す。遅れて終わった出力を行動へ適用しない。workerは実際のモデル処理が終わるまでsessionをbusyに保ち、その間の追加要求を`rejected`として返す。
- 各応答についてplan ID、worker ID、robot/episode/request、世代、worker instance ID、batch証拠と出力型を照合する。違う応答やworker再起動はrun失敗にする。resetで古い出力を無効にし、他robotの状態は維持する。

現行のdriverは1 runにつき各robot 1 episodeを実行する。worker側のepisode resetは試験済みだが、連続する実シミュレーションのepisode切替・状態復元は別の接続工程である。

### 保存物

| ファイル | 内容 |
| --- | --- |
| `execution_plan.json` | 入力fleet、R07の証拠付き配置、role割当、factory設定、派生経路、planハッシュ |
| `protocol.json` | 時間、timeout、要求数上限、行動の意味、観測の出所 |
| `source_manifest.json` | 実行に使ったPythonモジュールのSHA256 |
| `events.jsonl` | v2 request/action。各レコードをflushして保存 |
| `request_details.jsonl` | worker、入力・出力ハッシュ、S2/Safety/Monitorの意味出力、サーバー内時間 |
| `lifecycle.jsonl` | run開始、送信前intent、行動前intent、周期skip、run終了/失敗 |
| `workers_before.json` / `workers_after.json` | instance、PID、GPUリース、モデル呼出し・実バッチ数、backend宣言、ソースハッシュ |
| `audit.json` / `metrics.json` | 完了した監査コンテキストと評価結果 |
| `result.json` | 成否、要求状態別件数、drain状態、集計の要約 |

`events.jsonl`だけから、ログ欠落のない完了runだと推測しない。プロセスの強制終了・executor例外ではintentを残し、実行済みステップ数が不明なactionを補作しない。`audit.json`がなければ再集計を拒否する。正常な再集計でも生JSONLとaudit内のrecordsが一致しなければ拒否する。

`complete_with_request_failures`は失敗要求も含めて試行と集計を完了した状態で、全要求成功を意味しない。接続試験には正式測定の1000件等の合格基準を適用しておらず、常に`formal_performance_claim=false`を残す。

## R07から複数workerへの接続

`execution_plan.py`はR07の`feasible`出力だけを受け付け、既存の独立検査で全配置・route・Fmin・SLO・memory・閉ループ条件を再確認する。各robotはclass IDとlocal robot IDの衝突しない複合IDへ変換する。汎用の`obligation`は明示role mapでSafety/Monitorへ割り当てる。

派生した計画を実行直前に再生成し、入力・plan ID・頻度・route・GPU資源・batch設定の変更を検出する。R07のprofileで指定したbatch sizeと待機時間をworkerへ設定する。同じモデル重みでも、別component IDは別workerのまま維持する。

`fleet_service.py`は1 worker＝1 Serve deployment＝1 replicaを構成する。CPUモードは`num_gpus=0`、GPUモードは各workerへ`num_gpus=1`を明示する。[Rayの資源指定](https://docs.ray.io/en/latest/serve/resource-allocation.html)と[動的バッチAPI](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html)を使い、同じworkerで複数モデルバッチを同時実行しない。

GPUモードでは、空きGPU数の不足を開始前に拒否し、CPUへ自動的に切り替えない。worker内でCUDA可視数1、RayのGPUリース1、GPU機種名とprofileの機種名の一致、総メモリが必要量以上であることを確認してからモデルを読む。開始後もworkerの重複ハンドル、同じnode/GPUリースへの二重配置、driverとworkerのソース不一致を拒否する。

このGPUコードの検査条件はCPUの模擬inventoryで試験した。実CUDAデバイス、実VRAM使用量、複数GPUでの転送・推論・SLOは未検証である。実メモリの空き量を予約する仕組みではなく、モデルロードのOOMは起動失敗になる。

### backendの接続契約

各componentへ`{"factory":"package.module:factory", "kwargs":{...}}`を指定する。factoryはworker起動時に一度だけ呼ばれ、`describe()`と`predict_batch(payloads)`を持つモデルを返す。payloadは観測、plan本文、plan request IDを含む。返却数は入力batch数と同じで、順序も維持する。

| role | 1要求分の出力 |
| --- | --- |
| s1 | `action.*` → float32の`[1, horizon, dim]`配列 |
| s2 | `{"plan":"非空のplan本文"}` |
| safety | `{"stop":true/false,"reason":"非空の説明"}` |
| monitor | `{"text":"非空の報告"}` |

backendはstatelessであることを明示する。状態を持つモデルをrobotごとに無検証で共有しない。`describe()`には成分、role、モデルrevision、runtime fingerprintと実行種別を含め、計画と照合する。これはbackendの宣言との一致確認であり、重みの全ファイルやkernel動作の証明ではない。

S1用には`model_backends:groot_backend`を追加し、固定GR00T public APIへの委譲、観測の結合・actionの分割、既存の上流commit/config検査を接続した。`repo`、`model_path`、`instruction_key`、`stateless_verified`を明示する。CUDAローカル番号はRayが割り当てた可視デバイスの`cuda:0`を使う。こちらの実モデル経路はまだPod上で実行していない。

**S2・Safety・Monitorの実モデルfactoryは未接続**。今回動かしたのは型と依存を再現するCPUダミーである。モデル・revision・プロンプト・画像入力・生成設定・出力parserを固定して、この契約へ接続する必要がある。dummyをGPUとして実行する計画は拒否する。R07に渡す実測profileも、対応するbackend/runtimeで取得し直す。

## 再実行手順

Python 3.12を使用する。compilerとCPU単体試験は`requirements.lock`の環境、Ray試験は`requirements-serve-cpu.lock`の環境を使う。保存されたR07出力から計画を作るだけなら、Serve環境へOR-Toolsを追加する必要はない。今回この分離環境でもCLI生成結果が一致することを確認した。

```bash
# project root。出力先は未使用の名前を指定
PYTHONPATH=src <serve-venv>/bin/python scripts/pipeline_fleet.py compile \
  --fleet examples/runtime_mix4_fleet.json \
  --schedule examples/runtime_mix4_schedule.json \
  --roles examples/runtime_roles.json \
  --bindings examples/runtime_cpu_bindings.json \
  --output results/runtime-plan-new.json

# Mac上の新しいCPU専用Ray。外部clusterやPodに接続しない
PYTHONPATH=src <serve-venv>/bin/python scripts/verify_fleet_cpu.py \
  --plan results/runtime-plan-new.json --output results/ray-fleet-new

# 保存イベントから再集計
PYTHONPATH=src <cpu-venv>/bin/python scripts/pipeline_fleet.py reaudit \
  results/ray-fleet-new/pipeline --output results/reaudited-new.json

PYTHONPATH=src <cpu-venv>/bin/python -m pytest -q
```

`pipeline_fleet.py run`でも通常のbounded replayを実行できる。GPUモードは既存Ray clusterの`--address`と`--observation-manifest`を必須とし、Pod作成・起動・課金は行わない。manifestは`{"observations":[{"directory":"保存観測フォルダ","sha256":"digest_valueによる観測ハッシュ"}]}`形式で、既存のNPZ＋JSON保存観測を読む。実モデルのCLI runも仮想待機によるreplayであり、LIBERO成功率を測る実行器ではない。

本物の環境へ進む際はPython APIの`observation_source(robot_id, component_sequence)`と非同期`execute(robot_id, action, action_time_s)`を接続する。観測は各要求時点の状態を返すようにし、executorは実際の実行ステップ数を返す。`action_semantics='external_executor_reported_steps'`を明示する。既存T01の状態復元・終了条件・動画との統合は後続で検証する。

## 再開時の順序と未検証範囲

1. Pod再開後、固定モデル・上流ソース・dtype等を確認し、従来のR02出力照合とT01比較を行う。
2. S1のGR00T factoryとGPU batch 1/2/4等を検証する。S2/Safety/Monitorの実モデルfactoryを接続して意味出力とplan注入を確認する。
3. 各backendの実測profileを取得し、実際のGPU機種名・runtime fingerprintへ更新してR07計画を再計算する。人工profileは実測へ昇格しない。
4. GPU modeでresource inventoryを確認し、短いP2→P3→P4 replayを実行する。保存v2ログからQ/SLOを再集計し、R07予測との違いを調べる。その後に正式条件へ進む。

今回の接続は静的な配置・route・頻度である。動的な再配置、admission/failover、実行途中の計画差替え、実機の連続安全制御、正式なROSA性能再現は未完了。R07のヒューリスティック下限と全列挙の区別も維持する。

`ray-mix4-001`は開発途中の正常経路試験で、最終証拠は`ray-verified-001`。先行するv0.3/v0.4の結果はそれぞれの時点の履歴として保持している。最終Ray結果のソースハッシュは配布するv0.5ソースと一致する。
