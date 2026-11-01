> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# 閉ループ実験とS1 action証跡 — v0.11.0.dev1

2026-10-29。P1〜P4/R07の固定時間窓による実験ランナーと、S1の比較元・比較先action配列を保存する回帰経路を追加した。CPU検証の成果であり、GR00T・Qwen・LIBEROの実モデル実験、複数GPU性能、ROSA論文値の再現はPod再開後に検証する。

## 1. P1〜P4/R07の閉ループ実験

`scripts/pipeline_experiments.py`が、検査済みR07 planのworker配置・robot経路・頻度を使用する。robotごとに独立したシミュレータプロセスを持ち、reset→観測→S1推論→action適用→更新観測を繰り返す。GPU経路は固定GR00T/LIBEROの既存bridgeを使い、S1が実際にロードしたmodality設定を検査して受け取る。シミュレータPythonをdriver/workerと分離して指定できる。

- S2のplanは同じepisodeだけで利用し、S1呼出し回数でHを消費する。episode切替時にplan、Safety判定、worker sessionをresetする。
- Safety/Monitorはtrial全体の時計で周期起動する。遅れたslotは追い付くための連射をせず、欠測を残す。Safety未取得・期限切れ・stop判定時には次のaction chunkを抑止する。chunk途中の緊急停止を実装したという意味ではない。
- 環境が報告した実step数、成功・終了、最大step数を使用する。Monitorの文章から環境成功を補作しない。
- 観測保存、IPC、reset、action適用の時間を含む固定wall時間で測る。action chunkはplanで宣言した時間以上にpacingする。モデルだけのprofileとは別の測定条件である。
- 各初期・適用後観測、適用action配列、要求の観測参照・plan・context、worker/process情報、telemetry、失敗・timeout・終了理由を保存する。保存I/Oやシミュレータ描画による競合も測定に含む。

### 測定条件と終了

formalはGPUのみで、warmup最低30秒かつ各active class/componentで成功20件、計測最低120秒かつ成功1,000件、3反復以上を要求する。本プロジェクトの事前固定条件であり、非公開の論文評価コードと同一だとは主張しない。既定時間は最も低い公称要求レートから余裕を含めて算出する。S2が低レートの場合には長時間になるため、生成したsuiteの時間予算を確認する。

試行中の件数に応じた時間延長はしない。不足は`measurement_shortfall`となり、正式条件を満たした扱いにしない。各componentの成功数とp99の探索的表示、要求失敗を残す。requests・episode数・保存量・periodic-slot数の上限を事前検査し、実行時に上限へ達した場合も失敗として止める。上限を超える計画は条件・規模を見直して別suiteにする。

state bankの順序をrobot番号と反復番号で回転し、元のseedを保持して循環使用する。計測中に開始した**全episode**を成功率の分母に含め、cutoffまでに完了しないepisode、warmupからまたがるepisode、同じ状態の再利用を分けて報告する。再利用episodeを独立標本とする信頼区間は作らない。

### Pod再開後の準備と実行例

まず既存の[実モデル検証・profile取得](postpod-model-profiling.md)、[実観測・品質評価](prepod-readiness-1-2-3.md)、[profile採用ゲート](profile-adoption-workflow-report.md)を通す。実GPU inventoryと一致するplan、全workerの採用済みprofile、実LIBERO state bankが必要である。サンプルのCPU人工profileをGPU計画に流用しない。

`banks.json`はactive class IDから、`--root`に対するstate bankディレクトリへの対応を書く。例えばclass IDがc0〜c3で同じT01を使う場合は以下。

```json
{"c0":"banks/t01","c1":"banks/t01","c2":"banks/t01","c3":"banks/t01"}
```

各classに別taskを与える場合にはtask別bankを指定する。bankは既存`observations.py export-states`で作り、内容・seed・task・SHA256をsuiteへ固定する。

```bash
# 配布ソースのルートから実行。<...>は実在するパス/値へ置き換える。
PYTHONPATH=src <driver-python> scripts/pipeline_experiments.py prepare \
  --plan <gpu-plan.json> --banks <banks.json> --root <experiment-input-root> \
  --phase formal --repetitions 3 --setup-budget 600 --output <suite.json>

# 既定は計画表示だけ。Ray接続やモデル起動は行わない。
PYTHONPATH=src <driver-python> scripts/pipeline_experiments.py run \
  --suite <suite.json> --root <experiment-input-root> --output <run-dir> \
  --sim-python <absolute-libero-python> --sim-repo /workspace/Isaac-GR00T \
  --address <existing-ray-address> --adoptions <fleet-adoptions.json> --report <dry-run.json>

# suiteの全trial予算以上を明示し、まず1反復ずつ実行する。
PYTHONPATH=src <driver-python> scripts/pipeline_experiments.py run \
  --suite <suite.json> --root <experiment-input-root> --output <run-dir> \
  --sim-python <absolute-libero-python> --sim-repo /workspace/Isaac-GR00T \
  --address <existing-ray-address> --adoptions <fleet-adoptions.json> \
  --execute --max-new 1 --budget-seconds <wall-budget> --report <execution-001.json>
```

同じsuite/input/Python/run-dirで、未使用のreport名を指定して続行する。完了した反復は原本を再検査してskipする。予算不足なら次の反復を開始しない。GPUのtimeout/crashやcleanup未確認の試行は、記録に結び付いたcleanup確認が済むまで再実行しない。shared clusterの停止やPodの課金状態変更は行わない。

Ray workerにも同じ配布ソースとbackend依存環境が必要である。既存clusterの認証設定を維持し、各backendの`runtime_env`や配置パスを事前確認する。CPU試験のローカルRay起動とは区別する。シミュレータIPCはdriverと同一ホストの子プロセス間に限定される。

### 原本からの再検証

```bash
PYTHONPATH=src <python> scripts/pipeline_experiments.py reaudit \
  <run-dir>/closed-loop-000/attempt-001/trial/raw --output <recomputed.json>
```

イベントと要求journal、観測/contextの内容hash、適用配列とmodel出力hash、episode/step集計、固定時間窓、P1〜P4適格行動、sample不足を再計算・照合する。比較レポートはこの検査を通した結果を使用し、欠測・失敗反復を残す。R07予測objectiveと実測適格行動率は別の列で報告する。hashは対応・改変検出用であり、測定者の申告を認証する仕組みではない。

## 2. S1の数値回帰とbatch比較

更新版`r02_server.py`の待機中の同一モデルに対し、`r02_compare.py`が公式`get_action`と、GR00T backendも利用するテンソル結合・共有推論・分割経路を比較する。batch 1/2/4等を別runとして固定し、各入力とreference/candidateの**float32 action NPZ**、差分、seed、許容差、source/runtime証跡を保持する。前版のsummaryのみの証跡は新バージョンのS1採用条件を満たさない。

```bash
# 更新版serverはdocs/r02.mdの固定環境・全重み検証付き起動手順を使用。
<libero-python> scripts/r02_compare.py --repo /workspace/Isaac-GR00T \
  --observations <saved-observations> --batch-size 1 --seed 42 \
  --atol 0 --rtol 0 --port 5556 --output <s1-b1-result>
<libero-python> scripts/r02_compare.py --repo /workspace/Isaac-GR00T \
  --observations <saved-observations> --batch-size 4 --seed 42 \
  --atol 0 --rtol 0 --port 5556 --output <s1-b4-result>

# server/GPU不要。NumPyとパッケージの基本依存で保存配列を再比較。
<python> scripts/r02_compare.py --reaudit <s1-b4-result> --batch-size 4 \
  --output <s1-b4-recomputed.json>
```

観測数はbatch sizeで割り切れる必要がある。末尾の破棄やpaddingはしない。batch 1は同じseed・shapeで比較する。batch >1ではserial側のsampleごとのseedとbatched側のseed/shapeを記録するが、diffusion noiseのsample対応を保証できないため`rng_alignment=unverified_across_batch_shapes`とする。Python/NumPy/Torchのdefault RNG状態は各呼出し後に復元する。

不一致なら`mismatch`、途中通信失敗なら`failed`として成功済みcaseと配列を残す。オフライン再検証は保存済みcaseを再計算し、不一致・部分失敗の終了コードも2にする。許容差を自動で広げない。batch比較で数値一致しても、分布・意味品質・task successの同等性を証明した扱いにはしない。

S1 profile採用時は、再計算可能なbatch 1回帰が必要である。候補のbatch size >1ならquality requestに`batch_regression`ディレクトリも指定し、同じモデル・runtimeの該当batch比較を要求する。既存のLIBERO paired、backend/corpus association、人のレビューも引き続き必要である。観測・モデル条件を変更した場合には別runとして取得し直す。

## CPU検証と配布物

- 全テスト結果（非同梱の履歴資料: `../results/prepod-closed-loop-actions-20261029/pytest-release.xml`）
- 配布版でのCPU閉ループ・S1証跡検証（非同梱の履歴資料: `../results/prepod-closed-loop-actions-20261029/cpu-live-release/verification.json`）
- 欠測を残した比較レポート（非同梱の履歴資料: `../results/prepod-closed-loop-actions-20261029/cpu-live-release/comparison-report.md`）
- 配布ソース・証跡の最終検証（非同梱の履歴資料: `../results/prepod-closed-loop-actions-20261029/release_verification.json`）
- Pod配置用ソースZIP（非同梱の履歴資料: `../results/prepod-closed-loop-actions-20261029/source-release.zip`）

実行方法は`verify_closed_loop_cpu.py --driver-python <Ray Python> --sim-python <NumPy Python> --output <new-dir>`。起動側PythonにはTorch/NumPy、Ray側にはCPU Serve依存が必要である。CPU Ray 4 worker、独立simulator 4 process、P1〜P4の2反復、budget停止、再開skip、観測更新・episode切替、raw再検証、batch 1/2/4の配列比較を確認する。数値比較部分のtransportはCPU内の模擬clientであり、実GR00T/ZeroMQ/GPUの実行証拠ではない。

残るのはPod上での固定環境・重み・実観測照合、GPU kernel/通信/バッチの実測、LIBERO実taskとS2/Safety/Monitor品質確認、profile採用、正式条件での反復実験である。本ランナーの固定窓・状態再利用・計測I/Oを含む実験設計を、論文比較の条件差として明記する。
