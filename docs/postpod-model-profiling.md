> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# Pod再開後のモデル検証・プロファイル取得準備

再開時は[現行の配布設定・再開手順](prepod-final-integration.md)も参照する。以下の検証件数・版番号は当時の履歴。配布設定例は現行契約に更新済み。

> 本文はv0.7の履歴を含む。v0.8では[全重み検査・rate別時間上限・3分割の実観測評価](prepod-readiness-1-2-3.md)を追加した。下記の固定900秒上限、S1重み未検査factory、2分割については最新仕様を参照。

2026年10月28日、v0.7.0.dev1。S1（GR00T）とS2・Safety・Monitor（Qwen2.5-VL）を、既存のRay component worker経由で1成分ずつ検証・計測するスクリプトを追加した。今回はMacのCPUで検証しており、Pod起動・接続・課金操作、学習済みモデルのGPU推論は行っていない。

## 追加したもの

- [postpod_models.py](../scripts/postpod_models.py): template生成、観測の分割、preflight、試行準備・実行・再開、再集計、出力比較、負荷レートの候補、profile候補生成。
- [7つの設定template](../examples/postpod/): S1用1つ、S2/Safety/MonitorのTransformers・vLLM用各1つ。
- [profile_protocol.py](../src/rosa_repro/profile_protocol.py): 入力契約、観測の整合性、単一成分の計測計画、環境情報、GPU telemetry。
- [profile_runner.py](../src/rosa_repro/profile_runner.py): bounded負荷生成、期限・失敗・未送信を含むログ、監査可能なprofile候補。
- [verify_postpod_cpu.py](../scripts/verify_postpod_cpu.py): 実Ray上での再実行用CPU検証。

計測用workerを起動するために、未取得のR07 profileを作る必要はない。`unprofiled-probe`として起動し、実モデルfactory、セッション・応答ID検査、GPU割当、S1バッチ、VLM非同期処理は既存の実行経路を使う。S1のactionも保存するだけで、シミュレータやロボットへ適用しない。

## 再開後の順序

| 段階 | 目的 | 出力・次の判断 |
| --- | --- | --- |
| preflight | 依存版・モデルファイル・ソースの照合 | モデルをロードせず結果を保存。GPUロード成功の証明ではない |
| smoke | 固定観測で動作・出力・parserを確認 | 応答原文、意味出力、S1 action配列。判断品質は別途確認 |
| pilot | 指定stream数で閉ループの負荷探索 | 実測完了レートと10〜100%の負荷候補。最大能力の証明ではない |
| profile | 明示したレートでopen-loop計測 | 平均・p99・全要求の期限達成率、失敗・未送信、環境情報 |
| candidate | 複数反復を照合してR07形式へ整理 | 条件不足なら拒否。成立しても確認が必要な候補として保存 |

GPUのpilot/profileには、同じ設定・corpus・ソースで成功したsmokeの証拠を要求する。これは型・接続の確認であり、モデル判断の正しさを自動認定するものではない。S1の公式経路との数値照合とLIBERO成功率は、従来の[R02/T01手順](r02.md)を併用する。

## 設定と観測

`examples/postpod/s1-groot.json`、`s2-transformers.json`、`s2-vllm.json`などをコピーし、実際のGPU機種名・モデル絶対パス・GR00T repoを設定する。`SET_`や`/SET/`が残るtemplateは実行できない。S1では従来のR02検証に基づいてruntime fingerprintとstateless確認を設定する。

VLMは[v0.6の固定model/revision/processor/生成条件](vlm-backends.md)を維持する。backend設定を変えたら`runtime_fingerprint(config)`を再計算し、別条件として保存する。TransformersとvLLMのprofileを混ぜない。

S1のbatch size、batch wait、stream数は条件ごとに固定する。VLMは入場batch 1・wait 0msで、内部同時数はbackend configに記録する。`payload_overlay`には必要なS2 planや実行文脈を明示できる。例：

```json
{"plan":"保存観測に対応した次のsubgoal",
 "context":{"execution_history":"保存時点の実行履歴"}}
```

overlayは既定で空であり、存在しない履歴を生成しない。設定と入力ハッシュに含める。実P2〜P4のplan・履歴の長さや画像分布を代表する入力かどうかは、profile採用前に確認する。この計測器は現在の保存観測replayであり、閉ループの環境変化は再現しない。

既存T01の`observation.json`＋`arrays.npz`を使う。`index-observations`はtaskと初期状態のhashでgroupを作り、同じ状態の両arm・全frameを同じsplitに保つ。profile用とevaluation用はgroup単位で分離し、最低2 groupを要求する。smokeはevaluation、pilot/profileはprofile splitを使う。精度の正式評価やprompt調整には、さらに未使用のholdoutを確保する。

パスはcapture rootからの相対パスなので、corpus一式をPodへコピーできる。manifest hash、NPZ hash、復元した観測のcontent hash、group IDを実行前に再検査する。CPU合成観測をGPUの実観測として扱わない。

## 計測の定義

すべての比較時刻は同一driverのmonotonic ns。サーバー内時間は別欄に置く。画像・言語入力のhash、応答ID、生成原文とparser結果、S1 action配列を保存する。応答時刻はactionファイル保存より前に取得する。ログI/Oが負荷生成へ及ぼす影響はdispatch lagに現れる。

- smoke/pilotはstreamごとに前の応答を待つ。action待機・物理実行はない。
- profileは一定間隔の予定到着を作る。送信遅延上限を超えたslotや空きstreamがないslotは`not_sent`として残し、遅れた要求をまとめて送らない。
- warmupとmeasurementを分け、間で実処理をdrainする。計測終了後の応答も、予定時刻が窓内なら同じ要求コホートへ含める。
- timeoutは失敗として確定し、遅い成功に書き換えない。そのstreamをphase終了まで再利用せず、workerのbusy解除も待つ。drain不能なら試行を失敗にする。
- 平均/p99は成功応答だけの値と明記する。失敗・timeout・未送信数、全予定要求を分母とした期限達成率を併記する。
- `successful_response_cohort_rate_hz`は窓内到着の成功数÷窓長で、窓後の完了も含む。pilotの負荷候補には、窓内に実際に完了した要求数÷窓長を使う。
- 再監査では予定slotの欠落、重複ID、phase別完了数、原本のhashと集計の一致を確認する。

探索protocolはwarmup最低5秒・5成功（上限30秒）、計測最低10秒・10成功（上限30秒）。formal protocolはwarmup最低30秒・20成功（上限180秒）、計測最低120秒・1000成功（上限900秒）。各phaseには要求件数上限もある。低レートでは900秒でも1000件に届かない場合があり、その場合はshortfallのまま保存して候補生成を拒否する。必要なら計測protocolを改訂し、新しいsuite・ソース記録で実行する。時間を自動延長しない。formalという名称は計測条件を示し、未確認のkernelや判断精度を保証しない。

試行の時間上限にはモデルロード、Rayの配置待ち、起動・終了処理は含まない。`run --max-new`は既定1試行。30条件を一度に実行する既定動作にはしていない。停止・中断は新attemptに記録し、成功済みattemptを上書きしない。再開時に設定・ソース・corpus・driver依存情報が変わっていれば拒否する。

## GPU情報とprofile候補

worker環境のPython・全package版、backend attestation、GPUリース・機種・UUID・総メモリを保存する。GPU telemetryはUUIDを指定した読取専用の[nvidia-smi query](https://docs.nvidia.com/deploy/nvidia-smi/index.html)で、driver、総メモリ、使用メモリ、GPU使用率を約1秒ごとに採取する。取得不能は明示し、推定値で補わない。

使用メモリはデバイス全体の定期サンプルで、vLLM子プロセスや他の処理を含み、サンプル間のpeakは保証しない。候補profileの`memory_mib`は**1 workerがGPU全体を専有する予約量**として総メモリを採る。モデルの実peak使用量を計測した値にはしない。既存の[Ray GPU資源指定](https://docs.ray.io/en/latest/serve/resource-allocation.html)も1 worker＝1 GPUを維持する。

候補生成では、GPU実行、同一設定/corpus/source/worker環境、formal protocol、drain完了、全要求成功、欠落なし、dispatch lag上限、実モデルファイル検査、GPU telemetry、各rateに3反復以上を要求する。同じtrialや同じrate/repの二重投入は拒否する。

成立時はrateごとに反復間で最大の平均遅延・p99を採り、sample数と時間は合計する。平均・p99は予定到着から応答までを用いる。測定していないrate=0、単調化、外挿、`capacity_bound_hz`を追加しない。CPU・shortfall・失敗試行からはprofile本体を作らない。

候補は常に`automatic_scheduler_import_allowed=false`。実際のdtype/kernel/compile、背景負荷、queue安定性、SLO、入力分布、判断品質を確認してからR07へ取り込む。既存GR00T factoryは重み全shardの同一性をまだ証明していないため、現状ではS1候補に`model_artifact_verification_incomplete`が残る。S1の計測自体は可能だが、正式採用にはこの確認を補う必要がある。

## コマンド例

project rootから、必要な依存を入れた専用Pythonで実行する。モデル環境はGR00TとVLMを分ける。worker Pythonを分ける場合はbindingへ`runtime_env.py_executable`を設定し、[前版の環境条件](vlm-backends.md)を満たす。VLM suiteのdriverは入力検査にPillowも使うため、RayだけのCPU用環境ではなく、RayとPillowを含むVLM用環境を使用する。GPU環境の新規依存解決・インストールは今回実行していない。

```bash
# 実際のcapture rootでcorpus作成。配布templateをコピーして実パス等を解決する
PYTHONPATH=src <python> scripts/postpod_models.py index-observations \
  --root /workspace/captures --output results/corpus.json

# workerと同じPython環境で、ローカルファイル・依存を照合
PYTHONPATH=src <worker-python> scripts/postpod_models.py preflight \
  --config /workspace/configs/safety-vllm.json --output results/safety-preflight.json

# まずsmoke suiteを生成。prepareはモデルをロードしない
PYTHONPATH=src <python> scripts/postpod_models.py prepare \
  --config /workspace/configs/safety-vllm.json --corpus results/corpus.json \
  --stage smoke --output results/safety-smoke-suite.json

# 既存Ray clusterに接続して、1試行だけ実行
PYTHONPATH=src <serve-python> scripts/postpod_models.py run \
  --suite results/safety-smoke-suite.json --corpus-root /workspace/captures \
  --address auto --max-new 1 --output results/safety-smoke
```

`--address auto`はPod上に既に意図したRay clusterがある場合の例。実環境に合わせて明示する。このスクリプトはPodやGPU clusterを作成しない。GPU runの後は自身のServeアプリを削除してdriverを切断し、共有cluster全体を停止しない。

smokeの`<smoke-raw>`は、出力された`case_id/attempt-001/raw`ディレクトリを指す。生成原文・画像hash・Safety未知判定・Monitor報告を確認してから進む。

```bash
PYTHONPATH=src <python> scripts/postpod_models.py prepare \
  --config /workspace/configs/safety-vllm.json --corpus results/corpus.json \
  --stage pilot --output results/safety-pilot-suite.json
PYTHONPATH=src <serve-python> scripts/postpod_models.py run \
  --suite results/safety-pilot-suite.json --corpus-root /workspace/captures \
  --address auto --smoke-evidence <smoke-raw> --output results/safety-pilot
PYTHONPATH=src <python> scripts/postpod_models.py pilot-rates \
  <pilot-raw> --output results/safety-rate-hints.json
```

pilotの10〜100%は[ROSA §3.4.2](https://arxiv.org/html/2607.01088v1)の負荷grid準備に使えるが、1回のpilotを最大throughputの証明にしない。stream数・時間を変え、飽和と失敗率を確認して採用レートを選ぶ。下記`<r1> ... <r10>`には、その検討で選んだ実数値を入れる。

```bash
PYTHONPATH=src <python> scripts/postpod_models.py prepare \
  --config /workspace/configs/safety-vllm.json --corpus results/corpus.json \
  --stage profile --phase formal --rates <r1> <r2> <r3> <r4> <r5> <r6> <r7> <r8> <r9> <r10> \
  --repetitions 3 --output results/safety-profile-suite.json
PYTHONPATH=src <serve-python> scripts/postpod_models.py run \
  --suite results/safety-profile-suite.json --corpus-root /workspace/captures \
  --address auto --smoke-evidence <smoke-raw> --max-new 1 --output results/safety-profile
# 同じコマンドで次の未完了試行へ進む。失敗attemptは保持して再試行する
PYTHONPATH=src <python> scripts/postpod_models.py reaudit \
  <profile-raw> --output results/recomputed.json
PYTHONPATH=src <python> scripts/postpod_models.py profile-candidate \
  <profile-raw-rep0> <profile-raw-rep1> <profile-raw-rep2> --output results/profile-candidate.json
```

複数rateをまとめる際は各rateの全反復rawを渡す。`compare <left-raw> <right-raw> --output ...`では、同じ観測への出力hash集合・ばらつきを比較できる。異engine間の意味的等価性やGR00Tの確率的出力の正しさを認定する処理ではない。

## 検証結果と限界

最終結果と配布物のhashはrelease_verification.json（非同梱の履歴資料: `../results/prepod-profiling-20261028/release_verification.json`）に記録する。CPU回帰試験は**275件合格、失敗0・skip 0**（前版243件＋32件）。JUnit記録（非同梱の履歴資料: `../results/prepod-profiling-20261028/pytest-release.xml`）を保存した。試験は、template未解決・revision違い・観測改変・split漏洩・負荷未指定・timeout・過負荷・件数不足・証拠改変・重複反復・未送信欠落を検査する。exporterの正常系も、明示的なテスト用GPUメタデータで確認する。このfixtureは実GPU測定結果として配布しない。

最終CPU Ray検証（非同梱の履歴資料: `../results/prepod-profiling-20261028/ray-cpu-release/verification.json`）ではsmoke→pilot→profile、S1実batch 4、Monitor非同期処理、途中再開と完了済みskip、出力hash比較、再監査、CPU profile取り込み拒否が合格した。CPU試験はモデルの判断品質やGPU能力を評価していない。

```bash
PYTHONPATH=src <vlm-cpu-python> -m pytest -q
PYTHONPATH=src <serve-python> scripts/verify_postpod_cpu.py \
  --output results/postpod-cpu-new
```

既存P1〜P4の回帰（非同梱の履歴資料: `../results/prepod-profiling-20261028/fleet-regression-release/verification.json`）も合格した。4 worker、S1 52要求→batch 4×13回、S2 2回、P1〜P4各8適格仮想チャンク、生ログ再集計の一致を確認した。開発途中の失敗と修正は履歴（非同梱の履歴資料: `../results/prepod-profiling-20261028/development_history.json`）に残し、最終証拠と区別する。GPUロード・CUDA/vLLM kernel・実VRAM peak・実モデルの出力品質、R03正式profile、R07実GPUスケジュール、LIBERO成功率はPod再開後に残る。今回の追加だけではROSA性能再現の完了とはしない。
