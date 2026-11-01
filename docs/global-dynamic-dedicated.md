# 全体再配置・動的P2〜P4・専用GPU実行器（v0.17.0.dev1）

v0.16の有限pool/standby制御を残し、以下を順に追加した。実行コードとCPU検証であり、Pod、CUDA、GR00T/Qwenの実測結果ではない。結果原本・テスト件数は `results/prepod-global-dynamic-dedicated-20261029/SUMMARY.md` を参照する。

## 1. 全体の自動再配置

`global_planner.py` は固定カタログのクラス・profile・backend bindingと、現在のrobot集合・GPU予算からR07を再計算する。OR-Toolsは明示したPythonの子プロセスで時間制限付き実行。新規クラスや未測定profileを自動作成せず、得られた配置・頻度・全robot経路を独立検査する。結果は実行可能な下限であり最適解を保証しない。最小頻度は参加中のクラスに適用する。

`Supervisor` はjoin/leave/予算変更を直列化し、並行joinの取りこぼしを防ぐ。順序は「候補計算・検査 → 新規要求停止 → 進行中の要求と行動をdrain → 旧世代の全actor削除確認 → 新世代起動・session reset」。移行中のGPU二重予約は行わない。物理シミュレータの状態は保持し、S2/Safety cacheは破棄する。旧世代のticketは行動を実行できない。

候補不成立の通常joinは旧配置を維持する。新配置起動失敗時は、cleanupを確認でき、現在の予算に収まる旧配置に限りrollbackする。行動完了や資源所有が不明なら停止を保持する。`global_health.watch` は所有actorの応答・identityを検査し、異常時に同じ全体再計画を起動する。復旧回数に上限がある。物理GPU故障の診断・ブラックリスト化は行わず、GPU減数には明示した予算変更を使う。

これはcold replacementであり、model load時間も停止時間に含める。Pod外へ自動拡張せず、Podの起動・課金・停止は行わない。

削除済みactorへのRPCが完了しない場合、障害復旧では世代を無効化した後にモデルRPCのみキャンセルできる。物理行動の完了確認を省略するものではなく、旧actorの削除確認が取れるまで新世代を起動しない。この経路も実CPU Rayのactor削除試験で確認する。

## 2. 動的P2〜P4評価

`dynamic_runner.py` が状態を更新するシミュレータ、S1・S2、周期Safety/Monitor要求を駆動する。全世代共通の周期clockと事前固定した測定窓を使う。S1は送信時点のS2 request IDを保持し、S2 horizonは推論呼び出し回数で消費する。過去周期のcatch-up burstや、再配置後の旧周期への遡及送信をしない。

`global_audit.evaluate` が各世代のrawを再監査し、既存P1〜P4評価器へ変換する。P2はS1+Monitor、P3はS1+Safety+Monitor、P4は4成分。参加済robotは移行中にも周期要求を負う。停止区間を空episodeとして残し、未送信slotと壁時計時間を分母から削除しない。参加はcommit時点で受理し、参加前・離脱後はそのrobotへの需要に数えない。

`dynamic_evidence.reaudit` は送信観測・S2 plan文字列・contextのhash、適用したaction NPZ・simulatorの実行step数・時刻を制御ログと再照合する。行動完了不明や早期停止は `incomplete` / `metrics=null` とする。0スループットへの置換や測定窓の事後短縮はしない。保存容量・要求数・simulator呼び出し・全体wall timeを制限する。

動的試行はrobotごとに1つの保存初期状態を復元し、その物理episodeを再配置後も継続する。terminal後は別taskを自動開始しない。task成功率や論文と同等のGPU性能を主張しない。動的窓は診断用であり、定常profileのformal warmup/sample採用を代替しない。

## 3. 専用GPU実行器

`dedicated_execution.compile_plan` が従来の資源計画から全robotを含む実行計画を生成する。GPU不足なら拒否し、subsetに縮小しない。S1は `frequency_hz=null`、batch=1、batch wait=0。

| 方式 | 物理予約単位 | モデルの所有 |
| --- | --- | --- |
| gpu_per_robot | robotごとにGPU 1枚 / actor 1個 | 全モデルを同じprocessへロード |
| gpu_per_model_per_robot | robot×モデルごとにGPU 1枚 / actor 1個 | 個別モデルを個別processへロード |

同居groupのmodel呼び出しは1つのFIFO lockで直列化する。client timeout後もbackend処理が完了するまではlock/sessionを解放しない。GPU leaseとPIDがgroup内で同一、group間で別、logical model instanceはすべて別であることをinventoryから検証する。通常の1 model / 1 GPU実行器へ誤って流す経路は拒否する。

同居groupには共通Python環境が必要。GR00TとvLLMの環境が異なる場合、そのまま同居させない。vLLMの宣言メモリ予約合計がGPU全体以上になる構成も拒否する。設定を勝手に縮めず、共通環境・明示したメモリ配分で個別profileを再取得・採用する。

単独model profileのメモリ値の和を同居実測と扱わない。GPU同居は、個別profile採用後に `--residency-calibration` を明示して3回以上のformal閉ループを測定する。rawから全groupのdevice telemetry・同居latency・warmup・sample数・状態/行動証跡を再検証し、未承認reviewを生成する。人が余裕メモリと直列化した性能/task指標を確認後、`--joint-review` 付き比較実行を許可する。メモリはwhole-deviceの定期sampleであり瞬間peak保証ではない。CPU結果をGPU同居の承認根拠にできない。

直列化は本実装の方針であり、論文の未確認な同居kernel並列方式の再現は主張しない。GPUの実fit・OOM・CUDA kernel・実タスク品質はPod再開後に確認する。

## 実行入口

既存の `scripts/pod_sync.py` でsource releaseを作成・配置できる。接続先の再確認まではdeployしない。

```python
from rosa_repro.global_planner import catalog_from_plan
from rosa_repro.global_control import prepare
spec = prepare(catalog_from_plan(reference_plan), initial_robot_ids_by_class,
               gpu_budget, call_timeout_s=45, deployment_timeout_s=600)
```

```bash
python scripts/global_experiments.py prepare --spec spec.json --banks banks.json \
  --changes changes.json --root DATA_ROOT --warmup 30 --duration 120 --output scenario.json
# runの既定はdry run。実行には--executeと全試行を覆う--budget-secondsが必須。
python scripts/global_experiments.py run --scenario scenario.json --root DATA_ROOT \
  --solver-python /ABS/SOLVER_PYTHON --sim-python /ABS/SIMULATOR_PYTHON \
  --output NEW_RUN --report dry-run.json
# GPU実行時のみ、上に --address EXISTING_RAY_ADDRESS --adoptions bundle.json
# --sim-repo /ABS/PINNED_LIBERO_REPO を指定。自動GPU fault injectionは禁止。
python scripts/global_experiments.py reaudit NEW_RUN/trial --output audit.json

# 専用実行計画。続けて既存pipeline_experiments prepare/runを使用する。
python scripts/dedicated_baselines.py --reference reference.json --profiles batch1.json \
  --method gpu_per_robot --gpu-budget GPU_COUNT --compile --output dedicated.json
python scripts/pipeline_experiments.py prepare --plan dedicated.json --banks banks.json \
  --root DATA_ROOT --phase formal --repetitions 3 --output suite.json
# pipeline_experiments run: 個別profile採用bundle必須。
# 最初の同居測定は --residency-calibration、採用後は --joint-review review-bundle.json。
python scripts/dedicated_residency.py --plan dedicated.json --root RESULTS_ROOT \
  --directories raw-directories.json --output NEW_REVIEW
```

`banks.json` はclass ID→保存state bankの相対directory、`changes.json` のopは `join`, `leave`, `budget`, `replan`, CPU限定 `fault`。`review-bundle.json` は `evidence_root`, `candidate`, `review`（後二つはrootからの相対path）。rawとreview原本を保持する。GPU driver開始時には、将来参加するclassのS1 modalityも取得できる初期配置が必要。

動的runは新しい出力先のみを受け付ける。wall timeout後にcleanup確認がなければ `needs_cluster_cleanup=true` を残す。結果不明なGPU actorを残したまま自動再実行しない。

## CPU検証の再実行

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src SOLVER_PYTHON -m pytest -q -p no:cacheprovider
SOLVER_PYTHON scripts/prepare_dynamic_dedicated_cpu.py --output NEW_FIXTURES
SERVE_PYTHON scripts/verify_dynamic_dedicated_cpu.py --fixtures NEW_FIXTURES \
  --solver-python /ABS/SOLVER_PYTHON --output NEW_PROOF
```

Serve用PythonはRay+NumPy、solver用PythonはOR-Toolsを持つ既存環境を指定する。CPU試行はloopbackのみ・GPU予約0で自分のcluster/appだけを終了する。人工profileとdummy simulatorを性能の代用にしない。旧v0.16原本は変更せず、それぞれの同梱sourceで監査する。
