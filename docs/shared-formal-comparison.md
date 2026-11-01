# 共通環境・動的正式条件・専用方式比較（v0.18.0.dev1）

依頼された3項目を順に実装した。ここでいう正式条件は本プロジェクトの測定規約であり、論文との数値的一致を保証しない。Podの起動・接続、パッケージの導入、学習済み重みのロードは行っていない。

## 1. GR00T・vLLMの共通環境と同居設定

固定GR00T commit `944fe70c4fd54e3fcb80a7d7258a328f9c061176` の依存定義とvLLM 0.11.2を照合し、Python 3.12 / Linux x86_64 / CUDA 12.8の新規環境を作るrecipeを追加した。GR00Tが固定するTransformers 4.57.3は、vLLMの許容範囲に入る。従来VLMの4.57.6をそのまま併用するとGR00Tの固定条件に合わない。参照元は[固定GR00T pyproject](https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/944fe70c4fd54e3fcb80a7d7258a328f9c061176/pyproject.toml)、[vLLM CUDA依存](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.2/requirements/cuda.txt)、[vLLM共通依存](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.2/requirements/common.txt)。原文とSHA256もソースに保存した。

`shared_environment.py`は直接依存の整合性を検査し、requirements・constraints・build scriptを生成する。GR00Tのtraining/export依存も省かない。Torchを先に配置し、全依存を解決後、固定checkoutのGR00Tを導入してpip checkとfreezeを保存する。既存環境への上書き、異なるcommit、追跡ファイルの変更を拒否する。preflightはinstalled metadata・依存・import・CUDAを検査し、モデルはロードしない。

**これはLinuxで解決済みの完全lockではない。** 推移的依存、CUDA/FlashAttention ABI、FFmpeg/torchcodec、実GPUの同居fitはPodで検証する。build途中で失敗した環境は自動的に上書き再試行しない。古い環境を参照比較に保持したまま、別の新規pathを使う。

共通variant `groot-vllm-cu128-v1`を明示した場合だけVLMの版検査を変更する。variant自体をruntime fingerprintへ含め、メモリ値が同じでも旧profileと区別する。4モデルの`runtime_env.py_executable`を統一し、S1のR06回帰と全モデルのprofileを再取得する。元のprobe設定は変更しない。

生成する同居テンプレートはS1の計画枠35%、S2 15%、Safety 10%、Monitor 15%、余裕25%。vLLMの3値はengine予約設定へ接続するが、S1枠と余裕枠はallocatorの強制上限ではない。合計がGPU容量を超える宣言を拒否する。これは採用済み設定や実測fitではなく、モデルpath・GPU名・S1 stateless検証等の未解決欄を残す。単独profileの採用後、同居calibrationを3反復以上行い、既存のrawに結び付いたjoint residency reviewを通して比較へ進む。

```bash
python scripts/shared_environment.py prepare --repo /workspace/Isaac-GR00T \
  --environment /opt/p-ai-shared-v1 --uv /usr/local/bin/uv --output NEW_RECIPE
python scripts/shared_environment.py inspect --output dependency-inspection.json
# Pod再開後: recipeのbuild-on-pod.shを実行し、新環境のPythonでpreflight。
/opt/p-ai-shared-v1/bin/python scripts/shared_environment.py preflight --output preflight.json
```

## 2. 動的実験の正式評価条件

従来のdiagnostic実行を残し、`--phase formal`（GPUのみ）または`cpu_smoke`を追加した。3〜10反復、固定warmup・測定窓・変更時刻・許容dispatch遅れ・sample gateをscenarioに保存する。formalはwarmup 30秒以上、測定120秒以上、初期active class/componentごとにwarmup成功20要求以上。低頻度成分では30秒で20要求に届かないため、実測rateに基づいて事前に十分な時間を設定する。固定窓を結果に合わせて延長しない。

- warmupでは初期観測でモデルを実行し、物理行動を適用しない。cutoff時点までに返っていない応答は成功数へ加算しない。不足なら新しい測定要求・行動を開始せず早期停止する。
- 測定中に参加するrobotはcommit時点から需要を持つ。再配置・モデル再ロード中の停止時間、周期要求の欠落、失敗を測定窓から除外しない。
- 各robotは保存された1 episodeを継続する。terminal後は自動resetせず、離脱まで周期義務を継続する。steady-stateの能力値やタスク成功率をQだけから推定しない。
- 測定成功数の既定はclass/componentごと1要求以上。これは最低限の観測成立条件で、統計的十分性ではない。実験目的に応じて`--measure-min-success`で事前に増やす。p99は成分ごとに成功1000要求以上の場合だけ出力する。低頻度/短時間動的実験ではp99を空欄とし、Qと移行中の欠落を報告する。
- 全予定変更のdispatch・commitが規定窓内に収まること、反復ID・state/seed・source・入出力・cleanupを再監査する。失敗/欠測反復を残し、全反復が通るまでは正式集計値を出さない。ばらつきの単位は試行で、frame/requestを独立反復と数えない。同じtask/state bankを反復する場合の一般化にも限界がある。

```bash
python scripts/global_experiments.py prepare --spec spec.json --banks banks.json \
  --changes changes.json --root DATA_ROOT --phase formal --repetitions 3 \
  --warmup 120 --duration 300 --warmup-min-success 20 --measure-min-success 1 \
  --change-tolerance 1 --output scenario.json
# 上の時間は記法例。実測rate、変更/再ロード時間に合わせて事前に決める。
python scripts/global_experiments.py run --scenario scenario.json --root DATA_ROOT \
  --rep 0 --solver-python /ABS/SOLVER_PYTHON --sim-python /ABS/SIMULATOR_PYTHON \
  --output NEW_REP0 --report dry-run.json
python scripts/global_experiments.py report --scenario scenario.json --root RESULTS_ROOT \
  --directories repetitions.json --output dynamic-report.json
```

run既定はdry run。GPU実行時には既存clusterのaddress・adoptions・LIBERO repoを指定し、`--execute`と全試行を覆う`--budget-seconds`が必要。repごとに新規出力先を使い、`repetitions.json`は`{"0":"rep0/trial","1":"rep1/trial","2":"rep2/trial"}`のように全反復を列挙する。失敗反復を成功するまで自動的に置き換えない。

## 3. 専用GPU方式の比較実験への統合

`paper_experiments.py --dedicated-budgets`でR08/R09の各workload/robot数/反復へ2方式を追加できる。既定のROSA・Equal/Weighted・placement-only条件は維持する。`physical_gpus`はROSA/固定分割側の予算、`dedicated_gpu_budget`は専用方式の独立した予算であり、同じGPU数の比較とは扱わない。

| 専用方式 | 必要GPU数 | 実行制御 |
| --- | ---: | --- |
| gpu_per_robot | 全robot数 | 各robotのモデルを同居、group内FIFO直列化 |
| gpu_per_model_per_robot | 各robotのモデル数の合計 | 各モデルを別actor/GPUへ配置 |

例：P4・8 robotsなら8枚対32枚。P1では両方式のgroup構造は同じ。予算不足は`resource_infeasible`として計画とレポートに残し、robot数を減らした実行計画を生成しない。

`paper_family.py`は1 familyの全variant・全反復に同一の測定時間・gate・state bankを設定し、plan/suite、randomized順のdry-runコマンド、集計用directory対応を生成する。実行は既存`pipeline_experiments.py`へ接続する。コマンドには反復caseと時間予算を固定する。GPUでは実adoptions、環境、同居reviewを別途指定する。calibrationログは比較結果として受け付けない。

集計では実行計画・B1 profile・モデルbinding・source・state bank・測定規約・反復IDを照合し、rawから行動/Qを再計算する。測定前後のPID・GPU lease・worker identityも確認する。欠測・失敗・資源不足があればfamilyの比較倍率を保留し、baseline Q=0の場合の比は未定義として残す。

宣言GPU予算と実lease数を別々に報告する。GPU秒は「枚数×測定窓」の予約device-timeで、稼働率・setup時間・課金総額ではない。CPU検証は実GPU lease/秒を0、GPU効率を未算出とし、論理slot秒を別欄へ記録する。CPUの倍率は人工モデルの動作確認に限る。

```bash
# budgets.json: {"gpu_per_robot":8,"gpu_per_model_per_robot":32}
python scripts/paper_experiments.py --gpu-count 8 --repetitions 3 \
  --dedicated-budgets budgets.json --output paper.json
python scripts/paper_family.py --paper paper.json --reference adopted-reference.json \
  --family R08:P4:8 --profiles batch1.json --banks banks.json --root DATA_ROOT \
  --phase formal --controls controls.json --output DATA_ROOT/NEW_FAMILY
python scripts/paper_baselines.py report --paper DATA_ROOT/NEW_FAMILY/paper.json \
  --root DATA_ROOT --directories DATA_ROOT/NEW_FAMILY/directories.json \
  --references DATA_ROOT/NEW_FAMILY/references.json --output comparison.json
```

`controls.json`は`closed_loop_protocol.protocol`のkeyword引数（`warmup_s`,`duration_s`等）。suiteを生成した時点では実測条件の採用を意味しない。最初の実行を指すdirectory対応は、再試行時に実際のattemptに合わせて明示的に更新し、失敗証跡を保持する。

## 配布ソースからのCPU再検証

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src SOLVER_PYTHON -m pytest -q -p no:cacheprovider
SOLVER_PYTHON scripts/prepare_shared_formal_comparison_cpu.py --output NEW_FIXTURES
SERVE_PYTHON scripts/verify_shared_formal_comparison_cpu.py --fixtures NEW_FIXTURES \
  --solver-python /ABS/SOLVER_PYTHON --output NEW_PROOF
```

solver環境はOR-Tools、Serve環境はRay/NumPyを持つ既存のローカルPythonを使う。実CPU Rayを使うがGPU予約0、loopbackのみで実行し、自分が開始したcluster/appを終了する。動的3反復とwarmup不足1試行、P1・8 robotsの比較15試行を行う。P4の同居/分離構造と21条件の完全比較は別途ローカル統合テストで検証する。結果の実数と配布ハッシュは`results/prepod-shared-formal-comparison-20261029/SUMMARY.md`に保存する。
