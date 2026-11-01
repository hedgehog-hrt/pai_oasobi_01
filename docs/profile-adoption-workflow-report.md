> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# profile採用・再開ワークフロー・比較レポート

再開時は[現行の配布設定・再開手順](prepod-final-integration.md)も参照する。以下の検証件数・版番号は当時の履歴。配布設定例は現行契約に更新済み。

2026年10月29日、v0.10.0.dev1。Pod起動前の残作業3項目を実装した。これはNVIDIA/Stanford ROSAの再現実験を進めるための検証基盤であり、論文性能の再現完了を示すものではない。今回の実行はローカルCPUのみ。

## 1. R07へのprofile採用ゲート

[実装](../src/rosa_repro/profile_adoption.py)は、元の性能ログを再集計し、正式条件・反復数・失敗・telemetryを既存の`profile_candidate`で検査する。さらに以下を要求する。

- 同じ設定、ソース版、corpus、batch条件のGPU kernel診断。成功した終了codeと実CUDA kernel eventをtrace原本から再確認する。
- Python実体、主要package、GPU機種・memory・driver、Torch/CUDA/cuDNN・TF32等の観測状態の一致。param dtype・attention設定・compile markerも取得済みなら比較する。GPU UUIDやprofiler出力先など配置に依存する値は同一性の対象から外す。
- S2/Safety/Monitorは同じcorpusのholdoutを使い、raw予測・状態別ラベル・S2 rubricから品質評価を再計算する。欠測や未レビュー、基準未達は拒否する。kernelの入力もevaluation splitに属することを確認する。
- S1は同じ重み・観測Torch設定・dtype/attention/compile状態での公式経路との数値回帰とLIBERO paired実行、同じ初期状態、各outcomeと成功数の整合性を要求する。旧R02はraw action tensorを保存しないため数値比較を再計算することはできない。正確なbackend/corpus/現行ソースとの対応を記した人のassociationと、batch等価性・task successの明示レビューを別途要求する。
- 証跡全体のhashに結び付いた、名前と理由付きのレビュー。kernel・dtype・compile、入力の代表性、queue/SLO、GPUの他負荷、holdout品質、S1 task success/batch等価性の6項目。非S1の最後の項目も「対象外」の理由を人が記入する。未記入テンプレートは採用できない。

hashは保存後の改変検出と対応付けであり、測定者の申告が真実であることの証明ではない。vLLM子workerのdtype/compile等をfrontend属性から自動認定しない。代表性、queue安定性、診断で捉えられなかった演算、paired実験との対応はレビュー対象として残す。

```bash
# root配下の相対パスでrequestを作成。S2はreferencesの代わりにratingsを指定。
python scripts/profile_adoption.py inspect --root /workspace/p-ai-rosa/runs/<run-id> \
  --request request.json --output inspection.json
python scripts/profile_adoption.py review-template --root /workspace/p-ai-rosa/runs/<run-id> \
  --request request.json --output review.json
# 人がreview.jsonを確認・記入した後
python scripts/profile_adoption.py adopt --root /workspace/p-ai-rosa/runs/<run-id> \
  --request request.json --review review.json --output adopted.json
```

[VLM request例](../examples/adoption/vlm-request.template.json)、[S1 request例](../examples/adoption/s1-request.template.json)。S1 associationの項目は`config_sha256`、`corpus_id`、現行`runtime_source_manifest()`の`source_manifest`、`reviewer`、`reason`。この対応記述は自動承認しない。全ての入力証跡を同じroot内へ置く。symlinkとroot外参照は拒否する。採用IDはrootからの相対パスで決まり、rootの移動には依存しない。

`import`はfleetのprofilesを採用済みprofile集合に置き換え、Fleet契約を検査する。その後スケジューラを再実行してplanをcompileする。以前の人工profileに基づいたscheduleは使い回さない。

```bash
python scripts/profile_adoption.py import --root /workspace/p-ai-rosa/runs/<run-id> \
  --fleet fleet.json --adoptions s1-adopted.json s2-adopted.json safety-adopted.json monitor-adopted.json \
  --output measured-fleet.json
```

GPUの`pipeline_fleet.py run`には[adoptions対応表](../examples/adoption/fleet-adoptions.template.json)を`--adoptions`で渡す。起動前と`deploy_fleet`の内部で証跡を再検査し、実際に起動したworkerの観測環境も照合する。差異があればその実行で作ったworkerを片付けて停止する。backend、model revision、batch、SLOの一致と、配備stream数が測定stream数以下であることを要求する。共有GPUや別モデルへの流用はできない。

GPU planのcompileは配置のプレビューとして引き続き可能。未計測の単一成分probeも可能。実GPU R07の配備だけが採用ゲートを必須とする。profile採用は容量保証・ロボット安全保証・論文再現認定ではない。

## 2. 再開時の一括事前チェックと段階実行

[ワークフロー](../src/rosa_repro/resume_workflow.py)は、既存の[配置・ログ回収](runtime-kernel-and-pod-transfer.md)で配置した不変releaseから実行する。GPUモードではそのreleaseの全ファイルhashと、実行中コードがそのrelease内であることを検査する。SSH転送を暗黙に開始せず、`pod_sync deploy/init-run`後の境界で受け取る。モデル・観測・Python環境は既存の準備手順で配置する。

順序は環境収集→全重みpreflight→smoke各試行→独立kernel診断→pilot各試行→profile各試行。1 stepが1試行なので、低レートの長い正式profileも試行単位で予算・再開を管理する。profileには先行pilotが必要。smokeのみ、またはsmoke+pilotまでの計画も作成でき、pilot結果を確認してから正式rate gridを確定できる。

```bash
# Podの検証済みreleaseを作業ディレクトリにし、bytecodeをそこへ書かない
export PYTHONDONTWRITEBYTECODE=1
python scripts/resume_workflow.py prepare \
  --config /workspace/configs/safety.json --corpus-root /workspace/captures/data \
  --smoke-suite /workspace/configs/smoke.json --pilot-suite /workspace/configs/pilot.json \
  --profile-suite /workspace/configs/profile.json \
  --driver-python /opt/rosa-driver/bin/python --worker-python /opt/rosa-vlm/bin/python \
  --release /workspace/p-ai-rosa/releases/<bundle-id> --diagnostic-device 0 \
  --setup-budget 600 --kernel-budget 600 --output /workspace/configs/workflow.json
python scripts/resume_workflow.py check --plan /workspace/configs/workflow.json --report preflight-plan.json
# executeなしでは内容確認のみ。新規モデル実行もRay接続も行わない
python scripts/resume_workflow.py run --plan /workspace/configs/workflow.json \
  --output /workspace/p-ai-rosa/runs/<run-id>/safety-workflow --report dry-run.json
# 事前に出力された1 stepの最大時間以上のbudgetを指定して実行
python scripts/resume_workflow.py run --plan /workspace/configs/workflow.json \
  --output /workspace/p-ai-rosa/runs/<run-id>/safety-workflow --report step-001.json \
  --execute --max-new 1 --budget-seconds 1200 --address <existing-Ray-address>
```

`--setup-budget`は各試行のロード・hash検査・Ray準備等の追加上限。正式試行の測定・drain上限は低レート用の既存計算から取り込み、各stepに終了処理10秒分も予約する。残り予算内に最大時間が収まらないstepは開始しない。金額やPod全体の課金上限ではなく、このコマンドの処理時間予算である。

既定`--max-new 1`。成功済みstepはartifact hashと意味を再確認してskipする。失敗は番号の異なるattemptで再試行し、過去ログを上書きしない。途中のsource/config/corpus/Python実体変更は同じ計画で再開できず、新計画・新runが必要。報告JSONも毎回別名にする。通常成功はexit 0、失敗・予算停止・未解決条件はexit 2。

GPU子プロセスのtimeout/中断や未完attemptでは、共有Ray cluster上に残ったworkerを安全に自動停止できるとは限らない。このとき再開は`blocked_cluster_cleanup`になり、当該attemptのhashに対応した`cleanup_review`（`attempt_sha256`、`cleanup_confirmed: true`、`reviewer`、`reason`）が必要。実際の残存worker/GPU状態を確認した後だけ`--cleanup-review`で渡す。cluster全体や他の実験は停止しない。

観測採取、S1回帰・paired実験、品質ラベル作成、人のprofileレビューはこのワークフローの外にある既存工程。必要な証跡が揃うまでR07採用ゲートは通らない。

## 3. P1〜P4 / R07比較レポート

[レポート生成](../src/rosa_repro/comparison_report.py)は、事前に宣言したcondition・variant・反復・保存先の全セルを対象にする。結果の存在するセルだけを後から列挙する使い方はしない。run IDの重複、raw eventとauditの不一致、再計算と保存summaryの差異、異なる実行条件を検出する。

- P1〜P4はclass別の適格action chunk数・rate、R07は観測weighted Qとスケジューラの予測weighted objectiveを別列で表示する。
- 各反復の詳細、平均・標本標準偏差・最小最大、欠測・失敗・不正証跡の数を出力する。平均は成功反復だけなので分母も必ず表示する。未完groupを比較結果として扱わない。
- CPU/GPU、仮想action/外部executor、ソース、workload/hardware、測定protocol/provenanceのhashが同じcondition内で一致することを要求する。
- 少数反復からframe単位の信頼区間を作らず、論文の値やspeedupを推測で埋めない。既存pipeline runnerは接続検証用なので、実モデルを使っても自動的に正式な論文比較にはならない。

```bash
# conditionは予定した設定から固定する。calibrationを使う場合は本比較の反復から除外する。
python scripts/comparison_report.py condition <calibration-run> --output condition.json
# conditionをconditionsへ、全予定反復をcellsへ記載したmanifestを先に確定
python scripts/comparison_report.py build --manifest comparison-manifest.json \
  --root /workspace/p-ai-rosa/runs/<run-id> --output comparison-report
```

出力は`report.json`と`report.md`。欠測・失敗があっても報告を保存し、exit 2を返す。今回のCPU比較例（非同梱の履歴資料: `../results/prepod-adoption-workflow-report-20261029/installed-workflow-release/comparison-report.md`）は、予定3反復のうち実行した2反復と未実行1反復をそのまま示す。意図的に`incomplete`で、性能再現の結果ではない。

## 検証と残る実機工程

検証結果はrelease_verification.json（非同梱の履歴資料: `../results/prepod-adoption-workflow-report-20261029/release_verification.json`）に記録する。394件の単体・回帰試験、実CPU版Ray/Torchでの段階実行・再開、4 workerのRay接続、rawログの再集計を確認する。GPUを装った契約テストのfixtureはpytestの一時領域だけで使い、承認済みGPU profileとして配布しない。

Pod再開後には、GPU環境の確定、実モデルsmoke/CUDA trace、実観測・品質ラベル・S1回帰、正式profile反復、人の採用レビュー、複数GPU R07実行が必要。今回SSH接続・Pod起動・実学習済みモデル推論・人の承認代行は行っていない。
