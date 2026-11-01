# 共通モデル検査・Linux依存固定・動的異常系・P4/Mix2比較（v0.19.0.dev1）

依頼された1→2→3→4を順に実装。Pod接続、GPU実行、学習済み重みのロードは行わない。検証結果は配布に付属する`SUMMARY.md`・`release_verification.json`・原本ログを参照する。過去版の結果はその版の固定ソースで読む。

## 1. 共通環境をモデル事前検査へ接続

`profile_protocol.preflight`から`check_versions`へ`runtime_variant`を渡す。共通variantはTransformers 4.57.3、従来環境は4.57.6を検査する。variantを結果にも保存する。S2/Safety/Monitorそれぞれで、resume workflow→CLI→preflight→結果検査までを回帰テストする。逆の版や未知のvariantはcheckpoint検査前に拒否する。テストのパッケージmetadataとcheckpoint検査は明示的な人工fixtureであり、GPU・重みの証跡ではない。

FlashAttentionの配布wheel名にはCUDA/Torch/ABIのsuffixがあるが、取得したwheel内部の`METADATA`は`Version: 2.8.3`。誤った完全版指定を2.8.3に修正し、CUDA/Torch/ABI別の配布物はupstreamの固定URLとSHA256で拘束する。

## 2. Linuxでの推移的依存解決と固定ファイル

`shared_environment_lock.json`はGR00T固定commitの全直接依存（training/exportを含む）、vLLM 0.11.2、Ray 2.59.0、GR00Tビルド要件から解決した255パッケージの候補lock。Python 3.12.16/Linux x86_64/glibc 2.36、uv 0.13.6で実行した。Macの専用Colima profile内で、amd64公式PythonコンテナをRosetta経由で実行した。CPUのみであり、既存のdefault profileやPod環境は変更しない。作業用VMは終了後に停止する。

- イメージdigest：`python@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337`。
- 固定入力、upstream原文SHA256、resolverのargv/終了コード/log hash、Python/libc情報を保存。
- `requirements.lock`は全依存、`bootstrap.lock`はTorch・setuptools・pip・wheel・ninjaとその推移的依存の部分集合。両者の版とartifact hashを照合する。
- 固定ファイルを制約にして全入力から再解決し、同じpackage/extras/version/hash集合になったことを検査する。
- キャッシュだけでの再解決はTensorRT配布物不足で失敗した。`offline_recheck_passed=false`と失敗ログを残す。全wheelを同梱したオフライン導入用bundleではない。
- 全パッケージのインストール、`pip check`、native extensionのビルド/import、CUDA ABI、モデルロード、同居fitは未検証。最新の推移依存にはCUDA 13系のbinding/CUTLASS extrasも含まれる。metadata上の解決成功はCUDA 12.8上の実行互換性を保証しない。Podの検証失敗時には原因に基づいて制約を更新し、新lock・R06・profileを取り直す。

`shared_environment.py prepare`は既定でこのlockを検証・出力する。構築scriptは新規環境、固定GR00T checkout、uvの版、lockのSHA256を確認してから、両lockを`--require-hashes`で導入する。GR00T本体だけは固定checkoutから`--no-deps --no-build-isolation`で導入し、依存検査とfreezeを保存する。共通環境preflightでは直接pinに加え、全lock対象のinstalled versionの欠落・変化を検査する。既存環境の上書き再試行はしない。

```bash
# CPU側でrecipe生成。出力先は新規directory。
python scripts/shared_environment.py prepare --repo /workspace/Isaac-GR00T \
  --environment /opt/p-ai-shared-v19 --uv /usr/local/bin/uv --output NEW_RECIPE

# lock再生成の入力を保存する場合。
python scripts/shared_environment.py lock-inputs --output NEW_INPUTS
# 次はLinux x86_64 / Python 3.12 / uv 0.13.6のresolver環境で実行。
python src/rosa_repro/shared_lock.py --inputs NEW_INPUTS --output NEW_RESOLUTION \
  --uv /ABS/uv
```

再生成結果は同梱lockを自動上書きしない。失敗した試行も保存する。Pod再開後、レビューしたrecipeを実行し、新環境で`shared_environment.py preflight`、各roleのmodel stage、R06、profile測定、同居calibrationの順に進む。共通環境のPythonとruntime variantを全設定に渡す。

参考：[uv compile/target-platform仕様](https://docs.astral.sh/uv/reference/cli/)、[Colimaのprofile分離](https://colima.run/docs/profiles/)。uvのcross-platform指定でもsdistのビルドは実行ホストに依存するため、今回はLinuxのresolverを実行した。

## 3. 動的変更と異常時の採用判定

予定変更の開始・完了数、index、offset、dispatch遅れ、開始/完了の順序、固定窓内の完了を検査する。障害注入では`injected`だけでは採用しない。実際のhealth failureと復旧結果をworker ID・failed epochで結び付け、評価窓内に新しい世代のpopulation commitがあることを要求する。通常発生したhealth failureも、未復旧や窓外復旧なら採用しない。

join→budget変更→fault/recovery→leaveを最新のprospective scenario、3反復、state/seed割当、生ログ再監査、反復集計に通す。復旧不能、異なるworker/epoch、復旧証跡の欠落、遅いdispatch、窓外完了は不採用。raw Qは診断用に残し、採用Qや比較値へ混ぜない。遷移中の欠落は測定窓から除去しない。

サンプル境界とp99の999/1000/1001成功要求も検査する。1000未満はp99を出さず、1000以上はnearest rankで求める。warmup・測定ともcutoffまでに返らなかった応答を成功数へ入れない。GPUの正式反復、実際のモデル再ロード時間、実タスク品質は未検証。

## 4. P4・混在条件の専用方式比較

CPU検証はR08:P4:8とR09:Mix2:8。両familyともROSA、Equal/Weightedのfull/placement-only、2専用方式の7方式×3反復をすべて維持する。固定seedの事前ランダム順で実行し、欠測・資源不足・不適格な反復を除外して比較を成立させない。

| 条件 | robots | logical models | robotごとのactor | robot×modelごとのactor |
|---|---:|---:|---:|---:|
| P4 | 8 | 32 | 8 | 32 |
| Mix2（P2×4、P4×4） | 8 | 24 | 8 | 24 |

inventoryから実PID/node/worker identityを確認し、rawからS2参照付きS1、Safety、Monitor、適用行動と比較を再監査する。CPU実行ではGPU leaseは0、GPU効率は未算出。モデル数の多い専用方式が単なる論理ラベルになっていないことを確認する。GPU同居のメモリや性能は証明しない。

Monitor 0.5Hzのため、CPU試行も周期を覆う測定時間を事前に設定する。実Ray用fixtureはwarmup 2.5秒・測定6.5秒・class/componentごとwarmup/測定成功1以上。これはCPU動作確認の条件であり、正式p99条件ではない。

`paper_family.py`のjobsには`argv`（dry-run）と`execute_argv`（明示的実行）を別々に保存する。報告ファイルの衝突を避け、whole-trial予算に実行準備分の余裕を含める。GPUの場合は既存cluster、実profile採用、LIBERO、同居review等の追加条件を満たす必要がある。

```bash
# Solver/Test用Pythonでfixtureだけ生成。
python scripts/prepare_locked_dynamic_mixed_cpu.py --output NEW_FIXTURES
# Ray CPU用Pythonで実行。既定はdry-run。
python scripts/verify_locked_dynamic_mixed_cpu.py --fixtures NEW_FIXTURES \
  --output NEW_RESULTS --solver-python /ABS/SOLVER_PYTHON \
  --budget-seconds 3600 --execute
```

実CPU検証は動的3試行＋比較42試行。動的試行はsimulatorを独立プロセスで実行し、比較はrobotごとのin-process dummy episodeを使う。起動した専用Ray cluster/appだけを終了する。開発失敗や不採用のログは別保存し、最終配布の成功証跡と区別する。
