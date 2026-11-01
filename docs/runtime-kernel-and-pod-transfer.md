> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# 実行環境・kernelの証跡収集とPod配置・ログ回収

2026年10月28日、v0.9.0.dev1。依頼された順に、実行環境とkernelの証跡収集、次にPodへの配置・ログ回収を実装した。今回の実行はローカルCPUとローカルの模擬Podディレクトリだけ。SSH接続、Podの起動・課金、実CUDA推論は行っていない。

## 1. 実行環境とkernelの証跡

[環境収集モジュール](../src/rosa_repro/runtime_attestation.py)と[CLI](../scripts/runtime_evidence.py)を追加した。証跡を次の3種類に分ける。

| 種類 | 保存内容 | この情報だけで確定できないこと |
| --- | --- | --- |
| 実行環境 | Python実体のSHA256、全package名・version、OS、CPU architecture、ソースhash、限定した環境変数、取得可能ならdriver/CUDA tool・Nsight版 | どのkernelが実際に動いたか |
| ロード済み状態 | parameterのdtype・個数・要素数・device、attention設定、compile marker、Torch/CUDA/cuDNN版、GPU UUID・機種・能力、TF32/determinism/SDPA設定 | 設定されたbackendが全演算で使われたか、compile markerのない箇所で最適化がないか |
| 実行trace | Chrome trace原本、CPU operator名、GPU kernel eventの名前・件数・時間の合計、trace hash | 論文との最適化条件の等価性、未記録のkernel、通常時の性能・精度 |

既存のprofile/quality runnerのdriver・Ray worker環境証跡を拡張した。GR00TとTransformers VLMのattestationにはロード済みモデルの観測値を加えた。vLLMのモデルは別プロセスにあるため、frontendの属性を子プロセスのparameter観測として代用しない。

全環境変数、認証トークン、pipの認証付きURL、SSH秘密鍵は収集しない。CUDA/TF32/allocator/vLLMとPythonの読み込み経路に関係するallowlistだけを保存する。`collect`はローカルの読取りだけで、依存のインストールや設定変更はしない。CUDAがない、ツールがない、コマンドが失敗した場合はその状態を記録する。

```bash
# それぞれ実際に使用するworker Pythonで実行
PYTHONPATH=src <worker-python> scripts/runtime_evidence.py collect --torch \
  --output /workspace/p-ai-rosa/runs/<run-id>/environment.json
```

### traceは独立した短い診断実行にする

[PyTorch 2.9 profiler](https://docs.pytorch.org/docs/2.9/profiler.html)のCPU/CUDA activitiesとChrome trace出力を利用する。vLLM 0.11.2では[公式のprofiling機構](https://docs.vllm.ai/en/v0.11.2/contributing/profiling/)を有効にし、[AsyncLLMのstart_profile/stop_profile](https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/v1/engine/async_llm.py)経由で子workerのtraceを回収する。frontendのCPU traceに`cudaLaunchKernel`というAPI名があるだけでは、GPU kernel確認済みにしない。

診断は同じbackend factoryを新しい専用プロセスにロードし、evaluation splitの指定観測を使う。1 warmup batchの後に1 batchをtraceする。S1のbatch sizeは設定値を使う。Transformers/GR00Tは実際にモデル演算するthread内でprofilerを有効にする。vLLMはnative profilerをstart/stopし、失敗時も停止とengine shutdownを試みる。traceのstack記録は無効、shape記録は有効。raw generation/parser情報、出力hash、設定と入力hashも保存する。

この診断経路はRayの配置・ネットワーク・実シミュレータactionを含まない。trace取得による負荷があるため、この時間をR03/R07の性能profileへ混ぜない。通常の計測コマンドでprofilerを自動起動する変更はしていない。

```bash
# Pod上。意図した1 GPUだけを可視化し、既存のGPUジョブと競合しない状態で実行する
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src <worker-python> scripts/runtime_evidence.py probe \
  --config /workspace/configs/safety-vllm.json \
  --corpus /workspace/captures/corpus.json --root /workspace/captures/data \
  --sample 0 --timeout 600 --flush-timeout 120 \
  --output /workspace/p-ai-rosa/runs/<run-id>/safety-kernel
```

GR00TはS1設定、Transformersはその設定を渡す。既存preflightとsmokeを先に通し、S1 stateless確認も維持する。`runtime_env.py_executable`がある場合、そのPythonで診断する。GPU機種・可視GPU数・モデルrevision・実ファイル照合に失敗したらCPUへ切り替えない。

`--timeout`は10〜3,600秒、既定600秒。import・全重み検査・モデルロード・warmup・trace・flush・shutdownを含む子プロセスの時間予算である。timeoutやCtrl-Cで自身の子プロセスgroupを終了し、失敗と部分成果を保存する。終了処理に最大10秒の追加猶予がある。既存Ray clusterや他のGPUジョブは停止しない。

vLLMのtrace flushは時間がかかる場合があり、必要なら**事前に**wall/flush予算を変更する。backendの生成timeoutとmax token数は元の設定を保つため、診断中に出力が打ち切られた場合は失敗のまま残る。未記録のCUDA eventや不完全な生成を成功へ書き換えない。trace parserは既定128 MiBまで展開し、巨大・壊れたtraceを未確認のまま通さない。

出力は`condition.json`、`source_manifest.json`、`environment-before.json`、`diagnostic.json`、`result.json`、`execution.json`、`requests.jsonl`、trace原本、stdout/stderr、全ファイルの`evidence.json`。終了codeとtimeoutを反映した`result.json`が最終結果になる。診断に失敗した場合はexit 2。

```bash
PYTHONPATH=src <python> scripts/runtime_evidence.py trace-summary \
  /workspace/p-ai-rosa/runs/<run-id>/safety-kernel/traces/<trace.json.gz> \
  --output /workspace/p-ai-rosa/runs/<run-id>/trace-summary.json
PYTHONPATH=src <python> scripts/runtime_evidence.py compare \
  --diagnostic /workspace/p-ai-rosa/runs/<run-id>/safety-kernel \
  --profile /workspace/p-ai-rosa/runs/<run-id>/profile/<case>/attempt-001/raw \
  --output /workspace/p-ai-rosa/runs/<run-id>/kernel-profile-association.json
```

`compare`は改変検査の後、設定とソースが一致するかを示す。workerのPython/package・GPU identity、実kernel名、dtype、入力/batch代表性は別に確認する。`automatic_profile_acceptance=false`を維持し、kernel名を見ただけでFlashAttentionやCUDA Graphの完全な利用、ROSAとの一致を認定しない。kernel eventの時間合計は重複実行を含み得るため、end-to-end latencyやGPU使用率の代用にはしない。

## 2. ソース配置とログ回収

[pod_sync.py](../scripts/pod_sync.py)がローカル側、[pod_transfer.py](../src/rosa_repro/pod_transfer.py)が標準ライブラリだけで動くPod側helper。SSH実行時に同じhelperソースを送るため、初回配置前にプロジェクトをPodへインストールしておく必要はない。remote Pythonは3.9以上が必要。アプリ本体は従来通りPython 3.12以上。

全ネットワーク操作は既定でdry-run。`--execute`を付けた場合だけ接続する。SSHはBatchModeとStrictHostKeyCheckingを有効にし、認証情報をチャットやログへ出さない。初回やPod再作成後のhost keyは既存のSSH設定で確認・登録しておく。古いIP/portが再開後も有効とは仮定しない。[接続template](../examples/pod/connection.template.json)のhostを`SET-HOST`にしてある。

```bash
# ローカル。templateをコピーして、再開後に確認したhost/port・鍵パスを設定する
<python> scripts/pod_sync.py template --output work/pod-connection.json
# ソースのみの配布zipを作る。モデル・観測・過去の実験結果は含めない
<python> scripts/pod_sync.py pack --output work/rosa-source.zip
# 接続せずに、転送対象hashと配置先を確認
<python> scripts/pod_sync.py deploy --connection work/pod-connection.json \
  --bundle work/rosa-source.zip --report work/deploy-plan.json
```

`pack`の対象はsrc/scripts/tests/docs/examplesと明示したREADME・pyproject・requirements。model/cache/credentialディレクトリ、過去results、隠しファイル、symlinkを除外または拒否する。source bundleは50 MiB以下。bundle manifestの全ファイルSHA256を照合してから扱う。今回の配布一式には配置用source zip（非同梱の履歴資料: `../results/prepod-runtime-transfer-20261028/source-release.zip`）も含めた。

Pod再開後に実行する操作：

```bash
<python> scripts/pod_sync.py status --connection work/pod-connection.json \
  --execute --report work/pod-status.json
<python> scripts/pod_sync.py deploy --connection work/pod-connection.json \
  --bundle work/rosa-source.zip --execute --report work/deploy-receipt.json
<python> scripts/pod_sync.py init-run --connection work/pod-connection.json \
  --release-id <deployで返った64桁bundle_id> --run-id experiment-001 \
  --execute --report work/run-init.json
```

配置先は次の構成。既定baseは`/workspace/p-ai-rosa`だが、専用の空ディレクトリに変更できる。管理用markerのない既存の非空ディレクトリを転用しない。

```text
/workspace/p-ai-rosa/
  releases/<bundle_id>/       コード・設定・手順書
  runs/<run_id>/              その版に紐付けた実験出力
  incoming/<archive-sha>.part 転送途中のファイル
  exports/<archive-sha>.zip   回収用snapshot
```

配置はstagingで全内容を検証してからrenameする。同じ版を再送した場合は配置済み内容を照合して返す。変更された既存版を上書きしない。途中切断後は同じコマンドを新しいreport名で再実行し、所有する`.part`を先頭から再転送する。転送済みバイト位置からのresumeは実装していない。版ごとの配置なので、以前のコードや実験結果は保持される。依存インストール、モデルdownload、current symlinkの切替、実験の自動開始は行わない。

**releaseディレクトリへログ・結果・cacheを書き込まない。** 実行時は`PYTHONDONTWRITEBYTECODE=1`と絶対パスの`PYTHONPATH=<release>/src`を設定し、出力先を`runs/<run_id>/...`へ向ける。`__pycache__`等で版の内容を変えてしまった場合も、次の照合で変更として検出される。各workerも同じreleaseソースを読めるようにする。

### ログ回収

実験を停止または完了させてから回収する。指定run以外やホームディレクトリを一括回収しない。外部ファイルへのsymlink、activeな`.runner.lock`、未checkpointのSQLite sidecar、snapshot中のファイル変更があれば拒否する。ロックを使わない外部writerについては完全な停止をツールだけで保証できないため、結果は常に「安定して取得できたsnapshot」であり「実験成功」証明にはしない。

```bash
# まず接続しない計画。outputはローカル回収先
<python> scripts/pod_sync.py collect --connection work/pod-connection.json \
  --run-id experiment-001 --output results/recovered --report work/collect-plan.json
# 実回収
<python> scripts/pod_sync.py collect --connection work/pod-connection.json \
  --run-id experiment-001 --output results/recovered \
  --execute --report work/collect-receipt.json
```

回収先は`results/recovered/<run_id>/<bundle_id>/`。ZIP全体と各ファイルのSHA256を検査してから展開し、元の`evidence.json`等もそのまま保存する。同じsnapshotは照合して再利用し、新しいログは別snapshotとして保存する。既存ローカル結果を上書きせず、Pod側のログ・exportも削除しない。切断・破損の`.part`は未完了のまま保持し、再実行で先頭から回収する。

archiveは最大20 GiBの展開データ・50,000ファイルまで。path traversal、絶対パス、ZIP symlink、重複/case衝突、未登録ファイル、内容hash不一致を拒否する。SHA256は内容・転送の整合性検査であり、署名による作者認証ではない。接続先の認証はSSH host keyで行う。

`--report`は必須で既存reportを上書きしない。dry-run・実行有無・helper hash・対象hash・返却receipt・失敗理由を保存する。転送timeoutは既定600秒、1〜7,200秒で事前指定できる。失敗はexit 1。コマンド引数は構造化してSSHへ渡し、remote commandの引用はshell用に処理する。

## 検証結果と残る確認

全回帰試験（非同梱の履歴資料: `../results/prepod-runtime-transfer-20261028/pytest-release.xml`）は**359件合格、失敗0・skip 0**（前版324件＋35件）。CPUの実Torch profiler、native vLLM profilerの開始/停止契約を模した試験、timeout/Ctrl-Cの子process group停止、転送の改変・切断・lock・パス検査を含む。vLLM GPU子workerでの実traceはまだ取得していない。

配置から回収までの検証（非同梱の履歴資料: `../results/prepod-runtime-transfer-20261028/end-to-end-release/verification.json`）では、実際のsource bundleをローカル模擬Podへ配置し、配置したソースからCPU診断を実行して、そのtraceを回収する。sourceとraw traceのhash、再配置・再回収の冪等性、未解決接続設定でのdry-runを確認する。この検証はSSHサーバー・認証・ネットワークを模擬していない。通信部分の実確認はPod再開後に残る。

CPU版Rayの回帰（非同梱の履歴資料: `../results/prepod-runtime-transfer-20261028/ray-regression-release/verification.json`）では、拡張した環境証跡のもとでsmoke/pilot/profile、途中再開、同一出力比較、生ログ再監査を確認する。最終ソース・証拠hash（非同梱の履歴資料: `../results/prepod-runtime-transfer-20261028/release_verification.json`）と開発履歴（非同梱の履歴資料: `../results/prepod-runtime-transfer-20261028/development_history.json`）も保存する。過去版の証拠は当時のソースに対応する履歴として残した。

再開後はSSH接続確認→source配置→worker環境・モデル照合→smoke→短いkernel診断→既存の通常profileの順で進める。実GPU kernel・VRAM、実観測の品質、閉ループP2〜P4、R07の複数GPU実行は、引き続きPod上での検証が必要。
