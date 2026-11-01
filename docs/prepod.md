> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# Pod再開前の準備 1→2→3

2026年10月27日。ユーザー指定の順で、T01比較実行器、Ray ServeのCPU結合、実験条件生成・再開・集計を実装した。ローカルMacのCPUだけで実行し、Podへ接続・課金・起動操作は行っていない。

## 結果

| 項目 | 確認した内容 | 実行証拠 |
| --- | --- | --- |
| 1. T01比較の実行器 | 人工環境の10ケース×2経路を完走。状態・seed・初期観測の一致、実行ステップ、終了理由を記録 | paired-cpu（非同梱の履歴資料: `../results/prepod-20261027/paired-cpu/paired_summary.json`） |
| 推論前の観測保存 | 初期観測10件を保存。policy呼び出し0件・行動適用0件 | capture-cpu（非同梱の履歴資料: `../results/prepod-20261027/capture-cpu/capture_summary.json`） |
| 2. 実際のRay Serve | 12要求が4件×3バッチになった。応答対応、端数バッチ、重複拒否、推論中reset、別robotの状態維持を確認 | Ray結合試験（非同梱の履歴資料: `../results/prepod-20261027/ray-cpu-001/verification.json`） |
| 3. 実験自動化 | open-loop 2条件とclosed-loop 5条件。2条件で区切って再開し、残りだけを実行 | 最終集計（非同梱の履歴資料: `../results/prepod-20261027/suite-cpu-002/summary.json`）・CSV（非同梱の履歴資料: `../results/prepod-20261027/suite-cpu-002/summary.csv`） |
| 単体・回帰試験 | 117件合格。既存92件＋新規25件 | JUnit（非同梱の履歴資料: `../results/prepod-20261027/pytest-junit.xml`） |

実験自動化の7条件のうち、低頻度の1条件は3秒上限までに必要な8件に届かず、`insufficient_or_failed`として保存する。これは件数不足の検出結果であり、成功に読み替えない。SQLiteの`complete`は「試行を保存し終えた」という状態で、測定の合格は`result.json`の`status`と`requirements_met`を確認する。

人工環境での成功やCPUダミーモデルの遅延は、LIBERO成功率・GR00T性能・ROSA論文の数値再現ではない。今回の短い試行はソフトウェアの接続確認であり、p99や処理量の性能評価に使わない。

## 1. T01比較実行器

`paired.py`は初期状態をNPZとJSONで保存し、状態データのハッシュを検証してから各経路に同じケースを渡す。ケースごとに実行順を交互に入れ替える。両経路の復元後状態・初期観測・seedが一致しなければ`paired_comparison_valid=false`とする。例外、timeout、step上限、成功を区別し、失敗エピソードも開始数に残す。

各観測と推論行動を保存し、R00のrequest/actionログをクライアントの単調時計で記録する。行動は実行器の返す実ステップ数を1チャンクとして数える。CPU人工環境は11ステップで終了するため、8+3ステップの記録を検証した。

`libero_bridge.py`は固定済みGR00T/LIBEROのAPIに接続する。初期状態はLIBEROベンチマークの指定状態を使い、公式`LiberoEnv`・`MultiStepWrapper`とGymnasiumのspace-aware batchingを通す。初期状態をセットするresetだけを拡張し、画像変換やgripper変換は公式処理を使う。[LIBERO固定コミットの状態復元API](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py)を確認した。

この接続コードのMuJoCo描画・実モデルでの動作はPod再開後に検証する。今回の比較は、各要求でcase seed＋観測番号を使ってPython/NumPy/TorchのデフォルトRNGをそろえる回帰試験である。公式経路は公式public policy APIにメタデータ・seed制御を付加したもので、未変更の公式サーバーの遅延ベースラインではない。

Pod再開後、既存のR02サーバーを起動したうえで実行する例:

```bash
# GR00T repoとローカルCosmos alias、固定モデル、既存Pythonの確認後
/opt/p-ai-libero/.venv/bin/python /workspace/rosa_reproduction/scripts/r02_t01.py export-states \
  --repo /workspace/Isaac-GR00T --output /workspace/probes/r02-states-001 --count 10

/opt/p-ai-libero/.venv/bin/python /workspace/rosa_reproduction/scripts/r02_t01.py capture-observations \
  --repo /workspace/Isaac-GR00T --bank /workspace/probes/r02-states-001 \
  --output /workspace/probes/r02-observations-001

/opt/p-ai-libero/.venv/bin/python /workspace/rosa_reproduction/scripts/r02_compare.py \
  --repo /workspace/Isaac-GR00T --observations /workspace/probes/r02-observations-001 \
  --output /workspace/probes/r02-comparison-001

# 出力照合を確認してから、両経路の閉ループT01を実施
/opt/p-ai-libero/.venv/bin/python /workspace/rosa_reproduction/scripts/r02_t01.py run \
  --repo /workspace/Isaac-GR00T --bank /workspace/probes/r02-states-001 \
  --output /workspace/probes/r02-paired-001
```

各出力先は新しい名前を指定する。動画生成は今回の新ランナーには含めていない。前回の動画付き公式ランナーは保持しており、R06で統合する。

## 2. Ray Serveでの結合

Mac ARM64・Python 3.12・Ray 2.59.0を、既存環境から分離した`work/rosa-serve-venv`に導入した。依存版は[requirements-serve-cpu.lock](../requirements-serve-cpu.lock)に保存した。Mac用のCPU検証環境であり、このlockをPodのGPU環境へそのまま適用しない。

[Ray公式batch資料](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html)に従い、1 replica・1 model・同時1バッチで実行する。要求のNumPy配列をbatch次元で結合し、1回のmodel callの出力を元の要求へ分割する。単に逐次呼び出しを束ねてバッチと呼ぶ実装ではない。

モデル呼び出しは専用threadへ渡すため、その間にもresetを受け付ける。エピソード世代が変わった要求の出力は返さず、他のrobotは継続できる。異なるshapeの観測は同じworkerへ混ぜず拒否する。各応答に実バッチサイズとbatch IDを記録し、server内のqueue待ちとmodel壁時計時間はクライアント時計と分けて保存する。

試験はRayのDeploymentHandle経由で、HTTP proxyを無効にして行った。HTTP ingress、複数replica、GPU故障切替はこの試験の範囲外。実GR00T用policy factoryを差し込む口はあるが、GPU資源割当・実モデルのbatch挙動・runtime検証はPodでの結合が必要。

```bash
PYTHONPATH=src <serve-venv>/bin/python scripts/verify_ray_cpu.py --output results/ray-new-run
```

検証clusterは終了時に停止する。Rayの認証tokenは各実行process内で生成し、結果ファイルへ保存・表示しない。

## 3. 条件生成、負荷、再開、集計

`experiments.py`はバッチ・待機時間・ロボット数・要求頻度・反復seed・観測/実行環境fingerprintから条件IDを作る。

- R03は外部から与えた容量C_bの0.1〜1.0倍の10点。容量が未測定なら、そのバッチを未確定として残す。
- R04はロボット数1/2/4/8/16/32/64、行動時間0.2秒、頻度制限の有無を定義できる。明示入力した頻度を「ROSA最適」とは呼ばない。
- formal設定はwarmup 30秒以上・20正常要求以上、上限180秒。測定は120秒以上・1000正常要求以上、上限900秒。
- CPU接続確認には別名`cpu_smoke`の短い設定を使う。GPUのformal設定へ自動昇格しない。

`replay.py`はR03の周期open-loop負荷とR04のclosed-loop再生を実装する。R03ではinflight上限で拒否された要求も分母に残す。R04では応答後に仮想行動時間を待ち、必要なら送信頻度を制限する。R04のactionは`virtual_timed_chunks`で、実ロボットやLIBEROの物理動作とは区別する。

warmupに達しなければ測定を始めない。測定区間を切った後に進行中の要求を回収し、R00の送信区間ルールで再集計する。実バッチ分布、要求遅延、仮想適格行動量、inflight数の推移、server内時間を残す。

SQLiteと排他lockで1 driverだけを許可する。試行番号は保存領域の作成前に予約し、失敗・中断した試行を上書きしない。完了試行は全成果物のハッシュを検証してからskipする。条件、ソース、Ray版が変わった状態で同じ出力先をresumeしない。

```bash
# CPUのみ。最初の2条件でいったん区切る
PYTHONPATH=src <serve-venv>/bin/python scripts/prepod_experiments.py cpu-smoke \
  --output results/cpu-suite-new --max-new 2

# 同じソース・同じ環境で残りだけを実行
PYTHONPATH=src <serve-venv>/bin/python scripts/prepod_experiments.py resume \
  --output results/cpu-suite-new

# ログからJSON/CSVの集計を再生成
PYTHONPATH=src <cpu-venv>/bin/python scripts/prepod_experiments.py summarize \
  --output results/cpu-suite-new
```

初期状態のハッシュ単位でプロファイル用と評価用の観測を分ける。同じ状態の公式／共有の観測が別集合へ入らない。CPUの分割例はこちら（非同梱の履歴資料: `../results/prepod-20261027/dummy-observation-split.json`）。

```bash
PYTHONPATH=src <cpu-venv>/bin/python scripts/prepod_experiments.py split-observations \
  --observations path/to/paired-run --output results/observation-split.json
```

## GPU再開後に残ること

実モデルの出力照合とT01比較、Ray workerへのGR00T/GPU割当の結合、実モデルのbatch 1/2/4/8/16の成立確認、容量探索、dtype/最適化/VRAM証拠の取得が残る。現在のprofile出力は`profile_export_allowed=false`としており、CPUログや容量未確認の結果をR01へ実測GPUprofileとして渡さない。

GPU向け条件案（非同梱の履歴資料: `../results/prepod-20261027/gpu-matrix-unresolved.json`）にはR04の105条件を定義したが、実行可能フラグはfalse。R03の容量5件と実観測・runtime fingerprintが未確定である。条件案の生成はGPU実行を開始する操作ではない。まず短い探索で条件を絞り、料金と所要時間を確認してから正式測定へ進む。

この1〜3工程の完了時点ではP2〜P4評価器とR07は未実装だった。2026年10月28日の[4〜5工程の実装結果](prepod-4-5.md)でCPU実装・検証を追加した。
