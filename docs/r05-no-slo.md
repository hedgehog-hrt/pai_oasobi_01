# R05：SLOなし実行系への接続（v0.14.0.dev1）

2026-10-29。前版のR05条件表を、専用の実行plan・Ray Serve worker・固定時間ランナー・原本監査・比較レポートへ接続した。MacのCPUで検証したもので、Pod、GPU、学習済みモデル、実LIBEROは使用していない。

## 実行するもの

R05のK=1/4/16、行動時間0.05/0.2/1秒、最大batch=1/2/4/8/16を条件表から読み込む。単一workerがモデルを共有し、各robotは最大1要求だけを実行中にする。正常応答を受け取り、指定した仮想行動時間を待ち、その完了後に次の要求を送る。初期送信のずらし方と観測の循環順序はtrial seedから固定する。

- `latency_slo_ms=null`、頻度capなし。`deadline_ms`自体を設定に持たず、追加すると拒否する。200msや大きな代替期限を入れない。
- 既存の`ComponentWorker`、`ComponentCore`、GR00T backendを使う。batchはRay Serveが実際の同時要求から組み立てる。`deploy_r05`はR05専用planを検査する。
- GPU時はworker 1個に物理GPU 1枚を予約し、GPU種別、割当、モデルrevision、全重み検証、source、backend/runtimeを記録する。GPUが足りなければCPUへ切り替えない。R07のprofileを作り上げて起動する必要はない。
- 既存のSLO付きprobe／P1〜P4評価器の契約は変更しない。R05ログは専用の集計器で扱う。
- `batch_size`は上限。K=1・batch上限16なら実batchは1であり、padding・観測複製で16にしない。

入力は保存観測を再生する。返されたactionのshape・数値・ID対応を検査するが、actionをロボットやシミュレータへ適用することはない。`physical_actions_executed=0`、`action_semantics=virtual_timed_chunks`と明示する。R06のタスク成功率とは別の実験である。モデルの乱数をbatch間で整列したとは扱わない。

## 計測と停止

主指標は**固定計測区間内に開始・完了した仮想行動チャンク数／秒**。応答速度だけでなく行動待ち時間を含む閉ループの処理量を測る。正常な200ms超の応答も、行動開始時点が区間内なら利用する。区間外に完了した行動は別集計して主指標から除く。

要求は区間内送信を母集団にする。区間終了後の正常応答も遅延分布と件数には残すが、新たな行動を開始しない。成功応答cohortの完了レートは別名で表示し、窓内完了の行動レートと混ぜない。平均・p50・p95・p99、全status数、実batch分布、workerのbatch回数、queue待ち・モデル処理時間を保存する。

warmupと本計測は固定長であり、標本不足を理由に自動延長しない。warmup cutoffまでの成功応答が不足すれば、本計測前に停止する。この場合`metrics=null`であり、未実施の処理量を0とは記録しない。warmupで開始した要求・仮想行動を処理し終えてから本計測を始める。

`watchdog_s`は要求が戻らない場合の停止監視で、遅延SLOではない。期限に達した要求はtimeoutとして確定し、後から正常応答が来ても行動に使わない。試行を失敗として残し、drainを待つ。環境・inventory・resetなどの制御RPCには`max(5秒, watchdog_s)`の別の上限を使う。非常に短い要求watchdogの故障試験で、環境情報の採取自体が先にtimeoutすることを防ぐ。

親CLIは試行を別process groupで監督し、モデルロード・制御RPC・計測・drain・cleanupを含む壁時計予算で停止する。workerのbusy/inflightが残る場合はdrain成功としない。GPUで停止・cleanupが未確認のattemptは、原本hashに対応したcleanup確認がなければ再試行を止める。自分が作ったServe applicationだけを削除する。Podの起動・停止・レンタル・課金操作は行わない。

## 条件の準備

1. `paper_experiments.py`で論文比較条件表を生成する。
2. `r05_experiments.py template`で専用設定を作り、実環境のモデルpath、固定revision、runtime、GPU機種、stateless検証結果を設定する。
3. 保存観測corpusとtimingを結び付けてpilot suiteを作る。`--group`未指定なら9群すべて、指定すればその群だけを作る。
4. dry-runで条件・1試行上限・実行可能性を確認する。実行には`--execute`と明示した`--budget-seconds`が必要。新規試行は既定で1件。

```bash
python scripts/paper_experiments.py --gpu-count 1 --output paper.json
python scripts/r05_experiments.py template --output r05-config.json
python scripts/postpod_models.py index-observations --root /ABS/captures --output corpus.json
python scripts/r05_experiments.py prepare --paper paper.json --config r05-config.json --corpus corpus.json --stage pilot --group k4-a50 --output pilot-suite.json
python scripts/r05_experiments.py run --suite pilot-suite.json --corpus-root /ABS/captures --output runs/pilot --report pilot-preview.json
python scripts/r05_experiments.py run --execute --suite pilot-suite.json --corpus-root /ABS/captures --address EXISTING_RAY_ADDRESS --output runs/pilot --max-new 1 --budget-seconds 1800 --report pilot-first.json
```

未解決の`SET_`や`/SET/`を含む設定ではprepareを拒否する。ここでのpath・Ray address・予算は例であり、Pod再開後の実環境へ置き換える。driverとworkerには同じ凍結版の`rosa_repro`を利用可能にする。別worker Pythonは既存の`binding.runtime_env.py_executable`で指定できる。

### 低レートの時間予算

GPU confirmationはwarmup最低30秒・20成功、本計測最低120秒・1000完了行動を要求する。GPU pilotは最低5秒・5成功、10秒・10完了行動。CPU検証だけは明示した`cpu_smoke`条件を使う。

行動時間だけでも所要時間の下限がある。例えばK=1・行動時間1秒では120秒で1000完了行動は不可能なので、準備時に`executable=false`とする。推論と初期遅延が正であることを考慮した上限`K × (ceil(duration / action_time) − 1)`を用いる。これは実測能力の推定ではない。pilotの実測から十分な余裕を持つ固定期間を事前指定する。

`--protocol`には、`r05_protocol.timing`が生成するJSONを渡す。

```python
from rosa_repro.r05_protocol import timing
from rosa_repro.regression import write_json
write_json('pilot-timing.json', timing('pilot', warmup_s=30, duration_s=30))
write_json('confirmation-timing.json', timing('formal', warmup_s=60, duration_s=2000))
```

上記の秒数は指定例で、成功数を保証しない。速い群まで一律に長くせず、必要なら`--group`ごとにsuiteを分ける。時間上限・要求件数上限・全体予算を事前に確認する。実行中の標本数で期間を延ばさず、不足はshortfallとして残す。

## pilotから独立再測定への接続

confirmationには次が必要になる。

- 同じgroupのbatch上限1/2/4/8/16すべてについて、正常完了したpilotのraw directory。
- 同じモデル設定、batch待ち時間、corpus、source、観測されたbackend/runtime。
- 予測batchとその根拠artifact。論文条件表内の`prediction_evidence_sha256`を、実ファイルのSHA-256と一致させる。

予測artifactは以下の形で、数値は実測に基づく選定内容へ置き換える。

```json
{
  "kind": "r05_batch_prediction",
  "config_sha256": "R05設定のcanonical digest",
  "corpus_id": "保存corpusのID",
  "groups": {"k4-a50": 4},
  "method": "予測方法とその適用条件",
  "source_evidence_sha256": ["根拠となる測定証跡のSHA-256"]
}
```

接続層はartifact本文、設定・入力・選定batchの一致を検査する。予測手法の妥当性や大域最適性を自動保証するものではない。CPU実行証跡の候補4は接続確認用の明示値であり、GR00Tの最適batchと称していない。

`pilot-dependencies.json`はgroupから5つのraw directoryへの対応とする。`--evidence-root`配下の相対pathで、重複や不足を拒否する。各directoryは同じ固定releaseで原本監査され、成功したpilotだけを前提として使う。

```bash
# prediction.jsonのファイルSHAと選定batchをr05-selection.jsonに記載して条件表を更新する
python scripts/paper_experiments.py --gpu-count 1 --r05-selection r05-selection.json --output confirmation-paper.json
python scripts/r05_experiments.py prepare --paper confirmation-paper.json --config r05-config.json --corpus corpus.json --stage confirmation --group k4-a50 --protocol confirmation-timing.json --prediction-evidence prediction.json --output confirmation-suite.json
python scripts/r05_experiments.py run --execute --suite confirmation-suite.json --corpus-root /ABS/captures --address EXISTING_RAY_ADDRESS --pilots pilot-dependencies.json --evidence-root /ABS/experiment-root --output runs/confirmation --max-new 1 --budget-seconds 3600 --report confirmation-first.json
```

batch 1と予測候補をそれぞれ独立に3回以上測る。pilot値をconfirmationに再利用しない。予測候補がbatch 1ならaliasとして重複実行しない。suiteのseed付き順序を維持し、`--cell`で指定した条件だけを実行することもできる。

`ExperimentStore`がsingle-writer lock、immutable attempt、成功済みのhash検査とskipを担当する。再開は同じsuiteと出力先、新しい`--report`で実施する。失敗attemptは残し、次は新しいattempt番号を使う。dry-runや予算不足時にはモデルを起動しない。

## 原本監査と比較

```bash
python scripts/r05_experiments.py reaudit --raw runs/pilot/r05-XXXX/attempt-001/trial/raw --output audit.json
python scripts/r05_experiments.py report --suite confirmation-suite.json --directories confirmation-directories.json --root /ABS/experiment-root --output comparison.json
python scripts/statistical_evaluation.py latency --kind r05 --raw /ABS/trial/raw --block-ns 5000000000 --seed 42 --resamples 2000 --output latency-blocks.json
```

`confirmation-directories.json`はsuiteの全cell IDをraw directoryへ対応させた事前表。同一runの重複利用を拒否し、missing／failed／shortfall／invalidを行ごと残す。群がすべて完了した場合のみ、反復番号で対応させた「候補の行動完了レート／batch 1のレート」と、その反復間平均・標本SDを表示する。request数を独立反復数とみなさない。分母0は未定義として残す。

監査はファイルhashに加え、要求・episode・robot・workerの対応、同じrobotでの要求／行動の重複、観測の循環順序、固定window、実際の行動待ち時間、cutoff後の新規適用禁止、warmup gate、各batchのrow数とworker側counter、rawから再計算した集計を確認する。

遅延の時間ブロック評価は前版の統計評価器へ`--kind r05`で接続した。成功応答だけの条件付き分布であり、全要求の失敗数を残す。窓後に完了した要求も送信cohortに含める。block数不足時は区間算出不可とする。計測区間内のピークメモリや継続的な背景負荷の診断は、このR05の前後telemetryだけからは保証しない。

## 検証と残る実測

全体テスト511件。配布版からの実CPU Ray検証は14試行：5種類のpilot、独立confirmation 6試行、K=1・batch上限16・行動時間1秒の1試行、warmup早期停止1試行、停止監視timeout1試行。正常12試行と意図した停止2試行の証跡を残す。

200ms超の応答から行動が完了すること、実batch 4、上限16でもK=1なら実batch 1になること、1秒の実待機、warmup拒否後の計測禁止、遅い応答のdrainと不適用、dry-run・予算停止・再開skip、原本再集計、missing時の速度比抑止を確認する。

証跡は`results/prepod-r05-no-slo-20261029/`。R05のSLOなし実行系への接続は実装済みで、Pod再開後は実GR00T・実GPU・保存実観測を使い、時間予算と予測根拠を確定して同じ経路で検証する。今回のCPU速度比を論文の速度向上やGPU能力と読み替えない。Fig.14のworker数など未確定の論文条件があるため、1GPUへの条件変更を明記する。
