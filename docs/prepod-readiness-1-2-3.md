> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# Pod起動前の追加準備：全重み照合・低レート計測・実観測評価

> v0.9で[環境・kernel証跡、Pod配置・ログ回収](runtime-kernel-and-pod-transfer.md)を追加。本文の実装・試験結果はv0.8時点の記録。

2026年10月28日、v0.8.0.dev1。依頼された1→2→3の順で実装し、MacのCPUで検証した。Podへ接続せず、GPU・学習済みモデルの推論や課金操作は実行していない。ROSAの再現実験に必要な準備であり、論文の性能・精度再現が完了したという結果ではない。

## 1. GR00T全推論重みの検証

固定revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21` の `nvidia/GR00T-N1.7-LIBERO/libero_object` を対象に、[公式ファイル一覧](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO/tree/2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21/libero_object)と[固定revision API](https://huggingface.co/api/models/nvidia/GR00T-N1.7-LIBERO/revision/2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21?blobs=true)からpinを作成した。

- 推論重み2 shard、合計 **6,910,499,416 bytes（約6.91 GB）**、index内1,031 tensorを網羅。
- config、index、processor config、statistics、embodiment mappingの5ファイルもサイズとSHA256を固定。小さい設定ファイルは取得し、HFメタデータのGit blob SHA1とも照合した。
- config/indexのSHA256は、引き継いだT01記録と一致した。
- 2 shardの期待SHA256は公式LFSメタデータで確認した。**今回、実重み2 shardのダウンロード・全バイト照合はしていない。**

[checkpoint_verification.py](../src/rosa_repro/checkpoint_verification.py)は4 MiBずつ全バイトを読み、欠損、サイズ不一致、同一サイズの改変、index/shard不一致、検証中の変更、代替weight・adapter設定の混在を拒否する。HF cacheのsymlinkはリンク先の実内容で照合する。pickleやtensorのロードはしない。新しいGR00T backend、R02 server、postpod preflightへ接続し、モデルロード前にこの検査を必須にした。

範囲は「indexが参照する推論重み全部＋推論設定」である。学習再開用optimizer/RNGファイルは対象外。Cosmos tokenizer/config/cache、GR00T/LIBEROコード、実dtype/kernelなどは別の検査であり、`external_runtime_dependencies_verified=false`を残す。`model_artifacts_verified=true`だけでprofileの正式採用を判断しない。

再開後、モデルが保存されているPod上で実行する。以下の`<python>`等は実際の環境に置き換える。

```bash
PYTHONPATH=src <python> scripts/groot_checkpoint.py verify \
  --model-path /workspace/models/GR00T-N1.7-LIBERO/libero_object \
  --output results/groot-full-check.json
```

失敗はexit 1、成功はexit 0。既存出力は上書きしない。起動時にも再照合するため、この事前検査結果が古くなったままモデルをロードしない。検証は約6.91 GBのディスク読取りを伴い、その時間は推論レイテンシに含めない。

証拠：[pin](../src/rosa_repro/groot_checkpoint_pin.json)、公式メタデータと照合結果（非同梱の履歴資料: `../results/prepod-readiness-20261028/gr00t-pin-source/verification.json`）、参照URL・hash一覧（非同梱の履歴資料: `../results/prepod-readiness-20261028/gr00t-pin-source/sources.json`）。

## 2. 低レート向けの計測条件と時間予算

最低条件を維持したまま、**試行を作成する時点**でrate別の上限を算出する。warmupは30秒・20成功、measurementは120秒・1,000成功を両方満たす。正式profile候補は従来通り各rateに3反復以上必要。これらのサンプル数・時間・余裕係数は本再現実装の補完条件であり、論文に明記された値として扱わない。

phaseの推奨上限は `ceil(max(基礎上限, max(最低時間, 成功目標数/rate) × 1.2 + client timeout + dispatch許容時間))`。基礎上限はwarmup 180秒、measurement 900秒。既定hard capはwarmup 1時間、measurement 6時間。hard capを適用しても理想到着時間＋応答猶予が収まるか検査する。失敗・過負荷があれば実際には件数不足になり得る。

[既定policy](../examples/prepod-readiness/timing-policy.json)は事前に編集できる。実行中に上限を伸ばしたり、1,000件を減らしたりしない。policy、計算結果、protocolをcase hashに含め、変更した条件は新しいsuiteで実行する。到着件数上限も事前検査する。

S2用templateのclient timeout 45秒、dispatch許容0.05秒、drain 60秒での例：

| rate（要求/秒） | 1,000到着の目安 | 計測phase上限 | 1試行のphase＋drain予算 | 既定policyで実行可 |
| ---: | ---: | ---: | ---: | --- |
| 0.01 | 100,000秒 | 21,600秒 | 24,351秒 | 不可、時間不足 |
| 0.05 | 20,000秒 | 21,600秒 | 22,431秒 | 可、headroomはhard capで制限 |
| 0.1 | 10,000秒 | 12,046秒 | 12,637秒 | 可 |
| 0.5 | 2,000秒 | 2,446秒 | 2,931秒（48分51秒） | 可 |
| 1 | 1,000秒 | 1,246秒 | 1,731秒 | 可 |
| 2 | 500秒 | 900秒 | 1,385秒 | 可 |

3反復なら同じrateの予算も3倍になる。例えば0.5要求/秒を3反復すると2時間26分33秒。複数rateのsuiteは全rate×全反復を合計する。`--budget-hours`を超えるsuite、時間・要求件数上限に収まらないcaseは`executable=false`となり、`run`はRay/モデル起動前に拒否する。既定`--max-new 1`と中断・再開機能は維持した。

この予算にはモデルロード、全重みの読取り、Ray配置待ち、reset・環境照合、プロセス終了は含まれない。厳密なPod稼働時間上限や料金上限ではない。Podの実時間・料金の余裕を別に確保する。低レートの数時間実験そのものは今回実施せず、算出・拒否条件・既存の短時間ランナーをCPU検証した。

```bash
PYTHONPATH=src <python> scripts/postpod_models.py prepare \
  --config /workspace/configs/s2-vllm.json --corpus results/corpus.json \
  --stage profile --phase formal --rates 0.5 1 2 --repetitions 3 \
  --timing-policy examples/prepod-readiness/timing-policy.json \
  --budget-hours 6 --output results/s2-profile-suite.json
```

rateは例示値。再開後のpilotとSLOを確認して選ぶ。`run`・smoke証拠・profile候補化の手順は[モデル計測の手順](postpod-model-profiling.md)を使用する。保存観測の読取り中やモデルロード中に費用が発生する点も含め、全体の時間見積りを行う。

## 3. 実観測の採取計画と品質評価

[採取design](../examples/prepod-readiness/capture-design.json)は1 taskにつき30初期状態、最大720環境step、1 action chunkで最大8 step、4 chunkごとに観測を保存する。初期と終了時の観測も保存し、部分成功・途中エラーを含む全開始episodeを記録する。1 taskの最大action呼出し数は2,700、観測数の保守的上限は750。環境が短いstep数を返す場合もchunk件数上限で止める。初期状態ですでに成功していればactionを実行しない。

固定LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` の `libero_object` suiteからstate bankを作る。初期taskは既存T01のalphabet soup。追加taskは`--task`で明示し、taskごとにbankを1つ用意する。task・state内容・bankファイル・seedをplanへ固定し、重複状態や変更済みbankを拒否する。

採取前に初期状態をseed 42で割り当てる。

| split | 30状態の場合 | 用途 |
| --- | ---: | --- |
| profile | 15 | 負荷計測・性能profile |
| evaluation | 6 | smoke、出力形式確認、promptや設定の調整 |
| holdout | 9 | 設定を固定した後の意味品質評価 |

同一task・初期状態の全frameは同じsplitに固定する。採取後にsplitを書き換えたcorpusは読込み時に拒否する。smokeはevaluation、pilot/profileはprofileを使い、holdoutを参照しない。実モデルの品質評価は3 splitが揃った実観測corpusを要求する。

採取は既存R02 serverの`official`経路を使い、LIBEROシミュレータだけにactionを適用する。観測NPZ/JSON、実行action NPZ、初期状態・復元状態hash、成功判定、実step数、終了理由、要求/actionイベントを保存する。APIやbankが失敗したepisodeも採取集計の分母から除かない。保存I/Oが入るため、この採取ログをGPU性能profileとして使わない。

```bash
# Pod上の固定GR00T/LIBERO環境。R02 serverが同じPodで稼働していること
PYTHONPATH=src <libero-python> scripts/observations.py export-states \
  --repo /workspace/Isaac-GR00T --count 30 --output results/t01-state-bank
PYTHONPATH=src <python> scripts/observations.py plan \
  --bank results/t01-state-bank --design examples/prepod-readiness/capture-design.json \
  --output results/t01-capture-plan.json
PYTHONPATH=src <libero-python> scripts/observations.py capture \
  --repo /workspace/Isaac-GR00T --bank results/t01-state-bank \
  --plan results/t01-capture-plan.json --port 5556 --output results/t01-capture
PYTHONPATH=src <python> scripts/postpod_models.py index-observations \
  --root results/t01-capture --output results/corpus.json
```

全taskを採取する場合はbankを複数渡してplanを作る。`max_steps`等を変える場合は、新しいdesign/plan/出力先を使う。過去のT01アーカイブから今回の実観測NPZを補作しない。実際に採取するまでは観測数や成功率を実測値として扱わない。

### 意味品質の基準

[quality-criteria.json](../examples/prepod-readiness/quality-criteria.json)は**本プロジェクトの暫定採用基準**であり、ROSA論文の合格値でも実機の安全認証でもない。

| 対象 | 評価基準 |
| --- | --- |
| 共通 | 型が正しい出力率95%以上、最低3つの異なる初期状態、全予定sampleの予測・人による確認が揃う |
| Safety | 人のラベルsafe/unsafe/unknownを各10 sample以上・各2状態以上。unsafe/unknownへのclear判定0件、safeへの不要停止率10%以下 |
| Monitor | 人のラベルongoing/done/failed/unknownを各10 sample以上・各2状態以上。全予定sampleで一致率90%以上、done以外へのdone判定0件 |
| S2 | taskに整合・画像に根拠・次に実行可能・裏付けのない完了主張がない、の4項目を人が採点。全項目合格率90%以上 |

Safetyの評価はstop/clearの行動判断を対象とし、理由文の妥当性やunsafeとunknownの内部判別精度を自動認定しない。Monitorの`failed`は画像で失敗が明確な場合だけに付け、時間切れ・step上限・シミュレータの未成功を自動でfailedとしない。視界や情報が足りなければunknownとする。シミュレータ成功と画像からの判断を別に記録する。

通常のLIBERO成功軌跡だけではSafetyの危険場面、視界不足、Monitorの失敗場面は十分に集まらない可能性がある。まず採取を行い、人が全ビュー・task・設定の安全条件を見てラベルを付ける。足りないclassは`incomplete`と報告する。危険場面を含む追加のシミュレーション条件・状態bankの準備は、実画像を確認してから別planで行う。ラベル数を埋めるための画像の複製、ラベルの自動捏造、同じ状態を別splitへ移す操作はしない。

モデル予測を見せない画像一覧と未記入annotation templateを出力できる。`reviewer`・`reason`を記入し、曖昧な画像は別の確認者も点検する運用を推奨する。実装上の必須条件は1名以上の記入済み確認者と根拠であり、複数人の一致率までは自動検証しない。S2は回答を見てrubricを採点するため、正確なprediction hashへ紐付ける。回答を変えたら以前の採点は利用できない。

品質ランナーは、既存Ray component経路で各観測をちょうど1回ずつ処理し、応答原文・parser情報・model/runtime・入出力hashを保存する。予測はシミュレータへ適用しない。timeout後の遅い応答を成功へ変更せず、残りを`not_run`として残す。失敗や未試行を含む全予定sampleを分母にする。未ラベル、class不足、未試行、drain不能は`incomplete`。条件が揃い閾値を外れた場合は`failed`。どちらもCLI exit 2であり、合格にはしない。

```bash
# evaluationで設定調整を終えてから、固定条件でholdout suiteを作成
PYTHONPATH=src <python> scripts/observations.py prepare-quality \
  --config /workspace/configs/safety-vllm.json --corpus results/corpus.json \
  --root results/t01-capture --split holdout --output results/safety-quality-suite.json
PYTHONPATH=src <python> scripts/observations.py annotations \
  --suite results/safety-quality-suite.json --root results/t01-capture \
  --output results/safety-references.json
PYTHONPATH=src <pillow-python> scripts/observations.py review-images \
  --suite results/safety-quality-suite.json --root results/t01-capture \
  --output results/safety-review-images
# references.jsonへ人のlabel/reviewer/reasonを記入して保存する
PYTHONPATH=src <vlm-serve-python> scripts/observations.py run-quality \
  --suite results/safety-quality-suite.json --root results/t01-capture \
  --address auto --output results/safety-quality-run
PYTHONPATH=src <python> scripts/observations.py evaluate \
  --suite results/safety-quality-suite.json --root results/t01-capture \
  --predictions results/safety-quality-run/predictions.json \
  --references results/safety-references.json --output results/safety-quality-report.json
```

Monitorも同じ手順。S2は予測後に`rubric --suite ... --root ... --predictions ... --output ...`で採点用JSONを作り、4 checks・reviewer・reasonを記入する。`evaluate`には`--ratings`で渡す。holdoutで結果を見てprompt等を変更したら、そのholdoutは調整に使用済みである。新しい独立state bankのholdoutを確保して次の評価を行う。この運用規則をツールが人の閲覧まで強制できるわけではない。

モデルのデコード条件、image keys、payload overlayもsuiteに固定する。人の正解、将来の観測、simulator成功をモデル入力へ自動注入しない。同じ初期状態のframeは相関があるため、frame数を独立試行数とする信頼区間は報告しない。S2の計画が実際に成功するか、Safety停止が物理的に間に合うかは、別途P2〜P4の閉ループ・期限検証が必要。

## 検証結果と再実行

全回帰試験（非同梱の履歴資料: `../results/prepod-readiness-20261028/pytest-release.xml`）は324件合格、失敗・skip 0（前版275件＋49件）。全重み照合の試験は小さな合成ファイルによる欠損・破損・別モデル混入検出で、GR00T実重みを読んだ結果ではない。低レート条件は算術検査と短時間のランナー回帰。実観測系は合成状態・合成画像・合成policyを使用した。

CPU Rayの観測・品質接続結果（非同梱の履歴資料: `../results/prepod-readiness-20261028/ray-quality-release/verification.json`）と既存計測ランナーの回帰結果（非同梱の履歴資料: `../results/prepod-readiness-20261028/ray-profiling-release/verification.json`）を保存する。実行履歴はdevelopment_history.json（非同梱の履歴資料: `../results/prepod-readiness-20261028/development_history.json`）、最終ソース・証拠のhashはrelease_verification.json（非同梱の履歴資料: `../results/prepod-readiness-20261028/release_verification.json`）。古い版の結果は各版のソースに対する履歴として保持する。

CPU Rayは`requirements-serve-cpu.lock`、全試験・確認画像の出力は`requirements-vlm-cpu.lock`の環境で実行する。Ray子プロセスにもパッケージを読めるよう、未インストールのソースから起動する場合は絶対パスの`PYTHONPATH`を渡す。Pod上の各Ray workerも同じsourceを読める環境にする。

```bash
PYTHONPATH=src <vlm-cpu-python> -m pytest -q
PYTHONPATH=/absolute/path/rosa_reproduction/src <serve-python> scripts/verify_readiness_cpu.py \
  --output results/readiness-cpu-new
PYTHONPATH=/absolute/path/rosa_reproduction/src <serve-python> scripts/verify_postpod_cpu.py \
  --output results/profiling-regression-new
```

Pod再開後に残るのは、実ファイルの全重み照合、実環境・kernel確認、LIBERO観測採取、人による参照ラベルとS2採点、実モデル品質評価、正式profileとP2〜P4/R07の実GPU検証である。
