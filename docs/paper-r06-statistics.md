# 論文比較条件・R06動作回帰・統計評価（v0.13.0.dev1）

2026-10-29。Pod起動前に実装・検証できる3項目を、条件生成→動作回帰→統計評価の順に追加した。GPU、GR00T重み、LIBERO、RunPodにはアクセスしていない。CPU結果は機構の検証であり、論文性能や実タスクの成功率ではない。

## 3. 論文比較条件の生成

`paper_experiments.py`はR05・R08・R09の事前条件表を作る。条件表そのものは実行planではなく、`executable=false`である。乱数seed、反復番号、条件ID、条件全体のSHA-256を保存し、再生成と一致しない条件表を拒否する。

- R05：K=1/4/16、行動時間=0.05/0.2/1秒の9群。各群にbatch=1/2/4/8/16のpilotを置く。別のconfirmationでbatch 1と予測候補を各3回以上測る。pilot群の完了を前提にし、pilotとconfirmationの順序を分けて無作為化する。
- R05の遅延SLOは`null`、`slo_policy=none`。200msや大きな仮の期限を代入しない。1GPU条件への変更と、Fig.14のworker数が未確定であることを残す。
- 予測候補は外部で得たbatchと予測証跡SHA-256を指定する。未指定なら未確定の行を保持する。予測候補がbatch 1なら再測定を重複させずaliasで表す。未測定設定に対する大域最適性は主張しない。
- R08：P1〜P4、K=8/16/32/64。R09：Mix2=P2/P4を半数ずつ、Mix4=P1〜P4を4分の1ずつ、同じKの集合。重みは1、Fminは未確定として残す。
- Equal/Weightedの全方式比較はS1 batch 1、S1/S2の最適頻度capなし、周期モデルはTable 1の頻度。配置だけの比較はROSAのbatch・頻度を固定するため、その参照ハッシュを未確定欄として要求する。両者は別の行である。
- P4・8GPUはTable 3のEqual=2:2:2:2、Weighted=1:3:1:3（S1/S2/Safety/Monitor）。他は各成分最低1GPU＋残り最大剰余法、同率ならcomponent ID順という本案件の補完を使う。Weightedのサイズ未指定は未確定。必要成分数に満たないGPU数は明示的なinfeasible行になる。
- P1・1GPUのEqual/Weightedは重複実行しない。同じ7B系列のS2とMonitorは別成分として保持する。

既定の8GPU表は459条件（R05=99、R08=240、R09=120）。1GPU表はP1のaliasを反映して435条件。これは実行予約数でも請求額でもない。runtime、観測corpus、profile、測定時間・標本数、ROSAの選定結果を確定してから実行用planへ接続する。既存の期限必須のランナーにR05をそのまま渡すことはできず、SLOなしの実行系への接続は別途必要である。

```bash
python scripts/paper_experiments.py --gpu-count 8 --output paper-8gpu.json
python scripts/paper_experiments.py --verify paper-8gpu.json --output paper-8gpu-checked.json
```

任意の`--sizes`は各成分の正のモデルサイズをJSONで指定する。`--r05-selection`の形式は例えば`{"k1-a50":{"batch_size":4,"prediction_evidence_sha256":"64桁の実証跡ハッシュ"}}`。架空のprofile値で空欄を埋めない。

## 4. R06の拡張

T01（alphabet soup）、T02（tomato sauce）、T03（cream cheese）の固定状態bankを使い、official/shared_b1/selectedと同時環境数1/2/4の27条件を生成する。GPU smokeはT01の10件、formalは3task各50件を要求する。CPU検証はsynthetic bankと明示し、少数件で同じループを検証する。

各条件では同じ初期状態・seed集合を一度ずつ消費する。同時環境数で総ケース数を増やさない。子プロセスごとに独立した環境を作り、バッチに空きがあってもケースの複製やpaddingをしない。5件／同時4環境なら4件＋1件のwaveとなる。選定batchが同時環境数より大きい場合、実際のbatchは小さくなる。設定上限のbatchを実行したとは扱わない。

選定batchは動作回帰の候補であり、採用済みの最適batchという意味ではない。officialは公式の公開`policy.get_action`をB=1で呼び、sharedは本番と同じ`predict_policy_batch`を使う。共有モデルに明示的な同期batchを渡す回帰実験であり、Ray動的batchingや独立クライアントの性能試験ではない。

各requestはcase seed＋観測番号を記録し、batch呼び出しには先頭rowのseedを使う。batch形状間のモデル内部乱数は整列していない。結果差を自動的に乱数だけの差と断定しない。各条件の順序は固定seedで無作為化する。

保存する証跡は全観測配列・action配列、初期状態と復元状態のhash、task、seed、request/episode/robot ID、実際のbatch構成・seed、適用開始・終了・実step数、全予定ケースの結果、source、simulator環境、ファイルhashである。reset時点で成功なら推論しない。応答全体の対応を検査してからactionを適用する。timeout/異常時は以降のケースをnot_startedとして残す。適用開始後に返答が失われたstepはunknownとして記録し、ゼロstepだったとは扱わない。

CLIは指定した1条件だけを子プロセスとして実行し、時間予算を超えれば所有process groupを停止する。再実行には新しい出力先を使い、既存結果は上書きしない。既存のGPUモデルserverは外部プロセスのため、失敗後には推論が継続している可能性を記録する。子processの停止だけでGPU推論完了とは判定しない。

### Pod再開後

固定済みGR00T policy環境で更新版`r02_server.py`を起動すると、従来endpointに加えて`r06_reset`／`r06_predict_batch`／`r06_runtime`を公開し、`r06_evidence.json`に全重み検証・runtime・modality設定を保存する。モデルを追加で複製する機構ではない。

既存のsim環境で、taskごとにbankをexportする。`r02_t01.py`はT02/T03にも対応し、capture/run時もbank内のtaskを環境へ渡すよう修正した。

```bash
python scripts/r02_t01.py export-states --repo /ABS/GR00T --task T01 --count 50 --output banks/T01
python scripts/r02_t01.py export-states --repo /ABS/GR00T --task T02 --count 50 --output banks/T02
python scripts/r02_t01.py export-states --repo /ABS/GR00T --task T03 --count 50 --output banks/T03
python scripts/r06_experiments.py prepare --mode libero_gpu --phase formal --bank-root banks --selected-batch 4 --server-evidence /ABS/server/r06_evidence.json --output r06-suite.json
python scripts/r06_experiments.py check --suite r06-suite.json --bank-root banks --output r06-check.json
python scripts/r06_experiments.py run --suite r06-suite.json --bank-root banks --cell T01-4-selected --output r06-preview.json
python scripts/r06_experiments.py run --execute --suite r06-suite.json --bank-root banks --cell T01-4-selected --sim-python /ABS/sim-python --sim-repo /ABS/GR00T --output runs/T01-4-selected
python scripts/r06_experiments.py reaudit --raw runs/T01-4-selected/raw --output r06-audit.json
```

上記はpathを実環境へ置き換える例。serverの既定portは5556。driverのPythonには既存GR00Tの`PolicyClient`をimportできる環境が必要。シミュレータには別のPythonを指定できる。全27条件はsuiteの`randomized_order`に従い、各回の成功・cleanupと時間予算を確認して進める。最初のT01・10件smokeには別bankと`--phase smoke`を使う。

動画収録は今回の機械検証には含めていない。実モデルの動作説明用の別実行として取得する。今回のR06集計は既存profile採用ゲートの数値回帰・paired証跡・人によるreviewに追加する証跡であり、それらを自動的に置換しない。

## 5. 統計評価

### 対応のある成功率差

比較キーはtask・初期状態hash・seed。officialと各候補を、同じ同時環境数内で比較する。taskや同時環境数を跨いで同じ初期状態を独立標本として水増ししない。初期観測・復元状態が一致しなければ区間推定しない。

差は候補成功率−official成功率。不一致ペアの「改善」「悪化」の確率にそれぞれClopper–Pearson区間を作り、Bonferroniの和の上界で差の保守的95%区間を作る。95%差区間には各確率97.5%区間（片側tail 1.25%）を用いる。周辺確率同士の独立性は仮定しない。[SciPy公式資料](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)のClopper–Pearson定義と公開数値例を検算に使った。実装は標準Pythonによる二項CDFの反転で、新規SciPy依存は追加していない。

50組すべて両方式成功でも差の区間は約−8.39〜+8.39ポイントで、±5ポイント以内の同等性は未確定となる。点推定の低下が5ポイント超なら`investigate_drop`。区間全体が事前margin内のときだけ、独立で代表的なケースという標本仮定の下でmargin内と表示する。固定benchmarkの代表性やモデルseed間の変動は、この計算から証明できない。区間は比較ごとで、18比較全体の同時保証ではない。

未完了・failed・missing・invalid条件は残し、全予定ケースの分母を表示する。未完了の場合は効果の信頼区間を出さず、欠測結果がすべて成功／失敗の場合の識別範囲を表示する。これは信頼区間とは異なる。

```bash
python scripts/statistical_evaluation.py r06 --suite r06-suite.json --directories r06-directories.json --root . --output r06-statistics.json
```

`r06-directories.json`は実行前にsuiteの全cell IDをraw出力先へ対応させる辞書。例は`{"T01-4-selected":"runs/T01-4-selected/raw", ...}`。欠測は行ごと残り、directoryの重複利用は禁止する。保存済みsummaryの数値を転記せず、原本配列・イベントを再監査してから集計する。

### 時間相関を考慮した遅延

既存profileとP1〜P4/R07ログを原本から再監査する。profileはscheduled arrival、pipelineは既存評価器と同じ非warmup・測定区間内sendを母集団とする。未送信の周期義務は別のslot統計に保持する。遅延分布は成功応答に条件付けし、失敗／timeout／not_sentをその分布に混入させず、全体の件数には残す。

非重複の時間ブロック内の全robot・全requestをまとめて再標本化する。要求block長から`floor(全期間 / block長)`個のブロックを作り、全期間を等分する。丸めによる幅の差は最大1nsで、実際の幅を保存する。末尾を捨てない。seed・反復数・p50/p95/p99とpercentile区間を保存する。最低8個の成功を含むblock、100〜10000回の反復、計算量上限を要求する。成功を一つも含まない再標本が出た場合も、その再標本を捨てず区間算出不可とする。

[arch公式の時系列bootstrap資料](https://bashtage.github.io/arch/bootstrap/timeseries-bootstraps.html)を依存構造を保持する方法の参考にした。ここで実装したのは本案件の非重複・等時間block方式で、archのcircular/stationary方式そのものではない。block長が依存時間を十分覆うこと、近似的な定常性、block長を変えた感度分析は実測時に確認する。標本不足なら独立requestのbootstrapへ切り替えない。成功応答1000未満のp99にはexploratoryを付す。

```bash
python scripts/statistical_evaluation.py latency --kind pipeline --raw /ABS/trial/raw --block-ns 5000000000 --seed 42 --resamples 2000 --output latency-5s.json
python scripts/statistical_evaluation.py latency --kind profile --raw /ABS/profile/raw --block-ns 10000000000 --seed 42 --resamples 2000 --output latency-10s.json
```

block長5秒／10秒は指定例で、普遍的な推奨値ではない。区間はrun内の不確実性であり、既存比較レポートの独立反復ごとの平均・標本SDとは別に表示する。異なるrunをrequest単位で混ぜない。これらの区間からGPU性能、全ワークロードのSLO、profile採用を自動決定しない。

## 検証記録

全体テスト490件、凍結した配布版からのCPU検証を行う。CPU実行検証は27条件各5ケース＝135エピソード・270 actionを再監査し、追加で誤応答停止1条件とCLI監督実行1条件を確認する。合計29 trial、独立simulator子processの起動144回。CPUダミーであるためT01〜T03の物体操作は実行していない。

結果は`results/prepod-paper-r06-statistics-20261029/`に保存する。配布用source ZIP、展開版、全テストJUnit、条件表、各試験の生ログ・配列、再監査結果、統計計算の入力／出力、release verificationを同梱する。実モデル・CUDA・LIBEROの50初期状態比較、選定batchのtask品質、R05のSLOなし実行、R08/R09の複数GPU比較、動画はPod再開後の検証対象である。
