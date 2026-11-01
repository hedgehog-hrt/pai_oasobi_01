# 配布・論文比較・動的制御の統合（v0.16.0.dev1）

Pod起動前のCPU検証用。実GPU、実モデル性能、実ロボットの安全性を認定するものではない。

## 1. 配布設定からの再開

`python scripts/distribution_check.py --output distribution-check.json`は7種類の配布GPU設定をCLI生成物と照合する。S1の採用例にはR06 suite/bank/全cell対応表を必須とし、B>1例には数値回帰とcross-shape診断も含める。GPU設定例は`binding.kwargs.cuda_measurement=true`。未解決のパス・hardware・stateless確認を残した配布例は実行不能のままにする。

Pod再開後は配置した同一releaseから、(1)実パス・GPU名・検証済みrevision/runtimeとworker Pythonを設定、(2)全重み検査、(3)観測corpusと評価分割、(4)smoke、(5)kernel証跡、(6)pilot/profile、(7)R06・品質評価、(8)採用reviewの順に進む。認証情報は設定ファイルへ書かない。各スクリプトのdry-runを先行し、実行には既存Ray addressと明示予算を使う。`resume_workflow.py`が自動管理するのは環境・preflight・smoke・kernel・pilot/profileであり、R06・採用reviewを自動承認しない。

R06例は`examples/adoption/s1-request.template.json`と`s1-batch-request.template.json`。R06の`directories`は全cellを一度ずつrawディレクトリへ対応させる。B8/16には追加環境数を含む。`profile_adoption.py review-template`が生成する全`r06_checks`を人が確認する。CLI生成と配布例の差異を再発検出するテストを追加した。

## 2. R08/R09の固定配分比較

`paper_baselines.py plan --paper paper.json --reference rosa-plan.json --cell CELL --profiles batch1-profiles.json --output baseline-plan.json`でfull baselineを作る。profilesはcomponent ID→明示的なbatch-one profile ID。placement-onlyでは`--profiles`を省略し、referenceの各robot経路のprofile・batch待ちとクラス頻度を保持する。P4/8GPUの配分はTable 3。それ以外の端数配分は既存の固定規則を使う。referenceと比較cellのGPU予算、robot数、action時間、H、期限、周期、重み、Fmin=0の条件を照合する。

新しいplanはROSAのfeasible witnessとは区別する。baselineの構造・被覆・メモリ・モデル同一性を検査するが、SLO成立を事前認定しない。full baselineの頻度はnullであり、ゼロや巨大な擬似capではない。S1/S2は応答とactionの完了に従って次を発行し、Safety/Monitorの周期と評価期限は維持する。配置のみの比較で異なるbatch設定を維持するだけのGPU数が足りなければ拒否する。未測定profileや不一致を埋めない。

同じ`pipeline_experiments.py prepare/run`、`deploy_fleet`、固定window・warmup gate・raw再監査を利用できる。GPUでは既存のmodel/kernel/品質review済みprofileと同一のbindingを要求する。uncappedではaction-only上限を時間予算の目安に使い、実際の必要標本数は終了後に検査する。予測SLOやROSA目的値をbaselineに捏造しない。

各paper cellのrepeatに対応する`--case-id closed-loop-NNN`を指定して実行する。suiteは事前の反復数を保持する。モデル起動の前にdry-runと正の時間予算を要求し、成功済みattemptはスキップする。`paper_baselines.py report --paper ... --root ... --directories ... --references ...`は選択family内の全variant・全反復を要求する。familyは`R08:P4:8`等、referencesはfamily→ROSA planの相対path、directoriesはcell ID→raw path。全runでsource、bank、共通計測条件を照合し、同一runの流用、異なるrepeat、後から変更されたplanを拒否する。rawからQを再計算し、全条件完了時だけ反復対応のROSA/baseline比を出す。分母0は未定義のまま残す。

## 3. R10の受け入れ・standby切替

`control_experiments.py prepare`はcapped ROSA planを有限robot poolのblueprintにし、初期active robot一覧、primary→standby ID、standby込み物理GPU予算を固定する。join/leaveごとに既存配置の負荷、SLO、Fmin、平均closed-loop制約を再計算する。非単調profileも補間範囲内で扱う。未知robot・範囲外負荷・不成立は`requires_global_replan`で停止する。自動global replan/migrationは未実装であり、ここで暗黙の配置変更や性能保証を行わない。

worker隔離でepochを更新し、影響robotを停止、S2/Safety cacheを破棄する。同一backend・instance・sourceで起動済みのstandbyを検査して全componentをresetし、未参加robotの将来経路も更新する。非影響robotの経路は保持する。古い応答、superseded ticket、重複適用を拒否する。行動完了が不明なら再実行せず、外部の状態確認まで停止を保持する。standby不足や再起動も停止理由に残る。

`control_experiments.py run --spec ... --scenario ... --output ... --report ...`はdry-run。`--execute --budget-seconds ...`で自分のCPU Ray clusterだけにfault注入する。scenarioは最大256段階の予測・適用・join/leave・worker停止・healthcheck・failoverで、結果を事前指定する。停止は自分のServe applicationの削除であり、物理GPU故障の再現とは称さない。GPU向け`deploy_control` APIには既存profile adoption gateがあるが、このCPUシナリオCLIはGPUを受け付けない。実モデル・実simulatorの動的試験はPod再開後にadapter統合と実測確認が必要。

raw eventからP1期限内行動率、切替時間、復旧後最初の行動、故障前/中/後の区間を再計算する。P2〜P4のcache/安全停止制御は検証するが、この制御trace単独で全periodic-slot指標や実機安全性を認定しない。

## 4. 図表と独立した説明用収録

`result_media.py figures --request ... --root ... --output ...`は`pipeline_comparison`または`paper_comparison`の入力条件からrawを再監査し、ReportLab標準chartでSVGを生成する。元の再監査JSON、描画値、source/hashを併記する。Qとpaired速度比の誤差棒は独立反復のsample SD。未完了familyは描かずMISSINGと表示し、欠測を0に置換しない。CPUの図にはGPU性能ではない旨を表示する。SVGはそのまま文書へ組み込める。

`explanation_video.record`は実simulator/predict adapterを渡す独立の収録API。reset状態、seed、model/simulator provenance、観測配列、step、時刻を保存する。既存の性能実行系から呼ばず、全artifactを`performance_eligible=false`に固定する。GPUのprovenance文字列だけを実モデル検証の代用にしない。実GPU adapterの動作はPod後の確認事項。

`result_media.py record-cpu --output capture`でCPU収録の接続検証、`result_media.py video --capture capture --camera video.image --output video`でGIFに変換する。frame数/bytes/個別呼出時間を制限し、hash不一致、カメラ不一致、RGB形状不一致を拒否する。動画は一定fpsの説明再生で、実時間の速度比較や性能図には使用しない。欠測や上限到達を成功に読み替えない。GIFは追加encoder不要で、MP4は将来の配布形式の選択事項。

図の生成はoptional dependency `media`（ReportLab 4.4.9 / Pillow 12.3.0）を使う。今回の検証はローカルに既存のbundled packageを利用し、追加downloadやPod接続を行わない。

## 5. R05の負荷曲線と専用GPU比較の準備

`r05_curves.py prepare --pilots-suite ... --rates ... --repetitions 3 --output ...`で、1つの事前定義groupに対するB=1/2/4/8/16 × offered rate × repetitionを固定する。`run`は完了済み5pilot、対応corpus、同じsource/runtimeを確認してdry-runし、明示的`--case --execute --budget-seconds`で1caseを監督実行する。GPUには既存addressを必須とする。新しいraw専用directoryでのみ実行し、自動retryしない。中断時はcleanup記録を確認してから次の実行へ進む。

通常のR07 profileと区別した`r05_load_curve_calibration`ではstream数を対象Kのまま保ち、B>Kも実際のpartial batchで測る。diagnostic deadline 200msは旧probe集計の補助値であり、選択や行動のSLO条件には使用しない。CPUからGPUへの採用やR07への自動輸入は拒否する。GPU曲線には各rate3反復以上・1000成功・120秒以上・warmup20成功/30秒以上と計測範囲内のdispatch lagを要求する。timeout、missing arrival、source/runtime差異を除外して勝者を出すことはせず、予測生成を止める。

`r05_experiments.py predict --pilots ... --profiles ... --root ... --output prediction.json --selection-output selection.json`が新方式。profilesはgroup→batch文字列→raw path一覧。全capで同じrate/反復gridを使い、同一rateのmeanは反復最大値を採る。`λ=Kf`と`λ(t_action+L_B(λ))≤K`を各線形区間の二次式として解く。非単調も全区間を検査し、未測定範囲へ外挿しない。最大測定rateで決まる場合はright-censored。1%以内の候補では小さいbatchを選ぶ。元の5pilotは起動・runtimeの前提として残り、独立confirmationが別に必要。曲線上の推定はclosed-loopの到着分布や定常queueの成立を保証しない。

`dedicated_baselines.py --reference ... --profiles ... --method gpu_per_robot|gpu_per_model_per_robot --gpu-budget ... --output ...`は全robot/componentの専用配置を列挙し、物理GPU必要数・不足数・未使用数を記録する。同居モデルのsingle-model memory宣言を足してjoint peakを捏造しない。資源計画は`executable=false`で、専用実行adapterと実計測の確認待ち。同一GPUで複数モデルを動かすケースには、joint memory/latencyの追加計測が必要。資源不足時にrobot subsetへ縮小せず、要求条件全体が不足として残る。

## Pod前の完了範囲と実測待ち

この版で配布、固定配分比較、有限poolの動的制御、図/動画、R05負荷曲線のCPU接続を扱う。自動global replan/migration、動的P2〜P4の全評価ログ統合、専用GPU実行adapter、実モデルとの説明収録adapter、実GPU上の性能・memory・kernel・task successは完了を称さない。前者の追加実装と実モデルに依存する確認を混同せず、今回の制御/資源計画が持つ明示的な境界とする。8 H200、GR00T N1.6など論文と同じ条件の結果もまだ存在しない。
