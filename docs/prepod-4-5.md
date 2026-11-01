> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# Pod起動前の実装 4→5：P2〜P4評価器と異種スケジューラ

> 後続の[ログ収集・R07実行接続](pipeline-runtime.md)をv0.5で追加した。以下はv0.4時点の結果。

2026年10月28日、v0.4.0.dev1。P2〜P4の評価器を実装・CPU検証した後、異種タスクスケジューラR07を実装・検証した。[ROSA論文v1](https://arxiv.org/html/2607.01088v1)のTable 1と§3.4.3、および既存の再構成実験計画v3を基準とする。公式実装との一致を主張するものではない。

Podへの接続・起動・課金変更、新しいモデル推論は実施していない。使用した入力・プロファイルはすべて人工データである。GPU性能やLIBERO成功率の再現結果として扱わない。

## 結果と実行証拠

回帰試験は**165件合格（既存117件＋追加48件、失敗・skipともに0）**。追加には評価器28件、R07 16件、CLI 4件を含む。R07の1テスト内で32個の固定seed小規模構成を独立oracleと照合し、評価器の1テスト内で40個の周期欠落・遅延パターンを独立計算と比較した。

代表10ケースの検証結果（非同梱の履歴資料: `../results/prepod-4-5-20261028/verification.json`）に入力・ソースのSHA256を保存した。回帰試験のJUnit（非同梱の履歴資料: `../results/prepod-4-5-20261028/pytest-junit.xml`）も同梱する。

| 条件 | 結果 | 意味 |
| --- | --- | --- |
| P2正常、P3正常 | 各2チャンク適格 | 全要求成分を照合 |
| P4正常 | 10チャンク適格 | S2のH=10以内 |
| P2 Monitor遅延送信 | 適格0 | 応答自体は50msでも、予定時刻から2,040msのため違反 |
| P3 Safety未送信 | 適格0 | 記録のない周期スロットも欠落として生成 |
| P4 S2を11回使用 | 10適格、11回目不適格 | 未適用・warmupのS1呼出しも使用数に含む |
| Mix2：P2×2、P4×2、各2Hz | 4 worker、目的値8 | MonitorとS1のクラス間統合、成分ごとの完全な経路 |
| Mix4：P1〜P4各1、各2Hz | 4 worker、目的値8 | 同じ成分を共有し、各クラス固有の成分を保持 |
| 重み3:1、最低各1Hz、容量3Hz | 頻度2Hz/1Hz、目的値7 | 解析解・明示格子の全列挙と一致 |
| greedy統合の反例 | scheduler=`unknown`、全列挙は成立 | 2+2を先に統合すると失敗するが、4+2 / 3+2なら2GPUで成立 |

Mix2/Mix4の適応探索も有効な配置を返した。人工プロファイル上の目的値下限はそれぞれ15.4052734375 / 15.70068359375であり、最大値の証明ではない。固定2Hzの検算とは別に結果JSONへ記録している。評価器の短い人工ログは境界条件検証用で、論文の0.2秒行動を実行した測定ではない。

## 4. P2〜P4評価器

実装は`src/rosa_repro/multicomponent.py`。P1にも対応し、旧P1集計との縮退一致を検証した。v2 JSONはclass・episode・固定測定区間・監査終了時刻を明示し、request/actionに依存情報を記録する。旧v1ログからS2や周期要求を推測して埋めない。

| パイプライン | 必須成分と既定値 |
| --- | --- |
| P1 | S1、期限200ms |
| P2 | S1 + Monitor 0.5Hz、期限2秒 |
| P3 | S1 + Safety 2Hz、期限0.5秒 + Monitor |
| P4 | S1 + S2、期限2秒・H=10 + Safety + Monitor |

値は明示入力できるが、P1〜P4の役割集合は固定で検査する。モデルの正確さや安全判断の意味は評価器では判定せず、`valid_output`と`status`を上流が記録する。

### 周期スロットと適格行動の対応

予定時刻を `run_epoch_ns + phase_ns + slot_index × period_ns` と定義する。期限は予定時刻から計算するので、送信を遅らせても締切は延びない。episode内に予定されたスロットを先に生成し、未送信を`missing`、送信したが締切に間に合わなければ`violated`とする。同じスロット内のretryは適格スロット数を増やさない。監査終了より締切が後なら`pending`である。

行動開始時点で、期限が到来済みの最新スロットを各周期成分について参照する。未来スロットの早い成功で、過去の欠落を埋めることはできない。まだ期限到来済みのスロットがない場合はstartupとして不適格にし、startup robot秒を別記する。startup時間はQの壁時計分母から引かない。状態はepisode境界でリセットする。

周期集計の分母は、締切が測定区間`start,end)`にある期待スロット数。episode終了前に予定されたスロットは、その締切がepisode終了後でも監査台帳に残す。別episodeの行動には結び付けない。この対応付け・startup処理・retry規則は、論文で細部が公開されていない部分への本プロジェクトの補完仕様である。結果に`paper_official_metric_equivalence_verified=false`を残す。

### S2、停止、集計

S1要求は利用したS2のrequest IDを明示する。同じrobot・episodeのS2であり、S1送信前に有効な応答があり、S2自身の期限内で、使用数がH以下の場合だけ適格とする。HはS1呼出し回数で数え、warmup・失敗・未適用でも消費する。行動を取り消して使用数を戻さない。

`safety_stopped=true`の判断は別集計し、行動が適用されていたら入力矛盾として拒否する。モデルの時間条件を満たす停止判断も、実行した適格行動数には加えない。フラグの正しさや停止区間の実世界の挙動を評価器が確認する機能ではない。

Qは `Σ class_weight × 適格な適用チャンク数 / 固定測定秒数`。8制御ステップを8行動として数えない。S1送信と行動開始の両方が測定区間内で、warmupでないものを集計する。各成分の要求数、状態別件数、送信起点p99、有効応答数が少ない場合の探索的フラグ、予定時刻起点の周期達成率を分けて残す。ある成分の期間全体p99違反を理由に、期間内の適格行動を一括ゼロにはしない。

時計混在、重複適用、別episode/robotのS2参照、締切宣言の不一致、未確定の監査、同期S1や行動の重複は拒否する。全件を単一の比較可能な単調時計へ記録する必要があり、サーバー時刻を無補正で混ぜない。

## 5. 異種タスクスケジューラ R07

実装は`heterogeneous.py`、独立した検算は`heterogeneous_oracle.py`。ここで異種とは**タスククラス**の違いを意味し、GPUは同一機種を仮定する。目的関数は`Σ v_c K_c f_c`、最低量は`K_c f_c ≥ Fmin_c`である。`min_action_rate_hz`はクラス合計の最低量として入力する。

1. classごと、モデル成分ごとにOR-Toolsで独立配置を求める。義務モデルではworker数を優先し、S1/S2では平均遅延を優先する。
2. 義務モデルを先に確保し、同じ成分を扱うクラス間でgreedy統合する。S1/S2も必要なら統合する。低負荷workerを、高headroomの候補へ順に統合する。
3. 統合時に合計要求頻度、全クラスのうち最も厳しい締切、メモリ、各robotの閉ループ条件を再検査する。S2はf/H、周期モデルはクラスごとの指定周期を使う。
4. `f_c ≤ 1/(t_act,c + mean_S1 + mean_S2/H_c)`を満たす構成を返す。平均は閉ループ、p99は成分SLOに使い分ける。
5. Fminを満たす初期点から、重み×robots×未探索幅が最大の次元を選んで中点を試す。許容目的値幅、時間、反復上限で停止する。

同じ重みファイルを使っていても、S2とMonitorなど異なるcomponent IDを同じGPUへまとめない。共有component IDのrole・モデルrevision・runtime fingerprintがクラス間で異なる場合は入力を拒否する。1 workerに1 componentを置くモデルであり、モデル混載のVRAM最適化は範囲外。

返した解は、配置生成から独立した検査で、worker値、全robotの成分被覆、双方向route、rate、Fmin、p99、memory、閉ループを再計算する。

### 未確定と最適性の扱い

`feasible`は、与えたプロファイルに対して検査済みの正の証拠がある場合。`infeasible`は行動時間からの必要上限や必須成分数からのGPU下限など、証明できる必要条件違反に限定する。配置選択の失敗、greedyの統合失敗、未測定領域、時間切れは`unknown`。

論文の探索方針に沿って、既知候補でのgreedy失敗時は探索区間を縮めるが、これを大域的な不可能性証明とは扱わない。`heuristic_pruning_used`、`heuristic_rate_brackets_hz`、`unresolved_dimensions`を保存する。全体の出力は`validated_heuristic_lower_bound`、`optimality_proven=false`。義務モデルのクラス内最少配置とgreedy統合も、クラスを横断した全体最少GPU数の証明ではなく、`minimum_obligation_footprint_proven=false`である。

未知のプロファイル問い合わせから上限を捏造しない。非単調profileは適応探索を行わず`unknown`。初期Fmin=0でも全robotのrouteを要求する実装のため、0Hzのprofile点を含まない入力では初期点が`unknown`になる。未知の0Hz遅延を外挿せず、実測の初期点設計をPod再開後に確定する必要がある。

[保存したgreedy反例（非同梱の履歴資料: `../results/prepod-4-5-20261028/r07_greedy_counterexample.json`）は、一般規模での探索完全性を保証できないことを示す。自動的に全列挙解へ置き換えない。独立oracleは4 robots・6 GPU・100,000格子点以下に制限し、明示された周波数格子について全stream集合分割とprofile選択を列挙する。productionの候補生成・補間・配置・判定は呼ばない。`grid_optimal`は有限格子内の最大値で、連続最適値を意味しない。

## 再実行

プロジェクトrootでPython 3.12と`requirements.lock`を入れたCPU環境を使う。`<cpu-venv>`は環境パスへ置換する。

```bash
PYTHONPATH=src <cpu-venv>/bin/python -m rosa_repro.cli evaluate-pipeline examples/p4_horizon_audit.json
PYTHONPATH=src <cpu-venv>/bin/python -m rosa_repro.cli solve-heterogeneous examples/r07_weighted.json
PYTHONPATH=src <cpu-venv>/bin/python -m rosa_repro.cli solve-heterogeneous examples/r07_mix4.json --rates examples/r07_mix4_rates.json
PYTHONPATH=src <cpu-venv>/bin/python -m rosa_repro.cli oracle-heterogeneous examples/r07_greedy_counterexample.json --grid examples/r07_greedy_counterexample_grid.json
PYTHONPATH=src <cpu-venv>/bin/python scripts/verify_prePod45.py --output results/prepod45-new-run
PYTHONPATH=src <cpu-venv>/bin/python -m pytest -q --junitxml=results/prepod45-new-run/pytest-junit.xml
```

CLIの`--output`と代表検証スクリプトは既存出力を上書きしない。CLIの終了コード0は構文処理の成功であり、`status`を必ず読む。`unknown`もJSONとして正常に出力する。

## Pod再開後の残作業

まず既存のR02出力照合、固定初期状態でのT01比較、実GPUのバッチ・profile検証を進める。その後、Safety/Monitor/S2の実行器へv2イベントを記録する処理を接続し、実ログで周期スロット・S2 ID・停止フラグを検証する。R07にはその実測profileを渡し、返却route・頻度制御を複数GPUの実行環境へ接続して、予測したSLOと実測を比較する。admission/failoverや実タスク成功率の再現は未完了である。

1〜3工程のRay・ダミー試行の証拠は`results/prepod-20261027/`に履歴として保持した。今回Rayを再起動した試験ではなく、v0.3.0.dev1時点の結果である。ソースfingerprintが変わるため、v0.4.0.dev1で古いSQLite実験をresumeしない。旧試験を同じソースで再開する場合は、保存済みの1〜3工程ZIPを使う。現版の試験は新しい結果フォルダを指定する。
