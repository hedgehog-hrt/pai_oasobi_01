> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# 実計測記録・R06採用接続・R05候補生成（v0.15.0.dev1）

2026-10-29。依頼の3項目を順に実装した。Pod、GPU、GR00T重み、実LIBEROにはアクセスしていない。以下のCPU実行は機構の検証であり、実モデルの性能・成功率・採用結果ではない。

## 1. 実計測の記録補完

`ComponentCore`が各backend呼び出しに`backend_evidence.gpu_measurement`を付ける。同期backendは専有worker内で直列化し、`binding.kwargs.cuda_measurement: true`の場合にCUDA EventsとPyTorch allocatorのピークを取得する。新規GPU計測テンプレートはtrue。既存の設定は明示的にtrueにして新しいsuiteを生成する。

- `cuda_elapsed_ms`: current streamのEvent間隔。CPUのlaunch間隔を含み、他streamの処理は依存関係で結ばれる範囲のみ。純粋なkernel実行時間、要求の応答時間、GPU利用率ではない。
- `allocated_before_bytes` / `reserved_before_bytes`、`peak_allocated_bytes` / `peak_reserved_bytes`: 当該workerプロセスのPyTorch allocator値。常駐モデルを含む。デバイス全体や要求ごとの占有量ではない。
- 計測前後のdevice synchronizeとbatchごとのpeak resetによる負荷がある。設定と計測方式をruntime signatureに含め、計測の有無を混ぜてprofileを採用しない。
- CPU、無効設定、非同期vLLM、計測APIの失敗は数値をnullにして理由を記録する。vLLM engine subprocessの値をフロントプロセスから代用しない。既存のnative kernel trace取得は別経路。
- batch証跡は各応答へ同じbatch ID付きで保存し、集計時はbatch単位で重複除去する。SLO付きprofileとR05は要約を再計算できる。P2〜P4は既存のbackend evidence欄で保存する。失敗した同期推論にも計測証跡を付ける。

R05は開始前・実行中・終了後の`telemetry.jsonl`を追加。実行中はサンプル取得後1秒待ち、取得timeoutは5秒。nvidia-smi失敗も欠損として残す。短いCPU試験には実行中sampleがないことがある。これは離散サンプルで、瞬間ピークや無干渉を保証しない。allocator peakとの範囲の違いを維持する。元のbefore/afterファイルも維持する。

## 2. R06とprofile採用判定

S1採用では従来の同一shape数値回帰・cross-shape診断・paired記録に加え、次の`quality.r06`を必須にした。全パスは`--root`からの相対パス。

```json
{"r06":{"suite":"r06/suite.json","bank_root":"r06/banks","directories":"r06/directories.json"}}
```

`directories.json`はsuiteの各cell IDをrawディレクトリに対応させる。検査はbank、suite、全raw原本を読み、T01〜T03各50初期状態、formal/GPU、current source、数値回帰と同じserver checkpoint/runtime、実行前後に取得・保存したserver証跡、候補batch、state/seed、初期観測、全予定cellの完了を確認する。保存済み成功率summaryだけでは採用しない。証跡の全ファイルhashを採用レビューへ結び付ける。

基準の環境数1/2/4を保ち、B8またはB16候補にはそれぞれ8または16環境の条件を追加する（各taskで3 arm、計36 cell。B1/2/4は27 cell）。追加条件は本プロジェクトの品質検証用で、論文の実験条件との同一性を主張しない。各taskで候補の**実batchサイズが観測済み**であることが必要。capだけ一致しても通らない。paddingはしない。

`profile_adoption.py inspect`でrawから対応のある成功率差と区間を再計算する。`review-template`は各task・環境数・shared/selected armに`r06_checks`を生成する。未確定区間・低下判定をそのまま示し、各項目のacceptedと理由、および従来の全レビューを満たすまで`adopt`は通らない。レビューが受理されても統計的同等性の自動証明にはならない。50/50一致でも保守的95%区間は±5pp内に収まらない。

従来版の採用証跡は履歴として残す。v0.15のcurrent sourceで再計測・再検査が必要。

## 3. R05予測batch生成

```bash
python scripts/r05_experiments.py predict \
  --pilots evidence/pilot-dependencies.json --root evidence \
  --tie-tolerance 0.01 --output evidence/prediction.json \
  --selection-output evidence/selection.json
```

`pilot-dependencies.json`は`{"k4-a50":["pilot-b1/raw","pilot-b2/raw","pilot-b4/raw","pilot-b8/raw","pilot-b16/raw"]}`という形式。各groupの5種類を全て必要とする。異なるgroupをまとめる場合も、入力corpus・model/runtime・batch wait・sourceが一致していなければ拒否する。

各pilotの計測phaseのraw要求から平均応答時間を再計算し、[ROSA v1の式(4)](https://arxiv.org/html/2607.01088v1#S3.SS4.SSS2)を単一S1 workerへ限定した次式で比較する。

`予測総行動率 = ロボット数 / (動作時間 + 平均応答時間)`

応答時間にはqueue・batch待ち・通信が含まれる。CUDA Events値を代入しない。最大の点推定から事前指定の相対許容幅内にある最小のbatch capを候補とする（既定1%、許容0〜5%は本プロジェクトの規則）。実観測batch、要求数、式、原本hash、runtime、各候補の予測値を保存する。観測した完了行動率は診断用に保存するが、順位付けには使用しない。

これは**同じロボット数・動作時間でのpilot平均を使う候補予測**である。負荷別の開ループ曲線をfitする実装や、論文の複数worker配置最適化ではない。未測定batchや異なる条件へ外挿せず、Kより大きいcapで部分batchしか出なかった場合も実測内容を表示する。短いpilot、境界を跨ぐ応答、モデルの変動は予測誤差を生む。大域最適性、統計的優位性、論文Figure 14の再現達成を主張しない。

同時生成する`selection.json`を`paper_experiments.py --r05-selection evidence/selection.json`へ渡し、従来の`prepare --prediction-evidence`から確認suiteを作る。確認開始時と比較レポート生成時に、参照pilot原本からpredictionを完全に再生成して一致を要求する。任意JSON、hash書式だけの証拠、変更済みraw、異なるsource/runtime、失敗・欠落pilot、確認実験を選択に流用した証拠は拒否する。確認実験は従来どおり独立3反復、SLOなし、仮想行動。Pod実行には明示的な予算と`--execute`が必要。

## 検証と配布

全体テスト、固定配布版のテスト、実CPU Ray試験、CPU simulator subprocess試験の結果は検証記録（非同梱の履歴資料: `../results/prepod-measurement-quality-prediction-20261029/`）に保存する。source-release ZIPはコード・設定・ドキュメント・テストのみで、重みや認証情報を含まない。実CUDA値、実GR00T/LIBERO、採用レビュー、本測定はPod再開後に残る。
