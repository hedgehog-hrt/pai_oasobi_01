> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# S1バッチ比較の採用条件とwarmup早期停止 — v0.12.0.dev1

2026-10-29。Pod起動前の優先項目1、2をこの順に実装・CPU検証した。実GR00Tの数値結果、GPU品質、複数GPUの性能は今回の成果に含まない。

## 1. 同じバッチ条件での数値回帰と、バッチを変える診断

v0.11では、serial batch 1の各sampleをseed+i、共有batch Bをseedで実行した差を、batch Bの採用条件にも使っていた。この比較では乱数のsample対応が揃わず、確率的モデルの正常な実行も不一致になり得る。

v0.12は以下を区別する。

| 比較モード | 参照と候補 | 数値採用条件への利用 |
| --- | --- | --- |
| `same_shape`（既定） | 同一の入力順序・batch size・default RNG seedで、公式公開APIと共有経路を比較 | 全caseが許容差内であることを要求 |
| `serial_vs_batch`（B>1） | serial batch 1と共有batch B。各呼出しのseedを記録 | 完全な診断記録を要求。数値一致だけで採用せず、不一致だけでadapter不具合とも断定しない |

固定版`Gr00tSimPolicyWrapper`の公開入出力契約はbatch次元Bを持つ。参照側はこのAPIへ渡す配列の組み立て・分割を独立に実装し、productionのcombine/split helperを呼ばない。候補側はGR00T backendと同じ共有helperを通す。入力順序の取り違えや出力分割の誤りを、同じhelperを参照側でも呼ぶことで見逃さない。

モデル内部のnoiseを差し替えたり、計算精度・kernel・determinism設定を変更したりはしない。同じseed/shapeでも、専用Generator、状態付きpolicy、GPUの非決定的演算による差はあり得る。`same_shape`の不一致も原因調査対象であり、自動的にadapterだけの責任とは判定しない。

Python・NumPy・Torchのdefault RNG状態は正常時・例外時とも復元する。参照・候補action NPZ、観測、比較条件、seed、許容差、source/runtime情報を保存し、どちらの比較も配列から再計算できる。許容差は事前指定し、自動的に広げない。

### 取得例

更新版のR02 serverを、既存の[固定環境・全重み検査付き手順](r02.md)で起動する。以下はPod再開後に行う例であり、今回のGPU実行記録ではない。

```bash
# 同じ保存観測集合・seed・許容差を使用。観測数はBで割り切れること。
<client-python> scripts/r02_compare.py --repo /workspace/Isaac-GR00T \
  --observations <observations> --batch-size 1 --comparison same_shape \
  --seed 42 --atol 0 --rtol 0 --output <s1-same-shape-b1>
<client-python> scripts/r02_compare.py --repo /workspace/Isaac-GR00T \
  --observations <observations> --batch-size 4 --comparison same_shape \
  --seed 42 --atol 0 --rtol 0 --output <s1-same-shape-b4>
<client-python> scripts/r02_compare.py --repo /workspace/Isaac-GR00T \
  --observations <observations> --batch-size 4 --comparison serial_vs_batch \
  --seed 42 --atol 0 --rtol 0 --output <s1-serial-vs-b4>

# GPU/server不要の再計算。不一致なら証跡を出力してexit 2。
<python> scripts/r02_compare.py --reaudit <s1-serial-vs-b4> \
  --batch-size 4 --output <diagnostic-recomputed.json>
```

`passed`/`mismatch`は数値比較の結果であり、確率分布やtask successの同等性を意味しない。`numerical_admission_eligible`は、完全な`same_shape`回帰が合格した場合だけtrueになる。`serial_vs_batch`は数値が偶然一致してもfalseである。診断不一致の原因が乱数だけだという主張もしない。

旧schemaの配列記録はオフライン再計算できるが、旧B>1のserial比較を現行の同一shape回帰へ読み替えない。ソースを変更したため、旧版の採用済みprofileやsuiteは現行版に対して作り直す必要がある。

### S1 profileの採用条件

batch 1では`quality.regression`に現行版の同一shape回帰を指定する。batch B>1では次を揃える。

1. `regression`: 同じ入力集合のbatch 1数値回帰。
2. `batch_regression`: batch Bの`same_shape`数値回帰。全case合格が必要。
3. `batch_diagnostic`: batch Bの`serial_vs_batch`診断。全予定caseと配列が必要。不一致は保持するが、未完・timeout・改変は拒否する。
4. 既存の性能・kernel・LIBERO paired・state bank・association・人のレビュー。

3種類の数値記録は同じsource/server/runtime、入力順序・内容、seed、許容差に結び付ける。用途を取り違えた記録や、別条件の診断を拒否する。[batch用requestテンプレート](../examples/adoption/s1-batch-request.template.json)を追加した。

検査結果には同一shape回帰とcross-shape診断を別々に掲載する。人のレビューには「同一shapeの一致は共有経路の検査に限られる」ことを明記し、既存のtask品質・batch条件レビューを省略しない。Bを変えたときのtask品質の同等性は、この変更だけでは証明できない。R06の選定batch・複数環境での回帰拡張は後続項目として残る。

## 2. warmup終了時に本計測への移行を判定

閉ループランナーは固定warmup cutoffで、各active class/componentの正常・有効な応答数を検査する。cutoffより前に送信・完了した要求だけを数える。cutoffちょうどの完了、遅延完了、error、timeout、無効出力は数えない。

formalは従来どおり最低30秒・各成分20件を維持する。`--warmup-min-success`で最低件数を増やすことはできるが、formalの20件を下回る指定は拒否する。CPU smokeでは同じ停止経路を短時間で試せるよう、件数を明示指定できる。既定は従来どおり0件。

- 合格時: 既定のepoch・固定計測窓を維持して継続する。確認処理やI/Oの時間もwall時間に含む。
- 不足時: `warmup_shortfall`として停止し、本計測の新規要求・新規actionを開始しない。固定時刻を自動延長したり、閾値を下げたりしない。
- 本計測に移らなかった場合は`measurement_started=false`、`weighted_Q=null`、`qualified_action_chunks=null`とする。未実行を処理量0の成功結果にしない。
- 停止時も要求・episode記録を閉じ、期限付きで実行中モデル処理をdrainする。遅れて返ったactionを新しく適用しない。適用中のsimulator呼出しを中断した場合、適用step数が確定しないことを`steps_unknown`で残す。
- 同時停止・キャンセルでも開始済みepisodeの終了記録を必ず残す。drain/cleanup/telemetryの失敗があれば、単なるwarmup不足として隠さず`failed`にする。

`warmup_gate.json`にcutoff、判定時刻、成分別件数、必要件数、不足一覧を保存する。raw再検証は件数を再計算し、不合格cutoff以降に新しい要求/actionがないこと、本計測へ進んだという不正なラベルがないことも検査する。早期停止runは比較レポートの平均に入らない。

```bash
PYTHONPATH=src <driver-python> scripts/pipeline_experiments.py prepare \
  --plan <gpu-plan.json> --banks <banks.json> --root <input-root> \
  --phase formal --warmup-min-success 20 --repetitions 3 --output <suite.json>
```

実行・時間予算・再開・ログ回収の手順は[閉ループ実験の手順](closed-loop-and-action-regression.md)を使う。warmup不足を検出した実行はそこで止まり、残り反復を続けない。再試行は明示実行と新attemptで行い、元の失敗証跡は保持する。GPUでのcleanup不確定時の確認規則も維持する。

## 検証証跡

- 全テスト結果（非同梱の履歴資料: `../results/prepod-batch-warmup-20261029/pytest-release.xml`）
- 固定配布版でのCPU検証（非同梱の履歴資料: `../results/prepod-batch-warmup-20261029/cpu-release/verification.json`）
- warmup不足の原本再検証（非同梱の履歴資料: `../results/prepod-batch-warmup-20261029/cpu-release/warmup-stop-reaudit.json`）
- batch 4の同一shape回帰（非同梱の履歴資料: `../results/prepod-batch-warmup-20261029/cpu-release/actions-b4-reaudit.json`）
- batch 4のcross-shape診断（非同梱の履歴資料: `../results/prepod-batch-warmup-20261029/cpu-release/diagnostic-b4-reaudit.json`）
- 配布物の最終検証（非同梱の履歴資料: `../results/prepod-batch-warmup-20261029/release_verification.json`）
- Pod配置用ソースZIP（非同梱の履歴資料: `../results/prepod-batch-warmup-20261029/source-release.zip`）

CPU検証では、確率的模擬policyのbatch 1/2/4で同一shape回帰が通り、serial対batchでは差が残ること、入力・出力の順序破損を検出すること、既存採用ゲートを迂回しないことを確認する。GPUメタデータを必要とする採用ゲートの単体試験は、一時ディレクトリ内の明示的な偽fixtureを使い、実GPUの採用証拠として配布しない。

CPU版Rayでは4 worker＋4 simulator processで正常2反復と、意図的なwarmup不足1試行を行う。後者は予定した20秒の本計測に入らず停止する。いずれもGPU/SSH/Podへのアクセスは行わない。CUDAの非決定性、実モデルでの許容差・品質、物理GPU切替は実証していない。
