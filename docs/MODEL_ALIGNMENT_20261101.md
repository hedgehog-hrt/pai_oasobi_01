> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# ROSA論文・公開実装とのモデル照合

確認日: 2026-11-01 JST。対象はWenqi JiangほかのROSA、arXiv:2607.01088v1。実行環境の変更、重みのダウンロード、新しいGPU実験は行っていない。現在の構成は直近の配置計画と実行時attestationにより照合した。

本照合の後、同日にMuJoCo追加前の構成でQwen3-VL-4B/8BをSafetyへ置き換える実験（非同梱の履歴資料: `../results/pod-qwen3-20261101/SUMMARY.md`）を実施した。本計測のP3・P4停止率は両モデルとも100%で、既定モデルの変更には至っていない。以下はモデル照合を行った時点の記録である。

## 結論

S1は論文と異なる。論文の既定はGR00T N1.6、現在はNVIDIAのN1.7 LIBERO Object向け追加学習済み重みである。Safety・S2・Monitorは論文図3のモデル名・サイズと一致するが、論文実験の重みrevision、追加学習、完全なprompt/画像/生成条件の一致は未確認である。

| 成分 | 論文の記載 | 直近の実行構成 | 一致範囲 |
|---|---|---|---|
| S1 | GR00T N1.6を既定、図3はnvidia/GR00T-N1.6-3B | nvidia/GR00T-N1.7-LIBEROのlibero_object、実モデルclass Gr00tN1d7 | 世代・重みが異なる |
| Safety | 図3はQwen2.5-VL-3B-Instruct | Qwen/Qwen2.5-VL-3B-Instruct | 名前・サイズ一致、論文checkpointは未特定 |
| S2 | 図3はQwen2.5-VL-7B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct | 同上 |
| Monitor | 図3はQwen2.5-VL-7B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct | 同上 |

論文§4.1はS1の比較モデルとしてπ0.5も扱う。現在は未採用。S1の計算条件も、論文記載のFP16に対し、現在の実行時モデルはBF16であり、モデル名だけの差ではない。[論文§4.1](https://arxiv.org/html/2607.01088v1#S4.SS1)、[図3](https://arxiv.org/html/2607.01088v1#S3.F3)

現S1はNVIDIA公式のLIBERO追加学習モデルであり、無関係な汎用モデルをそのまま使っているわけではない。ただし、ROSA論文のN1.6と同一ではない。N1.7内部のCosmos-Reason2-2Bはバックボーンであり、別途起動するSafety/S2/MonitorのQwenとは区別する。[NVIDIAのN1.7-LIBEROモデルカード](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)

## P3が止まった直接原因

直近の確認300秒、P3の157判定は次の内訳だった。

| 実行ゲートの原因 | 件数 |
|---|---:|
| 指先とバスケットの距離が2mm未満になる予測 | 139 |
| 予測環境の履歴復元が2,000msを超過 | 18 |
| VLMの人物・視認性判定等による停止 | 0 |

同区間のSafety VLM出力600件はすべてvisibility=clear、person=absent、collision=unknownだった。衝突不明をMuJoCoへ委ねる現在の契約では、この出力だけでは停止しない。集計は実行直前のgate原因を数えており、区間末の未実行previewを含むpreview件数とは分母が異なる。

同じP3はwarmup中に2タスクを完了してから停止した。したがって「モデルが最初からアームを動かせない」という状態ではない。観測から言えるのは、生成動作が今回の独自2mm条件を満たさない状態に入り、停止・再試行だけでは抜け出せなかったこと、さらに復元時間の上限が停止に加わったことまでである。モデル性能だけを原因にする証拠、N1.6への変更で解決する証拠はまだない。監査済み結果（非同梱の履歴資料: `../results/pod-safety-hybrid-20261031/SUMMARY.md`）

ROSAの実機例は工具をバケツへ移すタスクと人の接近検知であり、今回のLIBERO缶投入・MuJoCoの2mm予測検査とは条件が異なる。また論文は停止後の再計画や安全状態への復帰も想定している。現プロトタイプで不足する回復接続と、S1のモデル選択を分けて評価する必要がある。[論文§3.2–3.3](https://arxiv.org/html/2607.01088v1#S3.SS3)、[§4.4](https://arxiv.org/html/2607.01088v1#S4.SS4)

## 公開コード・コミュニティの確認

NVIDIA Research、筆頭著者の論文一覧、GitHubを対象とする論文名・arXiv番号・ROSA/GR00T/reproduction/reimplementationの検索を実施した。公式ページと著者の当該項目には論文へのリンクがあり、今回の確認範囲ではROSA本体の公式コード、または採用モデルと実験設定を検証できる第三者のROSA再現実装を特定できなかった。公開されていないことの証明ではない。[NVIDIA Research](https://research.nvidia.com/publication/2026-07_rosa-robotics-foundation-model-serving-system-robot-factories)、[著者の論文一覧](https://wenqijiang.github.io/publications/)

GitHubで見つかる論文一覧・紹介記事は再現実装とは扱わない。またnasa-jpl/rosaはROS向け自然言語エージェント、aabbdev/rosaはRWKVのsuffix automatonで、対象論文とは別のプロジェクトである。[NASA/JPL](https://github.com/nasa-jpl/rosa)、[aabbdev](https://github.com/aabbdev/rosa)

比較に使える可能性がある別の成果として、コミュニティ製の`0xAnkitSingh/GR00T-N1.6-LIBERO`を確認した。作者はN1.6をFranka PandaでLIBEROの4系列へ追加学習したと説明している。これはROSAの再現実装でも、ROSA著者が使用した重みだと確認できるものでもない。現在は採用していない。作者のタスク成功率を、今回の2mm条件下での停止率へ読み替えることはできない。[モデル公開元](https://huggingface.co/0xAnkitSingh/GR00T-N1.6-LIBERO)

NVIDIA公式の`nvidia/GR00T-N1.6-3B`も公開されているが、基盤モデルを置き換えるだけで既存LIBERO入出力・追加学習・正規化が適合すると仮定しない。[公式N1.6モデル](https://huggingface.co/nvidia/GR00T-N1.6-3B)

## 次に比較すべき条件

追加の退避制御に進む前に、LIBERO適合性と入出力・正規化を検査したN1.6系と現N1.7を、固定初期状態・同じカメラ・同じ安全条件・同じ実行strideで比較するのが、モデル要因を切り分ける手順となる。失敗状態だけでなく事前固定した初期状態群を使い、タスク完了、元S1提案の拒否、実適用、予測復元時間超過を別集計する。N1.6系の採用を先に成功とみなさない。

S1 revision: `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`。Safety revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`。S2/Monitor revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`。これらは本プロジェクトの固定版であり、論文のrevisionではない。
