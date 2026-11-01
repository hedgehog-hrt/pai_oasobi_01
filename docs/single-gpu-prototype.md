> Publication note: historical evidence files referenced as paths below are not included in this public snapshot. See [publication scope](../PUBLICATION.md) and the [English report](../reports/ROSA_single_GPU_submission_EN_20261101.pdf).

# 単一GPUの実モデル検証

2026年10月31日、利用者が承認した完成範囲は「P1はROSAの1 GPU・1モデル配置で検証し、P2〜P4は単一GPUの拡張機能として検証する」。ROSA論文§3.4.1のモデル配置を、複数モデルのGPU同居と読み替えない。

以下は既定のFIFO方式。v0.20.6では明示的な実験オプションとして[役割別並行実行](parallel-role-prototype.md)も追加した。実測結果（非同梱の履歴資料: `../results/pod-throughput-20261031/SUMMARY.md`）では待ち時間を削減できたが、全期限達成と意味的品質の認定は残る。

## 検証用の実行経路

`prototype_plan.py`、`prototype_service.py`、`scripts/prototype_validation.py`は、未採用profileでも実モデルの接続を検証するための明示的な入口である。通常のR07配置・profile採用ゲートとは別で、`purpose=single_gpu_prototype_validation`を保存する。通常のfleet実行器はこのplanを拒否する。

| Pipeline | 使用する成分 |
| --- | --- |
| P1 | S1 |
| P2 | S1・Monitor |
| P3 | S1・Safety・Monitor |
| P4 | S1・S2・Safety・Monitor |

- P1は既存のRay Serve S1 workerを使う。複数のLIBERO環境が、単一のGR00Tモデルを共有する。
- P2〜P4は1つのRay actorが1枚のGPUを占有し、S1・S2・Safety・Monitorを役割ごとに1回ロードする。全モデル呼び出しをFIFOで直列化する。GPUの分割割当やモデル間の同時実行は行わない。
- 同居はbatch 1のみ。モデルごとのメモリ設定とS1用の余裕を検査するが、実際に同居できたことはGPU実行のinventoryとログで確認する。
- 各robotは独立したLIBEROプロセスと初期状態を持つ。実観測、実行動、実際に適用したステップ数、要求・応答・周期情報を保存し、通常のP1〜P4評価器で再監査する。

`gpu_connection`試験では、各クラスの各成分についてwarmupと測定区間に少なくとも1つの正常完了応答を要求する。warmup不足は固定cutoffで停止する。この小規模試験は、正式なp99・収容台数・品質の判定には使わない。正式条件のwarmup 20件、測定1,000件、3反復や、R06回帰・人による採用レビューを弱めるものではない。

## 実環境で必要になった変更

共通環境の互換性を確認し、PyArrow 19.0.1、CUTLASS DSL 4.2.1などを実行時lockへ固定した。テンプレートのPython環境は別に保持する。

固定版vLLM 0.11.2のQwen2.5-VL vision attentionは、このGPUとhead dimensionの組合せで起動に失敗した。`vision_attention_backend=TORCH_SDPA`を選ぶ場合だけ、専用workerがモデル構築時の選択を補正する。decoderのattention設定やupstreamファイル全体を書き換えない。選択・実行環境・CUDA kernelの証跡を別に保存する。

VLMのJSON出力には`role_json_compact_v1`を追加した。役割別JSON Schemaとxgrammarを使い、余分な空白生成を禁止する。文字列は192文字までとし、通常の厳格parserも通す。これは構造の制約であり、Safety判断・計画・終了判定の意味的正しさを保証しない。未注釈のholdoutデータから品質を推定しない。

探索的な単一モデルsmokeは、低レートの応答でも10件を収集できるよう測定上限を60秒にした。SLOと最低件数は維持する。期限超過、無効な出力、失敗、時間枠外の応答を成功率から除外しない。

v0.20.1のP1実機試験で、PyTorch 2.9が返す物理GPU UUIDに`GPU-`の接頭辞がないことが判明した。v0.20.2では監視時の照会にだけ接頭辞を補い、`nvidia-smi`の応答がそのUUIDと一致することを検査する。workerの元UUIDとNVIDIA表記を両方記録し、GPU番号による代用は行わない。推論に成功しても監視が欠けたv0.20.1の試験は、失敗記録として残す。

v0.20.2の8 Hz計測の2反復目では、event loopの起床遅延により次の未送信arrivalが30秒のwarmup下限より過去になり、cutoffを29.875秒へ戻していた。v0.20.3は早期終了境界が最低時間と必要応答数を満たした時刻より前にならないよう修正した。境界までの未送信arrivalは欠測として残し、上限時間・必要件数・失敗の分母は変更しない。修正前の反復を成功扱いせず、修正後は新しいsource/suiteで再計測する。

## 再実行と結果の扱い

モデルsnapshot・環境lock・ソースbundle・初期状態bankを固定し、`postpod_models.py`で個別モデルを検証した後、`prototype_validation.py prepare`でsuiteを作り、同じソースの`run`で実行する。GPU試験は明示した既存Ray cluster、simulator Python、LIBERO repoを要求する。CPU fixtureがGPU成功へ読み替わる経路はない。

実行ごとの`result.json`、`trial/raw/`、`trial/reaudit.json`、`trial/cleanup.json`を合わせて確認する。`status=complete`は試験の完走であり、SLO達成やタスク成功の意味ではない。数値・失敗を含む最終結果はPod実測報告（非同梱の履歴資料: `../results/pod-prototype-20261031/SUMMARY.md`）に記録する。
