## OBS のソース/シーン切り替え CLI (Python + uv)

OBS の WebSocket Server（デフォルト: host=localhost, port=4455）が有効になっている前提です。OBS の設定 > WebSocket Server でパスワードをメモしておきます。

### セットアップ
1. 依存を入れる: `uv sync`
2. 接続情報は環境変数に入れておくと便利です（任意）  
   - `OBS_WS_HOST`（省略可, default: localhost）  
   - `OBS_WS_PORT`（省略可, default: 4455）  
   - `OBS_WS_PASSWORD` または `OBS_WEBSOCKET_PASSWORD`（パスワード必須なら設定）

### 使い方（例）
- シーン一覧を確認  
  `uv run python main.py list-scenes --password "$OBS_WS_PASSWORD"`
- シーン内のソース一覧を確認  
  `uv run python main.py list-sources "待機" --password "$OBS_WS_PASSWORD"`
- 配信用（Program）シーンを切り替え  
  `uv run python main.py switch-scene "ゲーム配信" --password "$OBS_WS_PASSWORD"`
- スタジオモードのプレビューだけを切り替え  
  `uv run python main.py switch-scene "待機" --preview --password "$OBS_WS_PASSWORD"`
- シーン内のソース表示/非表示を切り替え（例: 待機シーンの BGM を消す）  
  `uv run python main.py source-visibility "待機" "BGM" --hide --password "$OBS_WS_PASSWORD"`  
  `--show` を付ければ再表示できます。`--host`/`--port` もオプションで変更可能です。
