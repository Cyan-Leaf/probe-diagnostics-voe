#!/usr/bin/env bash
# Control script for the human-evaluation platform.
#
# Port handling is the thing that bites here, so it is handled the same way the gsb
# instance next door does it: the dev box only maps the kubelet-assigned ports
# (WEBSERVER_PORT / AUTO_PORT*), and a service bound to anything else starts fine and is
# then simply unreachable from a browser.  `serve.py --ports` prints what is mapped.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$HERE")"
PY="$PROJ/.venv/bin/python"
PIDF="$HERE/data/server.pid"
LOG="$HERE/logs/server.log"

mkdir -p "$HERE/data" "$HERE/logs"

running() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case "${1:-}" in
  start)
    if running; then echo "已经在跑了 (pid $(cat "$PIDF"))"; exit 0; fi
    shift || true
    nohup "$PY" "$HERE/serve.py" "$@" >>"$LOG" 2>&1 &
    echo $! > "$PIDF"
    sleep 2
    if running; then
      echo "已启动 (pid $(cat "$PIDF"))"
      grep -m1 "评测平台已启动" "$LOG" | tail -1
      echo "日志: $LOG"
    else
      echo "启动失败，日志尾部："; tail -20 "$LOG"; exit 1
    fi
    ;;
  stop)
    if running; then kill "$(cat "$PIDF")"; sleep 1; rm -f "$PIDF"; echo "已停止";
    else echo "没在跑"; rm -f "$PIDF"; fi
    ;;
  restart) "$0" stop; sleep 1; shift || true; "$0" start "$@" ;;
  status)
    if running; then
      echo "在跑 (pid $(cat "$PIDF"))"
      grep -m1 "评测平台已启动" "$LOG" | tail -1
      "$PY" - <<'PYEOF'
import sqlite3, pathlib
db = pathlib.Path(__file__).parent if False else None
import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "humaneval.db")
PYEOF
      "$PY" -c "
import sqlite3, sys
from pathlib import Path
db = Path('$HERE/data/humaneval.db')
if db.exists():
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    rs = c.execute('''SELECT r.name, COUNT(a.pair_id) n, ROUND(AVG(a.confidence),2) conf
                      FROM rater r LEFT JOIN answer a ON a.rater=r.name
                      GROUP BY r.name ORDER BY n DESC''').fetchall()
    print(f'  评测人 {len(rs)} 名:')
    for r in rs: print(f'    {r[\"name\"]:16s} {r[\"n\"]:3d} 题   平均确定度 {r[\"conf\"]}')
else: print('  还没有数据')
"
    else echo "没在跑"; fi
    ;;
  ports) "$PY" "$HERE/serve.py" --ports ;;
  export)
    "$PY" -c "
import sys; sys.path.insert(0,'$HERE')
from serve import create_app
app = create_app()
with app.test_client() as c:
    import json; print(json.dumps(c.get('/api/export').get_json(), indent=2, ensure_ascii=False))
"
    ;;
  score)
    "$0" export >/dev/null
    echo '--- 跑归并评分 ---'
    "$PY" "$PROJ/scripts/t10_human_score.py" "$HERE"/data/export/*.csv
    ;;
  log) tail -f "$LOG" ;;
  *)
    cat <<EOF
用法: ./serve.sh <命令>

  start [--port N]  起服务（不带端口 = 自动挑 kubelet 映射的那个）
  stop / restart    停 / 重启
  status            在跑吗 + 每个人做了多少
  ports             本机开放哪些端口、会选哪个
  export            把答案导成 CSV（scripts/t10_human_score.py 的输入格式）
  score             导出并直接跑归并评分
  log               跟日志

数据都在 $HERE/data/（共享盘），换机器不丢。
EOF
    ;;
esac
