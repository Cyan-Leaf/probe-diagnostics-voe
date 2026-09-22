#!/usr/bin/env python3
"""Human-evaluation platform for the E-baseline (PROTOCOL_VALIDITY.md §5).

Design follows MIGRATION_PLAN.md §4.3 and the owner's 2026-09-19 decision:

  forced choice ..... every pair must be answered left/right.  There is deliberately no
                      "cannot tell" button: the claim we are testing is "humans cannot tell
                      these apart either", and letting raters skip the hard pairs would
                      bias accuracy upward on exactly those pairs (people only commit when
                      they see something).  The "cannot tell" information is not lost -- it
                      is carried by confidence = 1, whose label reads 完全分不出，纯猜.
  confidence 1-5 .... mandatory, so every pair yields both an unbiased correctness bit and
                      a subjective difficulty reading
  blind ............. the answer key never reaches the browser; /api/pairs strips it
  order randomised .. left/right already flipped per pair at generation time; the row order
                      is per-rater shuffled with a seed derived from the rater name, so two
                      raters do not see the same sequence but one rater's own sequence is
                      stable across reloads
  resumable ......... progress is keyed by rater name; reopening continues where they left

Operational choices copied from the gsb instance next door, which learned them the hard way:

  * the port MUST come from the kubelet-assigned env vars (WEBSERVER_PORT / AUTO_PORT*).
    Anything else starts fine and is then unreachable from a browser.
  * all state lives on the shared mount, so swapping machines costs one command.
  * SQLite in WAL mode, one row per answer, so a crashed browser loses nothing and several
    raters can work at once.

Run:  ./humaneval/serve.sh start
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import socket
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
DATA = ROOT / "data"
DB = DATA / "humaneval.db"
CLIPS = DATA / "clips"
PKG = PROJ / "results" / "T10_validity" / "human_eval"

CONF_LABELS = {
    1: "完全分不出，纯猜",
    2: "很不确定",
    3: "一般",
    4: "比较确定",
    5: "非常确定",
}


# ---------------------------------------------------------------------------- storage
def connect():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.execute("PRAGMA foreign_keys = ON")
    return db


def init_db():
    db = connect()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS rater (
        name        TEXT PRIMARY KEY,
        created_at  TEXT NOT NULL,
        note        TEXT
    );
    CREATE TABLE IF NOT EXISTS answer (
        rater       TEXT NOT NULL REFERENCES rater(name),
        pair_id     TEXT NOT NULL,
        choice      TEXT NOT NULL CHECK (choice IN ('left','rights','right')),
        confidence  INTEGER NOT NULL CHECK (confidence BETWEEN 1 AND 5),
        same_look   INTEGER NOT NULL DEFAULT 0,
        notes       TEXT,
        ms_spent    INTEGER,
        answered_at TEXT NOT NULL,
        PRIMARY KEY (rater, pair_id)
    );
    """)
    db.commit()
    db.close()


def load_pairs():
    """Answer key stays server-side; the browser only ever sees the stripped view."""
    key = json.loads((PKG / "ANSWER_KEY.json").read_text())
    return {p["pair_id"]: p for p in key}


def rater_order(name: str, pair_ids: list[str]) -> list[str]:
    """Stable per-rater shuffle: different raters see different orders, one rater sees the
    same order every time they reload."""
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    ids = list(pair_ids)
    random.Random(seed).shuffle(ids)
    return ids


# ---------------------------------------------------------------------------- web app
def create_app():
    from flask import Flask, jsonify, request, send_file, Response

    app = Flask(__name__, static_folder=None)
    pairs = load_pairs()
    init_db()

    @app.get("/")
    def index():
        return Response((ROOT / "index.html").read_text(), mimetype="text/html")

    @app.get("/api/meta")
    def meta():
        return jsonify({
            "n_pairs": len(pairs),
            "conf_labels": CONF_LABELS,
            "title": "物理视频配对评测",
        })

    @app.post("/api/login")
    def login():
        name = (request.json or {}).get("name", "").strip()
        if not name:
            return jsonify({"error": "请填写姓名"}), 400
        db = connect()
        db.execute("INSERT OR IGNORE INTO rater(name, created_at) VALUES (?,?)",
                   (name, datetime.now().isoformat(timespec="seconds")))
        db.commit()
        done = {r["pair_id"] for r in
                db.execute("SELECT pair_id FROM answer WHERE rater=?", (name,))}
        db.close()
        order = rater_order(name, list(pairs))
        return jsonify({"name": name, "order": order, "done": sorted(done),
                        "n_done": len(done), "n_total": len(order)})

    @app.get("/api/pair/<pid>")
    def pair(pid):
        p = pairs.get(pid)
        if not p:
            return jsonify({"error": "unknown pair"}), 404
        # Whitelist, not blacklist.  Besides the obvious answer fields, `scenario` and
        # `subgroup` are withheld too: knowing that several pairs come from the same
        # scenario lets a rater build a within-scenario reference ("in this one the ball
        # usually does X"), which is a different task from judging each pair on its own.
        return jsonify({k: p[k] for k in ("pair_id", "order")})

    @app.get("/video/<pid>/<side>")
    def video(pid, side):
        """Serve the transcoded H.264 copy, not the original.

        The LikePhys clips are mpeg4 Simple Profile (fourcc mp4v).  ffmpeg and decord read
        that fine -- which is why every server-side check passed -- but no browser decodes
        it, so the <video> element silently refused a perfectly valid 200 response.
        `transcode.py` converts the 96 referenced clips to H.264/yuv420p/faststart once.
        """
        if pid not in pairs or side not in ("left", "right"):
            return jsonify({"error": "not found"}), 404
        cached = CLIPS / f"{pid}_{side}.mp4"
        if not cached.exists():
            return jsonify({
                "error": "clip not transcoded yet",
                "fix": "run: python humaneval/transcode.py",
            }), 503
        return send_file(cached, mimetype="video/mp4", conditional=True)

    @app.post("/api/answer")
    def answer():
        d = request.json or {}
        name, pid = d.get("rater", "").strip(), d.get("pair_id", "")
        choice, conf = d.get("choice"), d.get("confidence")
        if not name or pid not in pairs:
            return jsonify({"error": "bad request"}), 400
        if choice not in ("left", "right"):
            return jsonify({"error": "必须二选一"}), 400
        if conf not in (1, 2, 3, 4, 5):
            return jsonify({"error": "确定度必须是 1-5"}), 400
        db = connect()
        db.execute("""INSERT INTO answer
                      (rater,pair_id,choice,confidence,same_look,notes,ms_spent,answered_at)
                      VALUES (?,?,?,?,?,?,?,?)
                      ON CONFLICT(rater,pair_id) DO UPDATE SET
                        choice=excluded.choice, confidence=excluded.confidence,
                        same_look=excluded.same_look, notes=excluded.notes,
                        ms_spent=excluded.ms_spent, answered_at=excluded.answered_at""",
                   (name, pid, choice, int(conf), int(bool(d.get("same_look"))),
                    (d.get("notes") or "").strip()[:500], d.get("ms_spent"),
                    datetime.now().isoformat(timespec="seconds")))
        db.commit()
        n = db.execute("SELECT COUNT(*) c FROM answer WHERE rater=?", (name,)).fetchone()["c"]
        db.close()
        return jsonify({"ok": True, "n_done": n, "n_total": len(pairs)})

    @app.get("/api/progress")
    def progress():
        db = connect()
        rows = db.execute("""SELECT r.name, COUNT(a.pair_id) n,
                                    AVG(a.confidence) conf
                             FROM rater r LEFT JOIN answer a ON a.rater=r.name
                             GROUP BY r.name ORDER BY n DESC""").fetchall()
        db.close()
        return jsonify({"n_total": len(pairs),
                        "raters": [{"name": r["name"], "n_done": r["n"],
                                    "mean_confidence": round(r["conf"], 2) if r["conf"] else None}
                                   for r in rows]})

    @app.get("/api/export")
    def export():
        """CSV in exactly the shape t10_human_score.py expects, one file per rater."""
        db = connect()
        rows = db.execute("SELECT * FROM answer ORDER BY rater, pair_id").fetchall()
        db.close()
        out = DATA / "export"
        out.mkdir(exist_ok=True)
        by = {}
        for r in rows:
            by.setdefault(r["rater"], []).append(r)
        written = []
        for name, rs in by.items():
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            p = out / f"{safe}.csv"
            with p.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["order", "pair_id", "left_clip", "right_clip",
                            "your_choice(left/right/cannot_tell)", "confidence(1-5)",
                            "same_look", "ms_spent", "notes"])
                for i, r in enumerate(rs, 1):
                    pr = pairs[r["pair_id"]]
                    w.writerow([i, r["pair_id"], pr["left_clip"], pr["right_clip"],
                                r["choice"], r["confidence"], r["same_look"],
                                r["ms_spent"] or "", r["notes"] or ""])
            written.append(str(p))
        return jsonify({"written": written, "n_raters": len(by), "dir": str(out)})

    return app


# ---------------------------------------------------------------------------- port
def open_ports() -> list[int]:
    """Kubelet only maps these; anything else binds fine and is then unreachable."""
    order = ["WEBSERVER_PORT", "AUTO_PORT1", "AUTO_PORT0", "AUTO_PORT2",
             "JUPYTER_PORT", "VSCODE_PORT"]
    seen, out = set(), []
    for k in order:
        v = os.environ.get(k)
        if v and v.isdigit() and v not in seen:
            seen.add(v)
            out.append((k, int(v)))
    return out


def free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def pick_port() -> int | None:
    for _, p in open_ports():
        if free(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--ports", action="store_true", help="list mapped ports and exit")
    args = ap.parse_args()

    if args.ports:
        print("本机对外开放的端口（kubelet 分配，别的端口浏览器连不上）：")
        for k, p in open_ports():
            print(f"  {k:16s} {p}  {'空闲' if free(p) else '被占用'}")
        print(f"\n会自动选：{pick_port()}")
        return

    port = args.port or pick_port()
    if port is None:
        sys.exit("没有可用端口；用 --port 指定，或先停掉占用的服务")
    if args.port and args.port not in [p for _, p in open_ports()]:
        print(f"⚠️  {args.port} 不在 kubelet 映射列表里，浏览器多半连不上", flush=True)

    if not PKG.exists():
        sys.exit(f"找不到评测包 {PKG}；先跑 scripts/t10_human_eval_prep.py")

    # Fail loudly here rather than letting raters stare at a black <video>: the source
    # clips are mpeg4 Simple Profile, which no browser decodes.
    n_pairs = len(load_pairs())
    n_clips = len(list(CLIPS.glob("*.mp4"))) if CLIPS.exists() else 0
    if n_clips < 2 * n_pairs:
        sys.exit(f"视频未转码（{n_clips}/{2*n_pairs} 个就位）。\n"
                 f"源文件是 mpeg4 Simple Profile，浏览器放不出来。先跑：\n"
                 f"  ./.venv/bin/python humaneval/transcode.py")

    app = create_app()
    print(f"评测平台已启动 →  http://{socket.gethostname()}:{port}/", flush=True)
    print(f"  数据库 {DB}", flush=True)
    print(f"  {len(load_pairs())} 对待评", flush=True)
    app.run(host=args.host, port=port, threaded=True)


if __name__ == "__main__":
    main()
