"""日记库：私聊长期记忆的可寻回形态（老板 2026-10-28 定案）。

设计要点：
- 每用户一个 SQLite（data/diaries/<uid>.db），与 chat_history/<uid>.json 同构：
  删一个用户=删一个文件，备份/导出/控制台按用户看都直观
- 表：entries（id, ts, title, tags, summary, tokens）+ FTS5 全文索引（标题+摘要+标签）
- 中文检索：FTS5 unicode61 按字切分（零依赖、零额外模型）；标签是自生成的规范词
- 三种寻回：时间范围 / 关键词 BM25 / 全部按时间倒序——都返回给模型可读的纯文本
- 纯标准库 sqlite3，无向量、无外部知识库、无额外模型（老板定案：模型只有一个）
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path

from .fts_zh import bigrams, grams_query

# ---------- 中文时间解析（寻回用，覆盖日常表达） ----------
_TIME_PATTERNS: list[tuple[re.Pattern, object]] = [
    (re.compile(r"今天"), lambda now: (now - 86400, now + 60)),
    (re.compile(r"昨天"), lambda now: (now - 2 * 86400, now - 86400 + 60)),
    (re.compile(r"前天"), lambda now: (now - 3 * 86400, now - 2 * 86400 + 60)),
    (re.compile(r"本周|这周|这个星期"), lambda now: (now - 7 * 86400, now + 60)),
    (re.compile(r"上周"), lambda now: (now - 14 * 86400, now - 7 * 86400 + 60)),
    (re.compile(r"本月|这个月"), lambda now: (now - 31 * 86400, now + 60)),
    (re.compile(r"上个月"), lambda now: (now - 62 * 86400, now - 31 * 86400 + 60)),
]

_DATE_RE = re.compile(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})")
_MONTH_RE = re.compile(r"(\d{4})[-/年.](\d{1,2})\s*月?")
_MONTH_ONLY_RE = re.compile(r"(\d{1,2})\s*月(?:份)?")


def parse_time_query(q: str, now: float | None = None) -> tuple[float, float] | None:
    """把「昨天/上周/10月20号/2026-10-20/10月」解析成时间范围。解析不了返回 None。"""
    now = now or time.time()
    q = (q or "").strip()
    for pat, fn in _TIME_PATTERNS:
        if pat.search(q):
            return fn(now)
    m = _DATE_RE.search(q)
    if m:
        import calendar
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        start = time.mktime((y, mo, d, 0, 0, 0, 0, 0, -1))
        return start, start + 86400
    m = _MONTH_RE.search(q)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        last = calendar.monthrange(y, mo)[1]
        start = time.mktime((y, mo, 1, 0, 0, 0, 0, 0, -1))
        return start, time.mktime((y, mo, last, 23, 59, 59, 0, 0, -1))
    m = _MONTH_ONLY_RE.search(q)   # 「10月」：默认今年
    if m:
        import calendar
        import datetime
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            y = datetime.datetime.fromtimestamp(now).year
            last = calendar.monthrange(y, mo)[1]
            start = time.mktime((y, mo, 1, 0, 0, 0, 0, 0, -1))
            return start, time.mktime((y, mo, last, 23, 59, 59, 0, 0, -1))
    return None


class DiaryStore:
    """单用户日记库。线程内使用（qqmind 全异步单线程事件循环，sqlite3 够用）。

    所有实例登记在 _OPEN，测试/关停时 close_all() 统一释放（Windows 下
    sqlite 句柄不关会锁住文件，导致临时目录清不掉）。
    """

    _OPEN: list["DiaryStore"] = []

    def __init__(self, diaries_dir: str | Path, uid: int | str,
                 *, name: str | None = None):
        """name 非空 → 公共库（文件名用 name，如 group_public.db，uid=0）。
        v0.3.0 公共群记忆：所有群共用一个库，entries.gid 记来源群。"""
        self.uid = int(uid) if not name else 0
        self.name = name or str(self.uid)
        self.path = Path(diaries_dir) / f"{self.name}.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.row_factory = sqlite3.Row
        self._init_schema()
        DiaryStore._OPEN.append(self)

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS entries (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      REAL NOT NULL,
            title   TEXT NOT NULL DEFAULT '',
            tags    TEXT NOT NULL DEFAULT '[]',
            summary TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            tokens  INTEGER NOT NULL DEFAULT 0,
            gid     INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_entries_ts ON entries(ts);
        CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
            grams, title, summary, tags,
            tokenize='unicode61',
            content='entries', content_rowid='id'
        );
        CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS trg_ai AFTER INSERT ON entries BEGIN
            INSERT INTO fts(rowid, grams, title, summary, tags)
            VALUES (new.id, new.grams, new.title, new.summary, new.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS trg_ad AFTER DELETE ON entries BEGIN
            INSERT INTO fts(fts, rowid, grams, title, summary, tags)
            VALUES ('delete', old.id, old.grams, old.title, old.summary, old.tags);
        END;
    """
    _FTS_VER = "2"   # meta.fts_ver：2=grams 列已建并回填

    @staticmethod
    def _grams_for(title: str, tags_json: str, summary: str) -> str:
        """FTS grams 列内容：标题+标签+摘要的二字组（检索面的全部文本）。"""
        try:
            tags = " ".join(json.loads(tags_json or "[]"))
        except Exception:
            tags = ""
        return bigrams(f"{title} {tags} {summary}")

    def _init_schema(self) -> None:
        row = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entries'"
        ).fetchone()
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(entries)")}
        upgrading_title = row is not None and "title" not in cols
        upgrading_content = row is not None and "content" not in cols
        # external content 的 fts 列必须真实存在于 content 表——grams 要加在 entries 上；
        # 已存在的旧 fts（无 grams）只能重建
        fts_cols = {r[1] for r in self._db.execute("PRAGMA table_info(fts)")}
        rebuilding = bool(fts_cols) and "grams" not in fts_cols
        if rebuilding:
            self._db.execute("DROP TRIGGER IF EXISTS trg_ai")
            self._db.execute("DROP TRIGGER IF EXISTS trg_ad")
            self._db.execute("DROP TABLE IF EXISTS fts")
        self._db.executescript(self._SCHEMA)
        if "grams" not in cols:
            self._db.execute(
                "ALTER TABLE entries ADD COLUMN grams TEXT NOT NULL DEFAULT ''")
        if upgrading_title:
            self._db.execute(
                "ALTER TABLE entries ADD COLUMN title TEXT NOT NULL DEFAULT ''")
        if upgrading_content:
            # v0.2.5 三层日记：content 列存原文选段（第三层），旧库默认空=没有全文
            self._db.execute(
                "ALTER TABLE entries ADD COLUMN content TEXT NOT NULL DEFAULT ''")
        if "gid" not in cols:
            # v0.3.0 公共群记忆：gid 记来源群，旧库补列（私聊库全 0，无影响）
            # 注意 cols 是建表前读的：新建库由 _SCHEMA 直接带 gid，别再 ALTER
            cols_now = {r[1] for r in self._db.execute("PRAGMA table_info(entries)")}
            if "gid" not in cols_now:
                self._db.execute(
                    "ALTER TABLE entries ADD COLUMN gid INTEGER NOT NULL DEFAULT 0")
        need_rebuild = (rebuilding or upgrading_title
                        or self._meta_get("fts_ver") != self._FTS_VER)
        if need_rebuild:
            # rebuild 从 entries 读 grams 列——先回填再重建
            self._backfill_grams()
            self._db.execute("INSERT INTO fts(fts) VALUES ('rebuild')")
            self._meta_set("fts_ver", self._FTS_VER)
        self._db.commit()

    def _meta_get(self, k: str) -> str | None:
        row = self._db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def _meta_set(self, k: str, v: str) -> None:
        self._db.execute(
            "INSERT INTO meta(k, v) VALUES (?,?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
        self._db.commit()

    # ---------- 用户画像（v0.2.6 数据导入配套） ----------
    def save_profile(self, profile: dict, uid: int = 0) -> None:
        """画像 JSON 存日记库 meta 表（与该用户日记同库同生命周期）。

        v0.4.0：一个库装所有人的记忆，画像按 QQ 号分键（`profile:<uid>`），
        否则最后导入的那个人会覆盖所有人的画像。uid=0 走旧单键（兼容）。
        """
        key = "profile" if not uid else f"profile:{int(uid)}"
        self._meta_set(key, json.dumps(profile, ensure_ascii=False))

    def set_meta(self, k: str, v: str) -> None:
        """公开 meta 写入口（证据链 evidence 等结构化扩展，v0.2.14）。"""
        self._meta_set(k, v)

    def get_meta(self, k: str) -> str | None:
        return self._meta_get(k)

    def del_meta(self, k: str) -> None:
        """删一条 meta（v0.4.0：合并时清掉单键 affection，防串号）。"""
        self._db.execute("DELETE FROM meta WHERE k=?", (k,))
        self._db.commit()

    def meta_keys(self, prefix: str = "") -> list[str]:
        """按前缀列 meta 键（v0.4.0：`affinity:` 键反查已知 uid）。"""
        rows = self._db.execute(
            "SELECT k FROM meta WHERE k LIKE ?", (f"{prefix}%",)).fetchall()
        return [r[0] for r in rows]

    def load_profile(self, uid: int = 0) -> dict:
        """读画像（v0.4.0：先读 `profile:<uid>`，没有才回落旧单键）。"""
        keys = ["profile"] if not uid else [f"profile:{int(uid)}", "profile"]
        for k in keys:
            try:
                d = json.loads(self._meta_get(k) or "{}")
            except Exception:
                continue
            if isinstance(d, dict) and d:
                return d
        return {}

    def profile_prompt(self, uid: int = 0) -> str:
        """画像 → system prompt 片段（空画像返回空串）。"""
        p = self.load_profile(uid)
        if not p:
            return ""
        lines = []
        if p.get("nickname"):
            lines.append(f"称呼：TA 常被叫「{p['nickname']}」")
        for k, label in (("traits", "特点"), ("interests", "常聊"),
                         ("facts", "要记住的事"), ("avoid", "忌讳")):
            vals = [str(x) for x in (p.get(k) or []) if str(x).strip()]
            if vals:
                lines.append(f"{label}：{'；'.join(vals)}")
        if not lines:
            return ""
        return "【关于这位用户的画像（来自历史记录）】\n" + "\n".join(lines)

    def _backfill_grams(self) -> None:
        """给 entries.grams 为空的行补二字组（迁移旧库；新写入在 add() 里带上）。"""
        for r in self._db.execute(
                "SELECT id, title, tags, summary FROM entries "
                "WHERE IFNULL(grams,'')='' ORDER BY id").fetchall():
            self._db.execute("UPDATE entries SET grams=? WHERE id=?",
                             (self._grams_for(r["title"], r["tags"], r["summary"]),
                              r["id"]))

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass
        try:
            DiaryStore._OPEN.remove(self)
        except ValueError:
            pass

    @classmethod
    def close_all(cls) -> None:
        for inst in list(cls._OPEN):
            inst.close()

    @staticmethod
    def merge_db(path: str | Path, target: "DiaryStore", *,
                 default_gid: int = 0, owner_uid: int = 0) -> int:
        """把一个旧库（私聊 <uid>.db 或群 <gid>.db）整体并进 target（v0.4.0）。

        - entries 逐行复制：gid 取行内值，为 0 时用 default_gid
        - meta：window/profile 仅在 target 还没有时搬（保留主人那条时间线）
        - affection 改写成 ``affinity:<uid>``（合并后一个库装所有人，单键会串号）
        返回并入篇数。调用方负责改名备份（.migrated）。
        """
        src_path = Path(path)
        if not src_path.exists():
            return 0
        rows = []
        meta: dict[str, str] = {}
        try:
            src = sqlite3.connect(str(src_path))
            src.row_factory = sqlite3.Row
            rows = src.execute("SELECT * FROM entries ORDER BY id").fetchall()
            try:
                meta = {r["k"]: r["v"] for r in src.execute("SELECT k, v FROM meta")}
            except sqlite3.Error:
                meta = {}
            src.close()
        except sqlite3.Error:
            return 0
        merged = 0
        for r in rows:
            cols = r.keys()
            gid = (int(r["gid"] or 0) if "gid" in cols else 0) or int(default_gid or 0)
            try:
                tags = json.loads(r["tags"] or "[]")
            except Exception:
                tags = []
            target.add(r["ts"], tags, r["summary"],
                       tokens=int(r["tokens"] or 0),
                       title=((r["title"] if "title" in cols else "") or ""),
                       content=((r["content"] if "content" in cols else "") or ""),
                       gid=gid)
            merged += 1
        try:
            window = meta.get("window")
            if window and owner_uid:
                # 私聊库搬过来的窗口条目没有 uid/src：补上归属，
                # 否则合并后按 uid 切片会把这段近期上下文整段丢掉
                try:
                    arr = json.loads(window)
                    for m in arr if isinstance(arr, list) else []:
                        if isinstance(m, dict):
                            m.setdefault("src", "private")
                            if not m.get("uid"):
                                m["uid"] = int(owner_uid)
                    window = json.dumps(arr, ensure_ascii=False)
                except Exception:
                    pass
            if window and not target.get_meta("window"):
                target.set_meta("window", window)
            if meta.get("profile"):
                # 画像也按人分键（v0.4.0）：私聊库的画像只属于 owner_uid，
                # 写成共享单键会让所有人读到同一个人的画像
                pkey = f"profile:{int(owner_uid)}" if owner_uid else "profile"
                if not target.get_meta(pkey):
                    target.set_meta(pkey, meta["profile"])
            if owner_uid and meta.get("affection"):
                key = f"affinity:{int(owner_uid)}"
                if not target.get_meta(key):
                    target.set_meta(key, meta["affection"])
            # 单键 affection 从合并库清掉：一个库装所有人，留着会被别人读到
            if meta.get("affection"):
                target.del_meta("affection")
        except Exception:
            pass
        return merged

    @staticmethod
    def migrate_group_dbs(diaries_dir: str | Path, target: "DiaryStore") -> int:
        """v0.3.0 公共群记忆迁移：把 <gid>.db 的群日记并入公共库。

        幂等：meta['migrated_gid_dbs'] 记录已并的库；并完改名 .migrated 备份。
        返回并入篇数。evidence 元数据不迁（旧证据链本就只服务最近一批）。
        """
        ddir = Path(diaries_dir)
        done = set((target.get_meta("migrated_gid_dbs") or "").split(","))
        done.discard("")
        merged = 0
        moved: list[str] = []
        for f in sorted(ddir.glob("*.db")):
            gid_s = f.stem
            if not gid_s.isdigit():
                continue
            if gid_s in done:
                # 上次已并入但改名被文件锁挡住：每次启动重试清理（数据不会重复）
                try:
                    os.replace(f, ddir / f"{gid_s}.db.migrated")
                except OSError:
                    pass
                continue
            try:
                src = sqlite3.connect(str(f))
                src.row_factory = sqlite3.Row
                rows = src.execute("SELECT * FROM entries ORDER BY id").fetchall()
                src.close()
            except sqlite3.Error:
                continue
            for r in rows:
                cols = r.keys()
                target.add(
                    r["ts"], json.loads(r["tags"] or "[]"), r["summary"],
                    tokens=int(r["tokens"] or 0),
                    title=(r["title"] if "title" in cols else "") or "",
                    content=(r["content"] if "content" in cols else "") or "",
                    gid=int(gid_s))
                merged += 1
            moved.append(gid_s)
        if moved:
            done.update(moved)
            target.set_meta("migrated_gid_dbs", ",".join(sorted(done)))
            for gid_s in moved:
                f = ddir / f"{gid_s}.db"
                try:
                    os.replace(f, ddir / f"{gid_s}.db.migrated")
                except OSError:
                    pass
        return merged

    # ---------- 写 ----------
    def add(self, ts: float, tags: list[str], summary: str,
            tokens: int = 0, title: str = "", content: str = "",
            gid: int = 0) -> int:
        tags = [t.strip() for t in (tags or []) if t and t.strip()][:8]
        title = (title or "").strip()[:40]
        tags_json = json.dumps(tags, ensure_ascii=False)
        grams = self._grams_for(title, tags_json, summary)
        # grams 是 entries 的真实列（external content 的 fts 从这里取），
        # INSERT 触发器自动带进 fts，无需手动同步
        cur = self._db.execute(
            "INSERT INTO entries(ts, title, tags, summary, tokens, grams, "
            "content, gid) VALUES (?,?,?,?,?,?,?,?)",
            (float(ts), title, tags_json,
             (summary or "").strip(), int(tokens), grams,
             (content or "").strip(), int(gid or 0)))
        self._db.commit()
        return cur.lastrowid

    def delete(self, entry_id: int) -> bool:
        cur = self._db.execute("DELETE FROM entries WHERE id=?", (int(entry_id),))
        self._db.commit()
        return cur.rowcount > 0

    def update(self, entry_id: int, *, title: str | None = None,
               tags: list[str] | None = None, summary: str | None = None,
               content: str | None = None, tokens: int | None = None,
               gid: int | None = None) -> bool:
        """原地改一篇日记（v0.4.0：摘录条目升级成正式日记用）。

        external content 的 FTS 不能直接 UPDATE——必须先按旧值发 'delete'
        指令再插新行，否则索引与内容错位（malformed）。
        """
        r = self._db.execute(
            "SELECT * FROM entries WHERE id=?", (int(entry_id),)).fetchone()
        if r is None:
            return False
        new_title = (str(title).strip()[:40] if title is not None
                     else (r["title"] or ""))
        if tags is None:
            new_tags = r["tags"] or "[]"
        else:
            clean = [str(t).strip() for t in tags if str(t).strip()][:8]
            new_tags = json.dumps(clean, ensure_ascii=False)
        new_summary = (str(summary).strip() if summary is not None
                       else (r["summary"] or ""))
        new_content = (str(content).strip() if content is not None
                       else (r["content"] or ""))
        new_tokens = int(tokens) if tokens is not None else int(r["tokens"] or 0)
        new_gid = int(gid) if gid is not None else int(r["gid"] or 0)
        grams = self._grams_for(new_title, new_tags, new_summary)
        try:
            self._db.execute(
                "INSERT INTO fts(fts, rowid, grams, title, summary, tags) "
                "VALUES ('delete', ?, ?, ?, ?, ?)",
                (r["id"], r["grams"] or "", r["title"] or "",
                 r["summary"] or "", r["tags"] or "[]"))
        except sqlite3.OperationalError:
            pass
        self._db.execute(
            "UPDATE entries SET title=?, tags=?, summary=?, content=?, "
            "tokens=?, gid=?, grams=? WHERE id=?",
            (new_title, new_tags, new_summary, new_content,
             new_tokens, new_gid, grams, int(entry_id)))
        self._db.execute(
            "INSERT INTO fts(rowid, grams, title, summary, tags) VALUES (?,?,?,?,?)",
            (int(entry_id), grams, new_title, new_summary, new_tags))
        self._db.commit()
        return True

    # ---------- 查 ----------
    @staticmethod
    def _scope_sql(scope: str, alias: str = "") -> str:
        """gid==0 → 含私聊内容；gid>0 → 纯群聊。scope 过滤用（v0.4.0 隐私边界）。"""
        col = f"{alias}.gid" if alias else "gid"
        if scope == "private":
            return f" AND {col} = 0"
        if scope == "group":
            return f" AND {col} > 0"
        return ""

    def _rows_to_text(self, rows: list[sqlite3.Row], header: str,
                      *, hide_private: bool = False) -> str:
        if not rows:
            return f"{header}\n（没有找到日记）"
        lines = [header]
        for r in rows:
            t = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
            tags = "、".join(json.loads(r["tags"]))
            title = (r["title"] or "").strip()
            head = f"{t} {title} #{r['id']}" if title else f"{t} #{r['id']}"
            private = not int(r["gid"] or 0)
            # 三层日记（v0.2.5）：#id 是钥匙，「▷」表示存有原话全文可展开
            # v0.4.0：群聊视角（hide_private）看不到私聊篇目的原话标记
            mark = " ▷" if ((r["content"] or "").strip()
                            and not (hide_private and private)) else ""
            lines.append(f"【{head}】{mark}标签：{tags}\n{r['summary']}")
            if private and hide_private:
                lines.append("（这篇含私聊内容，群里只给概要）")
        return "\n".join(lines)

    def expand(self, entry_id: int, *, allow_private: bool = True) -> str:
        """第三层：按 id 取单篇全文（原文选段）。没有全文返回提示。

        allow_private=False（群聊侧）：含私聊内容的篇目只给概要，不给原话。
        """
        r = self._db.execute(
            "SELECT * FROM entries WHERE id=?", (int(entry_id),)).fetchone()
        if r is None:
            return f"（日记 #{int(entry_id)} 不存在）"
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        tags = "、".join(json.loads(r["tags"]))
        head = f"{t} {(r['title'] or '').strip()} #{r['id']}".strip()
        body = (r["content"] or "").strip()
        if not allow_private and not int(r["gid"] or 0):
            return (f"【{head}】标签：{tags}\n（这篇含私聊内容，群里只给概要）\n"
                    f"{r['summary']}")
        if not body:
            return f"【{head}】标签：{tags}\n（这篇只有概要，没存原话）\n{r['summary']}"
        return f"【{head}】标签：{tags}\n概要：{r['summary']}\n原话：\n{body}"

    def entry_content(self, entry_id: int, limit: int = 0) -> str:
        """单篇的原文选段（摘录条目升级时读回原文用）。"""
        r = self._db.execute(
            "SELECT content FROM entries WHERE id=?", (int(entry_id),)).fetchone()
        if r is None:
            return ""
        body = (r["content"] or "").strip()
        return body[:int(limit)] if limit and limit > 0 else body

    def is_private_entry(self, entry_id: int) -> bool:
        """该篇是否含私聊内容（gid==0 保守判定，宁可少给原话也不漏）。"""
        r = self._db.execute(
            "SELECT gid FROM entries WHERE id=?", (int(entry_id),)).fetchone()
        return bool(r is not None and not int(r["gid"] or 0))

    def has_content(self, entry_id: int) -> bool:
        """该篇是否存了全文（决定回捞时要不要跑第三层）。"""
        r = self._db.execute(
            "SELECT content FROM entries WHERE id=?", (int(entry_id),)).fetchone()
        return bool(r and (r["content"] or "").strip())

    def recent(self, n: int = 5, *, scope: str = "any") -> str:
        """最近的日记（倒序），给模型做「标题行」提示或翻看。"""
        rows = self._db.execute(
            "SELECT * FROM entries WHERE 1=1" + self._scope_sql(scope) +
            " ORDER BY ts DESC, id DESC LIMIT ?",
            (max(1, min(int(n), 50)),)).fetchall()
        return self._rows_to_text(rows, f"（最近的日记·{len(rows)}篇）",
                                  hide_private=(scope == "group"))

    def recent_rows(self, n: int = 10, *, scope: str = "any") -> list[sqlite3.Row]:
        """最近日记的原始行（ts/tags/summary），给控制台 API 结构化输出用。"""
        return self._db.execute(
            "SELECT * FROM entries WHERE 1=1" + self._scope_sql(scope) +
            " ORDER BY ts DESC, id DESC LIMIT ?",
            (max(1, min(int(n), 50)),)).fetchall()

    def search(self, query: str, n: int = 5, *, scope: str = "any") -> str:
        """关键词寻回：FTS5 二字组 BM25（fts_zh.grams_query）；兜底 LIKE 子串。

        scope="private"/"group" 按 gid 过滤（v0.4.0 合并库后的隐私边界）。
        """
        q = (query or "").strip()
        if not q:
            return self.recent(n, scope=scope)
        match = grams_query(q)
        rows = []
        if match:
            try:
                rows = self._db.execute(
                    "SELECT e.* FROM fts JOIN entries e ON e.id = fts.rowid "
                    "WHERE fts MATCH ?" + self._scope_sql(scope, "e") +
                    " ORDER BY bm25(fts) LIMIT ?",
                    (match, max(1, min(int(n), 50)))).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if not rows:
            # 兜底：整串子串匹配（单字查询/超短词 FTS 命中不了）
            rows = self._db.execute(
                "SELECT * FROM entries WHERE (title LIKE ? OR summary LIKE ? "
                "OR tags LIKE ?)" + self._scope_sql(scope) +
                " ORDER BY ts DESC LIMIT ?",
                (f"%{q}%", f"%{q}%", f"%{q}%",
                 max(1, min(int(n), 50)))).fetchall()
        return self._rows_to_text(rows, f"（关键词「{q}」的日记·{len(rows)}篇）",
                                  hide_private=(scope == "group"))

    def by_time(self, query: str, n: int = 8, now: float | None = None,
                *, scope: str = "any") -> str:
        """时间寻回：「昨天/上周/2026-10-20/10月」→ 时间范围过滤，正序（叙事感）。"""
        rng = parse_time_query(query, now)
        if not rng:
            return f"（没看懂时间「{query}」，可以说：昨天/上周/上个月/2026-10-20/10月）"
        rows = self._db.execute(
            "SELECT * FROM entries WHERE ts >= ? AND ts <= ?" +
            self._scope_sql(scope) +
            " ORDER BY ts ASC, id ASC LIMIT ?",
            (rng[0], rng[1], max(1, min(int(n), 50)))).fetchall()
        return self._rows_to_text(rows, f"（这段时间的日记·{len(rows)}篇）",
                                  hide_private=(scope == "group"))

    def headers_for_prompt(self, n: int = 10, *, scope: str = "any") -> str:
        """给系统提示词的日记标题行：时间+标题（无标题用标签），不放正文。
        约 20 token/篇。标题不含原话，两种视角都能看。"""
        rows = self._db.execute(
            "SELECT ts, title, tags FROM entries WHERE 1=1" +
            self._scope_sql(scope) +
            " ORDER BY ts DESC, id DESC LIMIT ?",
            (max(0, min(int(n), 20)),)).fetchall()
        if not rows:
            return ""
        lines = []
        for r in rows:
            t = time.strftime("%m-%d", time.localtime(r["ts"]))
            label = (r["title"] or "").strip() or "、".join(json.loads(r["tags"]))
            lines.append(f"{t} {label}")
        return "\n".join(lines)

    def count(self, *, scope: str = "any") -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM entries WHERE 1=1" + self._scope_sql(scope)
        ).fetchone()
        return int(row[0]) if row else 0
