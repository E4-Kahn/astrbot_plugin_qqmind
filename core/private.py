"""记忆会话：私聊与群聊共用一条时间线 + 一个日记库（v0.4.0 合并）。

为什么合并：旧版私聊窗口存 `<uid>.db` 的 meta['window']、群聊窗口存
`group_memory/<gid>.json`、群日记另存 `group_public.db`——私聊里的 AI
不知道群里发生了什么，群里也接不上私聊的话。合并后只有一份窗口
（`memory.db` 的 meta['window']）和一份日记库，每条消息带来源标记
（src/gid/uid/nick），写日记时一起压，读的时候按场景过滤。

四层压缩（老板 2026-11-05 定案）：把「移出窗口」和「生成摘要」解耦。
旧版压缩失败就把原文留在窗口里等下一轮重试，模型一抽风窗口就无限涨；
现在：
  1. evict_oldest()  纯本地削窗口，先把原文存成「摘录条目」（零成本）
  2. summarize()     再调模型升级成正式日记；失败就留着摘录条目
  3. 熔断            连续失败 ``FAIL_LIMIT`` 次 → 冷却 ``COOLDOWN_SEC``，
                     期间只走本地摘录（与判读门的 _judge_fail 同构）
  4. 补升级          每轮顺带挑一条历史摘录条目升级，碎片慢慢变整齐
窗口 token 因此单调下降，与模型可用性无关。

隐私边界：entries.gid == 0 表示「含私聊内容」；群聊侧读日记时只给概要、
不给原话（`DiaryStore` 的 scope 参数负责）。群聊原文要进群聊 prompt 的
路径只有 `context(scope="group")`，它只取 src=="group" 的消息。
"""
from __future__ import annotations

import json
import re
import time

from .core import est_tokens
from .diary import DiaryStore

# 压缩提示词：一个模型走天下，输出严格 JSON（title 供控制台列表显示）
COMPRESS_PROMPT = (
    "你是记忆整理助手。把下面这段对话压缩成一篇日记，帮对方长期记住这件事。\n"
    "对话可能包含私聊和群聊，行首标了来源（「用户」「AI」「群123 昵称」）。\n"
    "只输出一行 JSON（不要代码块、不要解释）：\n"
    '{"title": "一句话标题，10字以内", "tags": ["标签1", "标签2"], '
    '"summary": "两三句话概括：聊了什么、有什么值得记住的'
    '（关键事实/约定/情绪/偏好/待办）", '
    '"content": "值得留原话的部分：直接引用1~3句重要原话，'
    '没有值得留的就输出空字符串"}\n'
    "私聊部分用第二人称「你」称呼对方；群聊部分用第三人称，保留说话人昵称。\n"
    "标签 2~5 个中文词。这段要是只有寒暄表情、没有值得记的事，"
    "summary 写「这段没有值得记的事」即可。"
)

# 兼容旧名（旧调用方/测试可能 import 这个常量）
MEMORY_COMPRESS_PROMPT = COMPRESS_PROMPT

FAIL_LIMIT = 3          # 连续失败几次触发熔断
COOLDOWN_SEC = 1800     # 熔断冷却时长（秒）
SLICE_MSGS = 30         # 一次最多压走多少条
RAW_SUMMARY_CHARS = 200      # 摘录条目的概要长度
RAW_CONTENT_CHARS = 3000     # 摘录条目保留的原文长度


def _truncate(text: str, cap_tokens: int) -> str:
    """超长消息截断保存（头 2/3、尾 1/3），不整条丢。"""
    head = max(200, cap_tokens * 2)
    tail = max(100, cap_tokens)
    if len(text) > head + tail:
        return text[:head] + "\n…（中间省略）…\n" + text[-tail:]
    return text


async def complete_text(llm, system: str, user: str,
                        max_tokens: int = 600, temperature: float = 0.3) -> str:
    """统一 LLM 调用：测试桩用 .complete，AstrBot Provider 用 .text_chat。"""
    out = await complete_text_resp(llm, system, user,
                                   max_tokens=max_tokens,
                                   temperature=temperature)
    return out[0]


async def complete_text_resp(llm, system: str, user: str,
                             max_tokens: int = 600,
                             temperature: float = 0.3) -> tuple[str, object]:
    """同 complete_text，但连 LLMResponse 一起返回（真实 usage 分桶记账用）。

    测试桩 .complete 路径没有 usage 对象时第二项为 None。

    v0.4.0：text_chat 分支补传 max_tokens/temperature——旧版这两个参数在
    Provider 路径上被静默丢弃，压缩输出长度完全靠模型自觉。部分 Provider
    签名不收这两个 kwarg，TypeError 时回落到原调用（行为与旧版一致）。
    """
    if hasattr(llm, "complete"):
        text = await llm.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature)
        return text, None
    try:
        out = await llm.text_chat(prompt=user, session_id=None,
                                  system_prompt=system, contexts=[],
                                  image_urls=[], max_tokens=max_tokens,
                                  temperature=temperature)
    except TypeError:
        out = await llm.text_chat(prompt=user, session_id=None,
                                  system_prompt=system, contexts=[],
                                  image_urls=[])
    return out.completion_text or "", out


class MemorySession:
    """一条时间线：私聊与群聊共用的 token 预算窗口 + 日记库。

    messages 元素：``{"role","content","ts","src","gid","uid"[,"nick"]}``
    - src="private"：私聊（gid=0，uid=说话人）
    - src="group"  ：群聊（gid=群号，uid/nick=说话人）
    - assistant 消息也带 src：私聊里回的记 private，群里回的记 group
    """

    def __init__(self, store, budget: int = 10000, *,
                 uid: int = 0, keep_ratio: float = 0.34,
                 max_msgs: int = 1000):
        """store 可以是 DiaryStore（合并后的公共库），也可以是目录（旧测试用法）。

        传目录时按旧语义懒建 `<uid>.db`（保持旧测试/旧调用可跑）。
        """
        if isinstance(store, DiaryStore):
            self.diary = store
        else:
            self.diary = DiaryStore(store, int(uid or 0))
        self.uid = int(uid or 0)
        self.budget = int(budget)          # 窗口 token 预算（可调）
        self.keep_ratio = keep_ratio       # 兼容旧配置（当前按预算切，不使用）
        self.max_msgs = int(max_msgs)
        self.messages: list[dict] = []
        self.compressing = False
        # 削窗口钩子（v0.4.0）：`on_evict(msgs) -> bool`，返回 True 表示这段
        # 已被外部接管（群聊分层模式的候选池），不要再生成摘录条目。
        self.on_evict = None
        self._fail_streak = 0
        self._fail_until = 0.0
        self._raw_pending: list[int] = []
        self._load_window()

    # ---------- 窗口持久化 ----------
    def _load_window(self) -> None:
        """从共享库 meta 恢复窗口（插件重载/重启不再失忆）。"""
        try:
            raw = self.diary.get_meta("window")
        except Exception:
            return
        if not raw:
            return
        try:
            data = json.loads(raw)
        except Exception:
            return
        if not isinstance(data, list):
            return
        msgs: list[dict] = []
        for m in data:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            try:
                ts = float(m.get("ts") or 0.0)
            except (TypeError, ValueError):
                ts = 0.0
            item = {"role": role, "content": content, "ts": ts,
                    "src": "group" if m.get("src") == "group" else "private",
                    "gid": int(m.get("gid") or 0),
                    "uid": int(m.get("uid") or 0)}
            if m.get("nick"):
                item["nick"] = str(m["nick"])
            if m.get("proactive"):
                item["proactive"] = True
            msgs.append(item)
        self.messages = msgs[-self.max_msgs:]
        self._raw_pending = self._load_pending()

    def save_window(self) -> None:
        """窗口写回共享库 meta（与日记同库同生命周期）。"""
        try:
            self.diary.set_meta(
                "window", json.dumps(self.messages[-self.max_msgs:],
                                     ensure_ascii=False))
        except Exception as e:
            print("[memory] save window error:", repr(e))

    def _load_pending(self) -> list[int]:
        try:
            data = json.loads(self.diary.get_meta("raw_pending") or "[]")
            return [int(x) for x in data if str(x).isdigit()][:50]
        except Exception:
            return []

    def _save_pending(self) -> None:
        try:
            self.diary.set_meta("raw_pending",
                                json.dumps(self._raw_pending[:50]))
        except Exception:
            pass

    # ---------- 写入 ----------
    def append(self, role: str, content: str, *, src: str = "private",
               gid: int = 0, uid: int = 0, nick: str = "", ts: float | None = None,
               cap_tokens: int = 0, **extra) -> None:
        """一条消息入窗口。cap_tokens>0 时超长截断（头 2/3、尾 1/3）。"""
        text = (content or "").strip()
        if not text:
            return
        if cap_tokens and est_tokens(text) > cap_tokens:
            text = _truncate(text, cap_tokens)
        item = {"role": role, "content": text,
                "ts": float(ts if ts is not None else time.time()),
                "src": "group" if src == "group" else "private",
                "gid": int(gid or 0), "uid": int(uid or 0)}
        if nick:
            item["nick"] = str(nick)
        item.update(extra)
        self.messages.append(item)
        if len(self.messages) > self.max_msgs:
            del self.messages[:len(self.messages) - self.max_msgs]

    def append_user_message(self, text: str, cap_tokens: int = 400, **kw) -> None:
        self.append("user", text, cap_tokens=cap_tokens, **kw)

    def append_assistant(self, text: str, cap_tokens: int = 0, **kw) -> None:
        self.append("assistant", text, cap_tokens=cap_tokens, **kw)

    # ---------- 读 ----------
    def _scoped(self, scope: str = "private", gid: int = 0) -> list[dict]:
        if scope == "group":
            return [m for m in self.messages
                    if m.get("src") == "group"
                    and (not gid or int(m.get("gid") or 0) == int(gid))]
        if scope == "private":
            return [m for m in self.messages if m.get("src", "private") == "private"]
        return list(self.messages)

    def context(self, *, scope: str = "private", gid: int = 0,
                limit: int = 0, skip_last: int = 0) -> list[dict]:
        """给 LLM 的逐字 contexts（按来源过滤；群聊只取本群）。"""
        msgs = self._scoped(scope, gid)
        if skip_last > 0:
            msgs = msgs[:-skip_last] if skip_last < len(msgs) else []
        if limit and limit > 0:
            msgs = msgs[-int(limit):]
        out = []
        for m in msgs:
            content = m.get("content") or ""
            if scope == "group" and m.get("role") == "user":
                nick = m.get("nick") or (f"QQ{m.get('uid')}" if m.get("uid") else "群友")
                content = f"{nick}：{content}"
            out.append({"role": m.get("role") or "user", "content": content})
        return out

    def recent_group_lines(self, n: int = 10, max_chars: int = 60) -> list[str]:
        """最近群聊压缩成一行一条（私聊 prompt 的「群里发生什么」速览）。"""
        msgs = [m for m in self._scoped("group")
                if m.get("role") == "user"]
        if n > 0:
            msgs = msgs[-int(n):]
        lines = []
        for m in msgs:
            nick = m.get("nick") or "群友"
            text = (m.get("content") or "").replace("\n", " ").strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "…"
            lines.append(f"群{m.get('gid') or 0} {nick}：{text}")
        return lines

    # ---------- 窗口 token 账 ----------
    def window_tokens(self) -> int:
        return sum(est_tokens(m.get("content") or "") for m in self.messages)

    def over_budget(self) -> bool:
        return self.window_tokens() > self.budget

    def slice_oldest(self, max_msgs: int = SLICE_MSGS) -> list[dict]:
        """最老的一段（给窗口留 2 条尾巴，方便接得上话）。"""
        total = len(self.messages)
        if total < 2:
            return []
        n = min(total, max(1, int(max_msgs)))
        if total > n and total - n < 2:
            n = max(1, total - 2)
        return self.messages[:n]

    # 旧名兼容
    oldest_slice = slice_oldest

    # ---------- 第 1 层：本地削窗口（零成本，不依赖模型） ----------
    def evict_oldest(self) -> tuple[list[dict], int | None]:
        """把最老一段移出窗口并存成「摘录条目」。失败也照样移出（不再堵窗口）。"""
        msgs = self.slice_oldest()
        if not msgs:
            return [], None
        taken = False
        if self.on_evict is not None:
            try:
                taken = bool(self.on_evict(msgs))
            except Exception:      # noqa: BLE001 钩子坏了也不许堵窗口
                taken = False
        if taken:
            del self.messages[:len(msgs)]
            self.save_window()
            return msgs, None
        text = self.excerpt(msgs)
        first_ts = msgs[0].get("ts") or time.time()
        tok = sum(est_tokens(m.get("content") or "") for m in msgs)
        summary = text[:RAW_SUMMARY_CHARS].strip()
        if len(text) > RAW_SUMMARY_CHARS:
            summary += "…"
        summary += "\n（自动摘录，尚未整理成日记）"
        eid = None
        try:
            eid = self.diary.add(
                first_ts, [], summary, tok,
                title=self.derive_title(text),
                content=text[:RAW_CONTENT_CHARS],
                gid=self._gid_for(msgs))
            self.diary.set_meta(f"raw_{eid}", "1")
            self._raw_pending.append(int(eid))
            self._save_pending()
        except Exception as e:
            print("[memory] evict archive error:", repr(e))
        del self.messages[:len(msgs)]
        self.save_window()
        return msgs, eid

    @staticmethod
    def _gid_for(msgs: list[dict]) -> int:
        """条目归属：含私聊内容一律记 0（群聊侧只给概要）；否则取群号多数。"""
        if any(m.get("src", "private") == "private" for m in msgs):
            return 0
        counts: dict[int, int] = {}
        for m in msgs:
            gid = int(m.get("gid") or 0)
            if gid:
                counts[gid] = counts.get(gid, 0) + 1
        if not counts:
            return 0
        return max(counts.items(), key=lambda kv: kv[1])[0]

    # ---------- 第 2 层：模型升级 ----------
    @staticmethod
    def excerpt(msgs: list[dict]) -> str:
        """把消息渲染成带来源的文本（压缩输入 / 摘录条目都用它）。"""
        lines = []
        for m in msgs or []:
            role = m.get("role")
            src = m.get("src", "private")
            if role == "assistant":
                who = "AI"
            elif src == "group":
                who = (f"群{m.get('gid') or 0} "
                       f"{m.get('nick') or ('QQ' + str(m.get('uid')) if m.get('uid') else '群友')}")
            else:
                who = "用户"
            lines.append(f"{who}：{m.get('content') or ''}")
        return "\n".join(lines)

    # 旧名兼容
    excerpt_for_compress = excerpt

    @staticmethod
    def derive_title(summary: str, limit: int = 14) -> str:
        """模型没给标题时的兜底：摘要第一句，截断到 limit 字。"""
        s = (summary or "").strip().split("\n")[0]
        for sep in "。！？!?；;，,":
            i = s.find(sep)
            if i > 0:
                s = s[:i]
                break
        s = s.strip(" ，,。.！!?？")
        return s[:limit] + ("…" if len(s) > limit else "")

    async def summarize(self, eid: int, msgs: list[dict] | None = None,
                        llm=None, text: str | None = None) -> int:
        """把摘录条目升级成正式日记。返回估算消耗 token（0=失败，条目保留）。"""
        if llm is None or not eid:
            return 0
        body = text if text is not None else self.excerpt(msgs or [])
        if not (body or "").strip():
            return 0
        ask = (f"当前时间 {time.strftime('%Y-%m-%d %H:%M')}。\n\n" + body)
        tok_in = est_tokens(body)
        try:
            out = await complete_text(llm, COMPRESS_PROMPT, ask,
                                      max_tokens=700, temperature=0.3)
        except Exception as e:
            print("[memory] summarize error:", repr(e))
            return 0
        parsed = self.parse_compress_output(out)
        if not parsed:
            return 0
        title, tags, summary, content = parsed
        old = self.diary.entry_content(eid, limit=RAW_CONTENT_CHARS)
        if not content:
            content = old[:600]        # 模型没给原话：至少留一段存档
        ok = self.diary.update(eid, title=title, tags=tags,
                               summary=summary, content=content)
        if not ok:
            return 0
        try:
            self.diary.set_meta(f"raw_{eid}", "0")
        except Exception:
            pass
        self._drop_pending(eid)
        return tok_in + 200

    def _drop_pending(self, eid: int) -> None:
        if eid in self._raw_pending:
            self._raw_pending = [x for x in self._raw_pending if x != eid]
            self._save_pending()

    def _pop_pending(self) -> int:
        while self._raw_pending:
            eid = int(self._raw_pending.pop(0))
            self._save_pending()
            if self.diary.entry_content(eid, limit=1):
                return eid
        return 0

    # ---------- 第 3 层：熔断 ----------
    def in_cooldown(self) -> bool:
        return time.time() < self._fail_until

    def _note_ok(self) -> None:
        self._fail_streak = 0
        self._fail_until = 0.0

    def _note_fail(self) -> None:
        self._fail_streak += 1
        if self._fail_streak >= FAIL_LIMIT:
            self._fail_until = time.time() + COOLDOWN_SEC
            self._fail_streak = 0
            print(f"[memory] compress breaker open "
                  f"({FAIL_LIMIT} fails, cooling {COOLDOWN_SEC}s)")

    # ---------- 总入口：稳压到预算内 ----------
    async def ensure_budget(self, get_llm=None) -> dict:
        """把窗口压回预算内（最多 12 轮），返回本次统计。

        get_llm：无参协程，返回 provider（可以是 None）。传 None 则只做本地
        摘录（不调模型）——调用方拿不到 provider 时也能保证窗口不涨。
        """
        stats = {"slices": 0, "upgraded": 0, "used": 0, "raw": 0,
                 "cooldown": self.in_cooldown()}
        if self.compressing:
            return stats
        self.compressing = True
        llm = None
        try:
            # v0.4.0 修复：provider 先取，别等窗口超预算才取——窗口已经清空时
            # 也要能继续升级历史摘录条目（第 4 层），否则熔断计数永远攒不满。
            if get_llm is not None and not self.in_cooldown():
                try:
                    llm = await get_llm()
                except Exception as e:     # noqa: BLE001
                    print("[memory] aux provider error:", repr(e))
                    llm = None
            guard = 0
            while self.window_tokens() > self.budget and guard < 12:
                guard += 1
                msgs, eid = self.evict_oldest()
                if not msgs:
                    break
                stats["slices"] += 1
                stats["raw"] += 1
                if eid is None or llm is None or self.in_cooldown():
                    continue
                used = await self.summarize(eid, msgs, llm)
                if used:
                    stats["used"] += used
                    stats["upgraded"] += 1
                    stats["raw"] -= 1
                    self._note_ok()
                else:
                    self._note_fail()
                    break
            # 第 4 层：顺带升级一条历史摘录（每次最多 2 条，别把一轮撑满）
            if llm is not None and not self.in_cooldown():
                for _ in range(2):
                    eid = self._pop_pending()
                    if not eid:
                        break
                    text = self.diary.entry_content(eid, limit=RAW_CONTENT_CHARS)
                    used = await self.summarize(eid, None, llm, text=text)
                    if used:
                        stats["used"] += used
                        stats["upgraded"] += 1
                        self._note_ok()
                    else:
                        self._raw_pending.append(eid)
                        self._save_pending()
                        self._note_fail()
                        break
        finally:
            self.compressing = False
        stats["cooldown"] = self.in_cooldown()
        return stats

    async def compress_oldest(self, llm) -> int:
        """兼容旧接口：压走一段并升级成日记，返回消耗 token（0=失败）。"""
        msgs, eid = self.evict_oldest()
        if not msgs or eid is None:
            return 0
        used = await self.summarize(eid, msgs, llm)
        if used:
            self._note_ok()
        else:
            self._note_fail()
        return used

    # ---------- 解析（第 2 层的一部分：多级降级） ----------
    _FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)```", re.S)
    _FIELD_RE_TMPL = r'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'

    @staticmethod
    def _unescape(s: str) -> str:
        try:
            return json.loads('"' + s + '"')
        except Exception:
            return s

    @classmethod
    def _salvage_object(cls, text: str):
        """从截断/带噪的 JSON 里捞第一个完整对象（字符级括号配平）。"""
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        return None
        return None

    @classmethod
    def _field(cls, text: str, key: str) -> str:
        m = re.search(cls._FIELD_RE_TMPL.format(key=key), text, re.S)
        if m:
            return cls._unescape(m.group(1)).strip()
        # 被截断在字段中间：吃到结尾
        m = re.search(r'"%s"\s*:\s*"([^"]*)$' % re.escape(key), text, re.S)
        return cls._unescape(m.group(1)).strip() if m else ""

    @classmethod
    def _tags(cls, text: str) -> list[str]:
        m = re.search(r'"tags"\s*:\s*\[(.*?)\]', text, re.S)
        if not m:
            return []
        return [cls._unescape(x).strip()
                for x in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
                if x.strip()][:8]

    @classmethod
    def parse_compress_output(cls, out: str) -> tuple[str, list[str], str, str] | None:
        """解析压缩输出，四级降级。返回 (title, tags, summary, content) 或 None。

        1. 剥代码块围栏 → json.loads
        2. 括号配平从截断文本里捞对象
        3. 正则逐字段抽取（summary 必须有）
        4. 散文兜底：没有 JSON 但内容是像样的一段话，直接当 summary
        content 是第三层原文选段（模型不输出该字段时为空串，三层退化为两层）；
        title 缺失时由摘要兜底生成。
        """
        t = (out or "").strip()
        if not t:
            return None
        m = cls._FENCE_RE.search(t)
        if m:
            t = m.group(1).strip()
        else:
            t = re.sub(r"^```[a-zA-Z]*\s*", "", t).strip()
            if t.endswith("```"):
                t = t[:-3].strip()

        data = None
        try:
            data = json.loads(t)
        except Exception:
            data = cls._salvage_object(t)

        if isinstance(data, dict):
            title = str(data.get("title") or "").strip()
            tags = [str(x) for x in (data.get("tags") or []) if str(x).strip()]
            summary = str(data.get("summary") or "").strip()
            content = str(data.get("content") or "").strip()
            if summary:
                return ((title or cls.derive_title(summary)),
                        tags[:8], summary, content[:600])

        # 第 3 级：逐字段正则
        summary = cls._field(t, "summary")
        if summary:
            title = cls._field(t, "title")
            content = cls._field(t, "content")
            return ((title or cls.derive_title(summary)),
                    cls._tags(t), summary, content[:600])

        # 第 4 级：散文兜底（模型没按格式来，但说了人话）
        if "{" not in t and 6 <= len(t) <= 800:
            return (cls.derive_title(t), [], t, "")
        return None


# 兼容旧名：main/其它模块一直 import PrivateSession
PrivateSession = MemorySession
