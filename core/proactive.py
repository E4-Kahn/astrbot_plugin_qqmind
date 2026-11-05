"""主动私信（老板 2026-10-30 定案：休息时段主动找熟人说句话）。

能力边界先说死（诚实账单原则）：
- QQ 协议拿不到好友实时在线状态 → 「在不在」只能启发式推断：
  最近一次发言距今 ≥ active_gap_hours 视为「现在闲」。
- 主动私信风控敏感 → 护栏不是选项是必须：
  * 只对白名单（proactive_friends）发，默认空=功能关
  * 每人每天最多 1 条（proactive_per_day）
  * 连续 no_reply_limit 次没回应 → 该用户冷却 7 天
  * 只在 quiet hours 时段发（默认 21:00-22:30）
- 带记忆：触发时读该用户日记寻回 + 窗口尾，生成「接上次话头」的问候
"""
from __future__ import annotations

import asyncio
import json
import random
import time

PROACTIVE_PROMPT = (
    "你是 {bot}。现在是 {now}，休息时段，你想主动找 {who} 说句话。\n"
    "【你们的历史】\n{memory}\n"
    "要求：只发一条，30 字以内，像朋友随手发的那种——"
    "接上次聊到的话头，或想到 TA 会感兴趣的事。别用「在吗」，"
    "别解释为什么发消息，别打扰感。直接输出要发的那句话。"
)

COOLDOWN_DAYS = 7


class ProactiveState:
    """护栏状态（内存态，重启清零；冷却可持久化到日记 meta）。"""

    def __init__(self, per_day: int = 1):
        self.per_day = max(0, int(per_day))
        self.day = ""
        self.sent: dict[int, int] = {}          # uid -> 今日已发
        self.no_reply: dict[int, int] = {}      # uid -> 连续未回应次数
        self.cool_until: dict[int, float] = {}  # uid -> 冷却截止 ts
        self.counted: dict[int, float] = {}     # 已记过未回应的那条主动消息 ts（幂等）

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            self.sent = {}

    def can_send(self, uid: int) -> bool:
        self._roll_day()
        if self.per_day <= 0:
            return False
        if self.sent.get(uid, 0) >= self.per_day:
            return False
        if time.time() < self.cool_until.get(uid, 0):
            return False
        return True

    def mark_sent(self, uid: int) -> None:
        self._roll_day()
        self.sent[uid] = self.sent.get(uid, 0) + 1

    def mark_no_reply(self, uid: int, limit: int) -> bool:
        """记录一次未回应；达到上限触发 7 天冷却，返回 True=已冷却。"""
        n = self.no_reply.get(uid, 0) + 1
        self.no_reply[uid] = n
        if n >= max(1, int(limit)):
            self.cool_until[uid] = time.time() + COOLDOWN_DAYS * 86400
            self.no_reply[uid] = 0
            return True
        return False

    def mark_replied(self, uid: int) -> None:
        self.no_reply[uid] = 0

    def sweep_no_reply(self, uid: int, last_pro_ts: float,
                       last_user_ts: float, limit: int) -> bool:
        """主动消息 24h 无回应判定（幂等：同一条主动消息只记一次）。

        返回 True = 本次判定触发 7 天冷却。发完主动消息没人理时由
        _proactive_tick 每轮调用——没有这步，mark_no_reply 就是死代码，
        「连续 no_reply_limit 次没回应 → 冷却 7 天」护栏形同虚设。
        """
        if not last_pro_ts or last_pro_ts <= (last_user_ts or 0):
            return False
        if time.time() - last_pro_ts < 86400:
            return False
        if self.counted.get(uid, 0) >= last_pro_ts:
            return False
        self.counted[uid] = last_pro_ts
        return self.mark_no_reply(uid, limit)


def in_quiet_hours(now: time.struct_time | None = None,
                   start: str = "21:00", end: str = "22:30") -> bool:
    """当前是否在安静时段（支持跨午夜，如 21:00→23:59 或 07:00→09:00）。"""
    t = now or time.localtime()
    cur = t.tm_hour * 60 + t.tm_min

    def _min(hhmm: str) -> int:
        try:
            h, m = hhmm.split(":")
            return int(h) * 60 + int(m)
        except Exception:
            return 0
    s, e = _min(start), _min(end)
    if s <= e:
        return s <= cur <= e
    return cur >= s or cur <= e   # 跨午夜


def pick_target(active_uids: list[tuple[int, float]],
                gap_hours: float = 2.0) -> int | None:
    """从最近活跃的用户里挑一个「现在闲」的：最近发言距 now ≥ gap_hours。

    active_uids: [(uid, last_active_ts)]；挑最久没聊的那个（更有话头）。
    """
    now = time.time()
    idle = [(ts, uid) for uid, ts in active_uids
            if now - ts >= gap_hours * 3600]
    if not idle:
        return None
    idle.sort()
    return idle[0][1]


async def build_greeting(provider, bot_name: str, who: str,
                         memory_text: str) -> str:
    """带记忆的问候语。失败返回空串（本次跳过，不打扰）。"""
    ask = PROACTIVE_PROMPT.format(
        bot=bot_name or "AI",
        now=time.strftime("%H:%M"),
        who=who or "朋友",
        memory=memory_text.strip() or "（还没有聊天记录，就打个自然的招呼）")
    try:
        out = await provider.text_chat(
            prompt=ask, session_id=None,
            system_prompt="你是对方的熟朋友，说话简短自然。",
            contexts=[], image_urls=[])
        reply = (out.completion_text or "").strip()
        return reply[:120] if reply else ""
    except Exception:
        return ""


def memory_digest(diary, window_tail: list[dict], n_topics: int = 3,
                  scope: str = "private") -> str:
    """触发时的记忆喂料：最近日记概要 + 窗口尾部对话。

    scope（v0.4.0 合并记忆）：默认只取私聊相关篇目——群聊内容不许被塞进
    主动私信（主人私下收到的问候不该带出群里的对话）。
    """
    parts = []
    if window_tail:
        lines = [f"{'用户' if m['role'] == 'user' else 'AI'}：{m['content']}"
                 for m in window_tail[-6:]]
        parts.append("最近聊过：\n" + "\n".join(lines))
    d = ""
    if hasattr(diary, "search"):
        try:
            d = diary.search("", n_topics, scope=scope)
        except TypeError:
            d = diary.search("", n_topics)
    if d and "没有找到日记" not in d:
        parts.append("日记里的旧事：\n" + d)
    return "\n\n".join(parts)
