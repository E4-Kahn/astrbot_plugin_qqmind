"""好感度系统（老板 2026-10-30 定案）。

- 跨群唯一：按 QQ 号一人一份，存该用户日记库的 meta 表（affection key），
  没私聊过也会懒建库，群聊互动照样累计
- 每份含：score（0~100，默认起点可配）、note（老板写的备注，如作息/称呼/雷点，
  会注入该用户相关回复的 system prompt）、schedule（作息，"HH:MM-HH:MM" 活跃段，
  主动私信只在对方作息内触发）
- 增减：判读判「是」+1（对方主动找 AI 说）、AI 回复后 +1、被 @ 额外 +2；
  判否不扣（不聊天≠讨厌）；可配每日上限防刷分
- 触发：主动私信候选=好感度 ≥ affinity_threshold 且当前时间在对方作息内
"""
from __future__ import annotations

import json
import re
import time

DEFAULT = {"score": 50, "note": "", "schedule": ""}
_HHMM_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def parse_schedule(s: str) -> tuple[int, int] | None:
    """"21:30-23:30" → (21*60+30, 23*60+30)。空串/非法返回 None（=不限制）。"""
    s = (s or "").strip()
    if not s:
        return None
    parts = s.replace("～", "-").replace("~", "-").split("-")
    if len(parts) != 2 or not _HHMM_RE.match(parts[0].strip()) \
            or not _HHMM_RE.match(parts[1].strip()):
        return None

    def m(x: str) -> int:
        h, mi = x.strip().split(":")
        return int(h) * 60 + int(mi)
    return m(parts[0]), m(parts[1])


def in_schedule(sched: str, now: time.struct_time | None = None) -> bool:
    """当前时间是否在作息段内；没配置作息=不限制（任何时间可以）。"""
    rng = parse_schedule(sched)
    if rng is None:
        return True
    t = now or time.localtime()
    cur = t.tm_hour * 60 + t.tm_min
    s, e = rng
    if s <= e:
        return s <= cur <= e
    return cur >= s or cur <= e   # 跨午夜


def clamp_score(v: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(v)))


def profile_to_prompt(data: dict, name: str = "") -> str:
    """好感度备注 → system prompt 片段（空数据返回空串）。"""
    lines = []
    if data.get("note"):
        lines.append(f"关于{name or '这位用户'}的备注：{data['note']}")
    if data.get("schedule"):
        lines.append(f"TA 的活跃时段通常是 {data['schedule']}，"
                     "避开 TA 作息外的深夜/清晨打扰")
    if not lines:
        return ""
    return "【相处须知】\n" + "\n".join(lines)


class AffectionStore:
    """好感度读写（借 DiaryStore 的 meta 表持久化，跨群按 uid 唯一）。"""

    def __init__(self, get_session, base: int = 50, daily_gain_cap: int = 10):
        self._get_session = get_session   # uid -> PrivateSession（懒建由调用方保证）
        self.base = clamp_score(base)
        self.daily_gain_cap = max(0, int(daily_gain_cap))
        self._gained: dict[int, int] = {}  # uid -> 今日已自然增长
        self._day = ""

    def _diary(self, uid: int):
        """PrivateSession（有 .diary）或 DiaryStore（自带 meta）都吃。"""
        obj = self._get_session(int(uid))
        return getattr(obj, "diary", obj)

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._gained = {}

    @staticmethod
    def _key(uid: int) -> str:
        """v0.4.0：一个库装所有人，好感度按 uid 分键（旧单键会串号）。"""
        return f"affinity:{int(uid)}"

    def get(self, uid: int) -> dict:
        d = self._diary(uid)
        raw = d._meta_get(self._key(uid))
        if not raw:
            # 旧库兼容：单用户库里的单键 affection（合并时已由 merge_db 搬好）
            legacy = d._meta_get("affection")
            if legacy:
                try:
                    _meta_set = d._meta_set
                except AttributeError:
                    _meta_set = None
                if _meta_set:
                    try:
                        _meta_set(self._key(uid), legacy)
                        if hasattr(d, "del_meta"):
                            d.del_meta("affection")
                        raw = legacy
                    except Exception:      # noqa: BLE001
                        raw = legacy
        if not raw:
            d0 = dict(DEFAULT)
            d0["score"] = self.base
            return d0
        try:
            out = json.loads(raw)
            return out if isinstance(out, dict) else dict(DEFAULT)
        except Exception:
            return dict(DEFAULT)

    def _put(self, uid: int, data: dict) -> None:
        self._diary(uid)._meta_set(self._key(uid),
                                   json.dumps(data, ensure_ascii=False))

    def set_note(self, uid: int, note: str) -> None:
        d = self.get(uid)
        d["note"] = (note or "").strip()[:300]
        self._put(uid, d)

    def set_schedule(self, uid: int, schedule: str) -> None:
        d = self.get(uid)
        d["schedule"] = (schedule or "").strip()[:40]
        self._put(uid, d)

    def add(self, uid: int, delta: int) -> int:
        """自然增长（受每日上限管）；delta<0 直接扣不设上限。返回新分数。"""
        d = self.get(uid)
        delta = int(delta)
        if delta > 0 and self.daily_gain_cap > 0:
            self._roll_day()
            room = self.daily_gain_cap - self._gained.get(int(uid), 0)
            if room <= 0:
                return clamp_score(d["score"])
            delta = min(delta, room)
            self._gained[int(uid)] = self._gained.get(int(uid), 0) + delta
        d["score"] = clamp_score(d["score"] + delta)
        self._put(uid, d)
        return d["score"]

    def candidates(self, uids: list[int], threshold: int) -> list[int]:
        """好感度达到阈值的用户（主动私信的候选池）。"""
        return [u for u in uids if self.get(u)["score"] >= threshold]
