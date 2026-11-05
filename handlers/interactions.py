"""交互链路：戳一戳反应 + 主动私信（C4 解耦自 main.py，v0.3.9）。

无装饰器钩子（戳一戳由判读门 on_message 分流，私信是自驱循环），
纯方法宿主，mixin 形式被 main.QQMindPlugin 继承。
本模块边界即未来散装包的「陪伴交互」件。
"""
from __future__ import annotations

import asyncio
import random
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain

from ..core import (PrivateSession, build_greeting, est_tokens,
                    in_quiet_hours, in_schedule, memory_digest, pick_target,
                    profile_to_prompt, split_bubbles)
from ..core.private import complete_text_resp
from ..main_helpers import _safe_int_id


class InteractionsMixin:

    # ---------- 戳一戳（v0.2.12 完整版） ----------

    _POKE_REACT_PROMPT = (
        "有人戳了戳你（QQ 戳一戳）。用一句话当场反应，10~25 字，"
        "不要输出任何别的。性格按你的人设来，反应要接得住对方的好感度档位：\n"
        "{stance}\n"
        "禁止解释自己是程序/机器人。")

    _POKE_STANCES = (
        ("陌生", "对方是陌生人：礼貌带点距离，客气地问对方有什么事"),
        ("熟识", "对方是熟人：自然随意，可以玩笑或吐槽对方的突袭"),
        ("亲近", "对方是你喜欢的人：开心、害羞或撒娇，按人设来"),
        ("生气", "对方惹过你/你正嫌弃 TA：炸毛、嫌弃，但别辱骂"),
    )

    async def _on_poked(self, gid: int, uid: int, event) -> None:
        """被戳反应：必回 + 好感度分档 + 概率回戳 + 进日记。

        独立于判读门（戳人不需要判是不是在叫它），但受每群频率上限约束；
        高频连戳只回第一下，冷却内静默（不刷屏）。
        """
        now = time.time()
        # 每人冷却 60s：连戳只认真反应一次，其余记数不回
        last = self._poke_last.get((gid, uid), 0.0)
        self._poke_last[(gid, uid)] = now
        if now - last < 60.0:
            return
        if not self._allow_reply(gid):
            return
        if self._budget_exhausted():
            return
        nick = (event.get_sender_name() or "").strip() or str(uid)
        self._note_group(gid, f"{nick}({uid})", "戳了戳机器人")

        # 好感度分档 → 反应基调（关好感度就一律熟识档）
        if self._cfg("affinity_enabled", False):
            score = self._affection.get(uid)["score"]
        else:
            score = self._affection.base
        if score >= 75:
            stance = self._POKE_STANCES[2][1]
        elif score >= 55:
            stance = self._POKE_STANCES[1][1]
        elif score <= 35:
            stance = self._POKE_STANCES[3][1]
        else:
            stance = self._POKE_STANCES[0][1]

        provider = self.context.get_using_provider()
        rt = self._rt(gid)
        origin = str(rt.get("origin") or "")
        if provider is None or not origin:
            return
        sys_prompt = self._group_system_prompt(gid, "", uid=uid, nick=nick)
        try:
            reaction, resp = await complete_text_resp(
                provider,
                sys_prompt or "你是群里的 AI 成员",
                self._POKE_REACT_PROMPT.format(stance=stance),
                max_tokens=80, temperature=0.9)
        except Exception as e:
            logger.warning(f"[qqmind] poke reaction llm failed: {e!r}")
            return
        # 戳一戳也走真实分桶账（v0.2.15 审查修）：system 稳定前缀这一笔
        # 不小（人设+活人模板），只记 est 会漏掉约 1000 tok/次的输入侧
        self._real_usage(resp, "reply")
        self._count_usage(sys_prompt, self._POKE_REACT_PROMPT, reaction,
                          gid=int(gid))
        reaction = split_bubbles(
            (reaction or "").strip(),
            max_total_chars=int(self._cfg("group_reply_max_chars", 80)))
        if not reaction:
            return
        body = reaction[0].strip()
        # 回戳：亲近档 50%、熟识档 25% 概率戳回去（OneBot group_poke）
        if ((score >= 75 and random.random() < 0.5)
                or (55 <= score < 75 and random.random() < 0.25)):
            await self._poke_back(gid, uid)
            body += "（戳回去）"
        ok = await self._send_bubble(gid, uid, body, True, origin)
        if not ok:
            return
        self._stat("replied", gid)
        if self._cfg("affinity_enabled", False):
            self._affection.add(uid, 1)
        # 进日记：轻描淡写一条（不为戳一戳跑压缩模型，省钱）
        if not self._is_light() and self._memory_mode() != "precise":
            gm = self._group_mem(gid)
            gm.diary.add(time.time(), ["戳一戳", "互动"],
                         f"{nick} 戳了机器人一下，"
                         f"机器人反应「{body[:30]}」",
                         tokens=0, title="被戳了一下", content="",
                         gid=gid)
            logger.info(f"[qqmind] poke reaction for {uid} in {gid}: "
                        f"{body[:40]}")

    async def _poke_back(self, gid: int, uid: int) -> None:
        """OneBot group_poke 戳回去；通道不可用/失败静默（文案已带「戳回去」）。"""
        rt = self._rt(gid)
        bot = rt.get("bot")
        self_id = str(rt.get("self_id") or "")
        sid = _safe_int_id(self_id)
        if bot is None or not sid or not hasattr(bot, "call_action"):
            return
        try:
            await bot.call_action("send_group_poke", self_id=sid,
                                  group_id=int(gid), user_id=int(uid))
        except Exception as e:
            logger.debug(f"[qqmind] poke back failed: {e!r}")

    # ---------- 主动私信（v0.2.6） ----------
    async def _proactive_loop(self) -> None:
        """安静时段守候：每 10 分钟醒来一次，挑一个「现在闲」的白名单
        用户发一条带记忆的问候。护栏全在 ProactiveState。"""
        try:
            while True:
                await asyncio.sleep(600)
                try:
                    await self._proactive_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as e:     # noqa: BLE001 守护循环不许死
                    logger.warning(f"[qqmind] proactive tick error: {e!r}")
        except asyncio.CancelledError:
            pass

    def _private_window(self, uid: int) -> list[dict]:
        """合并窗口里属于该 uid 的私聊消息（v0.4.0：窗口私聊/群聊共用）。"""
        try:
            msgs = self._session().messages
        except Exception:          # noqa: BLE001 兼容旧结构
            p = self.privates.get(uid)
            msgs = p.messages if p else []
        return [m for m in msgs
                if m.get("src", "private") == "private"
                and int(m.get("uid") or 0) == int(uid)]

    async def _proactive_tick(self) -> None:
        # 未回应冷却（v0.3.5 接线）：主动消息发出 24h 没人理 → 记一次，
        # 攒满 no_reply_limit 次 → 该用户冷却 7 天。之前这逻辑只有
        # mark_no_reply 方法、没有调用方，护栏形同虚设。
        limit = int(self._cfg("proactive_no_reply_limit", 3))
        if limit > 0:
            for uid in list(self.privates):
                pro_ts = 0.0
                user_ts = 0.0
                for m in self._private_window(uid):
                    if m.get("proactive"):
                        pro_ts = max(pro_ts, float(m.get("ts") or 0))
                    elif m.get("role") == "user":
                        user_ts = max(user_ts, float(m.get("ts") or 0))
                self._proactive.sweep_no_reply(uid, pro_ts, user_ts, limit)
        affinity_on = self._cfg("affinity_enabled", False)
        friends = [str(x) for x in (self._cfg("proactive_friends") or [])]
        if affinity_on:
            # 好感度模式（v0.2.8）：候选=有私聊窗口的活跃用户 ∩ 好感度达标 ∩ 作息匹配
            threshold = int(self._cfg("affinity_threshold", 70))
            actives = []
            for uid in list(self.privates):
                msgs = self._private_window(uid)
                if not msgs:
                    continue
                aff = self._affection.get(uid)
                if aff["score"] < threshold:
                    continue
                if not in_schedule(aff.get("schedule", "")):
                    continue
                actives.append((uid, max(
                    (m.get("ts") or 0) for m in msgs)))
        elif friends:
            # 旧白名单模式（兼容）：安静时段 ∩ 白名单
            actives = []
            for f in friends:
                uid = int(f) if f.isdigit() else 0
                if not uid or uid not in self.privates:
                    continue
                msgs = self._private_window(uid)
                if msgs:
                    actives.append((uid, max(
                        (m.get("ts") or 0) for m in msgs)))
        else:
            return
        if not affinity_on and not in_quiet_hours(
                start=str(self._cfg("proactive_start", "21:00")),
                end=str(self._cfg("proactive_end", "22:30"))):
            return
        gap = float(self._cfg("proactive_gap_hours", 2.0))
        uid = pick_target(actives, gap)
        if uid is None:
            return
        if not self._proactive.can_send(uid):
            return
        if self._budget_exhausted():
            return
        provider = self.context.get_using_provider()
        if provider is None:
            return
        p = self.privates[uid]
        history = self._private_window(uid)
        digest = memory_digest(p.diary, history, scope="private")
        bot_name = self._persona_block().split("\n")[0][:12] or "AI"
        greeting = await build_greeting(provider, bot_name, str(uid), digest)
        if not greeting:
            return
        origin = f"qqmind:friend:{uid}"
        # 平台前缀按实际平台适配器生成——从已知私聊事件里取真实 origin 更稳
        origin = self._latest_private_origin(uid) or origin
        try:
            from astrbot.api.event import MessageChain
            await self.context.send_message(origin, MessageChain(
                [{"type": "Plain", "text": greeting}]))
        except Exception as e:
            logger.warning(f"[qqmind] proactive send to {uid} failed: {e!r}")
            return
        self._proactive.mark_sent(uid)
        p.append_assistant(greeting, src="private", uid=int(uid),
                           proactive=True)
        p.save_window()
        self._STATS["proactive_sent"] = self._STATS.get("proactive_sent", 0) + 1
        logger.info(f"[qqmind] proactive greeting -> {uid}: {greeting[:40]}")

    def _latest_private_origin(self, uid: int) -> str:
        """该用户最近一条私聊事件的 unified_msg_origin（主动发送用它）。"""
        rt = self._runtime.get(("private", uid)) or {}
        return str(rt.get("origin") or "")

    async def _proactive_reply_watch(self, uid: int) -> None:
        """用户在主动消息后 24h 内回话 → 清零未回应计数（挂在私聊入口）。"""
        if self._proactive.no_reply.get(uid):
            self._proactive.mark_replied(uid)

