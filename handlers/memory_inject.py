"""记忆注入与落盘链路（C3 解耦自 main.py，v0.3.9）。

inject_memory（挂点 2）/ persist_round（挂点 3）及回捞/思考切换/私聊压缩的
宿主，以 mixin 形式被 main.QQMindPlugin 继承。装饰器壳留在 main.py
（AstrBot 按 handler.__module__ == 插件主模块过滤注册表）。

本模块边界即未来散装包 qqmind-diary + qqmind-persona 的边界。
"""
from __future__ import annotations

import time

from astrbot.api import logger
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

from ..core import (PrivateSession, est_tokens, parse_time_query,
                    profile_to_prompt, thinking_model)
from ..main_helpers import _is_detail_query, _quote_reply_prefix, _safe_int_id

def _m_re():
    """运行时取 main 模块的回捞正则（避免 handlers<->main 循环 import：
    模块加载期 main 尚在初始化，方法调用期必已完成）。"""
    from .. import main as _m
    return (_m._DIARY_CMD_RE, _m._EXPAND_CMD_RE, _m._GROUP_DIARY_CMD_RE,
            _m._MEMORY_Q_RE, _m._CROSS_DIARY_CMD_RE)
class MemoryInjectMixin:
    async def inject_memory_impl(self, event: AstrMessageEvent, req: ProviderRequest):
        uid = _safe_int_id(event.get_sender_id())
        if not uid:
            return
        if event.is_private_chat():
            p = self._private(uid)
            q = (event.message_str or "").strip()
            profile = self._private_system_prompt(p, q, uid=uid)
            # v0.3.2：记忆类提问先本地预检索（模型常忘记输出 <查日记> 标记）
            if q and _m_re()[3].search(q):
                try:
                    found = p.diary.search(q, n=3)
                except Exception as e:
                    found = ""
                    logger.debug(f"[qqmind] private pre-recall failed: {e!r}")
                if found and "没有找到日记" not in found:
                    profile = (profile + "\n【用户可能在问过去的事；下面是从日记"
                               "预检索到的结果，够答就直接答，不够再输出 "
                               "<查日记\"关键词\">】\n" + found).strip()
            # v0.3.7 市场整改：动态段（人设+日记预检索+窗口摘要）走官方
            # extra_user_content_parts（.mark_as_temp 只在本轮生效不落历史），
            # stable 人设骨架留 system_prompt；contexts 不再整体替换——
            # 官方会话历史照常携带，压缩预算照旧在本插件窗口上跑。
            if profile:
                req.extra_user_content_parts.append(
                    TextPart(text=profile).mark_as_temp())
            return
        gid = _safe_int_id(event.get_group_id())
        if not gid:
            return
        # 白名单隔离（v0.3.6 市场整改补漏）：allowed_groups 只在 on_message
        # 拦消息入口还不够——其他插件对白名单外群发起的 LLM 请求也会进本钩子，
        # 记忆/人设同样不许注入这些群。
        allowed = self._cfg("allowed_groups") or []
        if allowed and str(gid) not in [str(g) for g in allowed]:
            return
        gnick = (event.get_sender_name() or "").strip() or str(uid)
        # 原生管线的用户消息也可能带引用——主模型必须看得见被引用的那句
        gtext = event.message_str or ""
        quote = _quote_reply_prefix(event.get_messages() or [])
        if quote and quote not in gtext:
            gtext = f"{quote}{gtext}".strip()
        profile = self._group_system_prompt(gid, gtext, uid=uid, nick=gnick)
        # 引用过的原文进了 prompt，但 message_str 不动（AstrBot 自身管线兼容）；
        # 用户消息本体带引用前缀，模型才知道"他引用着那句话在说这个"
        if quote:
            req.prompt = f"{quote}{(req.prompt or '').strip()}".strip()
        # 好感度备注/作息（v0.2.8）：群聊回复对当前说话人生效——每轮变化，
        # v0.3.7 改走官方 extra_user_content_parts（不进 system_prompt 护缓存）
        if self._cfg("affinity_enabled", False):
            aff_line = profile_to_prompt(self._affection.get(uid))
            if aff_line:
                req.extra_user_content_parts.append(
                    TextPart(text=aff_line).mark_as_temp())
        if profile:
            req.system_prompt = (profile + "\n"
                                 + (req.system_prompt or "")).strip()
        req.contexts = self._group_contexts(gid, uid)
        self._apply_think_model(event, req)

    def _apply_think_model(self, event, req: ProviderRequest) -> None:
        """思考模式由判读控制（老板 2026-10-29 定案）。

        判读判「是|思考」的当轮：主回复切到思考模型（think_model 配置，
        缺省用 thinking_model() 从主模型名推导 nothink→thinking 别名）；
        判「是|直接」或 @ 路径未判读的：保持主模型不动。
        """
        if not getattr(event, "qqmind_think", False):
            return
        think_model = str(self._cfg("think_model", "") or "").strip()
        if not think_model:
            think_model = thinking_model(
                getattr(self.context.get_using_provider(), "get_model", lambda: "")())
        if think_model:
            req.model = think_model
            logger.debug(f"[qqmind] think round -> model={think_model}")

    def _diary_headers(self, p: PrivateSession) -> str:
        n = int(self._cfg("diary_headers_in_prompt", 10))
        # v0.4.0 合并记忆：私聊视角能看到全部篇目（群聊日记 + 私聊日记），
        # 群聊视角反过来只给纯群聊篇目（见 main._group_system_prompt）
        headers = p.diary.headers_for_prompt(n, scope="any")
        if not headers:
            return ""
        return (f"你有关于这位用户的日记（时间 标题，想细看就输出 <查日记\"关键词或时间\">）：\n"
                f"{headers}")

    def _group_glance(self, p: PrivateSession) -> str:
        """私聊 prompt 里的「群里最近发生了什么」速览（v0.4.0）。

        只给最近几条群消息的摘要行，不含私聊内容；条数由
        `private_group_glance_msgs` 控制（0 = 关闭）。
        """
        n = int(self._cfg("private_group_glance_msgs", 6))
        if n <= 0:
            return ""
        try:
            lines = p.recent_group_lines(n)
        except Exception as e:      # noqa: BLE001
            logger.debug(f"[qqmind] group glance failed: {e!r}")
            return ""
        if not lines:
            return ""
        return ("【群里最近在聊（供你了解语境，私聊里不用主动汇报）】\n"
                + "\n".join(lines))

    def _private_system_prompt(self, p: PrivateSession, query: str = "",
                               uid: int = 0) -> str:
        """私聊 system prompt 片段。

        uid 必须传（v0.4.0）：记忆合并后 `p` 是所有人共用的一个会话，
        `p.uid` 恒为 0，画像与好感度都得靠 event 的说话人 uid 定位。
        """
        uid = int(uid or p.uid or 0)
        parts = []
        persona = self._persona_block()
        if persona:
            parts.append(persona)
        profile_line = p.diary.profile_prompt(uid)
        if profile_line:
            parts.append(profile_line)
        # 好感度备注/作息（v0.2.8）：只对开启者注入
        if self._cfg("affinity_enabled", False):
            aff_line = profile_to_prompt(self._affection.get(uid))
            if aff_line:
                parts.append(aff_line)
        background = self._persona_background(query)
        if background:
            parts.append(background)
        headers = self._diary_headers(p)
        if headers:
            parts.append(headers)
        glance = self._group_glance(p)
        if glance:
            parts.append(glance)
        return "\n".join(parts)

    async def _recall_reply(self, reply: str, scope: str, uid: int,
                            gid: int = 0) -> str:
        """统一回捞：处理 <查日记>/<查群日记>，查库后带结果重答一轮。

        scope: "private" | "group"。原生管线（on_llm_response）与手动路径
        （攒批/随机读）共用，保证两条回复路径能力对等。
        返回清洗后的最终文本（不再含任何回捞标记）。
        """
        (_DIARY_CMD_RE, _EXPAND_CMD_RE, _GROUP_DIARY_CMD_RE,
         _MEMORY_Q_RE, _CROSS_DIARY_CMD_RE) = _m_re()
        cmd_re = _DIARY_CMD_RE if scope == "private" else _GROUP_DIARY_CMD_RE
        provider = self.context.get_using_provider()
        # 跨群寻回（v0.2.7）：<查别群日记"群号:关键词">——只有共享开关开着才响应
        # v0.3.0 公共记忆：不再有「查别群日记」——日记本就全群共用，
        # 直接 <查群日记> 检索整个公共库。旧标记剜掉防模型复读。
        if _CROSS_DIARY_CMD_RE.search(reply):
            reply = _CROSS_DIARY_CMD_RE.sub("", reply).strip() or reply
        m = cmd_re.search(reply)
        if not m:
            return reply
        query = m.group(1)
        if scope == "private":
            p = self._private(uid)
            # 私聊视角 = 全库（群聊日记也能查，主人自己的东西不设限）
            found = (p.diary.by_time(query, scope="any") if parse_time_query(query)
                     else p.diary.search(query, scope="any"))
            system = self._private_system_prompt(p)
            contexts = p.context(scope="private")
        else:
            store = self._group_diary()   # 统一库（v0.4.0）
            # 群聊视角 = 只有纯群聊篇目：含私聊内容的篇目（gid=0）不进群检索
            found = (store.by_time(query, scope="group") if parse_time_query(query)
                     else store.search(query, scope="group"))
            system = self._group_system_prompt(gid)
            contexts = self._group_contexts(gid, uid)
        if provider is None:
            return reply.replace(m.group(0), "").strip() or \
                "我翻了一下日记，但没找到合适的答案。"
        return await self._recall_answer(
            reply, query, found, system, contexts, provider,
            scope, uid, gid, gid, strip_re=cmd_re)

    async def _recall_answer(self, reply: str, query: str, found: str,
                             system: str, contexts: list, provider,
                             scope: str, uid: int, gid: int, diary_gid: int,
                             strip_re) -> str:
        """寻回后的重答（本群/跨群/私聊共用）：三层展开 + 记账 + 清洗标记。"""
        (_DIARY_CMD_RE, _EXPAND_CMD_RE, _GROUP_DIARY_CMD_RE,
         _MEMORY_Q_RE, _CROSS_DIARY_CMD_RE) = _m_re()
        # 三层日记（v0.2.5）：结果行带 ▷ 的篇目存有原话全文；模型想看细节
        # 就回 <看日记"#id">（最多两轮，防套娃烧 token）
        if scope == "private":
            expand_hint = ("结果里带 ▷ 的篇目存有当天的原话；"
                           "只有需要引用原话才能回答时，"
                           "才输出 <看日记\"#编号\"> 展开一篇（最多展开一篇）。"
                           "概要够答就直接答。")
        else:
            expand_hint = ("结果里带 ▷ 的篇目存有当时的群聊原话；"
                           "只有需要引用原话才能回答时，"
                           "才输出 <看日记\"#编号\"> 展开一篇（最多展开一篇）。"
                           "概要够答就直接答。")
        try:
            out = await provider.text_chat(
                prompt=(f"（你刚才想查记忆：{query}）\n查询结果：\n{found}\n"
                        + expand_hint
                        + "\n请根据查询结果，用自然的语气回答对方。"),
                session_id=None, system_prompt=system,
                contexts=contexts, image_urls=[])
            new_reply = (out.completion_text or "").strip()
            expand_m = _EXPAND_CMD_RE.search(new_reply)
            if expand_m:
                # 第三层：模型点名要看某篇全文——单篇展开后重答一次
                eid = int(expand_m.group(1))
                if scope == "private":
                    detail = self._private(uid).diary.expand(eid)
                else:
                    # 群聊视角不许展开含私聊内容的篇目（v0.4.0 隐私边界）
                    detail = self._group_diary().expand(eid, allow_private=False)
                # 证据链（v0.3.0）：命中日记的 evidence 消息 id →
                # 公共池原文 ±3 条一起给，概要之外的对话细节当场还原
                raw_ctx = ""
                if scope == "group":
                    raw_ctx = self._evidence_context(eid)
                    if raw_ctx:
                        detail = (detail + "\n\n（当时的上下文原文：\n"
                                  + raw_ctx + "）")
                out2 = await provider.text_chat(
                    prompt=(f"（你刚才想查记忆：{query}）\n"
                            f"查询结果（概要）：\n{found}\n"
                            f"你选中展开的第 {eid} 篇全文：\n{detail}\n"
                            "请结合这些信息，用自然的语气回答对方。"),
                    session_id=None, system_prompt=system,
                    contexts=contexts, image_urls=[])
                new_reply = (out2.completion_text or "").strip()
                self._count_usage(found + detail, new_reply, resp=out2,
                                  gid=int(gid) if scope == "group" else 0)
                clean = _EXPAND_CMD_RE.sub("", new_reply)
                clean = _DIARY_CMD_RE.sub("", clean)
                clean = _GROUP_DIARY_CMD_RE.sub("", clean)
                clean = strip_re.sub("", clean).strip()
                logger.info(f"[qqmind] diary recall '{query}' expanded #{eid}")
                return clean or new_reply
            if new_reply:
                self._count_usage(found, new_reply, resp=out,
                                  gid=int(gid) if scope == "group" else 0)
                clean = _DIARY_CMD_RE.sub("", new_reply)
                clean = _GROUP_DIARY_CMD_RE.sub("", clean)
                clean = strip_re.sub("", clean).strip()
                logger.info(f"[qqmind] diary recall '{query}' answered")
                return clean or new_reply
        except Exception as e:
            logger.warning(f"[qqmind] diary recall failed: {e!r}")
        stripped = reply.replace(strip_re.search(reply).group(0)
                                 if strip_re.search(reply) else "",
                                 "").strip()
        return stripped or "我翻了一下日记，但没找到合适的答案。"

    async def _resolve_recall(self, event: AstrMessageEvent, resp,
                              reply: str) -> str:
        """原生管线挂点：解析回捞标记并写回 resp.completion_text。"""
        uid = _safe_int_id(event.get_sender_id())
        scope = "private" if event.is_private_chat() else "group"
        gid = _safe_int_id(event.get_group_id())
        new_reply = await self._recall_reply(reply, scope, uid, gid)
        if scope == "group":
            if not gid:
                return reply
            # 群聊硬截断（v0.2.4）：AstrBot 渲染按 \n 分段发送，提示词压不住的
            # 长回复在这里按字符预算兜底（私有聊不限——日记问答允许展开说）。
            # v0.2.10：讲题/教程场景放宽上限——80 一刀切曾把数学推导腰斩。
            new_reply = self._clamp_group_reply(
                new_reply,
                detailed=_is_detail_query(event.message_str or ""))
        if new_reply != reply:
            resp.completion_text = new_reply
        return new_reply

    def _clamp_group_reply(self, reply: str, detailed: bool = False) -> str:
        """群聊回复字符预算：按行凑，超预算截断收尾。

        detailed=True（对方在讲题/求教程）时上限放宽到 group_reply_detail_chars：
        允许把步骤说完，但仍比私聊收紧，守群聊防刷屏底线。
        """
        limit = int(self._cfg("group_reply_max_chars", 80))
        if detailed:
            limit = max(limit, int(self._cfg("group_reply_detail_chars", 240)))
        if limit <= 0 or len(reply) <= limit:
            return reply
        lines = [l for l in reply.split("\n") if l.strip()]
        kept: list[str] = []
        used = 0
        for l in lines:
            room = limit - used
            if room <= 0:
                break
            if len(l) <= room:
                kept.append(l)
                used += len(l)
            else:
                kept.append(l[:room])
                used = limit
                break
        out = "\n".join(kept).rstrip("，、；：") + ("…" if used >= limit else "")
        logger.info(f"[qqmind] group reply clamped: {len(reply)} -> {len(out)} chars")
        return out

    async def persist_round_impl(self, event: AstrMessageEvent, resp):
        uid = _safe_int_id(event.get_sender_id())
        reply = (getattr(resp, "completion_text", "") or "").strip()
        if not uid or not reply:
            return
        is_private = event.is_private_chat()
        gid = 0 if is_private else _safe_int_id(event.get_group_id())
        if not is_private and not gid:
            return
        self._real_usage(resp, "reply")
        self._count_usage(event.message_str or "", reply, resp=resp,
                          gid=gid)
        reply = await self._resolve_recall(event, resp, reply)
        if not reply:
            return
        self._stat("replied", gid if not is_private else None)
        if self._cfg("affinity_enabled", False):
            self._affection.add(uid, 1)
        text = (event.message_str or "").strip()
        if event.is_private_chat():
            p = self._private(uid)
            cap = int(self._cfg("max_msg_tokens", 100)) * 4
            async with self._hist_lock:
                if text:
                    p.append_user_message(text, cap, src="private", uid=uid)
                p.append_assistant(reply, src="private", uid=uid)
                p.save_window()                        # v0.3.2：重启不失忆
        else:
            gid = _safe_int_id(event.get_group_id())
            key = (gid, uid)
            nick = (event.get_sender_name() or "").strip() or str(uid)
            cap = int(self._cfg("max_msg_tokens", 100))
            user_text = text if (text and est_tokens(text) <= cap) else ""
            async with self._hist_lock:
                self._persist_group(gid, uid, [(uid, nick, user_text)], reply)
            # 机器人自己的回复也进群缓冲：轻量档语境全靠它，
            # 判读提示词也承诺「含 AI 助手的发言」——缺了会重复回答/接不上话
            self._note_group(gid, self._bot_names.get(gid) or "AI", reply)
            s = self.sessions.get(key)
            if s:
                s.rounds_left = int(self._cfg("followup_rounds", 3))
                s.touch()
        # 统一稳压（v0.4.0）：私聊/群聊共用一条窗口，超预算就在后台削+压
        mem = self._session()
        if mem.over_budget() and not mem.compressing:
            self._spawn(self._compress_memory(mem))

    async def _compress_private(self, p: PrivateSession):
        """兼容旧名：统一压缩入口（v0.4.0 起私聊/群聊共用一个窗口）。"""
        await self._compress_memory(p)

