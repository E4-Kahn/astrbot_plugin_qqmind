"""聊天指令与 LLM 工具（C4b 解耦自 main.py，v0.3.9）。

/qqmind /账单 /日记 /群日记 + get_usage_stats 工具的实现体。
装饰器壳留在 main.py（AstrBot 按 handler.__module__ == 插件主模块过滤）。
指令是用户交互面，跟随框架主体，不单独散装。
"""
from __future__ import annotations

from astrbot.api.event import AstrMessageEvent

from ..core import normalize_preset, parse_time_query
from ..main_helpers import _safe_int_id


def _plugin_version() -> str:
    """运行时取 main.PLUGIN_VERSION（防 handlers<->main 循环 import）。"""
    from .. import main as _m
    return _m.PLUGIN_VERSION


class CommandsMixin:
    # ---------- 指令 ----------
    async def cmd_qqmind_impl(self, event: AstrMessageEvent):
        """查看插件状态与记忆账本。带「诊断」参数时输出排障摘要。"""
        uid = _safe_int_id(event.get_sender_id())
        arg = (event.message_str or "").strip()
        if arg in ("诊断", "diag", "诊断包"):
            d = self._diagnostics_snapshot()
            st = d["runtime"]["stats"]
            degraded = d["runtime"]["degraded_groups"]
            yield event.plain_result(
                f"qqmind 诊断 v{_plugin_version()}\n"
                f"套餐：{d['config'].get('group_preset')} | 读取："
                f"{d['config'].get('read_mode')} | 判读范围："
                f"{'全群' if d['config'].get('judge_scope') else '监控期'}\n"
                f"判读模型：{d['config'].get('judge_model') or '主模型'}"
                f" | 思考模型：{d['config'].get('think_model') or '未配'}\n"
                f"计数：判读{st['judged']} 拦截{st['blocked']} "
                f"回复{st['replied']} 思考轮{st['judge_think']} "
                f"限流弃{st['dropped_rate']} 预算弃{st['dropped_budget']}\n"
                f"降级群：{('、'.join(map(str, degraded)) if degraded else '无')}"
                f" | 判读失败计数：{d['runtime']['judge_fail_counts'] or '无'}\n"
                f"今日额度：{d['runtime']['budget']['used']}/"
                f"{d['runtime']['budget']['limit'] or '不限'}\n"
                f"最近日志：{len(d['logs'])} 条在环形缓冲"
                f"（完整诊断包请从 WebUI 控制台导出）")
            return
        n_mem = len(self.memories)
        n_priv = len(self.privates)
        diaries = self.privates[uid].diary.count() if uid in self.privates else 0
        b = self._budget().snapshot()
        budget_line = ("今日 token：不限" if b["limit"] <= 0
                       else f"今日 token：{b['used']}/{b['limit']}")
        yield event.plain_result(
            f"群聊记忆体 v{_plugin_version()}\n"
            f"群聊套餐：{normalize_preset(self._cfg('group_preset', 'light'))}\n"
            f"群聊记忆：{n_mem} 个群\n"
            f"私聊会话：{n_priv} 人\n"
            f"你的日记：{diaries} 篇\n"
            f"{budget_line}\n"
            f"判读门：{'开' if self._cfg('reply_judge_enabled', True) else '关'}"
            f"\n排障：/qqmind 诊断")

    async def tool_usage_stats_impl(self, event: AstrMessageEvent):
        """查询机器人今天的运行账单：读了多少条群消息、回复了多少条、共消耗多少 token。

        当群友问「今天读了多少」「回了几条」「用了多少 token」「省了多少」
        这类问题时先调用本工具，再用结果自然地回答。
        """
        if event.is_private_chat():
            gid = None
            scope_line = "全网总量"
        else:
            gid = _safe_int_id(event.get_group_id())
            scope_line = "本群"
        s = self._stats_for(gid)
        b = self._budget().snapshot()
        lines = (f"今日（{b['day']}）{scope_line}账单：读 {s['judged']} 条，"
                 f"回 {s['replied']} 条。")
        if gid is None:
            lines += f"全网共消耗 {b['used']} token。"
        else:
            lines += (f"{scope_line}消耗 {b.get('per_group', {}).get(gid, 0)}"
                      f" token（全网 {b['used']}）。")
        return lines

    async def cmd_bill_impl(self, event: AstrMessageEvent):
        """今日账单：群里=本群数据+全网总量；私聊=全网。"""
        s = self._STATS
        b = self._budget().snapshot()
        if event.is_private_chat():
            yield event.plain_result(
                f"🐋 今日账单（{b['day']}·全网）：读 {s['judged']} 条"
                f" · 回 {s['replied']} 条 · 用了 {b['used']} token")
            return
        gid = _safe_int_id(event.get_group_id())
        if not gid:
            return
        g = self._stats_for(gid)
        g_tok = b.get("per_group", {}).get(gid, 0)
        yield event.plain_result(
            f"🐋 本群今日账单（{b['day']}）：读 {g['judged']} 条"
            f" · 回 {g['replied']} 条 · 消耗 {g_tok} token\n"
            f"（全网：读 {s['judged']} · 回 {s['replied']}"
            f" · 用了 {b['used']} token）")

    async def cmd_diary_impl(self, event: AstrMessageEvent):
        """翻日记：/日记 最近 或 /日记 关键词 或 /日记 昨天 或 /日记 #3 看全文"""
        uid = _safe_int_id(event.get_sender_id())
        if not event.is_private_chat():
            yield event.plain_result("日记是私聊的记忆，私聊我再看哦")
            return
        if not uid:
            return
        p = self._private(uid)
        arg = (event.message_str or "").strip()
        if arg.startswith("#") and arg[1:].isdigit():
            yield event.plain_result(p.diary.expand(int(arg[1:])))
            return
        if not arg or arg in ("最近", "列表"):
            out = p.diary.recent(5, scope="any")
        elif parse_time_query(arg):
            out = p.diary.by_time(arg, scope="any")
        else:
            out = p.diary.search(arg, scope="any")
        out += "\n（看某篇原话：/日记 #编号）"
        yield event.plain_result(out)

    async def cmd_group_diary_impl(self, event: AstrMessageEvent):
        """翻本群日记（三层递进）：/群日记 最近|关键词|时间 或 /群日记 #3 看原话"""
        if event.is_private_chat():
            yield event.plain_result("群日记要在群里看哦，私聊找我看 /日记～")
            return
        gid = _safe_int_id(event.get_group_id())
        if not gid:
            return
        allowed = self._cfg("allowed_groups") or []
        if allowed and str(gid) not in [str(g) for g in allowed]:
            yield event.plain_result("这个群没有开通我的记忆功能～")
            return
        store = self._group_diary()
        arg = (event.message_str or "").strip()
        if arg.startswith("#") and arg[1:].isdigit():
            yield event.plain_result(
                store.expand(int(arg[1:]), allow_private=False))
            return
        if not arg or arg in ("最近", "列表"):
            out = store.recent(5, scope="group")
        elif parse_time_query(arg):
            out = store.by_time(arg, scope="group")
        else:
            out = store.search(arg, scope="group")
        out += "\n（▷=存有原话，看原话发：/群日记 #编号；日记全群共用，"
        out += "含私聊内容的篇目群里只给概要）"
        yield event.plain_result(out)
