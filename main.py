"""Token消耗高级优化（QQ机器人套件）—— AstrBot 插件（qqmind）。

三个挂点（全部走 AstrBot 官方扩展机制，不改 AstrBot 源码）：
1. @filter.event_message_type(ALL) 事件监听：浅层判读门。
   判否 event.should_call_llm(True)，只拦主模型、不终结事件链
   （其他插件与命令照常处理）；判是才交给 AstrBot 走 LLM。
2. @on_llm_request 记忆注入：替换 req.contexts 为我们自己的逐字窗口/私聊预算窗口，
   追加画像、日记标题行、群语境到 system_prompt。
3. @on_llm_response 落盘：把本轮对话写回窗口；预算溢出时后台压缩成日记。

核心逻辑（判读三态/日记库/预算窗口/回捞）在 core/ 包里，与 AstrBot 零耦合，
单测可脱离 AstrBot 跑。
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
import re
import shutil
import time

from astrbot.api import logger, star
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import on_llm_request, on_llm_response
from astrbot.api.message_components import At, Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.web import request as _web_request
from astrbot.core.agent.message import TextPart   # 官方推荐的动态上下文部件

from .core import (JUDGE_PROMPT, MAX_JUDGED_PER_ROUND, ChapterStore,
                   AffectionStore, CandidatePool, DailyBudget, GroupBuffer,
                   GroupMemory, ImportJob, MonitorSession, PrivateSession,
                   RateLimiter, batch_should_flush,
                   detect_wake, est_tokens, thinking_model,
                   keyword_wake_enabled, normalize_preset,
                   parse_judge_verdict, parse_time_query,
                   preset_memory_mode, resolve_read_mode, split_bubbles,
                   VisionBudget, extract_image_urls)
from .core.importer import normalize_messages, run_import
from .core.vision import describe_image
from .core.diagnostics import build_diagnostics
from .core import diagnostics
from .core import proactive
from .core.proactive import (build_greeting, in_quiet_hours, memory_digest,
                             pick_target)
from .core.affection import in_schedule, profile_to_prompt
from .core.private import complete_text, complete_text_resp

PLUGIN_VERSION = "0.3.8"
PLUGIN_NAME = "astrbot_plugin_qqmind"

# 数据根：官方规定 data/plugin_data/<插件名>（市场规范 2026-11-01 整改）。
# 老版本数据落在 data/ 根目录，首次运行整体搬进来（_migrate_legacy_data）。
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

# C2 解耦：基础 helper 移至 main_helpers（旧引用零改动，re-export）
from .main_helpers import (_DETAIL_QUERY_RE, _is_detail_query,  # noqa: F401
                           _is_poke_self, _quote_reply_prefix, _safe_int_id)
from .handlers.judge_gate import JudgeGateMixin
from .handlers.memory_inject import MemoryInjectMixin
from .handlers.interactions import InteractionsMixin
from .handlers.commands import CommandsMixin

def _plugin_data_root() -> str:
    try:
        root = os.path.join(get_astrbot_data_path(), "plugin_data", PLUGIN_NAME)
    except Exception:
        root = os.path.join("data", "plugin_data", PLUGIN_NAME)
    os.makedirs(root, exist_ok=True)
    return root

_PLUGIN_VERSION_ = None  # 版本走 PLUGIN_VERSION
JUDGE_PROMPT_LEN = est_tokens(JUDGE_PROMPT)   # 判读 system 提示词的 token 数（账单口径）

_DIARY_CMD_RE = re.compile(r'<查日记"([^"]{1,60})">')
_GROUP_DIARY_CMD_RE = re.compile(r'<查群日记"([^"]{1,60})">')
_CROSS_DIARY_CMD_RE = re.compile(r'<查别群日记"(\d{5,12})[:：]([^"]{1,60})">')
_EXPAND_CMD_RE = re.compile(r'<看日记"#?(\d{1,4})">')

# 私聊记忆提问识别（v0.3.2）：命中时先本地检索日记再答，不指望模型主动
# 输出 <查日记"…">（实测它经常不动）。只在命中时增加少量 token。
_MEMORY_Q_RE = re.compile(
    r"还记得|记得吗|记得么|记不记得|上次|上回|之前|以前|昨天|前天|"
    r"上周|上个月|我们说|我说过|你答应|你上次|聊过|提过|那次")

# 活人模式默认规则（v0.2.13）：从 4 群 3481 条真人消息画像提炼，
# 提炼依据与数据详见 docs/活人模式画像.md；配置 persona_live_mode 清空即关闭。
_LIVE_MODE_DEFAULT = (
    "1. 短：一句话够就别两句，能 15 字说完别写 30 字。\n"
    "2. 裸收尾：默认句尾不带标点，句号只在郑重时用；感叹号基本不用。\n"
    "3. 想说的多就拆成两三条短消息连发（像真人边想边打），别一次塞满一段。\n"
    "4. 接话直接回应上一条，用「不是/什么/怎么/可以」这类口语词推进，"
    "禁止书面衔接词（关于/总的来说/首先其次）。\n"
    "5. 情绪用词不用标点堆：惊讶「？？」「啊」，嘲笑「乐」，无奈「绷」，"
    "认可「6」，玩梗「草」；波浪号少用。\n"
    "6. 不每句自称名字/人设称谓；人设只在被问或被调侃时自然带出。\n"
    "7. 允许省主语、话说一半、无伤大雅的错别字——活人信号；"
    "禁止语法完美的长难句。\n"
    "8. 别人开玩笑就顺着接一句短的，别解释笑点；被戳到雷点才正经。")

# 详细场景识别（v0.2.10）：对方在要"讲清楚"的内容时放宽群聊截断。
# 命中面=求做法/求原因/求解释/讲题/求证/纠错，不含纯闲聊。
# 控制台可读写的配置键（与 _conf_schema.json 对齐）
_CONFIG_KEYS = (
    "persona_text", "persona_shallow", "persona_deep", "persona_pack",
    "persona_live_mode",
    "persona_packs_dir", "persona_headers_in_prompt", "persona_recall_top",
    "private_budget_tokens", "diary_headers_in_prompt", "diaries_dir",
    "private_group_glance_msgs",
    "group_preset", "wake_keywords", "group_buffer_size",
    "read_mode", "random_min_seconds", "random_max_seconds", "roam_enabled",
    "debounce_seconds", "batch_max_messages",
    "reply_judge_enabled", "judge_scope", "reply_rate_per_min",
    "max_judged_per_min", "max_judged_per_round",
    "judge_model", "think_model",
    "judge_no_think", "compress_no_think",
    "judge_fail_threshold", "bot_qq_ids", "allowed_groups",
    "group_flat_messages", "group_shallow_messages", "group_context_messages",
    "reply_context_messages", "judge_context_messages",
    "memory_layering", "precise_memory", "precise_budget_tokens",
    "followup_rounds", "max_msg_tokens",
    "group_reply_max_chars", "group_reply_detail_chars",
    "pool_trigger_msgs", "pool_max_msgs",
    "memory_prompt_budget",
    "vision_enabled", "vision_budget", "vision_model",
    "proactive_friends", "proactive_start", "proactive_end",
    "proactive_gap_hours", "proactive_per_day", "proactive_no_reply_limit",
    "affinity_enabled", "affinity_base", "affinity_threshold",
    "affinity_daily_gain",
    "daily_token_budget", "quota_message", "quota_notices_per_day",
    "bill_est_reply_tokens",
)


def _short_title(summary: str, limit: int = 18) -> str:
    """旧日记没有 title 时的显示兜底：摘要前 limit 字。"""
    s = (summary or "").strip().replace("\n", " ")
    return s[:limit] + ("…" if len(s) > limit else "")


def _strip_own_prefix(nick: str, content: str) -> str:
    """平铺窗口存的 content 已带「nick：」前缀，入候选池时剥掉——
    池压缩时会重新拼「nick：text」，不剥就变成「甲：甲：内容」双前缀。"""
    c = (content or "").strip()
    n = (nick or "").strip()
    if n and c.startswith(f"{n}："):
        c = c[len(n) + 1:].lstrip()
    elif n and c.startswith(f"{n}:"):
        c = c[len(n) + 1:].lstrip()
    return c


class QQMindPlugin(JudgeGateMixin, MemoryInjectMixin, InteractionsMixin,
                   CommandsMixin, star.Star):
    """Token消耗高级优化（QQ机器人套件）：判读门 + 深层记忆 + 私聊日记。"""

    def __init__(self, context: star.Context, config=None):
        super().__init__(context)
        self.plugin_config = config    # AstrBotConfig（dict 子类），带 save_config()
        self.sessions: dict[tuple[int, int], MonitorSession] = {}
        self.privates: dict[int, PrivateSession] = {}
        self.memories: dict[int, GroupMemory] = {}
        self._pub_pool: CandidatePool | None = None   # 公共候选池（v0.3.0）
        self._pub_diary = None                        # 统一记忆库（v0.4.0，memory.db）
        self._mem_session = None                      # 统一记忆窗口（v0.4.0）
        self._win_save_ts = 0.0                       # 窗口落盘节流（v0.4.0）
        self._flushing: dict[int, bool] = {}
        self.group_buffer = GroupBuffer(
            max_len=int(self._cfg("group_buffer_size", 60)),
            max_item_tokens=int(self._cfg("max_msg_tokens", 100)))
        self.rate_limiter = RateLimiter(int(self._cfg("reply_rate_per_min", 6)))
        self.judge_limiter = RateLimiter(int(self._cfg("max_judged_per_min", 30)))
        self._runtime: dict[int, dict] = {}
        self._bot_names: dict[int, str] = {}
        self._poke_last: dict[tuple[int, int], float] = {}   # (gid,uid)->上次反应
        self._judge_fail: dict[int, int] = {}
        self._degraded: set[int] = set()
        self._chapters: dict[str, ChapterStore] = {}
        self._import_jobs: dict[str, ImportJob] = {}   # 导入任务表（控制台轮询进度）
        self._data_root = _plugin_data_root()
        self._migrate_legacy_data()   # v0.3.6 一次性：老数据从 data/ 根搬入 plugin_data
        usage_dir = self._abs_dir("usage_dir", "qqmind_usage")
        self.budget = DailyBudget(os.path.join(usage_dir, "daily.json"), 0)
        self._hist_lock = asyncio.Lock()
        self._aux_lock = asyncio.Lock()   # 非思考辅助 provider 懒创建锁
        self._ring = diagnostics.attach()   # loguru sink（v0.3.7：不 import logging）
        self._vision = VisionBudget(int(self._cfg("vision_budget", 50)))
        self._proactive = proactive.ProactiveState(
            int(self._cfg("proactive_per_day", 1)))
        self._proactive_task: asyncio.Task | None = None
        self._pool_sweep_task: asyncio.Task | None = None
        self._bg_tasks: set[asyncio.Task] = set()   # create_task 强引用（防长任务被 GC 中断）
        self._last_platform = ""      # 最近一次事件的平台前缀（诊断包用）
        # 巡检随插件启动（不是懒启动）：池对象懒加载+安静群没人发消息，
        # 懒启动会让磁盘上的存量池永远等不到巡检；同步构造（无事件循环）
        # 时退回第一条消息再启动
        try:
            self._ensure_pool_sweep()
        except RuntimeError:
            pass
        self._STATS_G: dict[int, dict] = {}   # 分群计数子账（内存态，与 _STATS 同口径）
        self._affection = AffectionStore(
            self._private,
            base=int(self._cfg("affinity_base", 50)),
            daily_gain_cap=int(self._cfg("affinity_daily_gain", 10)))
        self._register_page_apis()

    # ---------- 配置与存储路径 ----------
    # 六个数据目录键（迁移/白名单共用）；default=老版本相对目录名
    _DIR_KEYS = (
        ("usage_dir", "qqmind_usage"),
        ("diaries_dir", "qqmind_diaries"),
        ("group_memory_dir", "qqmind_group_memory"),
        ("group_diaries_dir", "qqmind_group_diaries"),
        ("pool_dir", "qqmind_pool"),
        ("persona_packs_dir", "qqmind_persona_packs"),
    )

    def _migrate_legacy_data(self) -> None:
        """v0.3.6 一次性迁移（市场规范整改）：v0.3.5 及以前相对目录锚在
        data/ 根，官方规范要求数据落 data/plugin_data/<插件名>/。
        首次启动按配置里实际目录名逐个搬入（含用户自定义名，尊重绝对路径）。
        目标目录已存在则逐文件补缺（半途失败可重入），源搬空才删壳；
        单目录失败只告警不炸启动。"""
        for key, default in self._DIR_KEYS:
            name = str(self._cfg(key, default))
            if os.path.isabs(name):
                continue
            src = os.path.join("data", name)
            dst = os.path.join(self._data_root, name)
            if not os.path.isdir(src):
                continue
            try:
                if not os.path.isdir(dst):
                    shutil.move(src, dst)
                else:
                    for fn in os.listdir(src):
                        s, d = os.path.join(src, fn), os.path.join(dst, fn)
                        if not os.path.exists(d):
                            shutil.move(s, d)
                    if not os.listdir(src):
                        os.rmdir(src)
                logger.info(f"[qqmind] 数据已迁入 plugin_data: data/{name} "
                            f"-> plugin_data/{PLUGIN_NAME}/{name}")
            except OSError as e:
                logger.warning(f"[qqmind] data migration failed for {name}: {e!r}")

    def _cfg(self, key: str, default=None):
        if self.plugin_config is not None:
            return self.plugin_config.get(key, default)
        return default

    def _abs_dir(self, key: str, default: str) -> str:
        """配置目录 → 绝对路径。v0.3.6 锚定官方 plugin_data 目录：
        data/plugin_data/astrbot_plugin_qqmind/<子目录>（市场规范整改）。
        相对路径语义从「相对 data/ 根」改为「相对本插件数据根」，老数据由
        _migrate_legacy_data 一次性搬入；用户显式配绝对路径则原样尊重。"""
        d = str(self._cfg(key, default))
        if os.path.isabs(d):
            os.makedirs(d, exist_ok=True)
            return d
        d = os.path.join(self._data_root, d)
        os.makedirs(d, exist_ok=True)
        return d

    # ---------- 存储 ----------
    def _group_mem(self, gid: int) -> GroupMemory:
        """群记忆对象（v0.4.0 起只承担两件事：候选池合批压缩 + 群日记命令）。

        平铺/精确窗口已并入统一 `_session()` 的公共时间线，`flat`/`precise`
        不再写入；保留类与文件是为了 `compress_pool_call/compress_pool_batch`
        和旧数据兼容（磁盘上的 `<gid>.json` 只读不写）。
        """
        gid = int(gid)
        gm = self.memories.get(gid)
        if gm is None:
            gm = GroupMemory(
                self._abs_dir("group_memory_dir", "qqmind_group_memory"), gid,
                flat_limit=int(self._cfg("group_flat_messages", 60)),
                shallow_limit=int(self._cfg("group_shallow_messages", 20)),
                precise_budget=int(self._cfg("precise_budget_tokens", 4000)),
                diary=self._memory_store())
            self.memories[gid] = gm
        return gm

    def _memory_store(self):
        """统一记忆库（v0.4.0）：私聊与群聊共用一个 SQLite（`memory.db`）。

        entries.gid == 0 表示含私聊内容（群聊侧只给概要、不给原话）；
        entries.gid >  0 表示纯群聊，记来源群号。
        首次创建时把 `diaries_dir/<uid>.db` 与 `group_diaries_dir/*.db`
        全部并进来，原库改名 `.migrated` 备份（meta 记已并清单，幂等可重入）。
        """
        if self._pub_diary is None:
            ddir = self._abs_dir("diaries_dir", "qqmind_diaries")
            from .core.diary import DiaryStore
            store = DiaryStore(ddir, 0, name="memory")
            try:
                n = self._migrate_memory_dbs(store)
                if n:
                    logger.info(f"[qqmind] memory merged into memory.db: "
                                f"{n} entries")
            except Exception as e:     # noqa: BLE001 迁移失败不许影响启动
                logger.warning(f"[qqmind] memory migration failed: {e!r}")
            self._pub_diary = store
        return self._pub_diary

    _MIGRATED = "migrated_memory_dbs"

    def _migrate_memory_dbs(self, store) -> int:
        """把旧库并进统一库：群库（<gid>.db / group_public.db）与私聊库（<uid>.db）。"""
        from .core.diary import DiaryStore
        ddir = Path(self._abs_dir("diaries_dir", "qqmind_diaries"))
        gdir = Path(self._abs_dir("group_diaries_dir", "qqmind_group_diaries"))
        done = set((store.get_meta(self._MIGRATED) or "").split(","))
        done.discard("")
        plan: list[tuple[Path, int, int]] = []     # (路径, 默认 gid, 归属 uid)
        for d in {str(ddir.resolve()), str(gdir.resolve())}:
            p = Path(d)
            if not p.is_dir():
                continue
            for f in sorted(p.glob("*.db")):
                if f.stem in ("memory", "group_public"):
                    if f.stem == "group_public":
                        plan.append((f, 0, 0))
                    continue
                if str(p.resolve()) == str(ddir.resolve()) and f.stem.isdigit():
                    continue          # 私聊库，下面按 uid 单独处理
                plan.append((f, int(f.stem) if f.stem.isdigit() else 0, 0))
        if ddir.is_dir():
            for f in sorted(ddir.glob("*.db")):
                if f.stem.isdigit():
                    plan.append((f, 0, int(f.stem)))
        merged = 0
        moved: list[str] = []
        for f, gid, owner in plan:
            key = str(f.resolve())
            if key in done:
                continue
            try:
                merged += DiaryStore.merge_db(f, store, default_gid=gid,
                                              owner_uid=owner)
                moved.append(key)
            except Exception as e:     # noqa: BLE001
                logger.warning(f"[qqmind] merge {f.name} failed: {e!r}")
        if moved:
            done.update(moved)
            store.set_meta(self._MIGRATED, ",".join(sorted(done)))
            for key in moved:
                src = Path(key)
                try:
                    os.replace(src, src.with_suffix(src.suffix + ".migrated"))
                except OSError:
                    pass
        return merged

    def _group_diary(self):
        """群日记库（兼容旧名）：v0.4.0 起就是统一记忆库 `_memory_store()`。"""
        return self._memory_store()

    def _pool(self) -> CandidatePool:
        """公共候选池（v0.3.0）：所有群共用，消息自带 gid。

        v0.3.1：阈值每次读取都同步到已建实例——控制台改完立即生效，不必重载。
        """
        trig = max(5, int(self._cfg("pool_trigger_msgs", 1000)))
        mx = max(trig + 100, int(self._cfg("pool_max_msgs", 3000)))
        if self._pub_pool is None:
            self._pub_pool = CandidatePool(
                self._abs_dir("pool_dir", "qqmind_pool"),
                trigger_msgs=trig, max_msgs=mx)
        else:
            self._pub_pool.trigger_msgs = trig
            self._pub_pool.max_msgs = mx
        return self._pub_pool

    # ---------- 辅助任务 provider（v0.3.3） ----------
    @staticmethod
    def _is_deepseek_provider(provider) -> bool:
        cfg = getattr(provider, "provider_config", {}) or {}
        base = str(cfg.get("api_base") or cfg.get("base_url") or "")
        model = str(cfg.get("model") or "")
        return "deepseek" in base.lower() or model.lower().startswith("deepseek")

    async def _aux_provider(self, kind: str):
        """判读/压缩用的 provider：DeepSeek V4 默认思考，但这两类任务不需要。

        - 判读只是三选一（是否对 AI 说话、是否要思考），思考链纯烧 output；
        - 压缩只是把原文整理成 JSON，非思考模式足够。
        做法：克隆当前 provider，在请求里强制 thinking=disabled；回复仍走
        原 provider，判「是|思考」的当轮照常思考。
        kind: "judge" | "compress"；对应配置 judge_no_think/compress_no_think。
        """
        base = self.context.get_using_provider()
        if base is None:
            return None
        enabled = bool(self._cfg(f"{kind}_no_think", True))
        if not enabled or not self._is_deepseek_provider(base):
            return base
        pm = getattr(self.context, "provider_manager", None)
        if pm is None:
            return base
        clone_id = "qqmind-nothink"
        inst_map = getattr(pm, "inst_map", {}) or {}
        existing = inst_map.get(clone_id)
        if existing is not None:
            return existing
        async with self._aux_lock:
            existing = (getattr(pm, "inst_map", {}) or {}).get(clone_id)
            if existing is not None:
                return existing
            cfg = dict(getattr(base, "provider_config", {}) or {})
            if not cfg:
                return base
            cfg["id"] = clone_id
            cfg["enable"] = True
            raw_extra = cfg.get("custom_extra_body")
            extra = dict(raw_extra) if isinstance(raw_extra, dict) else {}
            extra["thinking"] = {"type": "disabled"}
            cfg["custom_extra_body"] = extra
            try:
                lock = getattr(pm, "reload_lock", None)
                if lock is not None:
                    async with lock:
                        await pm.load_provider(cfg)
                else:
                    await pm.load_provider(cfg)
            except Exception as e:     # noqa: BLE001
                logger.warning(
                    f"[qqmind] non-thinking provider init failed: {e!r}")
                return base
            inst = (getattr(pm, "inst_map", {}) or {}).get(clone_id)
            if inst is None:
                return base
            logger.info("[qqmind] judge/compress -> non-thinking provider clone")
            return inst

    async def _flush_pool(self) -> bool:
        """池子攒满（或凌晨残余兜底）→ 一次模型调用拆成多篇话题日记。

        返回 True=本批成功交割（含模型确认“没有值得记的内容”的合法空结果）；
        False=调用/入库失败或已有压缩在跑，消息仍在池里。
        """
        pool = self._pool()
        if not pool.msgs or self._flushing.get(0):
            return False
        self._flushing[0] = True
        batch: list[dict] = []
        confirmed = False
        try:
            # v0.3.1：单批最多 trigger_msgs 条——停机/故障后池子可能涨到
            # max_msgs，整批塞进一次请求会撑爆上下文；剩下的留给下一批
            batch = pool.take_batch(pool.trigger_msgs)
            pool.save()
            provider = await self._aux_provider("compress")
            if provider is None:
                pool.put_back(batch)
                pool.save()
                return False
            # 前情提要：上一批结尾 5 条，跨批话题不断头（v0.3.0）
            prelude = pool.prelude()
            gm0 = self._group_mem(batch[0]["gid"] if batch else 0)
            est, entries, resp = await gm0.compress_pool_call(
                provider, batch, prelude=prelude)
            if entries is None:          # 调用失败 / 输出捞不出 JSON
                pool.put_back(batch)
                pool.save()
                pool.mark_failed()       # 指数退避（5→10→20→40→60min）
                logger.warning(f"[qqmind] pool flush failed: "
                               f"{len(batch)} msgs put back, "
                               f"backoff streak={pool._fail_streak}")
                return False
            if resp is not None:
                self._real_usage(resp, "compress")
            n = 0
            if entries:                  # [] = 这批没有值得记的内容，正常清池
                n = gm0.compress_pool_batch(batch, entries)
                # 原文归档（v0.3.0）：保留最近 10 批（旧版只留 1 批，证据链
                # 寿命太短）；meta 存 JSON 数组的数组
                try:
                    import json as _json
                    store = self._group_diary()
                    raw = store.get_meta("evidence_archive")
                    batches = _json.loads(raw) if raw else []
                    if not isinstance(batches, list):
                        batches = []
                    batches.append(batch)
                    batches = batches[-10:]
                    store.set_meta(
                        "evidence_archive",
                        _json.dumps(batches, ensure_ascii=False))
                except Exception as e:
                    logger.debug(f"[qqmind] evidence archive failed: {e!r}")
            pool.confirm_taken()   # 正式交割，删 pending，记前情提要
            confirmed = True
            # 分群记预算：按批内各群消息占比分摊 est
            try:
                from collections import Counter
                cnt = Counter(int(m.get("gid") or 0) for m in batch)
                total = sum(cnt.values()) or 1
                for g, c in cnt.items():
                    self._budget().add(int(est * c / total), gid=g)
            except Exception as e:
                logger.debug(f"[qqmind] pool budget attribution failed: {e!r}")
                try:
                    self._budget().add(int(est), gid=0)
                except Exception:
                    pass
            logger.info(f"[qqmind] pool flush: {len(batch)} msgs "
                        f"-> {n} diary entries, est {est} tok")
            return True
        except asyncio.CancelledError:
            # 卸载/取消：把本批放回池，别让 pending 在下次 take_batch 时被覆盖
            if not confirmed and batch:
                pool.put_back(batch)
                pool.save()
            raise
        except Exception as e:
            # v0.3.1：压缩成功后的入库/归档异常也必须回滚——否则 pending
            # 会被下一批 take_batch 覆盖，整批永久丢失（BUG-6 同类）
            logger.warning(f"[qqmind] pool flush crashed, batch put back: {e!r}")
            if not confirmed and batch:
                pool.put_back(batch)
                pool.save()
                pool.mark_failed()
            return False
        finally:
            self._flushing[0] = False

    def _evidence_context(self, entry_id: int, window: int = 3,
                          max_chars: int = 900) -> str:
        """证据链回查（v0.3.0 公共记忆）：日记 evidence 消息 id → 原文 ±window。

        原文来源：公共候选池（还在池里）→ evidence_archive（最近 10 批）。
        都取不到返回空串（日记自带原话选段兜底）。
        """
        store = self._group_diary()
        ev_ids = self._diary_evidence(store, entry_id)
        if not ev_ids:
            return ""
        pool = self._pool()
        archived_list = []
        try:
            import json as _json
            raw = store.get_meta("evidence_archive")
            if raw:
                data = _json.loads(raw)
                # v0.3.0 存 [[批1],[批2],...]；v0.2.x 存单批 [msg,...]——兼容
                if isinstance(data, list) and data and isinstance(data[0], list):
                    archived_list = data
                elif isinstance(data, list):
                    archived_list = [data]
        except Exception:
            archived_list = []
        archived: dict[int, dict] = {}
        for batch in archived_list:
            for m in batch:
                try:
                    archived[int(m["id"])] = m
                except Exception:
                    continue
        lines, used = [], 0
        seen = set()
        for mid in ev_ids:
            ctx = pool.context_around(mid, window)
            if not ctx and mid in archived:
                ids = sorted(archived)
                idx = ids.index(mid)
                ctx = [archived[i] for i in
                       ids[max(0, idx - window):idx + window + 1]]
            for m in ctx:
                if m["id"] in seen:
                    continue
                seen.add(m["id"])
                line = f"[{m['id']}] {m.get('nick') or '群友'}：{m.get('text', '')}"
                if used + len(line) > max_chars:
                    return "\n".join(lines)
                used += len(line)
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _diary_evidence(store, entry_id: int) -> list[int]:
        raw = store.get_meta(f"evidence_{int(entry_id)}")
        if not raw:
            return []
        try:
            import json as _json
            v = _json.loads(raw)
            return [int(x) for x in v if str(x).isdigit()][:10]
        except Exception:
            return []

    def _session(self) -> PrivateSession:
        """统一记忆窗口（v0.4.0）：私聊与群聊共用一条时间线、一起写日记。

        窗口条目自带 `src`（private/group）、`gid`、`uid`、`nick`，
        注入时按作用域切片（`context(scope=...)`），压缩时统一成一篇日记，
        含私聊内容的篇目标 gid=0（群聊侧只给概要、不给原话）。
        """
        budget = int(self._cfg("private_budget_tokens", 10000))
        if self._mem_session is None:
            self._mem_session = PrivateSession(self._memory_store(), budget=budget)
        else:
            self._mem_session.budget = budget     # 控制台改预算热生效
        return self._mem_session

    def _private(self, uid: int) -> PrivateSession:
        """兼容接口：老代码按 uid 取会话；合并后全部指向同一个 `_session()`。"""
        s = self._session()
        self.privates[int(uid)] = s
        self._ensure_proactive_loop()
        return s

    def _spawn(self, coro) -> asyncio.Task:
        """create_task 并持有强引用：事件循环只持弱引用，纯 fire-and-forget
        的长任务（导入/池压缩）有被 GC 中断的风险（官方文档明确要求持有）。"""
        t = asyncio.create_task(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)
        return t

    def _ensure_pool_sweep(self) -> None:
        """残余兜底（v0.3.0）：每天凌晨 04:30 把不足一批的池底压一次。

        老板定案「一天一处理残余」：不再 5 分钟扫一次。压完即止，
        池子平时只受「攒够 pool_trigger_msgs 条」驱动。
        """
        if self._pool_sweep_task and not self._pool_sweep_task.done():
            return
        self._pool_sweep_task = asyncio.create_task(self._pool_sweep_loop())
        logger.info("[qqmind] pool daily sweep started")

    async def _pool_sweep_loop(self) -> None:
        try:
            # v0.3.1：启动补扫——04:30 已过且今天还没扫过，5 分钟后补一次。
            # 旧版只在进程活到 04:30 时才扫，开发期频繁重载会让残余等 24h。
            await asyncio.sleep(300)
            try:
                await self._sweep_residual_if_due()
            except Exception as e:     # noqa: BLE001
                logger.warning(f"[qqmind] startup residual sweep failed: {e!r}")
            while True:
                # 睡到下一个 04:30（本地时区）
                now = time.localtime()
                target = time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                                      4, 30, 0, 0, 0, -1))
                if target <= time.time():
                    target += 86400
                await asyncio.sleep(max(30.0, target - time.time()))
                try:
                    await self._sweep_residual_if_due(force=True)
                except Exception as e:     # noqa: BLE001
                    logger.warning(f"[qqmind] residual sweep failed: {e!r}")
        except asyncio.CancelledError:
            pass

    async def _sweep_residual_if_due(self, force: bool = False) -> None:
        """每天最多做一次残余兜底；启动补扫只在 04:30 之后执行。

        force=True 用于 04:30 定时触发；启动补扫用默认值（未过 04:30 就跳过）。
        一批最多压 trigger_msgs 条，最多连续 5 批；全部成功清空才记账，
        失败不记账——下次启动或明天还会再试。
        """
        pool = self._pool()
        if not force:
            now = time.localtime()
            if (now.tm_hour, now.tm_min) < (4, 30):
                return
        if pool.sweep_done_today():
            return
        if not pool.msgs:
            pool.mark_sweep_done()
            return
        logger.info(f"[qqmind] daily residual sweep: {len(pool.msgs)} msgs")
        for _ in range(5):
            if not pool.msgs:
                break
            ok = await self._flush_pool()
            if not ok:
                return          # 失败：不记账，等下次启动/明天补
        if not pool.msgs:
            pool.mark_sweep_done()

    def _ensure_proactive_loop(self) -> None:
        """守护循环懒启动（首个私聊会话建立时）；
        白名单非空 或 好感度模式开启 时跑。"""
        if self._proactive_task and not self._proactive_task.done():
            return
        if not (self._cfg("proactive_friends") or []) \
                and not self._cfg("affinity_enabled", False):
            return
        self._proactive_task = asyncio.create_task(self._proactive_loop())
        logger.info("[qqmind] proactive loop started")

    def _note_group(self, gid: int, nick: str, text: str):
        self.group_buffer.note(gid, nick, text)

    # ---------- 成本护栏 ----------
    def _budget(self) -> DailyBudget:
        self.budget.limit = max(0, int(self._cfg("daily_token_budget", 0)))
        return self.budget

    def _budget_exhausted(self) -> bool:
        return self._budget().exhausted()

    _STATS_KEYS = ("judged", "blocked", "replied", "judge_think",
                   "dropped_rate", "dropped_budget")

    def _stat(self, key: str, gid=None, n=1) -> None:
        """计数：全局聚合 + 分群子账（_STATS_G），一次调用双记。"""
        self._STATS[key] = self._STATS.get(key, 0) + n
        if gid is not None:
            g = self._STATS_G.setdefault(int(gid), {})
            g[key] = g.get(key, 0) + n

    def _stats_for(self, gid: int | None) -> dict:
        """账单口径：gid=None 全网汇总；gid=群号该群子账（缺键补 0）。"""
        if gid is None:
            return {k: self._STATS.get(k, 0) for k in self._STATS_KEYS}
        g = self._STATS_G.get(int(gid), {})
        return {k: g.get(k, 0) for k in self._STATS_KEYS}

    def _count_usage(self, *texts, resp=None, gid=None) -> int:
        """记一笔 token（总账必记；gid=群号同时记分群子账，0=私聊/后台）。"""
        usage = getattr(resp, "usage", None)
        n = (int(usage.total) if usage is not None
             else sum(est_tokens(t) for t in texts))
        if n:
            self._budget().add(n, gid=gid)
        return n

    def _real_usage(self, resp, kind: str) -> None:
        """真实 API usage 分桶落账（v0.2.14）：kind=judge/compress/reply。

        分三桶：in_hit（缓存命中）/ in_miss（未命中）/ out——对账对准
        DeepSeek 官网账单（命中价差 50 倍，混算看不出钱烧在哪）。
        存进 DailyBudget.usage_bucket，随 daily.json 落盘、跨日清零。
        """
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        b = self._budget()
        buckets = getattr(b, "usage_bucket", None)
        if buckets is None:
            buckets = b.usage_bucket = {}
        slot = buckets.setdefault(kind, {"in_hit": 0, "in_miss": 0, "out": 0})
        slot["in_hit"] += int(getattr(usage, "input_cached", 0) or 0)
        slot["in_miss"] += int(getattr(usage, "input_other", 0) or 0)
        slot["out"] += int(getattr(usage, "output", 0) or 0)
        # 不单独落盘：调用方紧跟的 _count_usage/budget.add 会 save，
        # 判读每条消息一次，单独 save 是无谓 IO（v0.2.14 审查修）

    def _maybe_quota_notice(self, gid: int, event: AstrMessageEvent) -> None:
        """额度耗尽时明说一次，每群每天最多 N 次。"""
        if not self._budget().allow_notice(
                gid, int(self._cfg("quota_notices_per_day", 2))):
            return
        msg = str(self._cfg("quota_message", "") or "").strip()
        if not msg or not event.unified_msg_origin:
            return
        self._spawn(self._send_bubble(
            gid, 0, msg, False, event.unified_msg_origin))

    def _is_bot(self, uid: int, event: AstrMessageEvent) -> bool:
        """不响应其他 bot（防互怼循环）：优先看原始 sender 标记，其次配置名单。"""
        raw = getattr(event.message_obj, "raw_message", None)
        try:
            sender = raw.get("sender") if hasattr(raw, "get") else None
        except Exception:
            sender = None
        if isinstance(sender, dict) and (sender.get("is_robot")
                                         or sender.get("is_bot")):
            return True
        bots = {str(x) for x in (self._cfg("bot_qq_ids") or [])}
        return str(uid) in bots

    # ---------- 读取轴 / 回复轴运行时 ----------
    def _read_mode(self) -> str:
        return resolve_read_mode(self._cfg("group_preset", "light"),
                                 self._cfg("read_mode", "batch"))

    def _rt(self, gid: int) -> dict:
        return self._runtime.setdefault(int(gid), {
            "origin": "", "batch": [], "batch_task": None,
            "random_task": None, "last_msg": 0.0})

    def _allow_reply(self, gid: int) -> bool:
        return self.rate_limiter.allow(
            int(gid), per_min=int(self._cfg("reply_rate_per_min", 6)))

    def _allow_judge(self, gid: int) -> bool:
        return self.judge_limiter.allow(
            int(gid), per_min=int(self._cfg("max_judged_per_min", 30)))

    def _has_active_session(self, gid: int) -> bool:
        now = time.time()
        alive = False
        for key, s in list(self.sessions.items()):
            if s.is_expired(now):
                self.sessions.pop(key, None)
                continue
            if key[0] == int(gid) and s.monitoring:
                alive = True
        return alive

    def _cancel_batch(self, gid: int) -> None:
        rt = self._rt(gid)
        task = rt.get("batch_task")
        if task and not task.done():
            task.cancel()
        rt["batch"] = []

    def _schedule_batch(self, gid: int) -> None:
        rt = self._rt(gid)
        task = rt.get("batch_task")
        if task and not task.done():
            task.cancel()
        rt["batch_task"] = asyncio.create_task(self._batch_flush_later(gid))

    def _flush_batch_now(self, gid: int) -> None:
        """合并上限到了：取消等待，立刻整批判读回复。"""
        rt = self._rt(gid)
        task = rt.get("batch_task")
        if task and not task.done():
            task.cancel()
        rt["batch_task"] = None
        self._spawn(self._flush_batch(gid))

    async def _batch_flush_later(self, gid: int) -> None:
        try:
            await asyncio.sleep(float(self._cfg("debounce_seconds", 10.0)))
            await self._flush_batch(gid)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[qqmind] batch flush error: {e!r}")

    def _batch_read_allowed(self, gid: int) -> bool:
        """随机读默认只在监控期内；常驻神游是高级开关。"""
        if self._read_mode() != "random":
            return True
        if self._cfg("roam_enabled", False):
            return True
        return self._has_active_session(gid)

    async def _flush_batch(self, gid: int) -> None:
        rt = self._rt(gid)
        # 先摘掉计时器引用：判读/回复期间新消息不能取消这次回包
        rt["batch_task"] = None
        batch = rt.get("batch") or []
        rt["batch"] = []
        if not batch or not self._batch_read_allowed(gid):
            return
        if self._budget_exhausted():
            # 熔断点补全：关判读时这里原来拦不住，攒批会继续烧主回复
            self._stat("dropped_budget", gid, len(batch))
            return
        if self._cfg("reply_judge_enabled", True):
            if not self._allow_judge(gid):
                self._stat("dropped_rate", gid, len(batch))
                logger.info(f"[qqmind] batch dropped (judge rate) in {gid}: "
                            f"{len(batch)} msgs")
                return
            # 攒批/随机读路径也要知道机器人在本群叫什么（判读提示词锚点），
            # 否则按名字喊机器人的消息只能靠 @ 判定，准确率系统性偏低
            bot_name = await self._bot_group_name_lazy(gid)
            ok, _ = await self._judge(gid, "", "", batch=batch,
                                      bot_name=bot_name)
            if not ok:
                return
        if not self._allow_reply(gid):
            self._stat("dropped_rate", gid, len(batch))
            logger.info(f"[qqmind] batch dropped (reply rate) in {gid}: "
                        f"{len(batch)} msgs")
            return
        last = batch[-1]
        prompt = "\n".join(f"{m['nick']}：{m['text']}" for m in batch)
        await self._reply_group(gid, last["uid"], prompt, rt.get("origin") or "",
                                batch=batch)

    def _ensure_random_loop(self, gid: int) -> None:
        rt = self._rt(gid)
        task = rt.get("random_task")
        if task and not task.done():
            return
        rt["random_task"] = asyncio.create_task(self._random_loop(gid))

    async def _random_loop(self, gid: int) -> None:
        while True:
            lo = max(1, int(self._cfg("random_min_seconds", 20)))
            hi = max(lo, int(self._cfg("random_max_seconds", 300)))
            await asyncio.sleep(random.uniform(lo, hi))
            if self._read_mode() != "random":
                self._rt(gid)["random_task"] = None
                return
            try:
                await self._flush_batch(gid)
            except Exception as e:
                logger.warning(f"[qqmind] random read error: {e!r}")

    async def _send_bubble(self, gid: int, uid: int, text: str,
                           first: bool, origin: str) -> bool:
        """主动发一条群消息：优先走 OneBot（明确 self_id），没有可用通道才退回 context。

        注意：OneBot 通道存在但调用失败（风控/断连/权限）时**不再重发**——
        兜底通道只补「通道缺失」，不补「通道被拒」，否则一次故障会变成双发
        （假 bot 里的 context_send 记录会把拒绝当成功，真实环境则可能重复刷屏）。
        """
        rt = self._rt(gid)
        bot = rt.get("bot")
        self_id = str(rt.get("self_id") or "")
        sid = _safe_int_id(self_id)
        if bot is None or not sid or not hasattr(bot, "call_action"):
            # 通道缺失（OneBot 断连 / bot=None）：走 context 兜底
            try:
                parts = ([At(qq=str(uid)), Plain(" " + text)] if first
                         else [Plain(text)])
                await self.context.send_message(origin, MessageChain(parts))
                return True
            except Exception as e:
                logger.warning(f"[qqmind] send failed: {e!r}")
                return False
        segs = []
        body = (" " + text) if first else text
        if first:
            segs.append({"type": "at", "data": {"qq": str(uid)}})
        segs.append({"type": "text", "data": {"text": body}})
        try:
            await bot.call_action("send_group_msg", self_id=sid,
                                  group_id=int(gid), message=segs)
            return True
        except Exception as e:
            logger.warning(f"[qqmind] onebot send failed: {e!r}")
            return False

    # ---------- 图片/动图识别（v0.2.6） ----------
    async def _vision_annotate(self, event: AstrMessageEvent,
                               text: str) -> str:
        """消息带图时把 [图片] 占位换成 [图片: 一句话描述]。

        预算内才调视觉模型；失败/超限静默降级为原占位，不影响消息链。
        描述同时进群缓冲与记忆（后续判读/回复都能看到图在表达什么）。
        """
        if not self._cfg("vision_enabled", True):
            return text
        segs = event.get_messages() or []
        urls = extract_image_urls(segs)
        if not urls:
            return text
        n_seen = int(self._STATS.get("vision_used", 0))
        if not self._vision.allow():
            self._STATS["vision_skipped"] = n_seen  # 保持结构存在
            return text
        provider = self.context.get_using_provider()
        if provider is None:
            return text
        vm = str(self._cfg("vision_model", "") or "").strip()
        descs = []
        for u in urls[:1]:   # 每条消息只解 1 张（多图只解首张，省预算）
            desc = await describe_image(provider, u, vm)
            if desc:
                descs.append(desc)
        if not descs:
            return text
        self._vision.spend(len(descs))
        self._STATS["vision_used"] = self._vision.used
        note = "；".join(descs)
        logger.info(f"[qqmind] vision: {note[:40]}")
        base = text.strip()
        return f"{base} [图片: {note}]".strip() if base else f"[图片: {note}]"

    async def _reply_group(self, gid: int, uid: int, prompt: str,
                           origin: str, batch=None) -> None:
        """攒批/随机读的主动回复路径（没有事件可交给 AstrBot 原生管线）。

        与原生管线共用同一条后处理链：回捞（<查群日记>）→ 分段 → 发送 → 落盘。
        """
        provider = self.context.get_using_provider()
        if provider is None or not origin:
            return
        try:
            out = await provider.text_chat(
                prompt=prompt, session_id=None,
                system_prompt=self._group_system_prompt(
                    gid, prompt, uid=uid),
                contexts=self._group_contexts(
                    gid, uid, skip_last=len(batch) if batch else 0),
                image_urls=[])
            reply = (out.completion_text or "").strip()
        except Exception as e:
            logger.warning(f"[qqmind] manual reply error: {e!r}")
            return
        if not reply:
            return
        # real 在 count 之前：count_usage/add 内部落盘，把 bucket 一并写盘
        self._real_usage(out, "reply")
        self._count_usage(prompt, reply, resp=out, gid=int(gid))
        reply = await self._recall_reply(reply, "group", uid, gid)
        if not reply:
            return
        # v0.2.10：详细场景放宽 @路径的气泡预算（与原生管线同一判定）
        limit = int(self._cfg("group_reply_max_chars", 80))
        if _is_detail_query(prompt):
            limit = max(limit, int(self._cfg("group_reply_detail_chars", 240)))
        bubbles = split_bubbles(reply, max_total_chars=limit)
        delivered = 0
        for i, b in enumerate(bubbles):
            ok = await self._send_bubble(gid, uid, b, i == 0, origin)
            if not ok:
                break
            delivered += 1
            if i < len(bubbles) - 1:
                await asyncio.sleep(0.4 + 0.5 * random.random())
        messages = ([(m["uid"], m["nick"], m["text"]) for m in batch]
                    if batch else [(uid, "", prompt)])
        async with self._hist_lock:
            self._persist_group(gid, uid, messages, reply)
        self._note_group(gid, self._bot_names.get(gid) or "AI", reply)
        if delivered:
            # 一段都没发出去的不算成功回复（避免失败轮虚增面板 delivered 数）
            self._stat("replied", gid)

    def _memory_mode(self) -> str:
        preset = normalize_preset(self._cfg("group_preset", "light"))
        if preset == "advanced":
            if self._cfg("precise_memory", False):
                # v0.4.0：窗口合并后「每人一条精确记忆线」并入公共时间线，
                # 精确记忆退化为平铺（配置保留，行为等同全量）
                return "flat"
            return "layered" if self._cfg("memory_layering", True) else "flat"
        mode = preset_memory_mode(preset) or "flat"
        return "flat" if mode == "precise" else mode

    def _is_light(self) -> bool:
        return normalize_preset(self._cfg("group_preset", "light")) == "light"

    def _group_contexts(self, gid: int, uid: int,
                        skip_last: int = 0) -> list[dict]:
        """回复的逐字 contexts（档位定案 2026-11-01；v0.4.0 改为公共窗口切片）：

        - 精确/全量：group_flat_messages 条
        - 轻量：[]（语境靠 system tail 的 20 条速览）
        - 分层：group_shallow_messages 条浅层
        - 高级：reply_context_messages 自选（默认 0 省）
        只取本群消息（合并窗口后按 gid 过滤），不会把别群的逐字带进来。
        """
        mode = self._memory_mode()
        preset = normalize_preset(self._cfg("group_preset", "light"))
        if self._is_light():
            return []
        if preset == "advanced":
            limit = int(self._cfg("reply_context_messages", 0))
        else:
            limit = (int(self._cfg("group_flat_messages", 60))
                     if mode == "flat"
                     else int(self._cfg("group_shallow_messages", 20)))
        if limit <= 0:
            return []
        return self._session().context(scope="group", gid=gid, limit=limit,
                                       skip_last=skip_last)

    # ---------- 人设三层 + 背景故事 ----------
    def _persona_block(self) -> str:
        """浅层（当前状态）+ 深层（核心性格）；都空时兼容旧版单卡。"""
        shallow = str(self._cfg("persona_shallow", "") or "").strip()
        deep = str(self._cfg("persona_deep", "") or "").strip()
        if not shallow and not deep:
            return str(self._cfg("persona_text", "") or "").strip()
        parts = []
        if deep:
            parts.append(deep)
        if shallow:
            parts.append("【当前状态】\n" + shallow)
        return "\n".join(parts)

    def _chapter_store(self) -> ChapterStore | None:
        pack = str(self._cfg("persona_pack", "") or "").strip()
        if not pack:
            return None
        return self._chapter_store_for(pack)

    def _chapter_store_for(self, pack: str) -> ChapterStore:
        pack = str(pack or "").strip() or "default"
        cs = self._chapters.get(pack)
        if cs is None:
            cs = ChapterStore(
                self._abs_dir("persona_packs_dir", "qqmind_persona_packs"),
                pack)
            self._chapters[pack] = cs
        return cs

    def _persona_background(self, query: str = "") -> str:
        """标题行常驻；当前消息命中关键词才带摘要+正文。"""
        cs = self._chapter_store()
        if cs is None:
            return ""
        parts = []
        n_headers = int(self._cfg("persona_headers_in_prompt", 20))
        headers = cs.headers_for_prompt(n_headers)
        if headers:
            parts.append("【背景故事篇目】\n" + headers)
        if query:
            hits = cs.search(query, int(self._cfg("persona_recall_top", 2)))
            if hits:
                parts.append("【与本条消息相关的背景】\n" + hits)
        return "\n".join(parts)

    def _group_system_prompt(self, gid: int, query: str = "",
                             uid: int = 0, nick: str = "") -> str:
        """群聊 system prompt（v0.2.14 缓存分层）：

        【稳定前缀】人设 + 发言风格 + 活人模板——长期不变，DeepSeek
        前缀缓存对这段生效（命中价 ≈ 未命中的 1/50）。
        【易变后缀】说话人锚定、好感度、召回背景、日记目录——每轮都变，
        放在最后，改动不影响前缀缓存。
        记忆注入总量受 _MEMORY_BUDGET_TOKENS 硬顶，超出按优先级裁。
        """
        # —— 稳定前缀 ——
        parts = []
        persona = self._persona_block()
        if persona:
            parts.append(persona)
        # 群聊长度约束（2026-10-29 群内实测：deepseek 爱列条目写长段）。
        # 无条件注入（v0.2.4：曾有「人设卡已有简短要求就跳过」的条件，但正则把
        # 「口语化」误判成长度规则导致整行从未生效）——和人格不冲突，直接加。
        # v0.2.10：加「废话预算」约束（截断器放宽了详细场景，提示词端同步
        # 压废话，两头一起防"说一半"）。
        parts.append("【发言风格（硬性）】群聊回复只许一小段：1~3 句、"
                     f"{self._cfg('group_reply_max_chars', 80)} 字以内，"
                     "禁止列条目/编号/多段落，禁止客套开场白和重复对方的话，"
                     "每个字都要有信息量，废话零容忍；"
                     "只有对方明确在求教程/讲题/求解释时才允许写长"
                     f"（上限 {self._cfg('group_reply_detail_chars', 240)} 字），"
                     "长也只写干货步骤或推导本身，写完就停，别总结别感叹。")
        # 活人模板（v0.2.13 提炼 / v0.2.14 抽成角色包微插件）：
        # 优先当前激活包里的 live_mode.md，其次全局配置，最后代码默认。
        live = self._live_mode_text()
        if live:
            parts.append("【像活人一样说话（硬性）】\n" + live)
        stable = "\n".join(parts)

        # —— 易变后缀（按预算从高到低优先级装填） ——
        tail: list[tuple[int, str]] = []
        if uid:
            who = f"{nick}（QQ:{uid}）" if nick else f"QQ:{uid}"
            tail.append((0, f"【当前说话人】现在是 {who} 在跟你说话；"
                         "群聊记录里每条开头的 昵称(QQ号) 就是发言者，"
                         "同一 QQ 号就是同一个人，别被改名片骗了。"))
        background = self._persona_background(query)
        if background:
            tail.append((1, background))
        mode = self._memory_mode()
        if mode == "layered":
            # 公共记忆（v0.4.0）：一个库装所有群 + 私聊，群聊视角只给标题行
            headers = self._group_diary().headers_for_prompt(
                int(self._cfg("diary_headers_in_prompt", 8)), scope="group")
            if headers:
                tail.append((3, "你有群聊日记（所有群共同的记忆；时间 标题，"
                                 "想细看就输出 <查群日记\"关键词或时间\">）：\n"
                                 + headers))
        if self._is_light():
            n_ctx = int(self._cfg("group_context_messages", 20))
        else:
            n_ctx = 0
        if n_ctx:
            ctx = "\n".join(self.group_buffer.recent(gid, n_ctx))
            if ctx:
                tail.append((2, f"【最近的群聊（供理解语境，不必逐条回应）】"
                                f"\n{ctx}"))

        # 硬预算裁剪（v0.2.14 审查修）：预算只约束 tail 增量（stable 段
        # 常超 700，以它起算会把所有后缀裁光）。
        # 优先级：0 锚定 > 1 召回背景 > 2 群聊窗口（理解语境必需）
        # > 3 本群日记目录 > 4 跨群目录；prio>=3 超预算才裁
        budget = int(self._cfg("memory_prompt_budget", 700))
        used = 0
        kept: list[str] = []
        for prio, text in sorted(tail, key=lambda x: x[0]):
            cost = est_tokens(text)
            if used + cost > budget and prio >= 3:
                continue
            used += cost
            kept.append(text)
        return stable + "\n" + "\n".join(kept)

    _LIVE_CACHE: dict[str, tuple[float, str]] = {}

    def _live_mode_text(self) -> str:
        """活人模板来源（v0.2.14 角色包微插件），优先级从高到低：

        1. 配置显式清空 = 强制关闭（包文件也不生效）
        2. 当前激活角色包目录下 live_mode.md（随包插拔）
        3. 全局配置 persona_live_mode
        4. 代码默认 _LIVE_MODE_DEFAULT
        """
        if "persona_live_mode" in (self.plugin_config or {}):
            configured = str(self.plugin_config.get("persona_live_mode")
                             or "").strip()
            if configured == "":
                return ""          # 显式清空 = 关
            global_src = configured
        else:
            global_src = _LIVE_MODE_DEFAULT
        pack = str(self._cfg("persona_pack", "") or "").strip()
        if pack:
            pdir = Path(self._abs_dir("persona_packs_dir",
                                      "qqmind_persona_packs")) / pack
            f = pdir / "live_mode.md"
            try:
                mtime = f.stat().st_mtime
                cached = self._LIVE_CACHE.get(pack)
                if cached and cached[0] == mtime:
                    return cached[1]
                text = f.read_text(encoding="utf-8").strip()
                self._LIVE_CACHE[pack] = (mtime, text)
                if text:
                    return text    # 包模板优先
            except OSError:
                pass
        return global_src

    # ---------- 记忆写入（v0.4.0：私聊/群聊共用一条时间线） ----------
    def _record_incoming(self, gid: int, uid: int, nick: str, text: str) -> None:
        """群消息进公共窗口（轻量档不记；削窗口由 `_compress_memory` 统一做）。

        每个条目自带 src/gid/uid/nick：注入时按作用域切片，压缩时混成一篇
        日记（含私聊内容的篇目记 gid=0，群聊侧只给概要）。
        """
        if not (text or "").strip() or self._is_light():
            return
        cap = int(self._cfg("max_msg_tokens", 100)) * 4
        s = self._session()
        s.append_user_message(text, cap_tokens=cap,
                              src="group", gid=gid,
                              uid=int(uid), nick=nick)
        # 落盘节流（v0.4.0）：群消息多，每条都写整窗太贵；5 秒内至多一次，
        # 崩溃最多丢 5 秒的窗口（旧版每条都写文件，但那时窗口只有 60 条）。
        now = time.time()
        if now - self._win_save_ts >= 5.0:
            self._win_save_ts = now
            try:
                s.save_window()
            except Exception as e:     # noqa: BLE001 落盘失败不许影响消息链
                logger.warning(f"[qqmind] save window failed: {e!r}")

    def _persist_group(self, gid: int, uid: int,
                       messages: list[tuple[int, str, str]], reply: str) -> None:
        """群回复落盘：助手消息也进同一条时间线（用户消息在入口已记）。"""
        if self._is_light():
            return
        if not (reply or "").strip():
            return
        self._session().append_assistant(
            reply, src="group", gid=int(gid),
            nick=self._bot_names.get(int(gid)) or "AI")
        try:
            self._session().save_window()
        except Exception as e:     # noqa: BLE001
            logger.warning(f"[qqmind] save window failed: {e!r}")

    def _pool_evict_handler(self, msgs: list[dict]) -> bool:
        """分层模式：窗口挤出的一整段群消息交给候选池攒批（池满才压一次）。

        返回 True 表示这段已被池接管，不再生成摘录条目；混了私聊的段落
        一律返回 False，走正常日记压缩（私聊原文不许进群池）。
        """
        if self._memory_mode() != "layered":
            return False
        if any(m.get("src", "private") != "group" for m in msgs):
            return False
        pool = self._pool()
        now = time.time()
        for m in msgs:
            pool.add(m.get("ts") or now, int(m.get("gid") or 0),
                     int(m.get("uid") or 0), m.get("nick") or "",
                     _strip_own_prefix(m.get("nick") or "",
                                       m.get("content") or ""))
        pool.save()
        if pool.should_flush():
            self._spawn(self._flush_pool())
        return True

    async def _compress_memory(self, s=None) -> None:
        """统一压缩入口（四层稳态）：本地削窗口 → 模型升级 → 熔断 → 碎片合并。

        窗口 token 单调下降，**与模型可用性解耦**：模型挂了也只是多几条
        摘录条目，不会像旧版那样「压不动就一直涨」。
        """
        s = s or self._session()
        if s.on_evict is None:
            s.on_evict = self._pool_evict_handler
        stats = await s.ensure_budget(lambda: self._aux_provider("compress"))
        if stats.get("used"):
            self._budget().add(int(stats["used"]), gid=0)
        if stats.get("slices") or stats.get("upgraded"):
            logger.info(f"[qqmind] memory compress {stats}")

    # ---------- 挂点 1：事件监听（判读门） ----------
    # C2 解耦：实现体在 handlers/judge_gate.JudgeGateMixin（self 即本类实例）。
    # 装饰器必须留在本模块：AstrBot 按 handler.__module__ == 插件主模块过滤
    # handler 注册表，mixin 模块里的方法不会被加载。
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        await self.on_message_impl(event)

    # ---------- 挂点 2：LLM 请求前（记忆注入） ----------
    # C3 解耦：实现体在 handlers/memory_inject.MemoryInjectMixin；装饰器壳
    # 必须留在本模块（AstrBot 按 handler.__module__ == 插件主模块过滤）。
    @on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest):
        await self.inject_memory_impl(event, req)

    # ---------- 挂点 3：LLM 回复后（落盘/压缩/额度刷新） ----------
    @on_llm_response()
    async def persist_round(self, event: AstrMessageEvent, resp):
        await self.persist_round_impl(event, resp)

    async def terminate(self) -> None:
        """插件卸载/重载：停掉随机读与攒批任务，关掉日记库句柄。"""
        if self._proactive_task and not self._proactive_task.done():
            self._proactive_task.cancel()
            self._proactive_task = None
        if self._pool_sweep_task and not self._pool_sweep_task.done():
            self._pool_sweep_task.cancel()
            self._pool_sweep_task = None
        for rt in self._runtime.values():
            for key in ("batch_task", "random_task"):
                t = rt.get(key)
                if t and not t.done():
                    t.cancel()
        self._runtime.clear()
        # v0.4.0：私聊/群聊共用一条会话与一个库，只落盘/关闭一次
        if self._mem_session is not None:
            self._mem_session.save_window()
            self._mem_session = None
        self.privates.clear()
        for gm in self.memories.values():
            gm.close()
        self.memories.clear()
        if self._pub_diary is not None:
            self._pub_diary.close()
            self._pub_diary = None
        self._pub_pool = None
        # v0.3.3：非思考辅助 provider 是本插件私有的运行时克隆，卸载时一并释放
        pm = getattr(self.context, "provider_manager", None)
        if pm is not None:
            try:
                await pm.terminate_provider("qqmind-nothink")
            except Exception as e:     # noqa: BLE001
                logger.debug(f"[qqmind] aux provider cleanup failed: {e!r}")
        for cs in self._chapters.values():
            cs.close()
        self._chapters.clear()
        if getattr(self, "_ring", None):
            diagnostics.detach(self._ring)
            self._ring = None

    # ---------- 配置页后端 API（views/console 通过面板桥调用） ----------
    _STATS = {"judged": 0, "blocked": 0, "replied": 0, "judge_think": 0,
              "dropped_rate": 0, "dropped_budget": 0,
              "judge_in": 0, "judge_out": 0, "saved_in": 0,
              "vision_used": 0, "vision_skipped": 0, "proactive_sent": 0}

    def _bill_saved(self, n_blocked: int) -> None:
        """拦截即省钱：被拦消息按「本来要一次完整回复」估算省下的输入量。

        口径（诚实基准）：单次回复均价 = 群聊 system 提示词 + 窗口上下文
        （用本群最近一次真实回复的计账，没有则用保守常数）。只算输入侧，
        输出侧省的是一次回复文本——不夸大。
        """
        if n_blocked <= 0:
            return
        per_reply = int(self._cfg("bill_est_reply_tokens", 650))
        self._STATS["saved_in"] += n_blocked * per_reply

    def _register_page_apis(self):
        """注册 views/console 页面用的后端接口（C1 解耦：处理函数在
        webapi/，这里只做注册；路径与解耦前完全一致）。"""
        from . import webapi
        webapi.register(self)

    def _diagnostics_snapshot(self) -> dict:
        """组装诊断包（WebUI 与命令共用）。隐私：日记正文不出包。"""
        diary_dir = self._abs_dir("diaries_dir", "qqmind_diaries")
        diary_total = 0
        try:
            for fn in os.listdir(diary_dir):
                if fn.endswith(".db") and fn[:-3].isdigit():
                    diary_total += 1
        except OSError:
            pass
        runtime = {
            "stats": dict(self._STATS),
            "degraded_groups": sorted(self._degraded),
            "judge_fail_counts": dict(self._judge_fail),
            "active_sessions": len(self.sessions),
            "group_runtimes": len(self._runtime),
            "diary_files": diary_total,
            "budget": self._budget().snapshot(),
        }
        provider = self.context.get_using_provider() if self.context else None
        model = ""
        if provider is not None:
            try:
                model = provider.get_model() or ""
            except Exception:
                model = ""
        ab_ver = ""
        try:
            from importlib.metadata import version as _pkgver
            ab_ver = _pkgver("astrbot")
        except Exception:
            ab_ver = ""
        return build_diagnostics(
            plugin_version=PLUGIN_VERSION,
            astrbot_version=ab_ver,
            platform=self._last_platform or "unknown",
            provider_model=model,
            config={k: self._cfg(k) for k in diagnostics.DIAG_CONFIG_KEYS
                    if k != "allowed_groups_count"},
            stats=dict(self._STATS),
            runtime=runtime,
            log_lines=self._ring.snapshot() if self._ring else [],
        )

    # ---------- 指令（C4b 解耦：实现体在 handlers/commands.CommandsMixin） ----------
    @filter.command("qqmind")
    async def cmd_qqmind(self, event: AstrMessageEvent):
        async for r in self.cmd_qqmind_impl(event):
            yield r

    @filter.llm_tool(name="get_usage_stats")
    async def tool_usage_stats(self, event: AstrMessageEvent):
        return await self.tool_usage_stats_impl(event)

    @filter.command("账单")
    async def cmd_bill(self, event: AstrMessageEvent):
        async for r in self.cmd_bill_impl(event):
            yield r

    @filter.command("日记")
    async def cmd_diary(self, event: AstrMessageEvent):
        async for r in self.cmd_diary_impl(event):
            yield r

    @filter.command("群日记")
    async def cmd_group_diary(self, event: AstrMessageEvent):
        async for r in self.cmd_group_diary_impl(event):
            yield r
