"""控制台 Web API（C1 解耦自 main.py，v0.3.9）。

views/console 页面的 15 个后端接口。全部是纯函数式闭包平移：
原 `_register_page_apis` 里的闭包改为模块级函数（第一个参数收插件实例），
路由与处理函数的绑定表在 `API_ROUTES`，由 main.QQMindPlugin._register_page_apis
遍历注册——路径、方法、处理逻辑与解耦前完全一致。

市场规范约定（三轮审查锤出来的，改这里前先看）：
- 数据一律经 plugin._abs_dir()（官方 plugin_data 目录）
- 会话历史只走官方 context.get_db().get_conversations()
- 日志只经 astrbot.api 的 logger
"""
from __future__ import annotations

import json
import os
import time

from astrbot.api import logger
from astrbot.api.web import request as _web_request

from ..core.importer import (chunk_messages, extract_conversations_json,
                             normalize_messages, run_import,
                             split_export_conversations)

# main 模块符号延迟解析（避免 webapi <-> main 循环 import）：
# register() 时经 main 模块对象取 _CONFIG_KEYS/_safe_int_id/_short_title
_API_ROUTES = []


def route(path: str, methods: list[str], desc: str):
    """路由登记装饰器：路径/方法/描述集中声明。"""
    def deco(fn):
        _API_ROUTES.append((path, fn, methods, desc))
        return fn
    return deco


# ---------- 今日账本 ----------

@route("/stats", ["GET"], "qqmind 今日账本")
async def api_stats(p):
    # v0.4.0：私聊/群聊共用一个库（memory.db），总数直接问那个库
    diary_total = 0
    try:
        store = p._memory_store()
        diary_total = store.count()
        # 私聊篇数（gid==0 保守口径）/群聊篇数分开报，控制台更好看
        priv_total = store.count(scope="private")
        group_total = store.count(scope="group")
    except Exception:      # noqa: BLE001
        priv_total = group_total = 0
    max_win = 0
    try:
        max_win = p._session().window_tokens()
    except Exception:      # noqa: BLE001
        pass
    return {"status": "ok", "data": {
        "judged": p._STATS["judged"],
        "blocked": p._STATS["blocked"],
        "replied": p._STATS["replied"],
        "judge_think": p._STATS["judge_think"],
        "dropped_rate": p._STATS["dropped_rate"],
        "dropped_budget": p._STATS["dropped_budget"],
        "diary_total": diary_total,
        "diary_private": priv_total,
        "diary_group": group_total,
        "max_window_tokens": max_win,
        "budget": p._budget().snapshot()}}


# ---------- 配置读写 ----------

@route("/config", ["GET"], "qqmind 读配置")
async def api_get_config(p):
    c = p.plugin_config or {}
    return {"status": "ok", "data": {k: c.get(k) for k in _CONFIG_KEYS}}


@route("/config", ["POST"], "qqmind 写配置")
async def api_set_config(p):
    payload = await _web_request.json(default={}) or {}
    c = p.plugin_config
    if c is None:
        return {"status": "error", "message": "config unavailable"}
    for key in _CONFIG_KEYS:
        if key in payload:
            c[key] = payload[key]
    try:
        c.save_config()
    except Exception as e:
        logger.warning(f"[qqmind] save_config failed: {e!r}")
    return {"status": "ok"}


# ---------- 诊断 ----------

@route("/diagnostics", ["GET"], "qqmind 一键诊断包")
async def api_diagnostics(p):
    try:
        return {"status": "ok", "data": p._diagnostics_snapshot()}
    except Exception as e:
        logger.warning(f"[qqmind] diagnostics failed: {e!r}")
        return {"status": "error", "message": repr(e)}


# ---------- 数据一键导入（v0.2.6） ----------

async def _conversation_rows(p, user_id: str = None):
    """经官方数据库接口读会话（市场规范整改：不再直连 data_v4.db）。
    db 不可用（测试环境/老版本 AstrBot）返回 None。"""
    try:
        db = p.context.get_db()
    except Exception:
        return None
    try:
        convs = await db.get_conversations(user_id=user_id)
    except Exception as e:
        logger.warning(f"[qqmind] get_conversations failed: {e!r}")
        return None
    out = []
    for c in convs:
        content = c.content if isinstance(c.content, list) else None
        out.append({
            "platform_id": c.platform_id,
            "user_id": c.user_id,
            "size": len(json.dumps(content, ensure_ascii=False))
                    if content is not None else 0,
            "updated_at": str(getattr(c, "updated_at", "") or ""),
            "content": content})
    return out


@route("/import/sources", ["GET"], "qqmind 可导入来源")
async def api_import_sources(p):
    """可导入来源：AstrBot conversations 表里的会话（按用户聚合）。
    v0.3.6 改走官方 get_db().get_conversations()。"""
    rows = await _conversation_rows(p)
    if not rows:
        return {"status": "ok", "data": {"sources": []}}
    latest: dict[str, dict] = {}
    for r in rows:
        prev = latest.get(r["user_id"])
        if prev is None or r["updated_at"] > prev["updated_at"]:
            latest[r["user_id"]] = r
    sources = []
    for r in latest.values():
        uid_part = str(r["user_id"] or "").split("!")[-1]
        sources.append({
            "id": str(r["user_id"] or ""),   # user_id 已含平台前缀，全局唯一
            "platform": r["platform_id"],
            "uid": uid_part,
            "size": int(r["size"] or 0),
            "updated": r["updated_at"]})
    sources.sort(key=lambda x: x["updated"], reverse=True)
    return {"status": "ok", "data": {"sources": sources[:200]}}


def _launch_import(p, msgs: list[dict], uid: int, label: str) -> str:
    from ..core.importer import ImportJob  # 局部导入避免环
    job = ImportJob(uid, msgs, len(chunk_messages(msgs)))
    import_id = f"imp_{int(time.time())}_{len(p._import_jobs)}"
    p._import_jobs[import_id] = job
    diary = p._private(uid).diary

    async def _run_and_persist():
        await run_import(job, diary, p.context.get_using_provider(),
                         budget=p.budget)
        if job.profile:
            try:
                diary.save_profile(job.profile, uid=uid)
                logger.info(f"[qqmind] import {import_id} "
                            f"profile saved for uid={uid}")
            except Exception as e:     # noqa: BLE001
                logger.warning(f"[qqmind] profile save failed: {e!r}")

    p._spawn(_run_and_persist())
    logger.info(f"[qqmind] import {import_id} started: uid={uid} "
                f"src={label} msgs={len(msgs)} "
                f"chunks={job.total_chunks}")
    return import_id


@route("/import/start", ["POST"], "qqmind 发起导入")
async def api_import_start(p):
    """发起导入。四种来源（v0.2.9 全兼容，自动识别）：
    - source: AstrBot conversations 的 user_id（桥接用户）
    - messages: 通用 [{role, content}, ...]
    - export: DeepSeek/ChatGPT 官方导出的 conversations.json 内容
      （多会话批量，每会话一个任务）
    - zip_b64: 官方导出 zip 的 base64（自动找里面的 conversations.json）
    """
    payload = await _web_request.json(default={}) or {}
    source = str(payload.get("source") or "").strip()
    uid = _safe_int_id(payload.get("uid"))
    if uid <= 0:
        return {"status": "error", "message": "缺少 uid（导入到谁）"}

    def _launch(msgs: list[dict], label: str) -> str:
        return _launch_import(p, msgs, uid, label)

    zip_b64 = str(payload.get("zip_b64") or "").strip()
    if zip_b64:
        import base64
        try:
            zbytes = base64.b64decode(zip_b64)
        except Exception:
            return {"status": "error", "message": "zip base64 无效"}
        raw = extract_conversations_json(zbytes)
        if raw is None:
            return {"status": "error",
                    "message": "zip 里没找到可解析的 conversations.json"}
        convs = split_export_conversations(raw)
        if not convs:
            return {"status": "error", "message": "包里没有可识别的对话"}
        ids = {}
        for i, c in enumerate(convs):
            ids[c.get("title") or f"会话{i + 1}"] = _launch(
                c["messages"], f"zip#{i + 1}")
        return {"status": "ok",
                "data": {"conversations": len(convs),
                         "import_ids": ids}}
    export = payload.get("export")
    if export is not None:
        convs = split_export_conversations(export)
        if not convs:
            return {"status": "error", "message": "没有可识别的对话内容"}
        ids = {}
        for i, c in enumerate(convs):
            ids[c.get("title") or f"会话{i + 1}"] = _launch(
                c["messages"], f"export#{i + 1}")
        return {"status": "ok",
                "data": {"conversations": len(convs),
                         "import_ids": ids}}
    if source:
        rows = await _conversation_rows(p, user_id=source)
        raw = None
        for r in rows or []:
            if str(r["user_id"]) == str(source):
                raw = r["content"]
                break
        if raw is None:
            return {"status": "error",
                    "message": f"会话不存在或数据库不可用：{source}"}
    else:
        raw = payload.get("messages")
        if raw is None:
            return {"status": "error", "message": "缺少 source 或 messages"}
    msgs = normalize_messages(raw)
    if not msgs:
        return {"status": "error",
                "message": "没有可识别的对话内容（role/content）"}
    import_id = _launch(msgs, "inline")
    return {"status": "ok", "data": {"import_id": import_id,
                                     "msgs": len(msgs),
                                     "chunks": len(chunk_messages(msgs))}}


@route("/import/status", ["GET"], "qqmind 导入进度")
async def api_import_status(p):
    qid = str(_web_request.query.get("id") or "").strip()
    if qid:
        job = p._import_jobs.get(qid)
        if not job:
            return {"status": "error", "message": "任务不存在"}
        return {"status": "ok", "data": job.progress()}
    return {"status": "ok", "data": {k: j.progress() for k, j in
                                     list(p._import_jobs.items())[-10:]}}


# ---------- 好感度 ----------

@route("/affection/update", ["POST"], "qqmind 编辑好感度/备注/作息")
async def api_affection_update(p):
    """编辑某用户好感度：note（备注）/schedule（作息）/score（手调）。"""
    payload = await _web_request.json(default={}) or {}
    uid = _safe_int_id(payload.get("uid"))
    if uid <= 0:
        return {"status": "error", "message": "缺少 uid"}
    if "note" in payload:
        p._affection.set_note(uid, str(payload.get("note") or ""))
    if "schedule" in payload:
        sched = str(payload.get("schedule") or "").strip()
        from ..core.affection import parse_schedule
        if sched and parse_schedule(sched) is None:
            return {"status": "error",
                    "message": "作息格式应为 HH:MM-HH:MM（如 21:30-23:30）"}
        p._affection.set_schedule(uid, sched)
    if "score" in payload:
        try:
            target = max(0, min(100, int(payload["score"])))
        except (TypeError, ValueError):
            return {"status": "error", "message": "score 必须是数字"}
        d = p._affection.get(uid)
        d["score"] = target       # 手调直写，不受每日自然增长上限管
        p._affection._put(uid, d)
    return {"status": "ok",
            "data": p._affection.get(uid)}


# ---------- 日记浏览 / 用户列表 ----------

@route("/diaries", ["GET"], "qqmind 日记浏览")
async def api_diaries(p):
    uid = str(_web_request.query.get("uid") or "").strip()
    try:
        store = p._memory_store()
    except Exception:      # noqa: BLE001
        return {"status": "ok", "data": {"entries": []}}
    # v0.4.0：私聊/群聊共用一个库，这里直接给全库最近篇目；
    # 每篇带 gid（0=含私聊内容，>0=纯群聊）供前端区分
    rows = store.recent_rows(10)
    entries = [{
        "ts": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"])),
        "title": (r["title"] or "").strip() or _short_title(r["summary"]),
        "tags": "、".join(json.loads(r["tags"])),
        "summary": r["summary"],
        "content": (r["content"] or "").strip(),
        "gid": int(r["gid"] or 0),
        "private": not int(r["gid"] or 0),
        "uid": uid} for r in rows]
    return {"status": "ok", "data": {"entries": entries}}


@route("/sessions", ["GET"], "qqmind 用户列表")
async def api_sessions(p):
    """用户列表（v0.4.0 合并记忆后）：uid 从窗口、好感度键与旧库文件三处汇总。

    日记数改为「共享库总篇数」（一个库装所有人，已无法按 uid 归属），
    每人的活跃度改看 `window_msgs`（该用户在窗口里的私聊消息数）。
    """
    diary_dir = p._abs_dir("diaries_dir", "qqmind_diaries")
    uids: set[int] = set(p.privates.keys())
    sess = None
    try:
        sess = p._session()
    except Exception:      # noqa: BLE001
        sess = None
    if sess is not None:
        for m in sess.messages:
            if m.get("src", "private") == "private" and int(m.get("uid") or 0):
                uids.add(int(m["uid"]))
    store = None
    try:
        store = p._memory_store()
        for k in store.meta_keys("affinity:"):
            try:
                uids.add(int(k.split(":", 1)[1]))
            except (IndexError, ValueError):
                pass
    except Exception:      # noqa: BLE001
        store = None
    # 兼容尚未迁移的旧库（<uid>.db）
    try:
        for fn in os.listdir(diary_dir):
            if fn.endswith(".db") and fn[:-3].isdigit():
                uids.add(int(fn[:-3]))
    except OSError:
        pass
    total = store.count() if store is not None else 0
    users = []
    for u in sorted(uids):
        msgs = [m for m in (sess.messages if sess else [])
                if m.get("src", "private") == "private"
                and int(m.get("uid") or 0) == int(u)]
        aff = p._affection.get(u)
        users.append({
            "uid": str(u), "diaries": total,
            "window_msgs": len(msgs),
            "window_tokens": (sess.window_tokens() if sess else 0),
            # QQ 头像直链（qlogo 公共 CDN，无需 API/无需下载中转）
            "avatar": f"https://q1.qlogo.cn/g?b=qq&nk={u}&s=100",
            "affinity": aff["score"],
            "note": aff.get("note", ""),
            "schedule": aff.get("schedule", "")})
    users.sort(key=lambda x: -x["window_msgs"])
    return {"status": "ok", "data": {"users": users}}


# ---------- 角色包 / 篇章 ----------

@route("/packs", ["GET"], "qqmind 角色包列表")
async def api_packs(p):
    packs_dir = p._abs_dir("persona_packs_dir", "qqmind_persona_packs")
    packs = []
    try:
        for fn in sorted(os.listdir(packs_dir)):
            if fn.endswith(".db"):
                name = fn[:-3]
                packs.append({
                    "name": name,
                    "chapters": p._chapter_store_for(name).count()})
    except OSError:
        pass
    active = str(p._cfg("persona_pack", "") or "").strip()
    names = {x["name"] for x in packs}
    if active and active not in names:
        packs.append({"name": active,
                      "chapters": p._chapter_store_for(active).count()})
        packs.sort(key=lambda x: x["name"])
    return {"status": "ok", "data": {"packs": packs, "active": active}}


@route("/chapters", ["GET"], "qqmind 篇章列表")
async def api_chapters(p):
    pack = str(_web_request.query.get("pack") or "").strip()
    if not pack:
        return {"status": "ok", "data": {"chapters": []}}
    rows = p._chapter_store_for(pack).list_rows(500)
    return {"status": "ok", "data": {"chapters": [
        {"id": r["id"], "title": r["title"],
         "summary": r["summary"], "content": r["content"]}
        for r in rows]}}


@route("/chapter", ["POST"], "qqmind 存篇章")
async def api_chapter_save(p):
    payload = await _web_request.json(default={}) or {}
    pack = str(payload.get("pack") or "").strip()
    if not pack:
        return {"status": "error", "message": "缺少角色包名"}
    title = str(payload.get("title") or "").strip()
    content = str(payload.get("content") or "").strip()
    if not title:
        return {"status": "error", "message": "标题不能为空"}
    cs = p._chapter_store_for(pack)
    cid = payload.get("id")
    if cid:
        ok = cs.update(int(cid), title,
                       str(payload.get("summary") or ""), content)
        return {"status": "ok" if ok else "error", "data": {"id": int(cid)}}
    new_id = cs.add(title, str(payload.get("summary") or ""), content)
    return {"status": "ok", "data": {"id": new_id}}


@route("/chapter/delete", ["POST"], "qqmind 删篇章")
async def api_chapter_delete(p):
    payload = await _web_request.json(default={}) or {}
    pack = str(payload.get("pack") or "").strip()
    cid = payload.get("id")
    if not pack or not cid:
        return {"status": "error", "message": "缺少参数"}
    ok = p._chapter_store_for(pack).delete(int(cid))
    return {"status": "ok" if ok else "error"}


def register(plugin) -> None:
    """由 main.QQMindPlugin._register_page_apis 调用：遍历路由表注册。

    main 符号（_CONFIG_KEYS/_safe_int_id/_short_title/PLUGIN_NAME）在此
    经 main 模块对象延迟解析，规避 webapi<->main 循环 import。
    处理函数经偏函数绑定 plugin 实例——注册出去的仍是零参数协程，
    对 WebUI 与测试的调用方式与解耦前完全一致。"""
    from .. import main as _m
    globals()["_safe_int_id"] = _m._safe_int_id
    globals()["_short_title"] = _m._short_title
    import functools
    for path, fn, methods, desc in _API_ROUTES:
        bound = functools.partial(fn, plugin)
        plugin.context.register_web_api(f"/{_m.PLUGIN_NAME}{path}", bound,
                                        methods, desc)
