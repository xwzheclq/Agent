"""
威胁情报分析 Agent - Streamlit UI
LangGraph Checkpointer 持久化 + 流式输出
"""
import queue
import threading
import asyncio

import streamlit as st

from context_manager import estimate_tokens, MAX_CONTEXT_TOKENS
import checkpoint_store as storage
import voice
voice.preload_model()

# 懒加载：避免启动时导入 torch / langchain 等重依赖
_run_off = None
_run_auto = None
_run_multi_agent = None

def _get_run_off():
    global _run_off
    if _run_off is None:
        from agent_core import run_off as _f
        _run_off = _f
    return _run_off

def _get_run_auto():
    global _run_auto
    if _run_auto is None:
        from agent_core import run_auto as _f
        _run_auto = _f
    return _run_auto

def _get_run_multi_agent():
    global _run_multi_agent
    if _run_multi_agent is None:
        from multi_agent import run_multi_agent as _f
        _run_multi_agent = _f
    return _run_multi_agent


# ========== 流式桥接：async → sync ==========

def _iter_stream(async_gen):
    """子线程跑 async generator，主线程从 queue 读取实时渲染"""
    import traceback
    import checkpoint_store
    q = queue.Queue()

    async def _runner():
        try:
            async for event in async_gen:
                q.put(event)
        except Exception as e:
            q.put({"type": "error", "message": str(e), "traceback": traceback.format_exc()})
        q.put(None)

    def _run_in_thread():
        # 每个新线程都有新的 event loop，必须重置 checkpointer 单例
        # 否则 AsyncSqliteSaver 的 asyncio.Lock 会绑死在旧 loop 上
        checkpoint_store._checkpointer = None
        asyncio.run(_runner())

    threading.Thread(target=_run_in_thread, daemon=True).start()

    while True:
        event = q.get()
        if event is None:
            break
        yield event


# ========== UI 工具函数 ==========

def _render_thinking(placeholder, thinking_lines: list, expanded: bool = False):
    if not thinking_lines:
        placeholder.empty()
        return
    tool_count = len([l for l in thinking_lines if l.startswith("🔧")])
    lines_html = "<br>".join(thinking_lines)
    open_attr = " open" if expanded else ""
    placeholder.markdown(
        f"<details{open_attr}><summary>🔍 检索过程 ({tool_count} 次工具调用)</summary><br>{lines_html}</details>",
        unsafe_allow_html=True,
    )


def _run_stream(mode: str, history: list, user_input: str, thread_id: str,
                thinking_ph, ans_ph, status_ph):
    """运行流式 agent，实时保存部分结果到 DB，返回 (answer, thinking_lines, truncated)"""
    if mode == "关闭":
        stream = _get_run_off()(history, user_input)
    elif mode == "多Agent验证":
        stream = _get_run_multi_agent()(user_input, thread_id=None)
    else:
        stream = _get_run_auto()(history, user_input, thread_id=thread_id)

    thinking_lines = []
    answer = ""
    truncated = False
    is_multi = (mode == "多Agent验证")
    interrupted = False

    storage.cleanup_partial(thread_id)
    partial_id = storage.save_partial_answer(thread_id, "", thinking_lines)

    try:
        for event in _iter_stream(stream):
            if event["type"] == "thinking":
                new_lines = event["lines"]
                # 多 Agent 模式的 thinking 事件是增量（每条只带当前阶段的行），自动模式是累积
                # 取累计条目更多的那份，防止增量事件覆盖掉之前的工具调用记录
                if len(new_lines) >= len(thinking_lines):
                    thinking_lines = new_lines
                else:
                    # 增量模式：把新行追加上去
                    thinking_lines = list(thinking_lines) + [l for l in new_lines if l not in thinking_lines]
                _render_thinking(thinking_ph, thinking_lines, expanded=True)
                storage.update_partial_answer(partial_id, answer, thinking_lines)

            elif event["type"] == "token":
                answer += event["text"]
                ans_ph.markdown(answer + " ▌")

            elif event["type"] == "phase":
                p = event["phase"]
                if p == "research_done":
                    status_ph.info("🔄 正在联网搜索，请稍候...")
                elif p == "verifying":
                    status_ph.info("🔍 正在对比本地库与联网结果...")
                elif p == "verify_pass":
                    status_ph.success("✅ 对比完成")
                elif p == "verify_fail":
                    status_ph.warning("⚠️ 验证未通过 — 回答可能存在事实错误，请查看下方标注")

            elif event["type"] == "done":
                answer = event["answer"]
                done_thinking = event.get("thinking", [])
                # 优先用 done 事件中更完整的 thinking 数据
                if len(done_thinking) > len(thinking_lines):
                    thinking_lines = done_thinking
                elif done_thinking:
                    thinking_lines = list(thinking_lines) + [l for l in done_thinking if l not in thinking_lines]
                truncated = event.get("truncated", False)
                break

            elif event["type"] == "error":
                interrupted = True
                err_msg = event.get("message", "未知错误")
                print(f"[_run_stream ERROR] {err_msg}")
                if event.get("traceback"):
                    print(event["traceback"])
                break
    finally:
        # 后处理
        if not answer:
            answer = "*(未生成回答)*"
        if truncated:
            answer += "\n\n> ⚠️ 回答达到长度上限，可能不完整"
        if interrupted:
            answer += "\n\n> ⚠️ 输出被中断"

        # 完成标记写入内容（持久化到 DB），多 Agent 验证结果已在 answer 中
        if not is_multi and not interrupted and not truncated:
            answer += "\n\n✅ 输出完成"

        # 写入 DB（try/finally 保证一定执行）
        storage.finalize_answer(partial_id, answer, thinking_lines)

        ans_ph.markdown(answer)
        _render_thinking(thinking_ph, thinking_lines)
        if is_multi:
            status_ph.empty()

    return answer, thinking_lines, truncated


# ===== 持久化辅助 =====

def _load_sessions():
    """从 DB 加载会话列表到 session_state（仅首次）"""
    if "sessions" not in st.session_state:
        sessions = storage.list_sessions()
        if not sessions:
            tid = storage.create_session(title="新对话", mode="自动（Agent 决定）")
            sessions = storage.list_sessions()
        st.session_state.sessions = {s["thread_id"]: s for s in sessions}


def _load_messages(thread_id: str) -> list:
    """加载对话消息"""
    msgs = storage.get_messages(thread_id)
    # 标记当前 thread 已加载
    st.session_state.loaded_thread = thread_id
    return msgs


def _ensure_current_session() -> str:
    """确保有当前会话，返回 thread_id"""
    if "current_thread" not in st.session_state:
        sessions = list(st.session_state.sessions.values())
        st.session_state.current_thread = sessions[0]["thread_id"]
    return st.session_state.current_thread


# ========== 初始化 ==========

storage.init()
st.set_page_config(page_title="威胁情报分析 Agent", page_icon="🛡️", layout="wide")
st.title("🛡️ 威胁情报分析 Agent")
_load_sessions()

if "current_thread" not in st.session_state:
    sessions = list(st.session_state.sessions.values())
    tid = sessions[0]["thread_id"] if sessions else storage.create_session()
    st.session_state.current_thread = tid

if "rag_mode" not in st.session_state:
    sess = st.session_state.sessions.get(st.session_state.current_thread, {})
    st.session_state.rag_mode = sess.get("rag_mode", "自动（Agent 决定）")

if "loaded_thread" not in st.session_state:
    st.session_state.loaded_thread = None

MODE_OPTIONS = ["关闭（纯 LLM）", "自动（Agent 决定）", "多Agent验证（Researcher + Verifier）"]

# ---- Sidebar ----
with st.sidebar:
    st.subheader("⚙️ 设置")

    mode = st.selectbox(
        "RAG 模式",
        MODE_OPTIONS,
        index=MODE_OPTIONS.index(st.session_state.rag_mode)
        if st.session_state.rag_mode in MODE_OPTIONS else 1,
    )
    st.session_state.rag_mode = mode

    st.divider()
    st.subheader("💬 会话")

    sessions = list(st.session_state.sessions.values())
    for sess in sorted(sessions, key=lambda s: s.get("updated_at", ""), reverse=True):
        tid = sess["thread_id"]
        c1, c2 = st.columns([4, 1])
        with c1:
            label = sess.get("title", "新对话")[:25]
            is_current = (tid == st.session_state.current_thread)
            if st.button(label, key=f"sw_{tid}",
                         use_container_width=True,
                         type="primary" if is_current else "secondary"):
                if tid != st.session_state.current_thread:
                    st.session_state.current_thread = tid
                    st.session_state.loaded_thread = None  # 触发重新加载
                    st.session_state.rag_mode = sess.get("rag_mode", "自动（Agent 决定）")
                    st.rerun()
        with c2:
            if len(st.session_state.sessions) > 1:
                if st.button("✕", key=f"del_{tid}"):
                    storage.delete_session(tid)
                    if st.session_state.current_thread == tid:
                        remaining = storage.list_sessions()
                        st.session_state.current_thread = remaining[0]["thread_id"] if remaining else storage.create_session()
                        st.session_state.loaded_thread = None
                    st.session_state.sessions = {s["thread_id"]: s for s in storage.list_sessions()}
                    st.rerun()

    if st.button("➕ 新建对话", use_container_width=True):
        tid = storage.create_session(title="新对话", mode=mode)
        st.session_state.current_thread = tid
        st.session_state.loaded_thread = None
        st.session_state.rag_mode = mode
        st.session_state.sessions = {s["thread_id"]: s for s in storage.list_sessions()}
        st.rerun()

    st.divider()
    mk = mode.split("（")[0]
    st.caption(f"当前模式: **{mk}**")
    if mk == "关闭":
        st.caption("纯 LLM 对话，不使用检索")
    elif mk == "自动":
        st.caption("Agent 自主判断是否需要检索")
    elif mk == "多Agent验证":
        st.caption("Researcher 检索+回答 → Verifier 联网核查事实")

# ---- 聊天区 ----
current_thread = st.session_state.current_thread

# 加载消息（仅在切换 thread 时）
if st.session_state.loaded_thread != current_thread:
    st.session_state.loaded_thread = current_thread

messages = storage.get_messages(current_thread)

# 上下文用量（sidebar 底部）
with st.sidebar:
    st.divider()
    # 直接用 content 文本估算 token
    all_text = "\n".join(m.get("content", "") for m in messages if m.get("content"))
    est_tokens = estimate_tokens(all_text) + len(messages) * 4
    pct = est_tokens / MAX_CONTEXT_TOKENS * 100
    st.caption(f"📊 上下文用量: **{pct:.0f}%** ({est_tokens}/{MAX_CONTEXT_TOKENS} tokens)")
    if pct > 80:
        st.progress(min(int(pct), 100), text="⚠️ 接近上限，旧消息将被压缩")
    else:
        st.progress(min(int(pct), 100))
VISIBLE_RECENT = 6  # 直接展示最近 6 条（3 轮对话）

if len(messages) > VISIBLE_RECENT:
    old_msgs = messages[:-VISIBLE_RECENT]
    recent_msgs = messages[-VISIBLE_RECENT:]
    with st.expander(f"📜 更早对话 ({len(old_msgs)} 条)", expanded=False):
        for msg in old_msgs:
            with st.chat_message(msg["role"]):
                thinking = msg.get("thinking", [])
                if thinking:
                    tool_count = len([l for l in thinking if l.startswith("🔧")])
                    with st.expander(f"🔍 检索过程 ({tool_count} 次工具调用)", expanded=False):
                        st.markdown("\n\n".join(thinking))
                content = msg.get("content", "")
                if msg.get("truncated") and not content.endswith("⚠️ 输出被中断"):
                    content += "\n\n> ⚠️ 输出被中断"
                st.markdown(content)
    for msg in recent_msgs:
        with st.chat_message(msg["role"]):
            thinking = msg.get("thinking", [])
            if thinking:
                tool_count = len([l for l in thinking if l.startswith("🔧")])
                with st.expander(f"🔍 检索过程 ({tool_count} 次工具调用)", expanded=False):
                    st.markdown("\n\n".join(thinking))
            content = msg.get("content", "")
            if msg.get("truncated") and not content.endswith("⚠️ 输出被中断"):
                content += "\n\n> ⚠️ 输出被中断"
            st.markdown(content)
else:
    for msg in messages:
        with st.chat_message(msg["role"]):
            thinking = msg.get("thinking", [])
            if thinking:
                tool_count = len([l for l in thinking if l.startswith("🔧")])
                with st.expander(f"🔍 检索过程 ({tool_count} 次工具调用)", expanded=False):
                    st.markdown("\n\n".join(thinking))
            content = msg.get("content", "")
            if msg.get("truncated") and not content.endswith("⚠️ 输出被中断"):
                content += "\n\n> ⚠️ 输出被中断"
            st.markdown(content)

# 语音输入（主区域，sidebar 有 JS bug）
audio_value = st.audio_input("🎤 语音提问")

if audio_value is not None:
    raw = audio_value.read()
    # 防止 st.audio_input 在 rerun 后仍返回旧值导致死循环
    audio_key = hash(raw)
    if audio_key != st.session_state.get("_last_audio_key"):
        st.session_state._last_audio_key = audio_key
        ready = voice.is_model_ready()
        status_text = "🔊 正在识别..." if ready else "⏳ 语音模型加载中（首次约30秒）..."
        with st.spinner(status_text):
            text = voice.transcribe(raw)
        if text:
            st.session_state.voice_text = text
            st.rerun()
        else:
            st.toast("语音识别失败，请重试", icon="⚠️")

if user_input := st.chat_input("输入问题，例如: APT29 使用了哪些恶意软件？"):
    text_to_send = user_input
elif "voice_text" in st.session_state:
    text_to_send = st.session_state.pop("voice_text")
else:
    text_to_send = None

if text_to_send:
    # 更新标题
    sess = st.session_state.sessions.get(current_thread, {})
    if sess.get("title") == "新对话":
        storage.update_session(current_thread, title=text_to_send[:30])

    # 保存用户消息
    storage.save_message(current_thread, "user", text_to_send)

    with st.chat_message("user"):
        st.markdown(text_to_send)

    with st.chat_message("assistant"):
        think_ph = st.empty()
        ans_ph = st.empty()
        status_ph = st.empty()

        history_messages = storage.get_messages(current_thread)
        history_for_agent = [m for m in history_messages[:-1]]  # 不含当前用户消息

        current_mode = mode.split("（")[0]

        _run_stream(
            current_mode, history_for_agent, text_to_send, current_thread,
            think_ph, ans_ph, status_ph,
        )

    # 刷新 session 列表
    st.session_state.sessions = {s["thread_id"]: s for s in storage.list_sessions()}
    st.rerun()
