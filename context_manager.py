"""
对话上下文管理：Token 估算 + 超限自动压缩

策略：
  1. 中英文混合 Token 估算（~2 chars/token for Chinese, ~4 for English）
  2. 总 Token 超过 MAX_CONTEXT * 0.8 时触发压缩
  3. 用 LLM 把最旧的消息压缩成摘要，保留最近 N 条原始消息
  4. 摘要作为 SystemMessage 注入，新对话继续

类似 Cursor 的上下文压缩：旧对话 → 一段摘要 + 最近的原始消息
"""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


# ========== 配置 ==========
MAX_CONTEXT_TOKENS = 30720   # 32K * 0.96，留一点余量给输出
COMPRESS_THRESHOLD = 0.8     # 超过 80% 就触发压缩
KEEP_RECENT = 4              # 压缩后保留最近 4 条原始消息


def estimate_tokens(text: str) -> int:
    """
    粗略 Token 估算。中文约 2 chars/token，英文约 4 chars/token。
    混合文本取加权平均 ~3 chars/token。
    """
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
    total_chars = len(text)
    english_chars = total_chars - chinese_chars
    return int(chinese_chars / 1.8 + english_chars / 3.5)


def estimate_messages_tokens(messages: list) -> int:
    """估算消息列表的总 Token 数"""
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += estimate_tokens(block["text"])
        # 每条消息加 4 tokens 的格式开销（role + 分隔符）
        total += 4
    return total


COMPRESS_SYSTEM_PROMPT = """你是对话摘要助手。请把以下对话历史压缩成一段简洁的摘要，保留关键信息：

- 用户问了什么问题
- 你给出了什么回答（核心观点、关键数据）
- 使用了哪些工具、得到了什么结果
- 任何重要的事实或结论

摘要控制在 500 字以内，用中文。只写摘要内容，不要加"摘要："前缀。"""


async def compress_history(messages: list, llm) -> list:
    """
    压缩消息列表：旧消息 → LLM 摘要 + 保留最近 KEEP_RECENT 条。

    返回新的消息列表：[(SystemMessage: 历史摘要), ...最近消息]
    """
    if len(messages) <= KEEP_RECENT + 4:
        return messages  # 太少，不压缩

    # 分离：旧消息（要压缩的） + 新消息（保留的）
    old_msgs = messages[:-KEEP_RECENT]
    recent_msgs = messages[-KEEP_RECENT:]

    # 把旧消息转成文本
    old_text_parts = []
    for msg in old_msgs:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(content, str) and content.strip():
            old_text_parts.append(f"[{role}]: {content[:500]}")
    old_text = "\n".join(old_text_parts)

    if len(old_text) < 200:
        return messages  # 太短不压

    # 用 LLM 生成摘要
    try:
        summary_llm = llm.bind(stop=None) if hasattr(llm, "bind") else llm
        resp = await summary_llm.ainvoke([
            SystemMessage(content=COMPRESS_SYSTEM_PROMPT),
            HumanMessage(content=old_text[:4000]),
        ])
        summary = resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        # LLM 调用失败，直接裁剪（保留最近消息）
        return recent_msgs

    summary_msg = SystemMessage(
        content=f"[历史对话摘要]\n{summary.strip()}\n---\n以下是最近的对话："
    )
    return [summary_msg] + list(recent_msgs)


def should_compress(messages: list, threshold: float = COMPRESS_THRESHOLD) -> bool:
    """判断是否需要压缩"""
    tokens = estimate_messages_tokens(messages)
    return tokens > MAX_CONTEXT_TOKENS * threshold


def get_context_info(messages: list) -> dict:
    """获取上下文使用情况（用于 UI 展示）"""
    tokens = estimate_messages_tokens(messages)
    return {
        "tokens": tokens,
        "max_tokens": MAX_CONTEXT_TOKENS,
        "usage_pct": tokens / MAX_CONTEXT_TOKENS * 100,
        "need_compress": tokens > MAX_CONTEXT_TOKENS * COMPRESS_THRESHOLD,
        "message_count": len(messages),
    }
