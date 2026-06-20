"""
Agent 工具 - 本地检索 + 知识图谱 + 联网搜索
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from rag_engine import query_graph, query_graph_2hop, query_graph_path, query_graph_by_type
from hybrid_retriever import hybrid_search
from tavily import TavilyClient

TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")
MAX_OUTPUT_CHARS = 800  # 每条结果最多800字，防止上下文溢出（模型 8192 tokens）


@tool
def search_vector_db(query: str) -> str:
    """
    从本地威胁情报文档库检索。输入是自然语言问题或关键词。
    适用于：已知APT组织（APT29、Lazarus等）、历史攻击事件、恶意软件家族、漏洞分析报告。
    """
    results = hybrid_search(query, top_k=3)  # top_k=3 限制上下文长度
    if not results:
        return "本地库未找到相关威胁情报，建议尝试联网搜索。"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", f"文档{i}")
        text = r['text']
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "..."
        lines.append(f"【来源 {i}: {title}】\n{text}")
    return "\n\n---\n\n".join(lines)


@tool
def search_web(query: str) -> str:
    """
    联网搜索最新的威胁情报信息。输入是自然语言问题或关键词。
    适用于：最近发生的安全事件（近6个月）、最新CVE漏洞详情、实时攻击活动、本
    地库中没有的新威胁。
    """
    if not TAVILY_KEY or TAVILY_KEY == "你的Tavily_API_Key":
        return "联网搜索未配置 API Key，请在 .env 中设置 TAVILY_API_KEY。"

    try:
        client = TavilyClient(api_key=TAVILY_KEY)
        response = client.search(query, max_results=3, search_depth="advanced")
    except Exception as e:
        return f"联网搜索失败: {e}"

    results = response.get("results", [])
    if not results:
        return "联网搜索未找到相关信息。"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        content = r.get("content", "")
        if len(content) > MAX_OUTPUT_CHARS:
            content = content[:MAX_OUTPUT_CHARS] + "..."
        url = r.get("url", "")
        lines.append(f"[W{i}] {title}\n{content}\n来源: {url}")

    return "\n\n".join(lines)


@tool
def query_knowledge_graph(entity: str) -> str:
    """
    在威胁情报知识图谱中查询指定实体的关联关系（支持1-hop直接关
    联和2-hop间接关联）。输入实体名称（APT组织名、恶意软件名、漏
    洞编号、资产名、公司名等）。返回多跳关系图，揭示间接攻击链和隐
    式关联。
    """
    # 1-hop 直接关系
    hits_1hop = query_graph(entity, top_k=10)
    # 2-hop 间接关系
    hits_2hop = query_graph_2hop(entity, top_k=8)

    if not hits_1hop and not hits_2hop:
        return f"知识图谱中未找到与 '{entity}' 相关的关联关系。"

    lines = [f"实体 '{entity}' 的知识图谱查询结果：\n"]

    # 1-hop
    if hits_1hop:
        lines.append("## 直接关联 (1-hop)")
        # 按关系类型分组
        by_rel = {}
        for h in hits_1hop:
            rel = h.get("predicate", "关联")
            by_rel.setdefault(rel, []).append(h)
        for rel, items in by_rel.items():
            entities_str = "、".join(
                f"{it['object']}({it.get('obj_type', '?')})"
                for it in items[:5]
            )
            lines.append(f"- {rel} → {entities_str}")
            if len(items) > 5:
                lines.append(f"  ...共 {len(items)} 个关联实体")

    # 2-hop
    if hits_2hop:
        lines.append("\n## 间接关联 (2-hop) — 揭示隐藏攻击链")
        seen = set()
        count = 0
        for h in hits_2hop:
            key = (h["source"], h["target"])
            if key in seen:
                continue
            seen.add(key)
            if count >= 8:
                lines.append("...(已截断)")
                break
            lines.append(
                f"- {h['source']} → [{h['rel1']}] → "
                f"{h['middle']}({h.get('mid_type', '?')}) → "
                f"[{h['rel2']}] → {h['target']}({h.get('target_type', '?')})"
            )
            count += 1

    return "\n".join(lines)


ALL_TOOLS = [search_vector_db, search_web, query_knowledge_graph]
