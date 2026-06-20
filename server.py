"""
FastAPI + WebSocket 流式服务
封装 agent_core / multi_agent，不改动任何现有代码
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json

import checkpoint_store as storage
from checkpoint_store import get_checkpointer, get_thread_config


# ---- 生命周期 ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init()
    storage._checkpointer = None
    import voice
    voice.preload_model()
    yield

app = FastAPI(title="威胁情报分析 Agent API", lifespan=lifespan)

# ---- 懒加载 agent ----

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


# ==================== REST API ====================


@app.get("/api/sessions")
async def list_sessions():
    return storage.list_sessions()


@app.post("/api/sessions")
async def create_session(data: dict | None = None):
    title = data.get("title", "新对话") if data else "新对话"
    mode = data.get("mode", "auto") if data else "auto"
    tid = storage.create_session(title=title, mode=mode)
    return {"thread_id": tid, "title": title, "mode": mode}


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str):
    storage.delete_session(thread_id)
    return {"ok": True}


@app.get("/api/sessions/{thread_id}/messages")
async def get_messages(thread_id: str):
    return storage.get_messages(thread_id)


# ==================== 知识图谱可视化 API ====================

@app.get("/api/graph")
async def get_graph(entity: str, mode: str = "expand", hops: int = 2):
    """
    两级交互：
      mode=search → 模糊搜索匹配的实体（只返回节点，无连线）
      mode=expand → 展开指定实体的完整关系图（支持 hops=1/2/3）
    """
    try:
        from rag_engine import _cypher_query
    except Exception:
        return {"nodes": [], "edges": []}

    # 加载关系中文名映射
    rel_map = {}
    rel_file = os.path.join(os.path.dirname(__file__), "kgc_dataset", "relation2id.txt")
    try:
        with open(rel_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    rel_map[parts[0]] = parts[2]
    except Exception:
        pass

    groups = {"APT_Group": "apt", "Attack_Event": "event",
              "Malware_Tool": "malware", "Asset": "asset",
              "Company": "company", "Country": "country",
              "Attack_Phase": "phase"}

    # 排除泛化名称节点（容易与搜索前缀混淆）
    EXCLUDED = {"APT", "APT攻击", "APT组织信息"}

    if mode == "search":
        # 模糊搜索：返回匹配的实体列表（节点，无连线），不区分大小写
        try:
            rows = _cypher_query("""
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($entity)
                  AND NOT n.name IN $excluded
                RETURN n.name AS name, n.type AS type
                ORDER BY n.type, n.name
                LIMIT 50
            """, entity=entity, excluded=list(EXCLUDED))
        except Exception:
            return {"nodes": [], "edges": []}

        nodes = [{"id": r["name"], "label": r["name"],
                  "group": groups.get(r["type"], "other"),
                  "title": f"{r['name']} ({r['type']})"}
                 for r in rows]
        return {"nodes": nodes, "edges": [], "mode": "search"}

    # mode == "expand": 1~3 hop 展开，保留真实边方向
    nodes = []
    edges_set = set()
    node_ids = set()

    def add_node(nid: str, label: str, etype: str):
        if nid not in node_ids:
            node_ids.add(nid)
            nodes.append({
                "id": nid, "label": label,
                "group": groups.get(etype, "other"),
                "title": f"{label} ({etype})",
            })

    def add_rows(rows):
        for row in rows:
            src = row["src"]; stype = row["stype"]
            rel = row["rel"]
            dst = row["dst"]; dtype = row["dtype"]
            add_node(src, src, stype)
            add_node(dst, dst, dtype)
            edge_key = f"{src}|{rel}|{dst}"
            if edge_key not in edges_set:
                edges_set.add(edge_key)

    # 1-hop：精确匹配（不区分大小写），若无精确匹配则整个 expand 返回空
    try:
        rows1 = _cypher_query("""
            MATCH (center:Entity)-[r]-(neighbor:Entity)
            WHERE toLower(center.name) = toLower($entity)
              AND NOT center.name IN $excluded
              AND NOT neighbor.name IN $excluded
            RETURN startNode(r).name AS src, startNode(r).type AS stype,
                   type(r) AS rel,
                   endNode(r).name AS dst, endNode(r).type AS dtype
            LIMIT 80
        """, entity=entity, excluded=list(EXCLUDED))
    except Exception:
        return {"nodes": [], "edges": []}

    if not rows1:
        return {"nodes": [], "edges": [], "mode": "expand"}

    add_rows(rows1)

    if hops >= 2:
        # 2-hop：邻居之间的关系（揭示隐藏关联），不区分大小写
        try:
            rows2 = _cypher_query("""
                MATCH (center:Entity)-[]-(mid:Entity)-[r]-(other:Entity)
                WHERE toLower(center.name) = toLower($entity)
                  AND other.name <> center.name
                  AND NOT mid.name IN $excluded
                  AND NOT other.name IN $excluded
                RETURN startNode(r).name AS src, startNode(r).type AS stype,
                       type(r) AS rel,
                       endNode(r).name AS dst, endNode(r).type AS dtype
                LIMIT 120
            """, entity=entity, excluded=list(EXCLUDED))
        except Exception:
            rows2 = []
        if rows2:
            add_rows(rows2)

    if hops >= 3:
        # 3-hop：更远一跳的关系网
        try:
            rows3 = _cypher_query("""
                MATCH (center:Entity)-[]-(mid:Entity)-[]-(other:Entity)-[r]-(third:Entity)
                WHERE toLower(center.name) = toLower($entity)
                  AND third.name <> center.name
                  AND NOT other.name IN $excluded
                  AND NOT third.name IN $excluded
                RETURN startNode(r).name AS src, startNode(r).type AS stype,
                       type(r) AS rel,
                       endNode(r).name AS dst, endNode(r).type AS dtype
                LIMIT 200
            """, entity=entity, excluded=list(EXCLUDED))
        except Exception:
            rows3 = []
        if rows3:
            add_rows(rows3)

    edges = [{"from": e.split("|")[0], "to": e.split("|")[2],
              "label": rel_map.get(e.split("|")[1], e.split("|")[1])}
             for e in edges_set]

    return {"nodes": nodes, "edges": edges, "mode": "expand"}


# ==================== WebSocket 流式对话 ====================


@app.websocket("/ws/chat/{thread_id}")
async def chat_websocket(ws: WebSocket, thread_id: str):
    await ws.accept()

    # 重置 checkpointer 单例（新 event loop）
    storage._checkpointer = None

    async def send_event(event: dict):
        await ws.send_text(json.dumps(event, ensure_ascii=False))

    try:
        while True:
            data = await ws.receive_json()

            if data.get("type") == "chat":
                query = data["query"]
                mode = data.get("mode", "auto")

                # 更新标题
                sess = _get_session(thread_id)
                if sess and sess.get("title") == "新对话":
                    storage.update_session(thread_id, title=query[:30])

                # 保存用户消息
                storage.save_message(thread_id, "user", query)

                # 取历史消息
                history_messages = storage.get_messages(thread_id)
                history_for_agent = [m for m in history_messages[:-1]]

                # 选择运行模式
                if mode == "off":
                    stream = _get_run_off()(history_for_agent, query)
                elif mode == "multi":
                    stream = _get_run_multi_agent()(query, thread_id=thread_id)
                else:
                    stream = _get_run_auto()(history_for_agent, query, thread_id=thread_id)

                # 流式输出
                storage.cleanup_partial(thread_id)
                partial_id = storage.save_partial_answer(thread_id, "", [])

                thinking_lines = []
                answer = ""
                truncated = False
                error = None

                try:
                    async for event in stream:
                        if event["type"] == "thinking":
                            new_lines = event["lines"]
                            if len(new_lines) >= len(thinking_lines):
                                thinking_lines = new_lines
                            else:
                                thinking_lines = list(thinking_lines) + [
                                    l for l in new_lines if l not in thinking_lines
                                ]
                            await send_event(event)
                            storage.update_partial_answer(partial_id, answer, thinking_lines)

                        elif event["type"] == "token":
                            answer += event["text"]
                            await send_event(event)

                        elif event["type"] == "phase":
                            await send_event(event)

                        elif event["type"] == "done":
                            answer = event["answer"]
                            done_thinking = event.get("thinking", [])
                            if len(done_thinking) > len(thinking_lines):
                                thinking_lines = done_thinking
                            elif done_thinking:
                                thinking_lines = list(thinking_lines) + [
                                    l for l in done_thinking if l not in thinking_lines
                                ]
                            truncated = event.get("truncated", False)
                            break

                        elif event["type"] == "error":
                            error = event.get("message", "未知错误")
                            await send_event(event)
                            break

                except Exception as e:
                    import traceback
                    error = str(e)
                    try:
                        await send_event({"type": "error", "message": error, "traceback": traceback.format_exc()})
                    except Exception:
                        pass
                finally:
                    # Properly close generator to cancel any ongoing LLM/tool calls
                    if hasattr(stream, 'aclose'):
                        try:
                            await stream.aclose()
                        except Exception:
                            pass

                # 后处理
                if not answer:
                    answer = "*(未生成回答)*"
                if truncated:
                    answer += "\n\n> ⚠️ 回答达到长度上限，可能不完整"
                if error:
                    answer += "\n\n> ⚠️ 输出被中断"

                if not error and not truncated and mode != "multi":
                    answer += "\n\n✅ 输出完成"

                # 持久化
                storage.finalize_answer(partial_id, answer, thinking_lines)
                try:
                    await send_event({"type": "done", "answer": answer, "thinking": thinking_lines, "truncated": truncated})
                except Exception:
                    pass

            elif data.get("type") == "voice":
                # 语音识别
                import voice
                audio_b64 = data["audio"]
                audio_bytes = __import__("base64").b64decode(audio_b64)
                text = voice.transcribe(audio_bytes)
                if text:
                    await send_event({"type": "voice_result", "text": text})
                else:
                    await send_event({"type": "voice_error", "message": "语音识别失败"})

            elif data.get("type") == "ping":
                await send_event({"type": "pong"})

    except WebSocketDisconnect:
        pass


def _get_session(thread_id: str) -> dict | None:
    sessions = storage.list_sessions()
    for s in sessions:
        if s["thread_id"] == thread_id:
            return s
    return None


# ==================== 静态文件 ====================

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
