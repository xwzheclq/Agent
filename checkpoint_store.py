"""
LangGraph Checkpointer + 对话管理
AsyncSqliteSaver 持久化 agent 状态 + sessions/messages 表
"""
import sqlite3
import uuid
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

DB_PATH = "/data/Agent/chat_history.db"

_checkpointer = None


async def get_checkpointer() -> AsyncSqliteSaver:
    """单例：已创建则复用，否则 aiosqlite.connect + AsyncSqliteSaver"""
    global _checkpointer
    if _checkpointer is None:
        conn = await aiosqlite.connect(DB_PATH)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        _checkpointer = AsyncSqliteSaver(conn)
    return _checkpointer


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA busy_timeout=10000")
    return c


def init():
    """初始化 sessions / messages 表（checkpoints 由 AsyncSqliteSaver 自动建表）"""
    import os as _os

    # --- PID 文件锁：防止多个 Streamlit 实例同时写 DB ---
    _pid_file = DB_PATH.replace(".db", ".pid")
    _my_pid = str(_os.getpid())

    if _os.path.exists(_pid_file):
        try:
            with open(_pid_file) as f:
                _old_pid = f.read().strip()
            if _old_pid and _old_pid != _my_pid:
                try:
                    _os.kill(int(_old_pid), 0)
                    print(f"[init] 发现旧 Streamlit 实例 PID={_old_pid}，正在终止...")
                    _os.kill(int(_old_pid), 9)
                    import time
                    time.sleep(2)
                except OSError:
                    pass
        except (ValueError, OSError):
            pass

    with open(_pid_file, "w") as f:
        f.write(_my_pid)

    # SQLite 自动从 WAL 恢复，无需手动清理
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                title TEXT DEFAULT '新对话',
                rag_mode TEXT DEFAULT '自动',
                created_at TEXT DEFAULT (datetime('now','localtime')),
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT DEFAULT '',
                thinking TEXT DEFAULT '[]',
                truncated INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (thread_id) REFERENCES sessions(thread_id)
            )
        """)
        try:
            c.execute("ALTER TABLE messages ADD COLUMN truncated INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        c.commit()


# ===== Session CRUD =====

def list_sessions() -> list[dict]:
    with _conn() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        )]


def create_session(thread_id: str = None, title: str = "新对话",
                   mode: str = "自动（Agent 决定）") -> str:
    tid = thread_id or str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO sessions(thread_id,title,rag_mode) VALUES(?,?,?)",
            (tid, title, mode),
        )
        c.execute(
            "UPDATE sessions SET updated_at=datetime('now','localtime') WHERE thread_id=?",
            (tid,),
        )
        c.commit()
    return tid


def update_session(thread_id: str, title: str = None):
    with _conn() as c:
        if title:
            c.execute(
                "UPDATE sessions SET title=?, updated_at=datetime('now','localtime') WHERE thread_id=?",
                (title, thread_id),
            )
        else:
            c.execute(
                "UPDATE sessions SET updated_at=datetime('now','localtime') WHERE thread_id=?",
                (thread_id,),
            )
        c.commit()


def delete_session(thread_id: str):
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
        c.execute("DELETE FROM sessions WHERE thread_id=?", (thread_id,))
        for _t in ("checkpoints", "writes"):
            try:
                c.execute(f"DELETE FROM {_t} WHERE thread_id=?", (thread_id,))
            except sqlite3.OperationalError:
                pass
        c.commit()


# ===== Message CRUD =====

def save_message(thread_id: str, role: str, content: str,
                 thinking: list = None, truncated: bool = False):
    import json
    with _conn() as c:
        c.execute(
            "INSERT INTO messages(thread_id,role,content,thinking,truncated) VALUES(?,?,?,?,?)",
            (thread_id, role, content, json.dumps(thinking or [], ensure_ascii=False),
             int(truncated)),
        )
        c.execute(
            "UPDATE sessions SET updated_at=datetime('now','localtime') WHERE thread_id=?",
            (thread_id,),
        )
        c.commit()


def save_partial_answer(thread_id: str, content: str, thinking: list = None) -> int:
    """流式过程中保存部分回答，返回 message id 供后续 finalize_answer 更新"""
    import json
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO messages(thread_id,role,content,thinking,truncated) VALUES(?,?,?,?,1)",
            (thread_id, "assistant", content,
             json.dumps(thinking or [], ensure_ascii=False)),
        )
        c.execute(
            "UPDATE sessions SET updated_at=datetime('now','localtime') WHERE thread_id=?",
            (thread_id,),
        )
        c.commit()
        return cur.lastrowid


def finalize_answer(msg_id: int, content: str, thinking: list = None):
    """流式完成后把部分回答更新为完整版，清除截断标记"""
    import json
    with _conn() as c:
        c.execute(
            "UPDATE messages SET content=?, thinking=?, truncated=0 WHERE id=?",
            (content, json.dumps(thinking or [], ensure_ascii=False), msg_id),
        )
        c.commit()


def update_partial_answer(msg_id: int, content: str, thinking: list = None):
    """流式过程中更新部分回答内容，保持 truncated=1"""
    import json
    with _conn() as c:
        c.execute(
            "UPDATE messages SET content=?, thinking=? WHERE id=?",
            (content, json.dumps(thinking or [], ensure_ascii=False), msg_id),
        )
        c.commit()


def cleanup_partial(thread_id: str):
    """删除指定 thread 中所有截断消息（切换会话/重新生成时清理）"""
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE thread_id=? AND truncated=1", (thread_id,))
        c.commit()


def get_messages(thread_id: str) -> list[dict]:
    import json
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, thinking, truncated FROM messages WHERE thread_id=? ORDER BY id",
            (thread_id,),
        ).fetchall()
    return [
        {"role": r[0], "content": r[1], "thinking": json.loads(r[2]),
         "truncated": bool(r[3])}
        for r in rows
    ]


def get_thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}
