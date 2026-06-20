"""
RAG Engine - LangChain FAISS 向量检索 + 知识图谱
"""
import os
from typing import List, Dict

# ========== 线程限制（避免 CPU 线程爆炸）==========
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ========== 路径 ==========
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))                          # /data/Agent
INDEX_DIR = os.path.join(AGENT_DIR, "faiss_index")                              # FAISS 索引

TOP_K = 5
MIN_SCORE = 0.5  # 内积分数阈值，低于此值视为不相关

# ========== 嵌入模型 ==========
_embeddings = None


def _get_embeddings() -> HuggingFaceEmbeddings:#bge-large-zh-v1.5模型
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ========== FAISS (load_local 磁盘加载) ==========
_vectorstore = None


def _load_vectorstore()-> FAISS:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    _vectorstore = FAISS.load_local(#持久化加载
        INDEX_DIR,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return _vectorstore
    print(f"[RAG] FAISS loaded: {_vectorstore.index.ntotal} vectors x {_vectorstore.index.d}d")


# ========== Neo4j 知识图谱（多跳查询）==========
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "newpassword123")

_driver = None


def _get_driver():
    global _driver
    if _driver is not None:
        return _driver
    _driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, max_connection_lifetime=3600)
    _driver.verify_connectivity()
    print("[RAG] Neo4j connected")
    return _driver


def _cypher_query(cypher: str, **params):
    """执行参数化 Cypher 查询，返回记录列表"""
    driver = _get_driver()
    with driver.session(database="neo4j") as session:
        result = session.run(cypher, **params)
        return [record.data() for record in result]


# ========== API ==========

def retrieve(query: str, top_k: int = TOP_K):
    _load_vectorstore()
    qv = _get_embeddings().embed_query(query)

    docs_with_scores = _vectorstore.similarity_search_with_score_by_vector(
        qv, k=min(top_k * 3, _vectorstore.index.ntotal)
    )

    results = []
    seen = set()
    for doc, score in docs_with_scores:
        text = doc.page_content.strip()
        if text in seen:
            continue
        seen.add(text)
        if score < MIN_SCORE:
            continue
        results.append({
            "id": f"E{len(results)+1}",
            "text": text[:300],
            "para_id": doc.metadata.get("para_id", ""),
            "title": doc.metadata.get("title", ""),
            "score": float(score),
        })
        if len(results) >= top_k:
            break
    return results


def query_graph(entity: str, top_k: int = 10) -> List[Dict]:
    """1-hop 查询：返回 entity 的直接邻居（兼容旧接口）"""
    cypher = """
    MATCH (n:Entity)-[r]-(m:Entity)
    WHERE toLower(n.name) CONTAINS toLower($entity)
    RETURN n.name AS subject, type(r) AS predicate, m.name AS object,
           labels(m)[1] AS obj_type, r.confidence AS confidence
    LIMIT $top_k
    """
    rows = _cypher_query(cypher, entity=entity, top_k=top_k)
    results = []
    for row in rows:
        results.append({
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "obj_type": row.get("obj_type", ""),
            "confidence": row.get("confidence"),
        })
    return results


def query_graph_2hop(entity: str, top_k: int = 15) -> List[Dict]:
    """2-hop 查询：entity → 中间节点 → 目标节点，展示间接关联"""
    cypher = """
    MATCH (n:Entity)-[r1]-(mid:Entity)-[r2]-(target:Entity)
    WHERE toLower(n.name) CONTAINS toLower($entity)
      AND n.name < target.name
    RETURN n.name AS source, type(r1) AS rel1, mid.name AS middle,
           labels(mid)[1] AS mid_type, type(r2) AS rel2,
           target.name AS target, labels(target)[1] AS target_type
    LIMIT $top_k
    """
    rows = _cypher_query(cypher, entity=entity, top_k=top_k)
    return [
        {
            "source": r["source"],
            "rel1": r["rel1"],
            "middle": r["middle"],
            "mid_type": r.get("mid_type", ""),
            "rel2": r["rel2"],
            "target": r["target"],
            "target_type": r.get("target_type", ""),
        }
        for r in rows
    ]


def query_graph_path(source_entity: str, target_entity: str, max_hops: int = 3) -> List[Dict]:
    """最短路径查询：发现两个实体之间的攻击链路径"""
    # shortestPath 不支持参数化跳数，仅整数字面量，此处 max_hops 由调用方控制
    cypher = f"""
    MATCH (s:Entity), (t:Entity),
          p = shortestPath((s)-[*1..{int(max_hops)}]-(t))
    WHERE toLower(s.name) CONTAINS toLower($source)
      AND toLower(t.name) CONTAINS toLower($target)
    RETURN [node in nodes(p) | node.name] AS path,
           [rel in relationships(p) | type(rel)] AS relations,
           length(p) AS hops
    LIMIT 5
    """
    rows = _cypher_query(cypher, source=source_entity, target=target_entity)
    return [
        {
            "path": r["path"],
            "relations": r["relations"],
            "hops": r["hops"],
        }
        for r in rows
    ]


def query_graph_by_type(entity: str, entity_type: str = None, top_k: int = 10) -> List[Dict]:
    """按实体类型过滤的图谱查询"""
    if entity_type:
        cypher = """
        MATCH (n:Entity)-[r]-(m:Entity)
        WHERE toLower(n.name) CONTAINS toLower($entity)
          AND $type IN labels(m)
        RETURN n.name AS subject, type(r) AS predicate, m.name AS object,
               labels(m)[1] AS obj_type, r.confidence AS confidence
        LIMIT $top_k
        """
        rows = _cypher_query(cypher, entity=entity, type=entity_type, top_k=top_k)
    else:
        return query_graph(entity, top_k=top_k)

    return [
        {
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "obj_type": r.get("obj_type", ""),
            "confidence": r.get("confidence"),
        }
        for r in rows
    ]
