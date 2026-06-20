"""
混合检索：BM25（稀疏）+ FAISS（稠密）+ RRF 融合 + BGE-Reranker 精排

管道：
  1. BM25 关键词检索    → 多召回（top_k * 3）
  2. FAISS 语义检索     → 多召回（top_k * 3）
  3. RRF 融合           → 合并排序，取 top_k * 3 候选
  4. BGE-Reranker 精排  → cross-encoder 逐条打分，取 top_k
"""
from langchain_community.retrievers import BM25Retriever
from rag_engine import _load_vectorstore, _get_embeddings
from sentence_transformers import CrossEncoder


# ========== BM25 检索器（懒加载） ==========
_bm25 = None


def _get_bm25():
    """
    从 FAISS 索引里读出全部 chunk 文本，建 BM25 检索器。
    FAISS 保存时把 Document 对象序列化在 index.pkl 里，直接读出来就能用。
    """
    global _bm25
    if _bm25 is not None:
        return _bm25

    vs = _load_vectorstore()
    all_docs = list(vs.docstore._dict.values())
    _bm25 = BM25Retriever.from_documents(all_docs)
    _bm25.k = 5
    return _bm25


# ========== BGE-Reranker（懒加载） ==========
_reranker = None


def _get_reranker():
    """
    BGE-Reranker 是 cross-encoder，基于 XLM-RoBERTa（BERT 架构）。
    把 (query, document) 拼成一对输入，输出相关性分数。
    比 bi-encoder（FAISS/BGE）准，但比它慢，所以只在候选集上跑。
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
    return _reranker


# ========== RRF 融合 ==========

def _rrf_fusion(results_a: list, results_b: list, k: int = 60) -> list:
    """
    RRF 融合两个排序好的检索结果。

    score = 1/(k+rank_a) + 1/(k+rank_b)

    谁在两个列表里都排前面，谁的 RRF 分就高。
    """
    scores = {}

    for rank, (text, _) in enumerate(results_a, start=1):
        scores[text] = scores.get(text, 0) + 1.0 / (k + rank)

    for rank, (text, _) in enumerate(results_b, start=1):
        scores[text] = scores.get(text, 0) + 1.0 / (k + rank)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [text for text, _ in sorted_items]


# ========== 混合检索 API ==========

def hybrid_search(query: str, top_k: int = 5) -> list:
    """
    BM25 + FAISS → RRF 融合 → BGE-Reranker 精排 → top_k 结果。

    返回格式（和原来的 rag_engine.retrieve 一致）：
      [{"id": "E1", "text": "...", "title": "...", "score": reranker_score}, ...]
    """
    # 1. 问题向量化（给 FAISS 用）
    qv = _get_embeddings().embed_query(query)

    # 2. BM25 关键词检索
    bm25 = _get_bm25()
    bm25.k = top_k * 3
    bm25_docs = bm25.invoke(query)
    bm25_pairs = [(doc.page_content.strip(), 0.0) for doc in bm25_docs]

    # 3. FAISS 语义检索
    vs = _load_vectorstore()
    faiss_results = vs.similarity_search_with_score_by_vector(qv, k=top_k * 3)
    faiss_pairs = [(doc.page_content.strip(), score) for doc, score in faiss_results]

    # 4. RRF 融合 → 候选集（多保留一些给 reranker）
    candidate_texts = _rrf_fusion(bm25_pairs, faiss_pairs)
    candidates = candidate_texts[: top_k * 3]

    # 5. BGE-Reranker 精排
    reranker = _get_reranker()
    pairs = [(query, text) for text in candidates]
    scores = reranker.predict(pairs)  # 返回 list[float]，越高分越相关

    # (text, score) 按分数降序
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # 6. 构建返回结果（从原始文档取元数据）
    all_docs = {doc.page_content.strip(): doc for doc in vs.docstore._dict.values()}

    results = []
    seen = set()
    for text, score in ranked:
        if text in seen:
            continue
        seen.add(text)

        doc = all_docs.get(text)
        title = doc.metadata.get("title", "") if doc else ""

        results.append({
            "id": f"E{len(results) + 1}",
            "text": text[:300],
            "title": title,
            "score": float(score),
        })

        if len(results) >= top_k:
            break

    return results
