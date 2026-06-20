"""
构建 RAG 向量索引（一次性脚本）

流程：原始段落 → RecursiveCharacterTextSplitter 分块 → BGE 嵌入 → FAISS → 保存到磁盘
"""
import os
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ========== 路径 ==========
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))          # /data/Agent
PARA_PATH = "/data/APTdata/get_data/all_paragraphs3_test.json"  # 原始段落
INDEX_DIR = os.path.join(AGENT_DIR, "faiss_index")              # 索引保存位置

# ========== 参数 ==========
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def main():
    # 1. 加载原始段落
    print(f"[1/4] 加载原始段落: {PARA_PATH}")
    with open(PARA_PATH, encoding="utf-8") as f:
        paragraphs = json.load(f)
    print(f"      共 {len(paragraphs)} 段")

    # 2. 转为 LangChain Document
    docs = []
    for p in paragraphs:
        docs.append(Document(
            page_content=p["text"],
            metadata={
                "title": p["metadata"]["title"],
                "para_id": p["id"],
            },
        ))

    # 3. RecursiveCharacterTextSplitter 分块
    print(f"[2/4] RecursiveCharacterTextSplitter 分块 (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "，", ",", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"      {len(docs)} 段落 → {len(chunks)} chunks")

    # 4. BGE 嵌入 + 构建 FAISS
    print(f"[3/4] BGE 嵌入 + FAISS 索引 (BAAI/bge-large-zh-v1.5)")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    print(f"      {vectorstore.index.ntotal} vectors x {vectorstore.index.d}d")

    # 5. 保存到磁盘
    print(f"[4/4] 保存到: {INDEX_DIR}")
    vectorstore.save_local(INDEX_DIR)
    print("      完成。")


if __name__ == "__main__":
    main()
