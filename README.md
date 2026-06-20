# 基于多Agent交叉验证的可信威胁情报分析系统

面向"人工智能系统安全与可信AI技术"主题，以威胁情报分析为验证场景，设计并实现基于 **LangGraph 多 Agent 交叉验证**的可信 AI 系统。核心思路：不再让单一模型自查自答，而是让三个拥有不同知识获取路径的 Agent 互相验证对方的输出。

## 核心指标（200题 × 3模式评测）

| 指标 | 纯LLM | 单Agent ReAct | 多Agent交叉验证（本方案） | 改善 |
|------|:---:|:---:|:---:|:---:|
| 幻觉率 | 34% | 24% | **14%** | ↓ 60% |
| 知识锚覆盖率 | 37% | 66% | **98%** | ↑ 165% |
| 不确定性表达率 | 35% | 53% | **54%** | ↑ 54% |
| 交叉纠正率 | — | — | **72% (71/98)** | — |

## 系统架构
<a href="https://raw.githubusercontent.com/xwzheclq/Agent/main/v/demo.mp4">
  <img width="1247" height="683" alt="1" src="https://github.com/user-attachments/assets/4afd9240-4d8c-4a19-a44c-10adbda064cc" />
</a>


用户输入
  │
  ▼
┌──────────────────────┐
│  FastAPI Server      │  WebSocket 实时流式 + REST API
│  三模式路由分发       │
├──────────────────────┤
│ off  │ auto │ multi  │  三种模式可对同一问题并行运行、实时对比
└──┬─────┬──────┬──────┘
   │     │      │
   ▼     ▼      ▼
 纯LLM  单Agent  多Agent交叉验证
        ReAct    ┌─ researcher（FAISS + Neo4j）
                ├─ web_search（Tavily 联网验证）
                └─ synthesizer（三段式对比输出）
                      │
         ┌────────────┴───────────────────────────────────┐
         │        RAG 基础设施                            │
         ├──────────────┬───────────────┬────────────────┤
         │ FAISS 向量库 │ Neo4j 知识图谱 │ Tavily 联网搜索 │
         │ BM25+RRF     │ 1-3跳实体查询  │ 并行3 query    │
         │ +BGE-Reranker│               │                │
         └──────────────┴───────────────┴────────────────┘
```

## 技术栈

| 组件 | 技术选型 |
|------|------|
| 大语言模型 | Qwen3.6-27B-FP8 (vLLM 推理) |
| Agent 框架 | LangGraph StateGraph（手写，非 create_agent 黑盒） |
| 向量检索 | FAISS + BGE-large-zh-v1.5 (1024维) |
| 混合检索管道 | BM25 + FAISS → RRF 融合 → BGE-Reranker 精排 |
| 知识图谱 | Neo4j (200+节点, Cypher 1-3跳查询) |
| 联网搜索 | Tavily Search API (asyncio 并行3 query) |
| 流式输出 | asyncio.Queue + ContextVar（绕过 checkpoint 序列化） |
| 持久化 | SQLite (AsyncSqliteSaver + sessions/messages 表) |
| 评测框架 | LLM-as-Judge 五维评分 + 3模式并行评测 |
| Web 服务 | FastAPI + WebSocket |
| 前端 | HTML + vis.js（知识图谱可视化） |

## 文件结构

```
├── server.py              # FastAPI 服务入口，WebSocket + REST 路由
├── multi_agent.py          # 多Agent交叉验证引擎（手写 StateGraph, 5节点）
├── agent_core.py           # 纯LLM模式 + 单Agent ReAct模式
├── tools.py                # 三个检索工具（向量库/知识图谱/联网搜索）
├── rag_engine.py           # FAISS 向量检索 + Neo4j Cypher 查询
├── hybrid_retriever.py     # 混合检索管道（BM25+FAISS+RRF+Reranker）
├── checkpoint_store.py     # SQLite 持久化（sessions/messages/checkpoints）
├── context_manager.py      # 上下文压缩（token估算 + LLM摘要）
├── eval_testset.py         # 评测流水线（3模式并行 + LLM-as-Judge + 自动报告）
├── build_testset.py        # 200题测试集生成器
├── build_rag_index.py      # FAISS 索引构建
├── neo4j_import.py         # Neo4j 知识图谱数据导入
├── finetune_bge.py         # BGE 嵌入模型微调
├── app.py                  # Streamlit UI（备选前端）
├── voice.py                # 语音输入模块
├── hotwords.txt            # 语音热词
├── requirements.txt        # Python 依赖
├── test_set.json           # 200题测试集（6类别）
└── demo_showcase.md         # 演示文档（例题+评测数据）
```

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env` 并填入实际值：

```bash
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=your_key
LLM_MODEL=/path/to/model
LLM_TEMPERATURE=0.7
TAVILY_API_KEY=your_tavily_key
```

### 3. 构建知识库

```bash
python build_rag_index.py      # 构建 FAISS 向量索引
python neo4j_import.py         # 导入 Neo4j 知识图谱
```

### 4. 启动服务

```bash
python server.py               # 启动 Web 服务（端口 8501）
```

### 5. 运行评测

```bash
python eval_testset.py --modes off,auto,multi --limit 200
```

## 评测方案

测试集覆盖 6 类 200 题，每道题含标准答案、易错陷阱和期望关键信息点：

| 类别 | 题目数 | 测试目标 |
|------|:---:|------|
| 简单事实 | 40 | 基础检索与事实准确性 |
| 多源交叉 | 50 | 多源信息综合与一致性判断 |
| 易混淆 | 41 | 相近实体区分（相近APT编号/工具名） |
| 时间线 | 25 | 攻击事件时间线精确性 |
| 归属争议 | 25 | 有争议的APT归属判断 |
| 对抗陷阱 | 20 | 嵌入虚假实体的反幻觉能力 |

## 典型交叉纠正案例

**T0034：2021年7月全球勒索攻击利用哪家厂商漏洞？**

| 模式 | 回答 |
|------|------|
| 纯LLM | "利用了Citrix ADC的CVE-2019-19781漏洞……"（编造） |
| 单Agent | "通常指WannaCry……或微软Exchange攻击……"（混淆） |
| **多Agent** | "**Kaseya VSA平台漏洞**，攻击者REvil，赎金7000万美元 [来源: W2, W3]" ✓ |

**T0007：NotPetya攻击起始于哪个国家？**

| 模式 | 回答 |
|------|------|
| 纯LLM | "起始于乌克兰……"（正确但无来源） |
| 单Agent | "Sandworm（也称Fancy Bear或APT28）开发……"（**三个不同组织混淆**） |
| **多Agent** | "俄罗斯Sandworm开发，通过乌克兰M.E.Doc传播。来源：《RansomBoggs》+ Web搜索" ✓ |

**T0182（对抗陷阱）：APT999使用哪些恶意软件？**

纯LLM和单Agent均正确告知"不存在APT999"，多Agent在此基础上提供三重确认（向量库×知识图谱×联网搜索），可信度最高。

---

 演示：`demo_showcase.md` | 测试集：`test_set.json`*
