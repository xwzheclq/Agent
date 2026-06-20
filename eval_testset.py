"""
评测脚本：对比三种模式在 200 条测试集上的可信度表现。

三种模式：
  off   — 纯LLM (run_off)
  auto  — 单Agent ReAct (run_auto)
  multi — 多Agent交叉验证 (run_multi，需server运行)

评测指标（LLM-as-judge）：
  1. accuracy (1-5): 与标准答案的一致性
  2. hallucinated (bool): 是否包含编造信息
  3. knowledge_anchored (bool): 是否有知识来源引用
  4. uncertainty_expressed (bool): 是否在不确时表达不确定性
  5. completeness (1-5): 关键信息覆盖度

汇总指标：
  - 幻觉率 = hallucinated / total
  - 准确率均值 = avg(accuracy)
  - 知识锚覆盖率 = knowledge_anchored / total
  - 不确定性表达率 = uncertainty_expressed / total
  - 交叉纠正率 = (auto误但multi对的题目数) / (auto误的题目数)

用法：
  python eval_testset.py                    # 跑全部
  python eval_testset.py --limit 20         # 先跑20条验证
  python eval_testset.py --modes off,auto   # 只跑特定模式
"""

import asyncio, json, os, sys, time, re
from typing import AsyncGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ===== LLM配置（评测用） =====
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))


def make_llm(streaming=False, temperature=0.1, max_tokens=2048):
    return ChatOpenAI(
        model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
        temperature=temperature, streaming=streaming, max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ===== 评测 Judge Prompt =====
JUDGE_SYSTEM = """你是威胁情报分析评测专家。你的任务是对AI助手的回答进行客观评分。

评分标准：
- accuracy (1-5): 与标准答案的事实一致性。5=完全一致，4=基本一致但有少量遗漏，3=部分正确有遗漏，2=大部分错误，1=完全错误
- hallucinated (true/false): 是否编造了不存在的信息（如捏造的APT编号/事件/恶意软件名/统计数据）。只要有1处编造就标true
- knowledge_anchored (true/false): 回答是否引用了具体知识来源（如文档名/URL/报告名）。泛泛而谈无来源标记=无锚
- uncertainty_expressed (true/false): 对于不确定或有争议的信息，是否明确表达了不确定性（如"待验证""据称""可能"等）
- completeness (1-5): 关键信息点的覆盖比例。5=覆盖所有得分点，3=覆盖一半，1=几乎无覆盖
- key_points_hit: 命中的关键信息点列表（从提供的key_points中提取）

输出必须是严格的JSON，不要有任何其他文本："""

JUDGE_FORMAT = """{
  "accuracy": 数字1-5,
  "hallucinated": true或false,
  "hallucination_detail": "如果hallucinated为true，具体写出编造了什么；否则写'无'",
  "knowledge_anchored": true或false,
  "uncertainty_expressed": true或false,
  "completeness": 数字1-5,
  "key_points_hit": ["命中的关键点1", "命中的关键点2"],
  "brief_reason": "一句话总结评分理由，20字以内"
}"""


async def judge_answer(question: str, ground_truth: str, answer: str,
                        key_points: list, trap: str, llm) -> dict:
    """用LLM-as-judge评测单个回答"""
    prompt = f"""【用户问题】
{question}

【标准答案】
{ground_truth}

【易错陷阱】
{trap}

【期望包含的关键信息点】
{key_points if key_points else "（无特定关键点要求）"}

【AI助手的回答】
{answer[:3000]}

请评测以上回答，输出严格JSON。"""

    msgs = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=prompt + "\n" + JUDGE_FORMAT),
    ]

    for attempt in range(3):
        try:
            resp = await llm.ainvoke(msgs)
            text = resp.content.strip()
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                result = json.loads(json_match.group(0))
                # Validate required fields
                for field in ["accuracy", "hallucinated", "knowledge_anchored",
                              "uncertainty_expressed", "completeness"]:
                    if field not in result:
                        result[field] = False if field in ("hallucinated", "knowledge_anchored", "uncertainty_expressed") else 3
                return result
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2)
                continue
            return {
                "accuracy": 3, "hallucinated": False, "knowledge_anchored": False,
                "uncertainty_expressed": False, "completeness": 3,
                "key_points_hit": [], "brief_reason": f"Judge error: {str(e)[:80]}",
                "judge_error": str(e),
            }
    return {}


# ===== 运行三种模式 =====

async def run_offline(text: str) -> str:
    """纯LLM模式：直接从agent_core调用"""
    from agent_core import run_off as _off
    answer = ""
    async for event in _off([], text):
        if event["type"] == "token":
            answer += event["text"]
        elif event["type"] == "done":
            if event.get("answer"):
                answer = event["answer"]
    return answer


async def run_auto_mode(text: str) -> str:
    """单Agent ReAct模式：直接从agent_core调用"""
    from agent_core import run_auto as _auto
    answer = ""
    async for event in _auto([], text, thread_id=None):
        if event["type"] == "token":
            answer += event["text"]
        elif event["type"] == "done":
            if event.get("answer"):
                answer = event["answer"]
    return answer


async def run_multi_mode(text: str, server_host="localhost:8501") -> str:
    """多Agent模式：通过WebSocket调用（需要server运行）"""
    import websockets
    import urllib.request

    # Create a session
    req_data = json.dumps({"title": "eval-test", "mode": "multi"}).encode()
    req = urllib.request.Request(
        f"http://{server_host}/api/sessions",
        method="POST", data=req_data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    session = json.loads(resp.read())
    tid = session["thread_id"]

    answer = ""
    try:
        async with websockets.connect(f"ws://{server_host}/ws/chat/{tid}") as ws:
            await ws.send(json.dumps({"type": "chat", "query": text, "mode": "multi"}))
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                event = json.loads(msg)
                if event["type"] == "token":
                    answer += event["text"]
                elif event["type"] == "done":
                    if event.get("answer"):
                        answer = event["answer"]
                    break
                elif event["type"] == "error":
                    break
    except Exception as e:
        answer = f"[MULTI ERROR: {str(e)[:200]}]"

    # Cleanup
    try:
        req2 = urllib.request.Request(
            f"http://{server_host}/api/sessions/{tid}",
            method="DELETE",
        )
        urllib.request.urlopen(req2)
    except Exception:
        pass

    return answer


# ===== 主评测流程 =====

async def evaluate_one_mode(mode: str, question: str, qid: str, server_host: str) -> dict:
    """单个模式的回答"""
    t0 = time.time()
    try:
        if mode == "off":
            answer = await run_offline(question)
        elif mode == "auto":
            answer = await run_auto_mode(question)
        elif mode == "multi":
            answer = await run_multi_mode(question, server_host)
        else:
            answer = f"Unknown mode: {mode}"
    except Exception as e:
        return {"mode": mode, "answer": f"[EXCEPTION: {str(e)[:200]}]", "error": True,
                "elapsed": time.time() - t0}
    return {"mode": mode, "answer": answer,
            "error": not answer or answer.startswith("[MULTI ERROR"),
            "elapsed": time.time() - t0}


# ===== 主评测流程 =====

async def evaluate_all(test_set: list, modes: list[str], limit: int = None,
                        server_host="localhost:8501"):
    """主评测循环 — 并行版本"""
    judge_llm = make_llm(streaming=False, temperature=0.1, max_tokens=512)
    if limit:
        test_set = test_set[:limit]

    results = []
    stats = {mode: {
        "accuracy_sum": 0, "hallucinated_count": 0,
        "knowledge_anchored_count": 0, "uncertainty_count": 0,
        "completeness_sum": 0, "total": 0, "errors": 0,
    } for mode in modes}

    sem = asyncio.Semaphore(3)  # 从2提升到3，多撑一点vLLM吞吐
    start_time = time.time()
    completed = 0
    lock = asyncio.Lock()
    all_results_flat = []  # 共享结果列表，供 autosave 用
    autosave_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "eval_results", "autosave.json")

    async def eval_one_question(item: dict):
        nonlocal completed
        async with sem:
            qid = item["id"]
            category = item.get("category", "未知")
            question = item["question"]
            ground = item["ground_truth"]
            key_pts = item.get("key_points", [])
            trap = item.get("trap", "")

            # 并行跑所有模式
            tasks = [evaluate_one_mode(m, question, qid, server_host) for m in modes]
            mode_results = await asyncio.gather(*tasks)

            item_results = []
            for mr in mode_results:
                mode = mr["mode"]
                answer = mr["answer"]
                elapsed = mr["elapsed"]
                error = mr["error"]

                if error:
                    stats[mode]["errors"] += 1
                    item_results.append({
                        "id": qid, "category": category, "mode": mode, "question": question,
                        "answer": answer, "error": True, "elapsed": elapsed,
                        "judge": None,
                    })
                    continue

                # Judge
                judge = await judge_answer(question, ground, answer, key_pts, trap, judge_llm)

                stats[mode]["total"] += 1
                stats[mode]["accuracy_sum"] += judge.get("accuracy", 3)
                stats[mode]["completeness_sum"] += judge.get("completeness", 3)
                if judge.get("hallucinated"):
                    stats[mode]["hallucinated_count"] += 1
                if judge.get("knowledge_anchored"):
                    stats[mode]["knowledge_anchored_count"] += 1
                if judge.get("uncertainty_expressed"):
                    stats[mode]["uncertainty_count"] += 1

                item_results.append({
                    "id": qid, "category": category, "mode": mode, "question": question,
                    "ground_truth": ground, "key_points": key_pts, "trap": trap,
                    "answer": answer[:3000], "elapsed": elapsed,
                    "judge": judge,
                })

            completed += 1
            # Progress line
            accs = "/".join(f"{r.get('judge',{}).get('accuracy','?')}" for r in item_results if r.get("judge"))
            halls = "/".join(str(r.get('judge',{}).get('hallucinated','?'))[0] for r in item_results if r.get("judge"))
            times = "/".join(f"{r.get('elapsed',0):.0f}s" for r in item_results)
            print(f"[{completed}/{len(test_set)}] {qid} [{','.join(modes)}] acc={accs} hall={halls} t={times}", flush=True)

            # Autosave every 10 items
            if completed % 10 == 0:
                async with lock:
                    summary_preview = {
                        "completed": completed,
                        "total": len(test_set),
                        "modes": modes,
                        "elapsed": time.time() - start_time,
                    }
                    os.makedirs(os.path.dirname(autosave_path), exist_ok=True)
                    with open(autosave_path, "w", encoding="utf-8") as f:
                        json.dump({"results": list(all_results_flat),
                                   "summary_preview": summary_preview},
                                  f, ensure_ascii=False, indent=2)
                    print(f"  💾 autosave: {completed}/{len(test_set)}", flush=True)

            all_results_flat.extend(item_results)
            return item_results

    # 并行处理所有题目
    all_item_results = await asyncio.gather(*[eval_one_question(item) for item in test_set])

    # 展平结果
    for item_results in all_item_results:
        results.extend(item_results)

    total_elapsed = time.time() - start_time

    # 计算交叉纠正率
    cross_corrected = 0
    auto_errors = 0
    for item in test_set:
        qid = item["id"]
        auto_result = next((r for r in results if r["id"] == qid and r["mode"] == "auto" and not r.get("error")), None)
        multi_result = next((r for r in results if r["id"] == qid and r["mode"] == "multi" and not r.get("error")), None)
        if auto_result and multi_result:
            auto_judge = auto_result.get("judge") or {}
            multi_judge = multi_result.get("judge") or {}
            auto_hall = auto_judge.get("hallucinated", False)
            multi_hall = multi_judge.get("hallucinated", False)
            auto_acc = auto_judge.get("accuracy", 3)
            multi_acc = multi_judge.get("accuracy", 3)
            if auto_hall or auto_acc < 3:
                auto_errors += 1
                if not multi_hall and multi_acc >= auto_acc:
                    cross_corrected += 1

    # 生成摘要
    summary = {
        "test_config": {"total_items": len(test_set), "modes": modes, "limit": limit},
        "elapsed_seconds": round(total_elapsed, 1),
        "cross_correction_rate": round(cross_corrected / max(auto_errors, 1), 3),
        "cross_correction_detail": f"{cross_corrected}/{auto_errors}",
        "modes": {},
    }

    for mode in modes:
        s = stats[mode]
        n = max(s["total"], 1)
        summary["modes"][mode] = {
            "total_answered": s["total"],
            "errors": s["errors"],
            "avg_accuracy": round(s["accuracy_sum"] / n, 2),
            "hallucination_rate": round(s["hallucinated_count"] / n, 3),
            "knowledge_anchor_rate": round(s["knowledge_anchored_count"] / n, 3),
            "uncertainty_rate": round(s["uncertainty_count"] / n, 3),
            "avg_completeness": round(s["completeness_sum"] / n, 2),
        }

    return {"results": results, "summary": summary}


# ===== 报告生成 =====

def generate_markdown_report(summary: dict, results: list, output_path: str):
    """生成可读的Markdown评测报告"""
    lines = []
    lines.append("# 威胁情报Agent可信度评测报告\n")
    lines.append(f"> 评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 测试集条目: {summary['test_config']['total_items']}")
    lines.append(f"> 评测模式: {', '.join(summary['test_config']['modes'])}")
    lines.append(f"> 总耗时: {summary['elapsed_seconds']:.0f}s\n")

    lines.append("## 1. 核心指标对比\n")

    run_modes = summary["test_config"]["modes"]
    mode_labels = {"off": "纯LLM", "auto": "单Agent", "multi": "多Agent"}
    header = "| 指标 |"
    sep = "|------|"
    for m in run_modes:
        header += f" {mode_labels.get(m, m)} |"
        sep += ":---:|"
    if len(run_modes) >= 2:
        header += " 提升 |"
        sep += ":---:|"
    lines.append(header)
    lines.append(sep)

    def get_vals(key):
        return [summary["modes"][m].get(key, 0) if m in summary["modes"] else 0 for m in run_modes]

    metrics = [
        ("幻觉率 ↓", "hallucination_rate", "{:.0%}"),
        ("准确率 ↑", "avg_accuracy", "{:.2f}"),
        ("知识锚覆盖率 ↑", "knowledge_anchor_rate", "{:.0%}"),
        ("不确定性表达率 ↑", "uncertainty_rate", "{:.0%}"),
        ("完整性 ↑", "avg_completeness", "{:.2f}"),
    ]

    for name, key, fmt in metrics:
        vals = get_vals(key)
        if len(run_modes) >= 2:
            if "↓" in name:
                improvement = f"-{(vals[0] - vals[-1]) / max(vals[0], 0.01):.0%}" if vals[0] > 0 else "N/A"
            else:
                improvement = f"+{(vals[-1] - vals[0]) / max(vals[0], 0.01):.0%}" if vals[0] > 0 else "N/A"
        vals_fmt = [fmt.format(v) for v in vals]
        row = f"| {name} | " + " | ".join(vals_fmt)
        if len(run_modes) >= 2:
            row += f" | {improvement} |"
        lines.append(row)

    lines.append(f"\n**交叉纠正率**: {summary['cross_correction_rate']:.0%} ({summary['cross_correction_detail']})")
    lines.append("  - 单Agent出错的题目中，多Agent交叉验证纠正的比例\n")

    lines.append("## 2. 按类别分组的幻觉率\n")
    cat_header = "| 类别 |"
    cat_sep = "|------|"
    for m in run_modes:
        cat_header += f" {mode_labels.get(m, m)} |"
        cat_sep += ":---:|"
    lines.append(cat_header)
    lines.append(cat_sep)

    cats = {}
    for r in results:
        item = next((t for t in results if t["id"] == r["id"] and t["mode"] == r["mode"]), r)
        # Actually let's get category from test_set
        pass

    # Category breakdown requires the test_set keys, which we store in results
    cat_stats = {}
    for r in results:
        if r.get("error"):
            continue
        cat = r.get("category", "未知")
        mode = r["mode"]
        if cat not in cat_stats:
            cat_stats[cat] = {}
        if mode not in cat_stats[cat]:
            cat_stats[cat][mode] = {"total": 0, "hall": 0}
        cat_stats[cat][mode]["total"] += 1
        judge = r.get("judge") or {}
        if judge.get("hallucinated"):
            cat_stats[cat][mode]["hall"] += 1

    for cat in sorted(cat_stats):
        row = f"| {cat} |"
        for mode in run_modes:
            s = cat_stats[cat].get(mode, {"total": 0, "hall": 0})
            rate = f"{s['hall'] / max(s['total'], 1):.0%}"
            row += f" {rate} |"
        lines.append(row)

    lines.append("\n## 3. 典型交叉纠正案例\n")
    corrected = []
    for r in results:
        if r.get("error") or r["mode"] != "auto":
            continue
        auto_r = r
        multi_r = next((x for x in results if x["id"] == r["id"] and x["mode"] == "multi" and not x.get("error")), None)
        if not multi_r:
            continue
        auto_j = auto_r.get("judge") or {}
        multi_j = multi_r.get("judge") or {}
        if auto_j.get("hallucinated") and not multi_j.get("hallucinated"):
            corrected.append({
                "id": r["id"], "question": r["question"],
                "auto_hall": auto_j.get("hallucination_detail", ""),
                "multi_answer": multi_r.get("answer", "")[:300],
            })

    for c in corrected[:5]:
        lines.append(f"### {c['id']}: {c['question'][:80]}...")
        lines.append(f"- **纯LLM幻觉**: {c['auto_hall'][:200]}")
        lines.append(f"- **多Agent纠正**: {c['multi_answer'][:300]}")
        lines.append("")

    lines.append("## 4. 错误分布\n")
    for mode in run_modes:
        s = summary["modes"].get(mode)
        if not s: continue
        error_cases = [r for r in results if r["mode"] == mode and r.get("error")]
        lines.append(f"- **{mode}**: {s['errors']} 错误 / {s['total_answered']} 回答")

    lines.append(f"\n---\n*由 eval_testset.py 自动生成*")

    md = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Report saved to {output_path}")
    return md


# ===== CLI =====

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit test set items")
    parser.add_argument("--modes", type=str, default="off,auto,multi",
                        help="Comma-separated modes to test")
    parser.add_argument("--server", type=str, default="localhost:8501",
                        help="Server host:port for multi mode")
    parser.add_argument("--output-dir", type=str, default="/data/Agent/eval_results",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to previous eval JSON to resume from")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from Nth item (0-indexed)")
    args = parser.parse_args()

    # Load test set
    test_set_path = "/data/Agent/test_set.json"
    with open(test_set_path, encoding="utf-8") as f:
        test_set = json.load(f)

    modes = [m.strip() for m in args.modes.split(",")]

    # Validate multi mode requires server
    if "multi" in modes:
        import urllib.request
        try:
            urllib.request.urlopen(f"http://{args.server}/api/sessions", timeout=5)
            print(f"Server OK at {args.server}")
        except Exception:
            print(f"WARNING: Server not reachable at {args.server}. "
                  f"Multi-agent mode will fail. Start with: python server.py")
            print("Removing 'multi' from modes. Add back with --modes off,auto,multi when server is running.")
            modes = [m for m in modes if m != "multi"]

    # Handle resume / start-from
    if args.resume:
        with open(args.resume, encoding="utf-8") as f:
            prev = json.load(f)
        prev_completed = {r["id"] for r in prev["results"] if not r.get("error")}
        print(f"Resuming from {args.resume}: {len(prev_completed)} already completed")
        test_set = [t for t in test_set if t["id"] not in prev_completed]
        args.start_from = 0  # override, we filter by id
    elif args.start_from > 0:
        test_set = test_set[args.start_from:]

    print(f"\nEvaluating {len(test_set[:args.limit])} items × {len(modes)} modes = "
          f"{len(test_set[:args.limit]) * len(modes)} answers\n")

    result = await evaluate_all(test_set, modes, args.limit, args.server)

    # Save detailed results (merge with previous if resuming)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"eval_{timestamp}.json")

    if args.resume:
        with open(args.resume, encoding="utf-8") as f:
            prev = json.load(f)
        prev_results = [r for r in prev["results"] if not r.get("error")]
        result["results"] = prev_results + result["results"]
        result["summary"]["test_config"]["total_items"] += len(prev_results)
        result["summary"]["test_config"]["resumed_from"] = args.resume

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to {json_path}")

    # Generate Markdown report
    md_path = os.path.join(args.output_dir, f"report_{timestamp}.md")
    md = generate_markdown_report(result["summary"], result["results"], md_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for mode in modes:
        s = result["summary"]["modes"][mode]
        print(f"\n{mode}:")
        print(f"  准确率: {s['avg_accuracy']:.2f}/5")
        print(f"  幻觉率: {s['hallucination_rate']:.0%}")
        print(f"  知识锚覆盖率: {s['knowledge_anchor_rate']:.0%}")
        print(f"  不确定性表达率: {s['uncertainty_rate']:.0%}")
        print(f"  完整性: {s['avg_completeness']:.2f}/5")
    print(f"\n交叉纠正率: {result['summary']['cross_correction_rate']:.0%}")

if __name__ == "__main__":
    asyncio.run(main())
