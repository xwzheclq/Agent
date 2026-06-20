"""
SimKGC-style 对比学习微调 BGE 嵌入模型 v2
BiEncoder + LoRA + InfoNCE + pre-batch negatives
+ 数据增强 + 过滤评估 + 实体类型前缀
"""
import os
from collections import deque
from typing import List, Dict, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

# ========== 配置 ==========
KGC_DIR = os.path.join(os.path.dirname(__file__), "kgc_dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "finetuned_bge")

MODEL_NAME = "BAAI/bge-large-zh-v1.5"
BATCH_SIZE = 64
EPOCHS = 15
LR = 2e-5
TEMPERATURE = 0.1
PRE_BATCH_CACHE_SIZE = 64
MAX_LEN = 128
GRAD_ACCUM = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========== 数据加载 ==========
def load_mappings():
    entities = {}   # id -> name
    etypes = {}     # id -> type
    with open(os.path.join(KGC_DIR, "entity2id.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("entity"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                eid = int(parts[1])
                entities[eid] = parts[0]
                etypes[eid] = parts[2].replace("-", " ")

    relations = {}  # id -> name
    with open(os.path.join(KGC_DIR, "relation2id.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("adopts"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                relations[int(parts[1])] = parts[0]

    return entities, etypes, relations


def load_triples(filename) -> List[Tuple[int, int, int]]:
    triples = []
    with open(os.path.join(KGC_DIR, filename), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return triples


# ========== 数据增强 ==========
def augment_reciprocal(triples: List[Tuple[int, int, int]],
                       relations: Dict[int, str]) -> List[Tuple[int, int, int]]:
    """为语义可逆的关系生成反向三元组 (t, r_rev, h)"""
    # 可逆关系映射
    rev_map = {
        "alias_of": "alias_of",
        "associated_with": "associated_with",
        "targets": "targets",
    }
    rid_to_name = {rid: name for rid, name in relations.items()}
    name_to_rid = {name: rid for rid, name in relations.items()}

    augmented = list(triples)
    for h, r, t in triples:
        rname = rid_to_name.get(r, "")
        if rname in rev_map:
            rev_rname = rev_map[rname]
            rev_rid = name_to_rid.get(rev_rname)
            if rev_rid is not None:
                augmented.append((t, rev_rid, h))
    return augmented


class KGCTripleDataset(Dataset):
    def __init__(self, triples: List[Tuple[int, int, int]],
                 entities: Dict[int, str], etypes: Dict[int, str],
                 relations: Dict[int, str]):
        self.triples = triples
        self.entities = entities
        self.etypes = etypes
        self.relations = relations

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h_id, r_id, t_id = self.triples[idx]
        h_name = self.entities.get(h_id, str(h_id))
        h_type = self.etypes.get(h_id, "")
        r_name = self.relations.get(r_id, str(r_id))
        t_name = self.entities.get(t_id, str(t_id))
        t_type = self.etypes.get(t_id, "")
        # 实体类型前缀增强语义
        query = f"[{h_type}] {h_name} {r_name}"
        tail = f"[{t_type}] {t_name}"
        return query, tail, h_id, t_id


def collate_fn(batch):
    queries, tails, h_ids, t_ids = zip(*batch)
    return list(queries), list(tails), list(h_ids), list(t_ids)


# ========== BiEncoder ==========
class BiEncoder(nn.Module):
    def __init__(self, model_name: str, use_lora: bool = True,
                 lora_r: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.use_lora = use_lora
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModel.from_pretrained(model_name)

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "key", "value", "dense"],
                lora_dropout=0.1,
            )
            self.encoder = get_peft_model(base_model, lora_config)
            self.encoder.print_trainable_parameters()
        else:
            self.encoder = base_model

    def mean_pooling(self, hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_emb = torch.sum(hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask

    @torch.no_grad()
    def encode(self, texts: List[str], max_len: int = MAX_LEN) -> torch.Tensor:
        """批量编码文本"""
        embs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_len, return_tensors="pt"
            ).to(DEVICE)
            outputs = self.encoder(**tokens)
            emb = self.mean_pooling(outputs.last_hidden_state, tokens["attention_mask"])
            embs.append(F.normalize(emb, p=2, dim=1))
        return torch.cat(embs, dim=0)

    def forward(self, query_texts: List[str], tail_texts: List[str]):
        all_texts = query_texts + tail_texts
        tokens = self.tokenizer(
            all_texts, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        ).to(DEVICE)
        outputs = self.encoder(**tokens)
        emb = self.mean_pooling(outputs.last_hidden_state, tokens["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        n = len(query_texts)
        return emb[:n], emb[n:]


# ========== InfoNCE Loss ==========
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.cache = deque(maxlen=PRE_BATCH_CACHE_SIZE)

    def push_cache(self, tail_emb: torch.Tensor):
        self.cache.append(tail_emb.detach())

    def forward(self, query_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        batch_size = query_emb.size(0)
        sim_inbatch = torch.matmul(query_emb, tail_emb.T) / self.temperature

        if self.cache:
            cached = torch.cat(list(self.cache), dim=0)
            sim_cached = torch.matmul(query_emb, cached.T) / self.temperature
            logits = torch.cat([sim_inbatch, sim_cached], dim=1)
        else:
            logits = sim_inbatch

        labels = torch.arange(batch_size, device=DEVICE)
        return F.cross_entropy(logits, labels)


# ========== 评估（过滤协议）==========
@torch.no_grad()
def evaluate(model: "BiEncoder",
             test_triples: List[Tuple[int, int, int]],
             entities: Dict[int, str], etypes: Dict[int, str],
             relations: Dict[int, str],
             all_train_triples: List[Tuple[int, int, int]],
             all_valid_triples: List[Tuple[int, int, int]],
             top_k: int = 10):
    """
    标准 KGC 评估（filtered）：
    对每个 test triple (h,r,t)，过滤掉所有已知的 (h,r,*) 正样本后再排名
    """
    # 构建所有尾实体候选 embedding
    all_entity_ids = list(set(entities.keys()))
    all_entity_names = [f"[{etypes.get(eid, '?')}] {entities[eid]}" for eid in all_entity_ids]
    all_emb = model.encode(all_entity_names)  # (N_entities, D)

    # 构建 (h, r) → {已知 tails} 映射（train + valid + test 全部）
    known_tails: Dict[Tuple[int, int], Set[int]] = {}
    for triples_set in [all_train_triples, all_valid_triples, test_triples]:
        for h, r, t in triples_set:
            known_tails.setdefault((h, r), set()).add(t)

    hits1 = hits10 = 0
    mrr_sum = 0.0
    total = 0

    for h_id, r_id, t_id in test_triples:
        h_name = entities.get(h_id, str(h_id))
        h_type = etypes.get(h_id, "")
        r_name = relations.get(r_id, str(r_id))

        query_text = f"[{h_type}] {h_name} {r_name}"
        query_emb = model.encode([query_text])

        sim = torch.matmul(query_emb, all_emb.T).squeeze(0)

        # 过滤: 除当前 t_id 外的所有已知正样本
        filtered = known_tails.get((h_id, r_id), set()) - {t_id}
        if filtered:
            filtered_indices = [i for i, eid in enumerate(all_entity_ids) if eid in filtered]
            sim[filtered_indices] = -float("inf")

        ranked = torch.argsort(sim, descending=True)
        ranked_entities = [all_entity_ids[i] for i in ranked]

        try:
            rank = ranked_entities.index(t_id) + 1
        except ValueError:
            rank = len(ranked_entities)

        if rank <= 1:
            hits1 += 1
        if rank <= top_k:
            hits10 += 1
        mrr_sum += 1.0 / rank
        total += 1

    return {
        "Hits@1": hits1 / total,
        f"Hits@{top_k}": hits10 / total,
        "MRR": mrr_sum / total,
        "total": total,
    }


# ========== 训练主循环 ==========
def train():
    entities, etypes, relations = load_mappings()
    train_triples = load_triples("train.txt")
    valid_triples = load_triples("valid.txt")
    test_triples = load_triples("test.txt")

    print(f"Original train: {len(train_triples)}, valid: {len(valid_triples)}, test: {len(test_triples)}")

    # 数据增强：反向三元组
    train_triples = augment_reciprocal(train_triples, relations)
    print(f"After augmentation: {len(train_triples)} triples")
    print(f"Entities: {len(entities)}, Relations: {len(relations)}")

    dataset = KGCTripleDataset(train_triples, entities, etypes, relations)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    print(f"Batches per epoch: {len(loader)}")

    model = BiEncoder(MODEL_NAME, use_lora=True)
    model.to(DEVICE)
    model.train()

    loss_fn = InfoNCELoss(temperature=TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_mrr = 0.0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        loss_fn.cache.clear()

        for step, (queries, tails, h_ids, t_ids) in enumerate(loader):
            query_emb, tail_emb = model(queries, tails)
            loss = loss_fn(query_emb, tail_emb)
            loss = loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_fn.push_cache(tail_emb)
            epoch_loss += loss.item() * GRAD_ACCUM

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{EPOCHS} | loss={avg_loss:.4f} | cache={len(loss_fn.cache)}")

        # Eval from epoch 3 onwards
        if (epoch + 1) >= 3:
            model.eval()
            metrics = evaluate(model, test_triples, entities, etypes, relations,
                               train_triples, valid_triples)
            model.train()
            print(f"  Eval: Hits@1={metrics['Hits@1']:.4f} "
                  f"Hits@10={metrics['Hits@10']:.4f} MRR={metrics['MRR']:.4f}")

            if metrics["MRR"] > best_mrr:
                best_mrr = metrics["MRR"]
                model.encoder.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
                model.tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))
                print(f"  Saved best (MRR={best_mrr:.4f})")

    print(f"\nBest MRR: {best_mrr:.4f}")

    # ===== 基线对比：无 LoRA 的原始 BGE =====
    print("\n=== Baseline (pretrained BGE, no LoRA) ===")
    baseline = BiEncoder(MODEL_NAME, use_lora=False)
    baseline.to(DEVICE)
    baseline.eval()
    base_metrics = evaluate(baseline, test_triples, entities, etypes, relations,
                            train_triples, valid_triples)
    print(f"  Baseline: Hits@1={base_metrics['Hits@1']:.4f} "
          f"Hits@10={base_metrics['Hits@10']:.4f} MRR={base_metrics['MRR']:.4f}")

    # ===== 微调模型 =====
    print("\n=== Fine-tuned (best checkpoint) ===")
    best_model = BiEncoder(MODEL_NAME, use_lora=True)
    best_model.encoder.load_adapter(os.path.join(OUTPUT_DIR, "adapter"))
    best_model.to(DEVICE)
    best_model.eval()
    ft_metrics = evaluate(best_model, test_triples, entities, etypes, relations,
                          train_triples, valid_triples)
    print(f"  Fine-tuned: Hits@1={ft_metrics['Hits@1']:.4f} "
          f"Hits@10={ft_metrics['Hits@10']:.4f} MRR={ft_metrics['MRR']:.4f}")

    delta_h1 = ft_metrics['Hits@1'] - base_metrics['Hits@1']
    delta_mrr = ft_metrics['MRR'] - base_metrics['MRR']
    print(f"\n  Improvement: dHits@1={delta_h1:+.4f}, dMRR={delta_mrr:+.4f}")
    print(f"  Relative: Hits@1 {ft_metrics['Hits@1']/max(base_metrics['Hits@1'], 1e-6):.1f}x, "
          f"MRR {ft_metrics['MRR']/max(base_metrics['MRR'], 1e-6):.1f}x")


if __name__ == "__main__":
    train()
