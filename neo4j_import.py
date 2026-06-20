"""
Neo4j 知识图谱导入 — CTI-KGC 数据集
实体类型标签 + 英文关系类型 + 置信度属性 + 索引
"""
import os
from py2neo import Graph

KGC_DIR = os.path.join(os.path.dirname(__file__), "kgc_dataset")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "newpassword123")


def load_mappings():
    """加载 entity2id, relation2id 映射表"""
    entities = {}  # id -> (name, type)
    with open(os.path.join(KGC_DIR, "entity2id.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("entity"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                name, eid, etype = parts[0], int(parts[1]), parts[2]
                entities[eid] = (name, etype)

    relations = {}  # id -> (name, chinese)
    with open(os.path.join(KGC_DIR, "relation2id.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("adopts"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                rname, rid = parts[0], int(parts[1])
                relations[rid] = rname

    return entities, relations


def load_triples(filename):
    """加载三元组文件，返回 [(head_id, rel_id, tail_id, confidence), ...]"""
    triples = []
    with open(os.path.join(KGC_DIR, filename), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                conf = float(parts[3]) if len(parts) >= 4 else 1.0
                triples.append((h, r, t, conf))
    return triples


def import_all():
    entities, relations = load_mappings()

    # 加载所有三元组（train + valid + test）
    all_triples = []
    for fname in ["train_confidence.txt", "valid_confidence.txt", "test_confidence.txt"]:
        fpath = os.path.join(KGC_DIR, fname)
        if os.path.exists(fpath):
            all_triples.extend(load_triples(fname))

    print(f"Loaded {len(entities)} entities, {len(relations)} relations, {len(all_triples)} triples")

    graph = Graph(NEO4J_URI, auth=NEO4J_AUTH)

    # 清空数据库
    print("Clearing database...")
    graph.run("MATCH (n) DETACH DELETE n")

    # 删除旧约束
    try:
        graph.run("DROP CONSTRAINT entity_name_unique IF EXISTS")
    except Exception:
        pass

    # 创建约束和索引
    graph.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.eid IS UNIQUE")
    graph.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)")

    # 创建实体节点（带类型标签）
    print("Creating entity nodes...")
    tx = graph.begin()
    for eid, (name, etype) in entities.items():
        safe_type = etype.replace("-", "_").replace(" ", "_")
        # 每个实体同时有 Entity 主标签和具体类型标签
        query = f"""
        CREATE (e:Entity:{safe_type} {{eid: $eid, name: $name, type: $type}})
        """
        tx.run(query, eid=eid, name=name, type=etype)
    tx.commit()
    print(f"  Created {len(entities)} typed entity nodes")

    # 创建关系（批量，每 500 条一个事务）
    print("Creating relationships...")
    batch_size = 500
    for batch_start in range(0, len(all_triples), batch_size):
        batch = all_triples[batch_start:batch_start + batch_size]
        tx = graph.begin()
        for h_id, r_id, t_id, conf in batch:
            rel_name = relations.get(r_id, f"rel_{r_id}")
            # 用反引号包裹关系类型以支持特殊字符
            safe_rel = rel_name.replace("`", "``")
            query = f"""
            MATCH (s:Entity {{eid: $h_id}})
            MATCH (o:Entity {{eid: $t_id}})
            MERGE (s)-[r:`{safe_rel}`]->(o)
            SET r.confidence = $conf
            """
            tx.run(query, h_id=h_id, t_id=t_id, conf=conf)
        tx.commit()
        print(f"  Imported {batch_start + len(batch)}/{len(all_triples)} triples")

    # 验证
    node_count = graph.run("MATCH (n) RETURN count(n) AS c").evaluate()
    rel_count = graph.run("MATCH ()-[r]->() RETURN count(r) AS c").evaluate()
    print(f"\nImport complete: {node_count} nodes, {rel_count} relationships")

    # 按类型统计
    print("\nEntity types:")
    for record in graph.run("""
        MATCH (n:Entity)
        RETURN n.type AS type, count(n) AS cnt
        ORDER BY cnt DESC
    """):
        print(f"  {record['type']}: {record['cnt']}")

    # 关系类型统计
    print("\nRelation types (top 10):")
    for record in graph.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS t, count(r) AS cnt
        ORDER BY cnt DESC
        LIMIT 10
    """):
        print(f"  {record['t']}: {record['cnt']}")


if __name__ == "__main__":
    import_all()
