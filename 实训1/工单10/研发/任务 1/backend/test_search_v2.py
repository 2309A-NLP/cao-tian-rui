"""向量搜索对比测试：兴图新科 vs 力源信息"""
import sys, json
sys.path.insert(0, '.')

from pymilvus import connections, Collection
connections.connect(host='127.0.0.1', port=19530)
col = Collection('doc_chunks')
col.load()

# 加载 embedding 模型
from embedding_provider import EmbeddingFactory
with open('../config.json', encoding='utf-8') as f:
    cfg = json.load(f)

factory = EmbeddingFactory.create(
    model_type='bge-m3',
    model_path=cfg['embedding_model_path'],
    device='auto',
    batch_size=4,
)
factory.load()

# 测试查询
queries = [
    ("Q260 兴图-军用收入", "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入分别是多少？"),
    ("Q1 力源-发行股数", "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"),
    ("Q33 兴图-收入占比", "报告期内，武汉兴图新科电子股份有限公司来自军用领域的收入占主营业务收入的比重分别是多少？"),
    ("Q95 兴图-技术标准", "武汉兴图新科电子股份有限公司参与制定了哪个技术标准？"),
    ("Q543 兴图-注册资本", "武汉兴图新科电子股份有限公司注册资本是多少？"),
]

for label, q in queries:
    emb = factory.encode_query(q)
    col.load()
    results = col.search(
        data=[emb], anns_field="vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=5, output_fields=["chunk_id", "page", "text"]
    )
    print(f'\n=== {label} ===')
    print(f'  query: {q[:50]}...')
    for hit in results[0]:
        cid = hit.entity.get('chunk_id')
        page = hit.entity.get('page')
        text_raw = hit.entity.get('text') or ''
        text_preview = text_raw[:120].replace('\n','|')
        print(f'  score={hit.score:.4f} page={page} chunk={cid}: {text_preview}')
