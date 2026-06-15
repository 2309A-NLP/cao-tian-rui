import json
from pathlib import Path

chunk_file = Path('output/chunks/all_chunks.json')
print('chunk_file exists:', chunk_file.exists())
if not chunk_file.exists():
    raise SystemExit(1)
with chunk_file.open('r', encoding='utf-8') as f:
    data = json.load(f)
print('total chunks:', len(data))

queries = [
    '本次发行', '发行股数', '募集资金', '发行价格', '招股说明书',
    '法定代表人', '注册资本', '电子信息行业', '国家科技进步一等奖',
    '大客户销售部', '本公司的主营业务', '主营业务收入',
]
for q in queries:
    hits = [c for c in data if q in c.get('text', '')]
    print('\nQUERY:', q, 'count=', len(hits))
    for i, c in enumerate(hits[:5]):
        text = c.get('text', '').replace('\n', ' ').replace('\r', ' ')[:240]
        print('  ', i, 'page=', c.get('page'), 'chunk_id=', c.get('chunk_id'), 'field_type=', c.get('field_type'), 'text=', text)

# Also check common template words to see whether chunks are mostly cover/table-of-contents
words = ['目录', '声明', '董事会', '律师', '承诺', '保荐机构', '风险因素']
for w in words:
    cnt = sum(1 for c in data if w in c.get('text', ''))
    print('WORD', w, 'count=', cnt)

# Distribution of field_type
from collections import Counter
ft = Counter(c.get('field_type', 'body') for c in data)
print('\nfield_type distribution:')
for k, v in ft.most_common():
    print(' ', k, v)

# Chunk length distribution sample
lengths = [len(c.get('text', '')) for c in data]
print('\nchunk length stats:', min(lengths), max(lengths), sum(lengths)//len(lengths), 'avg approx')
