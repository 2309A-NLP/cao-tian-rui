# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""
知识库增强器 - 去重、摘要、父子块

负责：
- 文档去重（精确哈希 + 简化Jaccard相似度）
- 为每个文档添加摘要（第一段/前200字）
- 创建父子块结构（用于检索时父子块联合召回）

⚠️ 常改动的地方：
1. 停用词列表（stopwords）可补充更多无意义词，用于关键词抽取时的过滤
2. 去重相关阈值：
   - 过滤短文档的最小长度（当前硬编码 50）
   - 精确去重使用的字符数（当前 md5 基于前300字符）
   - 相似度去重的 Jaccard 阈值（当前 0.85，超过即视为重复）
3. 摘要生成策略：当前取第一段或前200字符，可改为调用 LLM 生成摘要
4. 父子块分块参数：parent_chunk_size=1000, overlap=100；child_chunk_size=300, overlap=50

⚠️ 注意事项：
1. 去重算法中的相似度计算基于中文二字及以上词汇，对英文/数字敏感度低
2. 父子块创建后，通常需要分别向量化，并在检索时关联使用（当前仅生成结构，未集成到检索流程）
3. 摘要信息保存在文档的 metadata["summary"] 中，可供前端展示或提示词使用
4. 所有增强方法都是原地修改文档（添加 metadata）或返回新文档列表
"""

import re
import hashlib
from typing import List, Set, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeBaseEnhancer:
    """知识库增强器：去重、加摘要、父子块"""

    def __init__(self):
        # ⚠️ 常改动：可扩充停用词，中文分词时可过滤
        self.stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人'}

    def deduplicate(self, docs: List[Document]) -> List[Document]:
        """
        文档去重：先按长度过滤，再按哈希精确去重，再按Jaccard相似度去重
        ⚠️ 常改动：
            - min_len = 50（过滤短文档）
            - hash_range = 300（用于精确去重的内容长度）
            - similarity_threshold = 0.85（Jaccard相似度阈值）
        """
        if len(docs) <= 1:
            return docs

        # 1. 过滤短文档（长度<50字符的认为是无效/无意义文档）
        docs = [d for d in docs if len(d.page_content.strip()) >= 50]

        # 2. 精确去重：基于内容前300字符的MD5哈希
        seen = set()
        unique = []
        for doc in docs:
            h = hashlib.md5(doc.page_content[:300].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(doc)

        # 3. 相似度去重（简化版Jaccard，基于中文二字及以上词汇）
        final = []
        keywords_list = []  # 存储每篇文档的关键词集合
        for doc in unique:
            # 提取中文词汇（长度>=2的中文字符串）
            words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', doc.page_content))
            # 去除停用词（可选，这里未实现）
            is_dup = False
            for kw_set in keywords_list:
                # 计算 Jaccard 相似度 = 交集大小 / 并集大小
                intersection = len(words & kw_set)
                union = len(words | kw_set)
                sim = intersection / union if union > 0 else 0
                if sim > 0.85:   # ⚠️ 常改动：相似度阈值
                    is_dup = True
                    break
            if not is_dup:
                keywords_list.append(words)
                final.append(doc)

        print(f"[KB增强] 去重: {len(docs)} -> {len(final)}")
        return final

    def add_summaries(self, docs: List[Document]) -> List[Document]:
        """
        为每个文档添加摘要（保存在 metadata["summary"]）
        ⚠️ 常改动：可改为调用 LLM 生成更准确的摘要，或取文档的前几句
        当前策略：取第一段（按换行分割）的前200字符，若没有换行则直接取前200字符
        """
        for doc in docs:
            text = doc.page_content
            # 如果文档中有换行，取第一段；否则取前200字符
            summary = text.split('\n')[0][:200] if '\n' in text else text[:200]
            doc.metadata["summary"] = summary
        return docs

    def create_parent_child_chunks(self, docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        创建父子块结构（用于层级检索）
        - 父块：较大粒度（1000字符），保留全局语义
        - 子块：较小粒度（300字符），用于精确匹配
        ⚠️ 常改动：分块参数（chunk_size, overlap）可根据文档类型调整
        ⚠️ 注意事项：该方法会为每个父块生成唯一的 parent_id，子块通过 metadata["parent_id"] 关联
        """
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

        parent_docs = []
        child_docs = []

        for doc in docs:
            # 对每个输入文档先生成父块
            parents = parent_splitter.split_documents([doc])
            for parent in parents:
                # 基于父块内容前100字符生成唯一ID
                parent_id = hashlib.md5(parent.page_content[:100].encode()).hexdigest()
                parent.metadata["chunk_type"] = "parent"
                parent.metadata["parent_id"] = parent_id
                parent_docs.append(parent)

                # 对每个父块再切分子块
                children = child_splitter.split_documents([parent])
                for child in children:
                    child.metadata["chunk_type"] = "child"
                    child.metadata["parent_id"] = parent_id
                    child_docs.append(child)

        print(f"[KB增强] 父子块: {len(parent_docs)}父, {len(child_docs)}子")
        return parent_docs, child_docs


# 全局单例实例
kb_enhancer = KnowledgeBaseEnhancer()