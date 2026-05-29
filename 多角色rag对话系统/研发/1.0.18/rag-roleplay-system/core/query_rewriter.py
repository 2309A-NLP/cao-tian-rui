# -*- coding: utf-8 -*-
"""
查询改写器 - 提升检索召回率

负责：
- 查询标准化（替换口语化表达）
- 查询扩写（生成同义词变体）
- 为检索提供多元查询，提高召回率

⚠️ 常改动的地方：
1. 同义词库（synonyms）：可针对业务领域（法律、医疗等）不断扩充
2. 口语化替换表（replacements）：根据实际用户输入补充常见口语/错别词
3. 扩写的最大变体数量（当前硬编码 [":3]），可调整
4. use_llm 参数（当前保留但未实现），未来可集成 LLM 进行智能改写

⚠️ 注意事项：
1. 当前实现仅基于规则（字符串替换和同义词替换），不依赖 LLM
2. 扩写时会基于原查询中的关键词进行同义词替换，生成多个变体
3. 去重通过 dict.fromkeys 保持顺序，最终返回最多3个变体
4. 同义词库中每个词最多取前2个同义词进行替换
"""

import re
from typing import List, Dict


class QueryRewriter:
    """查询改写器：支持标准化、扩写、关键词提取"""

    def __init__(self, use_llm: bool = False):
        """
        ⚠️ 常改动：use_llm 当前为保留参数，未来可扩展基于 LLM 的智能改写
        """
        self.use_llm = use_llm
        self._init_synonyms()

    def _init_synonyms(self):
        """初始化同义词库（可根据业务持续扩充）"""
        self.synonyms = {
            # 法律领域
            "离婚": ["解除婚姻", "婚姻破裂"],
            "捡到": ["拾得", "发现"],
            "失物": ["遗失物", "丢失物品"],
            "合同": ["协议", "契约"],
            "违约": ["违反约定", "不履行"],
            "侵权": ["侵害权益", "损害"],
            "未成年人": ["儿童", "青少年"],
            "保护": ["保障", "维护"],

            # 医疗领域
            "高血压": ["血压高", "高压"],
            "心衰": ["心力衰竭", "心脏衰竭"],
            "治疗": ["诊疗", "医治"],
        }

    def rewrite(self, query: str) -> str:
        """
        改写查询 - 标准化表达（替换常见口语/错别词）
        ⚠️ 常改动：replacements 列表可不断补充
        """
        rewritten = query

        # 口语化/不规范表达替换表
        replacements = [
            ("咋办", "怎么办"),
            ("啥是", "什么是"),
            ("为啥", "为什么"),
            ("咋样", "怎么样"),
            ("咋回事", "怎么回事"),
            ("我捡到", "捡到"),
            ("我是个老师", "老师"),
            # 可继续添加
        ]
        for old, new in replacements:
            rewritten = rewritten.replace(old, new)

        return rewritten

    def expand(self, query: str) -> List[str]:
        """
        扩写查询，生成同义变体
        ⚠️ 常改动：每个关键词最多取的同义词数量（当前 [":2]），以及最终变体数量限制（当前 [":3]）
        """
        variants = [query]

        # 对每个关键词，用其前2个同义词生成新查询
        for word, synonyms in self.synonyms.items():
            if word in query:
                for syn in synonyms[:2]:
                    variants.append(query.replace(word, syn))

        # 去重（保持顺序）并限制最多3个变体
        variants = list(dict.fromkeys(variants))[:3]

        return variants

    def optimize(self, query: str) -> Dict:
        """完整优化：返回标准化后的查询和扩写变体，以及被匹配到的关键词"""
        rewritten = self.rewrite(query)
        expanded = self.expand(rewritten)

        return {
            "original": query,
            "rewritten": rewritten,
            "variants": expanded,
            "keywords": [w for w in self.synonyms.keys() if w in query]
        }