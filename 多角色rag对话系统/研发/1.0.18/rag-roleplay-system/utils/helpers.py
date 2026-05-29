# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-   # 注意：重复的编码声明，保留原样
"""辅助函数

提供常用的通用功能：
- 密码哈希（SHA256）
- 关键词提取（中文文本，基于词频，过滤停用词）

⚠️ 常改动的地方：
1. 停用词列表（stopwords）可根据需求扩充，常见中文字词
2. 提取关键词的最大数量（max_words，默认 5）
3. 关键词提取的正则模式（当前匹配长度>=2的中文字符）

⚠️ 注意事项：
1. 密码哈希使用 SHA256，不是加盐的（仅用于简单场景，生产环境建议加盐）
2. 关键词提取基于词频，未使用 TF-IDF 或更复杂算法，适用于短文本
3. 停用词包含部分常见虚词，可根据具体业务补充（如“也”、“又”等）
"""

import hashlib
import re
from typing import List


def hash_password(password: str) -> str:
    """密码哈希：返回 SHA256 十六进制字符串"""
    return hashlib.sha256(password.encode()).hexdigest()


def extract_keywords(text: str, max_words: int = 5) -> List[str]:
    """
    从中文文本中提取高频关键词（长度≥2的中文字符，过滤停用词）

    参数:
        text: 输入文本
        max_words: 最多返回的关键词数量，默认 5

    返回:
        关键词列表（按出现频率降序）

    ⚠️ 常改动：停用词集合和 max_words 可根据业务需求调整
    """
    # 停用词集合（可在此扩充）
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '上'}

    # 提取所有长度 >= 2 的中文字符序列
    words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)

    # 统计词频（跳过停用词）
    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1

    # 按频率降序排序，取前 max_words 个
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:max_words]]