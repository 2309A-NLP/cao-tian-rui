# -*- coding: utf-8 -*-
"""
提示词管理器

负责：
- 管理不同角色的系统提示词
- 判断是否为法律相关问题（关键词匹配 + 正则）
- 构建普通/法律场景的最终提示词（包含短期记忆、长期记忆、知识库内容）

⚠️ 常改动的地方：
1. LEGAL_KEYWORDS 列表（在 config.py 中定义），用于识别法律相关问题
2. ROLE_PROMPTS 角色提示词内容（在 config.py 中定义）
3. 身份问题关键词列表（is_identity_question 中的关键词）
4. 长期记忆注入时的内容长度阈值（len(content) > 10 和 > 10 的不同处）
5. 法律场景下的回答要求（build_law_prompt 中的要求措辞）

⚠️ 注意事项：
1. 长期记忆格式化时会过滤长度小于10字符的内容（避免无意义记忆）
2. 法律专用提示词中，如果用户问“我是谁/我的职业”等身份问题且有历史记忆，则完全跳过知识库
3. 普通提示词中，记忆部分置于历史对话之前
4. 提示词中的【知识库内容】可能为空字符串，模型应处理“无相关知识”的情况
"""

import re
from typing import List, Dict

from config import ROLE_PROMPTS, LEGAL_KEYWORDS


class PromptManager:
    """提示词管理器（单例，全局使用）"""

    def __init__(self):
        # 角色提示词字典，从 config 导入
        self.role_prompts = ROLE_PROMPTS
        # 法律关键词列表（支持字符串直接包含和正则匹配）
        self.legal_keywords = LEGAL_KEYWORDS

    def get_role_prompt(self, role_type: str) -> str:
        """
        获取角色提示词，若角色不存在则返回 "friend" 角色的提示词
        """
        return self.role_prompts.get(role_type, self.role_prompts["friend"])

    def is_law_question(self, question: str) -> bool:
        """
        判断用户问题是否涉及法律相关内容
        使用关键词包含和正则匹配（支持如 '第.*条' 的模式）
        ⚠️ 常改动：关键词列表在 config.py 的 LEGAL_KEYWORDS 中调整
        """
        question_lower = question.lower()
        for keyword in self.legal_keywords:
            # 直接包含匹配（不区分大小写）
            if keyword in question_lower:
                return True
            # 正则匹配（用于复杂模式，如 '第.*条'）
            if re.search(keyword, question_lower):
                return True
        return False

    def _format_memories(self, long_memories: List[Dict]) -> str:
        """
        将长期记忆列表格式化为可在提示词中使用的文本块
        ⚠️ 常改动：记忆条目的内容长度过滤阈值（当前 len(content) > 10）
        ⚠️ 注意事项：只返回非空的记忆，每条以 "- " 开头
        """
        if not long_memories:
            return ""

        items = []
        for mem in long_memories:
            content = mem.get('content', '')
            # 过滤掉过短的记忆（可能是无效信息）
            if content and len(content) > 10:
                items.append(f"- {content}")

        if items:
            return "【用户之前告诉我的重要信息（请优先使用）】\n" + "\n".join(items) + "\n"
        return ""

    def build_law_prompt(self, role_prompt: str, context: str, long_memories: List[Dict],
                         history_text: str, message: str, has_knowledge: bool) -> str:
        """
        构建法律问题的专用提示词 - 记忆优先

        参数:
            role_prompt: 角色系统提示词
            context: 知识库检索到的相关文本（可能为空）
            long_memories: 长期记忆列表
            history_text: 短期记忆格式化的文本
            message: 用户当前问题
            has_knowledge: 是否检索到有效知识（未使用，保留为扩展）

        ⚠️ 常改动：身份问题的关键词列表（'我是谁', '我是什么', '我的职业', '我的名字', '外号', '身份'）
        ⚠️ 注意事项：
            1. 如果用户问身份类问题且存在长期记忆，则完全忽略知识库，直接基于记忆回答
            2. 否则，将知识库内容作为主要信息来源
            3. 提示词中明确禁止使用 emoji
        """
        memory_section = self._format_memories(long_memories)

        if memory_section:
            print(f"[提示词] 已注入 {len(long_memories)} 条历史记忆")

        # 判断是否是询问个人身份的问题
        is_identity_question = any(
            kw in message for kw in ['我是谁', '我是什么', '我的职业', '我的名字', '外号', '身份']
        )

        if is_identity_question and memory_section:
            # 身份问题且有记忆，直接基于记忆回答（不参考知识库）
            return f"""{role_prompt}

{memory_section}

【用户问题】
{message}

【回答要求】
请直接从【用户之前告诉我的重要信息】中查找答案。
如果信息中有用户的职业、身份、外号，请直接告诉用户。
不要使用知识库内容。
禁止使用emoji。

请回答："""

        # 普通法律问题（包含知识库内容）
        # 注意：context 可能为空字符串，模型应处理“无相关知识”的情况
        return f"""{role_prompt}

{memory_section}
【历史对话】
{history_text}

【知识库内容】
{context if context else "（无相关知识）"}

【用户问题】
{message}

【回答要求】
1. 如果用户问的是关于他自己的信息（如职业、身份），请从【用户之前告诉我的重要信息】中回答
2. 否则，基于【知识库内容】回答
3. 禁止使用emoji

请回答："""

    def build_normal_prompt(self, role_prompt: str, context: str, long_memories: List[Dict],
                            history_text: str, message: str) -> str:
        """
        构建普通角色（非法律）的提示词
        ⚠️ 注意事项：记忆部分置于历史对话之前，但实际顺序为：角色提示词 → 长期记忆 → 短期历史 → 知识库内容 → 问题 → 要求
        """
        memory_section = self._format_memories(long_memories)

        return f"""{role_prompt}
{memory_section}
{history_text}

【相关资料】
{context}

【用户问题】
{message}

请回答："""


# 全局单例实例
prompt_manager = PromptManager()