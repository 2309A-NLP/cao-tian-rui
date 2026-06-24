"""
LLM Provider — OpenAI 兼容接口 + Function Calling 支持
工单编号：人工智能 NLP-Agent 数字人项目-记账本任务
"""
import json
import requests
from typing import Optional, Generator
from backend.logger import get_logger

logger = get_logger("llm")


class LLMProvider:
    """OpenAI 兼容 LLM 调用（支持 tool calls）"""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict] = None,
             tool_choice: str = "auto") -> dict:
        """
        调用 chat/completions（非流式），返回完整响应。

        返回格式:
        {
            "finish_reason": "stop" | "tool_calls",
            "content": "回复文本（stop时）",
            "tool_calls": [...]  # tool_calls时
        }
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            finish = choice.get("finish_reason", "stop")
            msg = choice.get("message", {})

            result = {"finish_reason": finish}
            if finish == "tool_calls":
                result["tool_calls"] = msg.get("tool_calls", [])
                # tool_calls 可能也带文字，用于追问
                if msg.get("content"):
                    result["content"] = msg["content"]
            else:
                result["content"] = msg.get("content", "")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API 请求失败: {e}")
            raise

    def chat_stream(self, messages: list[dict], tools: list[dict] = None,
                    tool_choice: str = "auto") -> Generator[dict, None, None]:
        """
        流式调用 chat/completions。

        Yields:
            {"type": "token", "content": "..."} 或
            {"type": "tool_calls", "tool_calls": [...]} 或
            {"type": "done", "finish_reason": "..."}
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        try:
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=120, stream=True
            )
            if not resp.ok:
                logger.error(f"LLM 流式响应错误 {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            resp.encoding = "utf-8"

            # 流式聚合：tool_calls 需要跨 chunk 拼起来
            accumulated_tool_calls = {}
            content_parts = []

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})

                    # 文本 token
                    if delta.get("content"):
                        token = delta["content"]
                        content_parts.append(token)
                        yield {"type": "token", "content": token}

                    # tool_calls（可能分片）
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": "", "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                }
                            entry = accumulated_tool_calls[idx]
                            if tc.get("id"):
                                entry["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                entry["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                entry["function"]["arguments"] += fn["arguments"]

                    # 检查是否结束
                    finish = choice.get("finish_reason")
                    if finish:
                        if accumulated_tool_calls:
                            tcs = [accumulated_tool_calls[i] for i in sorted(accumulated_tool_calls)]
                            yield {"type": "tool_calls", "tool_calls": tcs}
                        yield {"type": "done", "finish_reason": finish}

                except json.JSONDecodeError:
                    continue

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM 流式请求失败: {e}")
            yield {"type": "token", "content": f"【错误】LLM 请求失败: {e}"}
            yield {"type": "done", "finish_reason": "error"}
