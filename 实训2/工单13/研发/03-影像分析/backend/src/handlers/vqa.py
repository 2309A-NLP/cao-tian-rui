"""
工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
VQA Handler：医疗影像视觉问答（Visual Question Answering）。

VQA 是最简单的影像分析任务：
- 输入：医疗影像 + 用户问题
- 输出：基于影像内容的文字回答

本处理器直接将影像和问题发送给 VLM，无需检索步骤，是三种任务中最轻量的。
"""

# 从父包导入所需模块（使用相对导入）
from ..config import DISCLAIMER         # 医疗免责声明文本
from ..vlm_client import get_vlm_client  # VLM 客户端工厂函数

# ── VQA 任务的系统提示词 ──
# 规范模型的行为：只基于影像作答、识别非医疗图片、处理低质量图片、附加免责声明
VQA_SYSTEM_PROMPT = (
    "你是一位经验丰富的医学影像分析助手。请根据用户提供的医疗影像回答问题。\n"
    "回答规则：\n"
    "1) 仅基于影像可观察到的内容作答，不要臆造未见的信息；\n"
    "2) 若图片明显不是医疗影像（如风景、文字截图、动漫等），先明确告知用户，再简要描述所见；\n"
    "3) 若图像质量过差（全黑、全白、严重模糊）导致无法分析，明确说明并请用户重新上传；\n"
    "4) 回答简洁准确，中文回答中文提问，英文回答英文提问；\n"
    "5) 涉及诊断或建议时，务必附上：本回答仅供参考，不构成医疗诊断建议。"
)


class VQAHandler:
    """
    医疗影像视觉问答处理器。

    将影像和用户问题直接发送给 VLM，返回基于影像内容的回答。
    这是三种任务中逻辑最简单的处理器，无额外处理步骤。
    """

    def __init__(self):
        """初始化处理器，获取 VLM 客户端单例。"""
        self.vlm = get_vlm_client()  # 获取全局 VLM 客户端（单例，避免重复创建）

    def run(self, image_b64: str, query: str, image_format: str = "jpeg") -> str:
        """
        执行 VQA 任务：基于影像回答用户问题。

        参数：
            image_b64 (str)：图片的 base64 编码字符串
            query (str)：用户问题文本
            image_format (str)：图片格式（"jpeg" 或 "png"），默认 "jpeg"

        返回值：
            str：模型生成的回答文本（已追加免责声明）
        """
        # 调用 VLM 进行图文理解
        # 用户问题直接作为 prompt，系统提示词规范模型行为
        answer = self.vlm.chat_vision(
            image_b64=image_b64,
            prompt=query,              # 用户问题作为输入 Prompt
            system=VQA_SYSTEM_PROMPT,  # VQA 专用系统提示词
            max_tokens=1024,           # 回答 token 上限（VQA 回答通常简短）
            temperature=0.3,           # 中等温度：平衡准确性和流畅度
            image_format=image_format,
        )

        # 检查模型回答是否已包含免责声明
        # 若未包含，则自动追加（防止模型忘记遵守规则5）
        if "仅供参考" not in answer and "遵医嘱" not in answer:
            answer = f"{answer}\n\n⚠️ {DISCLAIMER}"  # 追加免责声明

        return answer
