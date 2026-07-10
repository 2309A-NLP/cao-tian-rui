# 工单编号：人工智能NLP-Agent数字人项目-医疗智能体-影像分析
# 生成边界测试所需的图片/文件 fixture（测试夹具）
#
# fixture 是测试中使用的预制数据文件，此脚本生成三种特殊图片：
# - normal_photo.jpg：普通非医疗图片（用于测试非医疗影像的处理）
# - black.jpg：全黑图（用于测试低质量图片的处理）
# - tiny.jpg：超低分辨率（用于测试分辨率校验逻辑）
#
# 运行方式：python tests/create_fixtures.py

# os：Python 内置模块（此处导入但未使用，可忽略）
import os

# pathlib.Path：面向对象的文件路径操作
from pathlib import Path

# PIL（Pillow）：Python 图像处理库
# Image：核心图像操作类，支持创建新图片和保存文件
# 安装方式：pip install Pillow
from PIL import Image

# 图片输出目录：tests/images/（相对于本脚本所在目录）
IMG_DIR = Path(__file__).parent / "images"

# 若目录不存在则创建（exist_ok=True 避免目录已存在时报错）
IMG_DIR.mkdir(exist_ok=True)


def make_normal_photo():
    """
    创建一张普通非医疗图片（纯色绿色方块，256×256）。

    用途：测试 VQA/MRG/RAG 接口对非医疗影像的处理，
    模型应识别出这不是医疗影像并给出相应提示。
    """
    p = IMG_DIR / "normal_photo.jpg"
    if not p.exists():  # 文件不存在时才创建（避免重复覆盖）
        # Image.new("RGB", (256, 256), color=(34, 139, 34))：
        # 创建一张 256×256 像素的 RGB 图片，颜色为深绿色 (R=34, G=139, B=34)
        img = Image.new("RGB", (256, 256), color=(34, 139, 34))
        img.save(p, "JPEG")  # 以 JPEG 格式保存
        print(f"[创建] {p}")


def make_black():
    """
    创建一张全黑图片（128×128）。

    用途：测试模型对低质量/无效影像的处理逻辑，
    模型应指出图像质量过差，建议用户重新上传。
    """
    p = IMG_DIR / "black.jpg"
    if not p.exists():  # 文件不存在时才创建
        # 创建 128×128 的纯黑色图片，RGB 全为 0
        img = Image.new("RGB", (128, 128), color=(0, 0, 0))
        img.save(p, "JPEG")
        print(f"[创建] {p}")


def make_tiny():
    """
    创建一张超低分辨率图片（16×16，低于配置中 32×32 的最小分辨率阈值）。

    用途：测试 validate_image 中的分辨率校验逻辑，
    此图片应被 API 拒绝并返回 422 IMAGE_INVALID 错误。
    """
    p = IMG_DIR / "tiny.jpg"
    if not p.exists():  # 文件不存在时才创建
        # 创建 16×16 的浅灰色图片（低于 MIN_IMAGE_RESOLUTION=32 的最小要求）
        img = Image.new("RGB", (16, 16), color=(200, 200, 200))
        img.save(p, "JPEG")
        print(f"[创建] {p}")


# 直接运行脚本时依次生成三种图片
if __name__ == "__main__":
    make_normal_photo()  # 生成普通照片
    make_black()          # 生成全黑图
    make_tiny()           # 生成超小图
    print("fixture 生成完成，路径：", IMG_DIR)
