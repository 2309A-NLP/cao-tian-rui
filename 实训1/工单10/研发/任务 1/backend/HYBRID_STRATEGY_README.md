# PaddleOCR-VL 混合策略使用说明

## 🎯 策略概述

优化后的混合策略采用**智能路由处理**，针对不同类型的页面使用最合适的处理方法：

### 📋 处理逻辑

| 页面类型 | 处理方法 | 优势 | API依赖 |
|---------|---------|------|---------|
| **纯文字页** | pymupdf直接提取 | 高效，无需API调用 | ❌ 无需 |
| **表格页** | PaddleOCR-VL识别 | 更准确的表格结构识别 | ✅ 需要 |
| **图表页** | PaddleOCR-VL识别 | 图表内容智能理解 | ✅ 需要 |
| **混合页** | 分离处理（文字+表格） | 完整的内容提取 | ✅ 需要 |

## 🔧 使用方法

### 1. 基本使用

```bash
# 启用混合策略扫描索引
python main.py --scan --hybrid --api-key <your_key> --api-secret <your_secret>

# 单文件混合策略处理
python main.py --hybrid --index "path/to/file.pdf"

# 重新索引（混合策略）
python main.py --reindex --hybrid
```

### 2. 配置文件

```json
{
  "hybrid_processor_enabled": true,
  "paddleocr_api_key": "your_api_key",
  "paddleocr_api_secret": "your_api_secret",
  "paddleocr_api_base": "https://api.paddlecloud.com/v1/ocr",
  "hybrid_batch_size": 10,
  "hybrid_text_density_threshold": 0.3,
  "hybrid_table_rows_min": 3
}
```

### 3. 命令行参数

- `--hybrid`: 启用混合策略
- `--api-key`: 覆盖配置文件的API密钥
- `--api-secret`: 覆盖配置文件的API密钥

## 📊 效果对比

### 标准策略 vs 混合策略

| 内容类型 | 标准策略 | 混合策略 | 提升效果 |
|---------|---------|---------|---------|
| 纯文字 | pymupdf提取 | pymupdf提取 | 相同，高效 |
| 表格 | 无法识别 | PaddleOCR-VL | ✅ 大幅提升 |
| 图表 | 无法识别 | PaddleOCR-VL | ✅ 大幅提升 |
| 混合内容 | 混合提取 | 智能分离 | ✅ 更好组织 |

## ⚠️ 注意事项

### 1. API配额限制
- 每个模型有日解析上限
- 达到上限返回429错误，自动重试
- 建议控制单文件在100页内

### 2. 页数处理
- 单文件最多处理100页
- 超出部分自动截断并提示
- 大文件建议分批处理

### 3. 回退机制
- API不可用时自动回退到pymupdf
- 表格识别失败时回退到基础提取
- 保证处理流程的稳定性

## 🚀 适用场景

### 推荐使用混合策略的场景：
- **财务报告**：大量表格和图表的智能识别
- **学术论文**：图表+文字的完整提取
- **技术文档**：混合内容的结构化处理
- **法律文件**：纯文字的精确识别

### 不推荐使用的场景：
- 纯文字文档（标准策略足够）
- 预算有限无法使用API
- 对处理速度要求极高

## 📈 性能优化

### 1. 节省API成本
- 纯文字页面不消耗API配额
- 只对复杂内容使用API

### 2. 提升处理速度
- pymupdf提取比OCR更快
- 减少不必要的API调用

### 3. 保证准确性
- 表格和图表使用专门API
- 避免通用OCR的识别错误

## 🔍 故障排除

### 常见问题

1. **429错误**
   - API配额已用完
   - 等待配额重试或升级套餐

2. **表格识别不准确**
   - 检查图片清晰度
   - 尝试调整页面分类阈值

3. **混合页处理不理想**
   - 调整文字密度阈值
   - 检查API配置是否正确

### 日志分析

处理时会输出详细日志：
```
[  1/50] Type=pure_text    Time=0.05s  # 纯文字页面
[  2/50] Type=table        Time=2.30s  # 表格页面（API调用）
[  3/50] Type=image         Time=1.80s  # 图表页面（API调用）
[  4/50] Type=mixed        Time=2.10s  # 混合页面（分离处理）
```

通过日志可以清楚看到每种页面的处理方式和耗时。