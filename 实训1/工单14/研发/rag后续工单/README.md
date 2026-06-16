# RAGFlow 工单：修复低质量工业 PDF 的解析与信息丢失

> 项目位置：`E:/rag后续工单`　｜　起始日期：2026/06/14
> 基于开源项目 [RAGFlow](https://github.com/infiniflow/ragflow)

## 这个工单要干什么

针对 IMDR 数据集中**低分辨率、图片型的工业 PDF**，修复 RAGFlow 解析流水线中
导致图文信息丢失/关联错误的缺陷，让 6 个测试问题的问答精度达到 **100%**。

## 目录结构

```
E:/rag后续工单/
├── ragflow/        # RAGFlow 源码（git clone 而来，任务一源码分析 + 部署用）
├── docs/           # 产出文档
│   ├── 任务一_技术总结/   # deepdoc 模块、Redis Stream 队列、do_handle_task 分析
│   └── 任务二_测试报告/   # 6 问多轮测试、原因分析、优化方案与效果
├── data/           # 测试数据（CN100342976C.pdf 等，需另行获取）
└── README.md       # 本文件
```

## 三大任务与验收

| 任务 | 内容 | 验收 |
|---|---|---|
| 一 | 部署 RAGFlow + 梳理 deepdoc 技术方案 | 文档讲清：分块策略入 Redis Stream、do_handle_task 流程、DeepDoc 解析器 |
| 二 | 上传 CN100342976C.pdf，调 6 问到 100% | 问答准确率 100% |
| 三 | 演示视频 + 6 问检索精度展示 | 响应 <3s、高并发稳定 |

## 当前进度

- [x] 阶段 0：建目录 + clone 源码
- [x] 任务一文档：`docs/任务一_技术总结/RAGFlow技术实现方案总结.md`（三问全覆盖）
- [ ] 阶段 1：部署 RAGFlow（docker compose）
- [ ] 阶段 2：任务二 6 问调优至 100%（**阻塞：缺 CN100342976C.pdf / IMDR 数据集**）
- [ ] 演示视频：**由用户自行录制**，本项目只保证系统能跑出 6 问精度
