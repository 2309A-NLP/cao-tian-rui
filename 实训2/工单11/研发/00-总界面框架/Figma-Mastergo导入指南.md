# Figma / Mastergo 导入指南

> 本指南说明如何将 HTML 原型设计规范导入 Figma 或 Mastergo，以便设计团队协作和开发对接。

---

## 一、快速导入路径

```
设计Tokens (design-tokens.json)
        |
        v
[Figma Tokens插件] 或 [Mastergo 变量面板]
        |
        v
创建样式库 (Color/Text/Effect)
        |
        v
按页面规范搭建 Frame + 放置组件
        |
        v
配置交互原型 (页面跳转/组件状态)
```

---

## 二、Figma 导入步骤

### Step 1: 安装 Tokens 插件
1. 打开 Figma，进入 `Plugins` -> `Browse plugins`
2. 搜索 `Tokens Studio for Figma`，安装并打开
3. 在 Tokens Studio 面板中，点击右上角 `...` -> `Import`
4. 选择本目录下的 `design-tokens.json` 文件
5. 导入后，所有 Token 会自动映射为 Figma 变量

### Step 2: 创建本地样式库
基于导入的 Token，创建以下本地样式（Local Styles）:

**颜色样式 (Color Styles)**
| 样式名称 | Token 路径 | 用途 |
|----------|-----------|------|
| Primary/600 | color.primary.600 | 主按钮、导航选中 |
| Primary/900 | color.primary.900 | 标题、重要文本 |
| Secondary/500 | color.secondary.500 | 次级按钮、图标 |
| Surface/Base | color.surface.base | 卡片背景 |
| Surface/Background | color.surface.background | 页面背景 |
| Text/Primary | color.text.primary | 标题文字 |
| Text/Secondary | color.text.secondary | 正文文字 |
| Accent/500 | color.accent.500 | 警告、高亮 |
| Success/500 | color.success.500 | 成功状态 |
| Error/500 | color.error.500 | 错误状态 |

**文字样式 (Text Styles)**
| 样式名称 | 字号 | 字重 | 行高 | 用途 |
|----------|------|------|------|------|
| Heading/1 | 24px | 600 | 1.25 | 页面大标题 |
| Heading/2 | 20px | 600 | 1.25 | 模块标题 |
| Heading/3 | 18px | 600 | 1.3 | 卡片标题 |
| Body/Normal | 14px | 400 | 1.5 | 正文 |
| Body/Medium | 14px | 500 | 1.5 | 强调正文 |
| Caption | 12px | 400 | 1.5 | 辅助说明 |

**效果样式 (Effect Styles)**
| 样式名称 | 值 | 用途 |
|----------|-----|------|
| Shadow/SM | 0 1px 3px rgba(0,0,0,0.05) | 卡片默认阴影 |
| Shadow/MD | 0 2px 8px rgba(13,148,136,0.1) | 悬浮阴影 |
| Shadow/LG | 0 4px 20px rgba(13,148,136,0.15) | 弹窗阴影 |

### Step 3: 创建组件库
在 Figma 中，按以下层次创建组件（Components）:

**原子组件 (Atoms)**
- `Button/Primary` - 主按钮 (高 44px, 圆角 12px, 渐变背景)
- `Button/Outline` - 描边按钮
- `Button/Small` - 小号按钮 (高 32px)
- `Input/Default` - 输入框 (高 44px, 圆角 22px)
- `Input/Select` - 下拉选择框
- `Tag/Primary` - 主标签
- `Tag/Success` - 成功标签
- `Tag/Warning` - 警告标签
- `Avatar/SM` - 小头像 (32px)
- `Avatar/MD` - 中头像 (36px)
- `Avatar/LG` - 大头像 (80px)
- `IconButton` - 图标按钮 (36px 圆形)

**分子组件 (Molecules)**
- `NavItem/Default` - 导航项 (高 48px, 宽 196px, 圆角 12px)
- `NavItem/Active` - 导航项选中态 (渐变背景 + 阴影)
- `ChatBubble/AI` - AI 对话气泡 (左侧头像 + 内容卡片)
- `ChatBubble/User` - 用户对话气泡 (右侧头像 + 渐变背景)
- `InfoCard` - 信息卡片 (圆角 12px, 内边距 16px)
- `StatCard` - 统计卡片 (圆角 12px, 居中对齐)
- `DataTable/Row` - 表格行
- `DataTable/Header` - 表格头

**有机体组件 (Organisms)**
- `Sidebar` - 侧边导航 (宽 220px, 高 100%, 白色背景)
- `TopBar` - 顶部栏 (高 60px, 宽 100%)
- `ChatInputBar` - 聊天输入栏 (高 60px)
- `ModulePanel` - 模块内容面板 (自适应宽高)

### Step 4: 搭建页面
创建以下 Frame（建议画布尺寸 1440x900）:

| Frame 名称 | 尺寸 | 说明 |
|-----------|------|------|
| `01-首页-数字人大厅` | 1220x840 | 主内容区（含侧边栏外区域） |
| `02-挂号管理` | 1220x840 | 号源查询/预约/取消 |
| `03-健康咨询` | 1220x840 | 知识图谱问答 |
| `04-影像分析` | 1220x840 | 影像上传/分析 |
| `05-出行服务` | 1220x840 | 地图/路线/周边 |
| `06-语音助手` | 1220x840 | 实时录音/纪要 |

> 每个 Frame 的左侧预留 220px 给 Sidebar 组件，上方预留 60px 给 TopBar。

### Step 5: 配置原型交互
使用 Figma 的 Prototype 模式配置：

1. **导航切换**: Sidebar 的 NavItem 点击 → 对应 Frame 跳转 (Instant)
2. **Tab 切换**: 各模块内部的 Tab 点击 → 同 Frame 内组件显隐 (Smart Animate)
3. **按钮反馈**: Button hover → 状态变体 (while hovering)
4. **弹窗显示**: 预约按钮点击 → Overlay 弹窗 (Open overlay)
5. **输入聚焦**: Input 点击 → 边框颜色变化 (while focusing)

---

## 三、Mastergo 导入步骤

### Step 1: 创建设计规范
1. 在 Mastergo 中创建新项目 `医疗智能体 Agent`
2. 进入 `资源` -> `样式` 面板，新建以下样式分类：

**颜色样式** (与 Figma 相同，参见上方表格)

**文字样式** (与 Figma 相同，参见上方表格)

**效果样式** (与 Figma 相同，参见上方表格)

### Step 2: 创建组件
Mastergo 的组件创建逻辑与 Figma 类似：

1. 先绘制原子元素（如一个圆角矩形按钮）
2. 右键 -> `创建组件`（快捷键 Ctrl/Cmd + Alt + K）
3. 在右侧属性面板中为组件添加变体（Variant）
4. 组件命名规范：`[类别]/[名称]/[状态]`，例如：
   - `Button/Primary/Default`
   - `Button/Primary/Hover`
   - `Button/Primary/Disabled`
   - `NavItem/Default`
   - `NavItem/Active`

### Step 3: 搭建页面
1. 创建画板（Artboard），尺寸建议 1440x900
2. 从组件库中拖拽 `Sidebar` 到画板左侧（x=0, y=60, w=220, h=840）
3. 拖拽 `TopBar` 到画板顶部（x=0, y=0, w=1440, h=60）
4. 在内容区域放置各模块的组件组合
5. 使用 Mastergo 的 `自动布局`（Auto Layout）功能确保响应式适配

### Step 4: 配置交互
Mastergo 原型模式支持以下交互：

1. **页面跳转**: 选择元素 -> `交互` -> `页面跳转` -> 选择目标画板
2. **组件状态**: 选择组件 -> 创建状态变体 -> 配置状态切换
3. **弹出层**: 选择触发元素 -> `交互` -> `打开弹窗` -> 选择弹窗组件
4. **滚动**: 内容超出可视区域时，启用画板的 `垂直滚动`

---

## 四、各模块页面结构速查

每个模块的详细页面结构规范，请参见对应文件夹中的 `页面结构规范.md`：

| 模块 | 规范文档 | 原型文件 |
|------|---------|---------|
| 总框架 | `00-总界面框架/设计说明文档.md` | `index.html` |
| 挂号管理 | `01-挂号管理/页面结构规范.md` | `prototype.html` |
| 健康咨询 | `02-健康咨询/页面结构规范.md` | `prototype.html` |
| 影像分析 | `03-影像分析/页面结构规范.md` | `prototype.html` |
| 出行服务 | `04-MCP/页面结构规范.md` | `prototype.html` |
| 语音助手 | `05-实时语音识别/页面结构规范.md` | `prototype.html` |

---

## 五、从 HTML 获取精确数值的方法

由于 HTML 原型中已包含所有精确尺寸，您可以通过以下方式获取：

### 方法1：浏览器开发者工具（推荐）
1. 用 Chrome 打开对应的 `prototype.html`
2. 右键点击任意元素 -> `检查`
3. 在 Elements 面板中查看 Computed 样式
4. 关键属性：`width`, `height`, `padding`, `margin`, `border-radius`, `background`, `color`, `font-size`, `font-weight`

### 方法2：截图参考
1. 按 `F12` 打开开发者工具
2. 按 `Ctrl/Cmd + Shift + P` -> 输入 `full size screenshot` -> 回车
3. 获取整页截图后，在 Figma/Mastergo 中作为参考图导入

### 方法3：使用浏览器扩展
安装 `CSS Peeper` 或 `VisBug` 扩展，可快速提取页面中所有 CSS 属性

---

## 六、设计-开发交接清单

完成设计后，请确保以下交付物完整：

- [ ] 所有页面 Frame/画板已完成
- [ ] 组件库已整理为可复用状态
- [ ] 颜色/文字/效果样式已统一
- [ ] 原型交互已配置（至少包含主要流程）
- [ ] 标注了关键尺寸和间距（或使用 Figma 的 Dev Mode）
- [ ] 导出所有图标为 SVG 格式
- [ ] 导出所有图片资源（如头像、背景图）
- [ ] 与设计 Tokens JSON 中的值保持一致

---

*文档版本: v1.0*
*生成时间: 2026-07-05*