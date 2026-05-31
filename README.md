# PDF 雷同性检测工具

一款基于 **Tauri 2 + React + Rust** 的本地桌面应用，专为批量检测 PDF 文件之间的文本与内嵌图片相似度而设计。所有计算均在本地完成，无需联网，数据不上传至任何服务器。

---

## 功能特性

- **批量 PDF 导入**：一次性导入任意数量的 PDF 文件，支持拖拽或文件选择框
- **多维度相似度计算**：综合文本覆盖率与内嵌图片相似度，给出最终雷同分数
- **图片重复检测**：使用 SHA-256 识别完全重复图片，使用 pHash 识别轻微压缩、缩放后的近似图片
- **三档分析深度**：快速（Fast）、均衡（Balanced）、深度（Deep），灵活权衡速度与精度
- **实时进度反馈**：分阶段展示分析管道进度（扫描 → 索引 → 候选召回 → 精算 → 图聚类）
- **雷同组聚类**：基于图算法将相互雷同的文件归入同一组，并标注 Extreme / High / Medium / Low 四级风险
- **文本证据展示**：逐对展示匹配文本片段及对应页码，支持直观比对
- **报告导出**：支持导出 JSON 结构化报告与 Word 格式报告，可选是否包含文本证据
- **完全本地**：基于 Rust 原生解析 PDF（lopdf），零网络依赖

---

## 技术架构

```
前端：React 18 + TypeScript + Vite + Lucide React
后端：Rust（Tauri 2 命令层 + lopdf PDF 解析）
通信：Tauri IPC（invoke）
构建：Cargo + npm
```

### 核心算法流水线

| 阶段 | 说明 |
|------|------|
| 快速扫描 | 逐页抽取 PDF 文本、文档元数据与内嵌图片 |
| 全局索引 | 基于 Shingle 的文本索引与图片指纹索引构建 |
| 候选召回 | 文本、图片候选合并 + 公共特征降噪 + Top-K 截断 |
| 候选精算 | 精确页、近似文本覆盖率与图片一对一匹配计算 |
| 图聚类 | 构建相似度图，输出雷同组与风险等级 |

---

## 运行环境

- macOS 12 及以上（已提供 aarch64 Apple Silicon DMG）
- 需安装 PDF 文件，纯图片扫描件仅在 PDF 内含可提取图片对象时可参与图片比对

---

## 快速开始

### 直接使用（macOS）

从 `release/` 目录下载 `PDF雷同性检测工具_1.0.0_aarch64.dmg`，双击安装后启动即可。

### 开发环境搭建

**前提条件**

- [Node.js](https://nodejs.org/) >= 18
- [Rust](https://rustup.rs/) 工具链（stable）
- [Tauri CLI](https://tauri.app/start/prerequisites/) 依赖（macOS 需 Xcode Command Line Tools）

**安装依赖并启动开发服务**

```bash
npm install
npm run tauri dev
```

**构建生产包**

```bash
npm run tauri build
```

构建产物位于 `src-tauri/target/release/bundle/`。

---

## 使用说明

1. **添加文件**：点击左侧"添加文件"按钮，选择 2 个或以上 PDF 文件
2. **选择分析深度**：
   - `快速`：速度优先，阈值较宽松，适合初步筛查
   - `均衡`（默认）：速度与精度均衡，日常推荐
   - `深度`：精度优先，适合高要求场景，耗时较长
3. **启动分析**：点击"开始分析"，实时查看各阶段进度
4. **查看结果**：
   - **雷同组**标签页：查看被归组的文件及整体风险等级
   - **文件对**标签页：查看任意两个文件的详细相似度分数与文本证据
5. **调整阈值**：底部滑块可实时过滤低于指定分数的文件对
6. **导出报告**：点击右上角导出按钮，选择目标目录与导出格式（JSON / Word）

---

## 参数说明

| 参数 | 含义 | 默认值（均衡） |
|------|------|---------------|
| `text_threshold` | 文本相似度子分阈值 | 0.72 |
| `image_threshold` | 图像相似度子分阈值 | 0.80 |
| `final_threshold` | 最终综合分阈值（界面过滤用） | 0.50 |
| `shingle_size` | Shingle 粒度（字符 n-gram） | 5 |
| `min_shared_shingles` | 最少共同 shingle 数（候选召回） | 3 |
| `candidate_top_k_per_file` | 每文件最多保留候选对数量 | 20 |
| `max_matches_per_pair` | 每对文件最多展示文本证据条数 | 30 |

高级参数可通过界面右上角"设置"面板调整。

---

## 项目结构

```
file_compare/
├── src/                  # React 前端源码
│   ├── App.tsx           # 主界面组件
│   ├── main.tsx          # 应用入口
│   └── styles.css        # 全局样式
├── src-tauri/            # Rust 后端源码
│   ├── src/
│   │   ├── lib.rs        # Tauri 命令注册与任务管理器
│   │   ├── analysis.rs   # 核心分析算法（PDF 解析、索引、相似度计算、报告生成）
│   │   └── bin/
│   │       └── analyze_demo.rs  # 命令行 Demo 工具
│   ├── Cargo.toml
│   └── tauri.conf.json
├── analysis_results/     # 示例分析结果（JSON）
├── demo_files/           # 示例 PDF 文件
├── release/              # 已构建的 DMG 安装包
├── package.json
└── vite.config.ts
```

---

## 主要依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| tauri | 2.8.4 | 跨平台桌面框架 |
| lopdf | 0.40.0 | Rust 原生 PDF 解析 |
| image | 0.25 | JPEG 与图片像素解码 |
| sha2 | 0.10 | 图片原始流 SHA-256 指纹 |
| serde / serde_json | 1.0 | 序列化 / 反序列化 |
| uuid | 1.10.0 | 任务 ID 生成 |
| parking_lot | 0.12 | 高性能 Mutex |
| chrono | 0.4 | 时间戳处理 |
| react | 18.3 | 前端框架 |
| lucide-react | 0.468 | 图标组件库 |

---

## 注意事项

- 内嵌图片 SHA-256 与 pHash 已启用；整页渲染比对和 OCR 尚未接入
- 纯扫描 PDF 若没有可提取的内嵌图片对象，建议使用带文字识别（OCR）的 PDF
- 文件数量较多（> 100 份）时，建议使用"快速"档位或减小 `candidate_top_k_per_file`
- 导出 Word 报告依赖 zip 格式写入，无需安装 Office

---

## License

MIT
