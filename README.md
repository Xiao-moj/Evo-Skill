# Evo-skill

从真实对话中自动提炼、存储、进化可复用技能的终身学习框架。

专为 CS 学习者设计：每次对话结束后，Evo-skill 会自动识别算法、数学公式、ML 概念等技术知识，提炼为带有可运行脚本的技能单元，并在下次对话时自动召回注入，让 AI 越用越懂你。

---

## 快速开始

### 1. 进入项目目录

```bash
cd ~/Desktop/evoskill
```

### 2. 安装依赖

Evo-skill 核心不依赖任何第三方库（纯 urllib + 标准库）。
根据你选择的 Embedding 方案，按需安装：

```bash
# 方案 A：使用 OpenAI Embedding API（无需额外安装）
# 什么都不用装

# 方案 B：使用本地 BGE-M3 模型
pip install FlagEmbedding        # 推荐
# 或
pip install sentence-transformers  # 备选
```

### 3. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

然后编辑 `.env`，填入你的 API Key 和模型配置：

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.openai.com
LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxx
EMBEDDING_MODEL=text-embedding-3-small

EXTRACTOR_MODE=learner   # CS 学习者模式
```

### 4. 运行

```bash
# 启动对话 chat（主功能）
python main.py chat

# 列出所有已保存的技能
python main.py list

# 删除一条技能
python main.py delete <skill_id>

# 导出所有技能为 JSON 文件
python main.py export skills_backup.json

# 根据已保存的技能脚本组合解决方案
python main.py compose "对数组先排序再二分查找目标值"
```

---

## 配置说明

所有配置均通过 `.env` 文件或环境变量控制，无需修改代码。

### LLM 提供商

| `LLM_PROVIDER` | 说明 |
|---|---|
| `openai` | OpenAI 或任何兼容 OpenAI 接口的模型（DeepSeek、Qwen 等） |
| `anthropic` | Anthropic Claude |

**使用 OpenAI 兼容接口（如 DeepSeek）：**
```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

**使用 Anthropic Claude：**
```env
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-xxxxxxxx
LLM_MODEL=claude-sonnet-4-6
```

### Embedding 提供商

| `EMBEDDING_PROVIDER` | 说明 | 需要 API Key |
|---|---|---|
| `openai` | OpenAI `text-embedding-3-small` 等 | 是 |
| `bge-m3` | 本地 BAAI/bge-m3 模型（从 HuggingFace 缓存加载） | 否 |
| `hashing` | 哈希向量（仅用于测试，不需要任何配置） | 否 |

**使用本地 BGE-M3：**
```env
EMBEDDING_PROVIDER=bge-m3
# BGE_M3_DEVICE=cpu   # 或 cuda
# BGE_M3_USE_FP16=false
```

模型路径自动从 `~/.cache/huggingface/hub/models--BAAI--bge-m3` 加载。

### 提取模式

| `EXTRACTOR_MODE` | 说明 |
|---|---|
| `learner` | CS 学习者模式：提取算法、ML、数学等技术知识，并生成可运行脚本 |
| `default` | 偏好模式：提取用户工作流程和个人习惯 |

### 检索参数

```env
MIN_SCORE=0.4        # 技能相似度阈值（低于此值不注入），范围 0~1
TOP_K=3              # 每轮 embedding 检索的技能数量
REWRITE_MODE=always  # 查询改写：always / auto / never
SKILL_SCOPE=user     # 检索范围：user / library / all
```

### 提取参数

```env
EXTRACTOR_MODE=learner          # 提取模式：learner / default
EXTRACT_MODE=auto               # 提取时机：auto / always / never
MAX_CANDIDATES_PER_INGEST=5     # 单次最多提取几个技能
EXTRACTOR_TIMEOUT_S=240         # 提取 LLM 超时时间（秒）
USAGE_TRACKING=false            # 是否追踪技能使用情况
```

---

## Chat 内置命令

进入 `python main.py chat` 后，支持以下命令：

| 命令 | 说明 |
|---|---|
| `/help` | 查看帮助 |
| `/skills` | 列出当前用户的所有技能 |
| `/compose <task>` | 根据已保存技能脚本，组合生成完整解决方案 |
| `/extract_now [hint]` | 立即从当前对话历史提炼技能 |
| `/clear` | 清空本次对话历史 |
| `/exit` 或 `/quit` | 退出 |

---

## 技能银行结构

所有技能以 `SKILL.md` 文件形式存储在 `./SkillBank/` 目录下。
在 `learner` 模式下，每个技能还会附带一个可运行的 Python 脚本：

```
SkillBank/
├── Users/
│   └── u1/
│       ├── quicksort/
│       │   ├── SKILL.md
│       │   └── scripts/
│       │       └── quicksort.py
│       └── kl_divergence/
│           ├── SKILL.md
│           └── scripts/
│               └── kl_divergence.py
├── vectors/
│   └── skills-openai-text-embedding-3-small-xxxxxx.vecs.f32
└── index/
    └── skills-bm25.bin
```

每个 `SKILL.md` 是人类可读、可手动编辑的文件。
每个 `scripts/*.py` 是完整的、可独立运行的 Python 模块，可被 `/compose` 命令调用组合。

---

## 工作原理

```
用户输入
  ↓
[Query Rewriting]           仅多轮对话时生效，解决"它/这个"等指代问题
  │                         输出：[Query rewritten] xxx
  ↓
[Capability Decomposition]  将 query 分解为所需子能力列表
  │                         输出：[Capabilities] 计算梯度, 更新权重, ...
  ↓
[两阶段召回]
  ├─ Embedding 检索          向量相似度搜索 top-k 候选技能
  └─ Capability Pre-recall   扫描所有技能的 capabilities 标签，补充 embedding 未召回的技能
                             输出：[Capability pre-recall] 梯度下降, ...
  ↓
[Skill Selection]           LLM 从候选技能中精选真正有用的
  │                         输出：[Skills selected for context] xxx
  ↓
[Context Inject]            将技能注入 system prompt
  ↓
LLM 生成更贴合你知识背景的回复
  ↓
[Extraction]（后台异步）     判断本轮对话是否包含可提炼的技术知识
  │                         输出：[Extracting skills...]
  ↓
[Maintenance]               相似技能合并（版本升级 + capabilities 合并），新技能直接保存
  ↓
[Review]（后台异步）         审查每条技能的质量
  ├─ 新技能：检查描述、capabilities、脚本完整性
  │          score < 0.6 → 自动删除
  │          输出：[Skill saved (new): xxx] 或 [Skill rejected (new): xxx] reason
  └─ 合并技能：对比合并前后，检测功能是否回退
             regression → 标记警告，由用户决定
             输出：[Skill saved (merge): xxx] 或 [Skill saved (merge): xxx] ⚠ regression: reason
```

---

## 常见问题

**Q: 什么样的对话会被提炼成技能？（learner 模式）**

任何包含"有名字、可实现、值得复用"的技术概念都会被提炼，例如：
- 排序、搜索、图算法、动态规划等经典 CS 算法
- KL 散度、梯度计算、交叉熵等 ML/数学概念
- 注意力机制、Transformer 组件、优化器等深度学习知识

一次性的问答（"今天天气怎么样"）不会被提炼。

**Q: 技能是否会自动更新？**

会。当新对话与已有技能相似度超过阈值时，系统会自动合并：版本号递增，`capabilities` 标签取并集，技能内容越来越丰富。

**Q: Capability Pre-recall 是怎么工作的？**

每个技能在保存时会附带 `capabilities` 标签（如 `["sort an array", "partition elements"]`）。检索时，LLM 先将用户 query 分解为所需子能力，再与所有技能的 `capabilities` 做词级匹配，将命中的技能加入候选集，作为 embedding 检索的补充。

**Q: 生成的 Python 脚本在哪里？**

在 `SkillBank/Users/<user_id>/<skill_name>/scripts/` 目录下，也可以用 `/compose <task>` 命令自动组合多个脚本生成完整解决方案。

**Q: 技能审查机制是怎么工作的？**

每次技能保存后会异步触发 LLM 审查：
- **新技能**：检查描述清晰度、capabilities 是否具体、脚本是否完整。综合评分 < 0.6 的技能自动删除
- **合并技能**：对比合并前后的内容，检测是否有功能回退（regression）。发现回退时打印 ⚠ 警告，不自动删除，由用户决定是否手动处理

**Q: 如何完全禁用技能提取？**

```env
EXTRACT_MODE=never
```

**Q: BGE-M3 首次运行很慢？**

正常，首次运行需要将模型权重从磁盘加载到内存，之后同进程内复用。

**Q: 如何换一个用户？**

```env
USER_ID=u2
```

每个用户的技能独立存储，互不影响。
