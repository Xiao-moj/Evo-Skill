# EvoSkill

EvoSkill 是一个从真实对话中自动提炼、存储、召回并持续演化技能的长期记忆框架。它既可以作为终端聊天工具使用，也可以嵌入自己的 agent 或应用里。

适合两类场景：
- 把对话中反复出现的知识、流程、代码模式沉淀为可复用技能
- 为 AI 助手补一层“长期记忆”，让它在后续对话里自动召回你的技能库

## 特性

- 从多轮对话中提炼技能，并保存为人类可读的 `SKILL.md`
- `learner` 模式面向 CS 学习场景，可为技能生成可运行的 Python 脚本
- 检索链路包含 query rewrite、embedding 检索、capability pre-recall 和 LLM selection
- 自动做去重、合并、版本升级，并在保存后异步审查技能质量
- 支持 OpenAI 兼容接口、Anthropic、本地 `bge-m3` 和测试用 `hashing` embedding
- 可选使用 `llm`、`codex`、`claude-code` 作为聊天执行后端
- 支持导入/导出 anthropics/skills 风格的技能目录

## 快速开始

要求：Python 3.10+

下面默认使用 `python main.py ...`。如果你执行了 editable install，也可以把命令替换成 `evoskill ...`。

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

如果你想用本地 `bge-m3` 作为 embedding，再额外安装一个可选依赖：

```bash
python -m pip install -e ".[flagembedding]"
# 或
python -m pip install -e ".[sentence-transformers]"
```

说明：
- EvoSkill 调用模型接口时不依赖官方 Python SDK，核心网络请求基于标准库实现
- 项目本身仍然依赖 `jinja2` 和 `pydantic`

### 2. 配置环境变量

```bash
cp .env.example .env
```

最小可用配置如下：

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.openai.com
LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxx
EMBEDDING_BASE_URL=https://api.openai.com
EMBEDDING_MODEL=text-embedding-3-small

EXTRACTOR_MODE=learner
```

如果只想快速本地试跑，也可以把 embedding 改成：

```env
EMBEDDING_PROVIDER=hashing
```

它适合测试，不适合正式检索。

### 3. 启动

```bash
python main.py chat
```

进入交互模式后，先试几个基础命令：

```text
/help
/skills
/extract_now
```

## 常用命令

### CLI

| 命令 | 说明 |
| --- | --- |
| `python main.py chat` | 启动交互式聊天 |
| `python main.py list` | 列出当前用户的技能 |
| `python main.py delete <skill_id>` | 删除指定技能 |
| `python main.py export skills.json` | 导出当前用户技能为 JSON |
| `python main.py compose "<task>"` | 根据已保存的技能脚本组合解决方案 |

### Chat 内命令

| 命令 | 说明 |
| --- | --- |
| `/help` | 查看帮助 |
| `/skills` | 查看当前用户技能 |
| `/compose <task>` | 在聊天里直接调用技能脚本组合方案 |
| `/extract_now [hint]` | 立即从当前对话历史触发一次提炼 |
| `/clear` | 清空当前对话历史 |
| `/exit` / `/quit` | 退出 |

## 配置速查

所有配置都来自 `.env` 或环境变量；常用项如下。

### 模型与 embedding

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` 或 `anthropic` |
| `LLM_API_KEY` | 空 | 也会回退读取 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| `LLM_BASE_URL` | `https://api.openai.com` | OpenAI 兼容接口可替换 |
| `LLM_MODEL` | `gpt-4o-mini` | 主对话、改写、技能选择、审查使用的模型 |
| `EMBEDDING_PROVIDER` | `openai` | `openai` / `bge-m3` / `hashing` |
| `EMBEDDING_API_KEY` | 空 | `openai` embedding 时需要 |
| `EMBEDDING_BASE_URL` | `https://api.openai.com` | OpenAI 兼容接口可替换 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding 模型名 |

### 检索与提取

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `USER_ID` | `u1` | 当前用户，决定技能隔离范围 |
| `SKILL_BANK_PATH` | `./SkillBank` | 本地技能库目录 |
| `SKILL_SCOPE` | `user` | `user` / `library` / `all` |
| `REWRITE_MODE` | `always` | `always` / `auto` / `never` |
| `MIN_SCORE` | `0.4` | 技能注入阈值 |
| `TOP_K` | `3` | 每轮 embedding 检索数量 |
| `EXTRACTOR_MODE` | `learner` | `learner` 或 `default` |
| `EXTRACT_MODE` | `auto` | `auto` / `always` / `never` |
| `MAX_CANDIDATES_PER_INGEST` | `5` | 单次最多提炼多少个候选技能 |
| `EXTRACTOR_TIMEOUT_S` | `240` | 提取阶段 LLM 超时秒数 |
| `USAGE_TRACKING` | `true` | 是否记录技能是否真正被回复用到 |

### 可选：Agent 后端

如果你不想用内置 `llm` 回复，也可以把聊天执行阶段切到 `codex` 或 `claude-code`。这部分依赖 Docker 运行时。

```env
AGENT_BACKEND=codex
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5
AGENT_WORKSPACE_DIR=.
AGENT_LOGS_DIR=./trail
```

说明：
- `AGENT_BACKEND` 可选 `llm`、`codex`、`claude-code`
- 即使聊天后端不是 `llm`，`LLM_*` 配置仍会用于 query rewrite、技能选择、技能审查等流程

更完整的配置项见 [`.env.example`](./.env.example)。

## 工作方式

EvoSkill 的主流程可以概括成 5 步：

1. 用户输入后，系统会按需做 query rewrite，解决多轮对话里的指代问题。
2. 检索阶段先做 embedding 搜索，再根据 capability 标签做一次 pre-recall 补召回。
3. LLM 从候选技能里筛选真正相关的技能，并把它们注入当前上下文。
4. 对话结束后，后台异步提炼候选技能，并执行去重、合并、版本升级。
5. 新技能或合并后的技能会再经过一次质量审查；低质量技能会被删除，疑似回退会给出警告。

如果你只关心“它会不会影响当前回答”，可以记住一条：
技能检索发生在回答前，技能提炼和审查发生在回答后。

## SkillBank 结构

技能默认保存在 `./SkillBank`。每条技能至少包含一个 `SKILL.md`，在 `learner` 模式下还可能附带可运行脚本。

```text
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
└── index/
```

补充说明：
- `SKILL.md` 是人类可读、可手动编辑的
- `scripts/*.py` 是可独立运行的技能实现，可被 `/compose` 或 `compose` 命令复用
- 技能目录格式兼容 anthropics/skills 风格，便于导入导出
