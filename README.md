# LearnEnByPrompt — 从 Prompt 实践中学习英语

在日常使用 Codex、Claude Code 等 AI 编程助手时，我们会写下大量的英文 prompt。这些 prompt 是真实的、即学即用的英语写作素材。本项目将这些散落在各处的 prompt 收集起来，配合中英翻译对照，帮助你在回顾自己写过的 prompt 的过程中：

- **提升 prompt 表达的准确性** — 通过翻译对照，发现措辞问题，逐步改掉中式英语
- **积累技术英语词汇与句式** — 每一条 prompt 都是一个真实场景下的英语写作样本
- **建立个人 prompt 语料库** — 按天、周、月回顾，观察自己表达习惯的变化

## 使用场景

### 1. 收集：自动汇聚你的历史 prompt

服务扫描 `~/.codex` 和 `~/.claude` 下的会话历史文件，将你写过的每一条 prompt 提取出来，按时间分组展示。你可以按今天、本周、本月、全部或指定日期范围过滤，快速找到想要回顾的 prompt。

### 2. 对照：中英翻译

对单条或批量 prompt 调用翻译服务，生成中文译文。翻译结果持久化保存，下次查看时直接呈现中英对照。通过对照翻译，你可以：

- 发现自己表达不清或语法错误的地方
- 学习更地道的英文技术表达
- 逐步形成"先想中文、再写英文"的翻译思维

### 3. 回顾：导出报告

可将收集到的 prompt 导出为纯文本报告，方便离线复习或归档整理。

## 快速开始

```bash
uv sync              # 首次安装依赖
uv run server.py     # 默认 127.0.0.1:8030，浏览器打开 http://localhost:8030
```

## 配置

环境变量优先级高于 `config.json`（已被 `.gitignore` 忽略，可从 `config.example.json` 复制）。

```json
{
  "server": { "port": 8030, "host": "127.0.0.1" }
}
```

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `8030` | 主服务监听端口 |
| `HOST` | `127.0.0.1` | 主服务监听地址 |
| `TRANSLATE_URL` | — | 翻译接口地址（OpenAI 兼容 `/v1/chat/completions`） |
| `TRANSLATE_MODEL` | 自动探测 | 指定模型名，缺省时取 `/v1/models` 第一个 |
| `TRANSLATE_TIMEOUT` | `180` | 翻译请求超时秒数 |

## API

| 路径 | 方法 | 说明 |
| --- | --- | --- |
| `/api/prompts?range=day\|week\|month\|all\|YYYY-MM\|YYYY-MM-DD` | GET | 刷新收集（JSON） |
| `/api/prompts/{id}/translate?force=false` | POST | 翻译单条（命中缓存则直接返回） |
| `/api/report?range=...` | GET | 纯文本报告 |
| `/api/translations` | GET | 已记录的全部翻译 |
| `/api/health` | GET | 服务与翻译接口状态 |
