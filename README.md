# LearnEnByPrompt (loco) — 从 Prompt 实践中学习英语 · 纯本地算力

在日常使用 Codex、Claude Code、Antigravity 等 AI 编程助手时，我们会写下大量的英文 prompt。这些 prompt 是真实的、即学即用的英语写作素材。本项目将这些散落在各处的 prompt 收集起来，配合**纯本地运行**的中英翻译与 TTS 语音朗读，帮助你在回顾自己写过的 prompt 的过程中：

- **提升 prompt 表达的准确性** — 通过翻译对照，发现措辞问题，逐步改掉中式英语
- **积累技术英语词汇与句式** — 每一条 prompt 都是一个真实场景下的英语写作样本
- **建立个人 prompt 语料库** — 按天、周、月回顾，观察自己表达习惯的变化
- **听读结合** — TTS 语音朗读帮助你建立语感，掌握自然语调与节奏

> **loco 分支特点**：翻译和语音合成全部使用本地模型运行，**无需任何外部 API、不发送数据到云端**，隐私零泄露。适合离线环境或对数据隐私有严格要求的场景。

## 使用场景

### 1. 收集：自动汇聚你的历史 prompt

服务扫描 `~/.codex`、`~/.claude` 和 `~/.gemini/antigravity` 下的会话历史数据，将你写过的每一条 prompt 提取出来，按时间分组展示。可按今天、本周、本月、全部或指定日期范围过滤。

### 2. 对照：本地中英翻译

内置基于 HuggingFace T5 模型的本地翻译服务，随主服务自动启动。对单条或批量 prompt 生成中文译文，结果持久化保存，下次查看时直接呈现中英对照。通过对照翻译，你可以：

- 发现自己表达不清或语法错误的地方
- 学习更地道的英文技术表达
- 逐步形成"先想中文、再写英文"的翻译思维

### 3. 朗读：本地 TTS 语音合成

集成 Matcha-TTS，为英语 prompt 提供自然语音朗读，支持语速调节（0.5×–2.0×）。听读结合，加深语感。

### 4. 回顾：导出报告

可将收集到的 prompt 导出为纯文本报告，方便离线复习或归档整理。

## 鸣谢

本项目依赖以下优秀的开源项目，在此致以感谢：

### [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

由 Shivam Mehta 等人开发的基于条件流匹配（Conditional Flow Matching）的快速文本到语音合成系统。本项目以 Git submodule 形式引入，用于为英语提示词生成语音朗读。

- 模型：默认使用 `matcha_ljspeech` 单说话人模型 + `hifigan_T2_v1` 声码器
- 兼容性修复：针对 PyTorch 2.6+ 的 `weights_only=True` 默认值和 `inference_mode` tensor 问题做了 monkeypatch

### [t5_translate_en_ru_zh_small_1024](https://huggingface.co/utrobinmv/t5_translate_en_ru_zh_small_1024)

由 [utrobinmv](https://huggingface.co/utrobinmv) 发布的多语言 T5 翻译模型，支持英语、俄语、中文之间的互译。本项目通过 `transformers` 库加载该模型，提供 `translate_server.py` 本地翻译服务。

- 模型大小约 420MB，支持 CPU 推理
- 翻译格式：`translate to <lang>: <text>`（lang 为 `en`/`zh`/`ru`）

## 快速开始

```bash
# 1. 克隆项目（含 submodule）
git clone --recurse-submodules <repo-url>
cd LearnEnByPrompt

# 2. 安装依赖（主项目 + Matcha-TTS 子模块）
./install_deps.sh

# 3. 配置（可选）
cp config.example.json config.json
# 编辑 config.json 调整端口、模型等

# 4. 启动服务（自动启动本地翻译服务）
./start.sh              # 默认 127.0.0.1:8030，浏览器打开 http://localhost:8030
./start.sh --tts        # 同时启动 TTS 语音合成服务
```

## 配置

项目根目录下的 `config.json`（已被 `.gitignore` 忽略）集中管理所有配置，`config.example.json` 为模板。环境变量优先级高于配置文件。

```json
{
  "server": { "port": 8030, "host": "127.0.0.1" },
  "tts": { "port": 8052 },
  "local_translate": {
    "enabled": true,
    "port": 8053,
    "model_name": "utrobinmv/t5_translate_en_ru_zh_small_1024",
    "cpu_only": true
  }
}
```

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `8030` | 主服务监听端口 |
| `HOST` | `127.0.0.1` | 主服务监听地址 |
| `TRANSLATE_PORT` | `8053` | 本地翻译服务端口 |
| `TRANSLATE_MODEL_NAME` | `utrobinmv/...` | T5 模型名 |
| `TRANSLATE_CPU` | `1` | 强制 CPU 推理 |
| `TRANSLATE_TIMEOUT` | `120` | 翻译超时秒数 |
| `TRANSLATE_MAX_TOKENS` | `4096` | 单次请求最大 token |
| `TRANSLATE_TEMPERATURE` | `0.1` | 翻译温度 |
| `TTS_PORT` | `8052` | TTS 服务端口 |

## 本地翻译服务

`translate_server.py` 使用 HuggingFace T5 模型提供本地翻译，不依赖外部 API：

```bash
# 启动本地翻译服务
.venv/bin/python translate_server.py

# 测试
curl "http://127.0.0.1:8053/translate?text=Hello+world&target_lang=zh"
# {"text":"Hello world","target_lang":"zh","translation":"你好世界"}

curl -X POST http://127.0.0.1:8053/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"你好世界","target_lang":"en"}'
# {"text":"你好世界","target_lang":"en","translation":"Hello world"}
```

## TTS 语音合成

`tts_server.py` 使用 Matcha-TTS 模型为英语提示词提供语音朗读，支持语速调节（0.5×–2.0×）：

```bash
# 使用 Matcha-TTS 的 venv 运行
PYTHONPATH=./Matcha-TTS Matcha-TTS/.venv/bin/python tts_server.py

# 测试
curl "http://127.0.0.1:8052/synthesize?text=hello+world&speed=1.0" --output out.wav
```

## API

| 路由 | 方法 | 说明 |
| --- | --- | --- |
| `/api/prompts?range=day\|week\|month\|all` | GET | 刷新收集（JSON） |
| `/api/prompts/{id}/translate?force=false` | POST | 翻译单条提示词 |
| `/api/report?range=...` | GET | 纯文本报告 |
| `/api/translations` | GET | 已记录的全部翻译 |
| `/api/health` | GET | 服务与翻译接口状态 |
| `/api/translate/start` | POST | 启动本地翻译子进程 |
| `/api/translate/stop` | POST | 停止本地翻译子进程 |
| `/api/translate/status` | GET | 翻译服务状态 |
| `/api/tts/start` | POST | 启动 TTS 子进程 |
| `/api/tts/stop` | POST | 停止 TTS 子进程 |
| `/api/tts/status` | GET | TTS 服务状态 |
| `/api/tts/synthesize?text=...&speed=1.0` | GET | 合成语音（WAV） |

## 项目结构

```
LearnEnByPrompt/
├── server.py              # 主 Web 服务
├── collect_prompts.py     # 提示词收集器
├── translate_server.py    # 本地 T5 翻译服务
├── tts_server.py          # Matcha-TTS 语音合成服务
├── install_deps.sh        # 一键安装脚本
├── start.sh               # 一键启动脚本
├── config.json            # 配置文件（gitignore）
├── config.example.json    # 配置模板
├── pyproject.toml         # 项目元数据
├── static/                # 前端页面
├── data/                  # 翻译数据与日志（gitignore）
└── Matcha-TTS/            # Git submodule
```
