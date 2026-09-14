#!/usr/bin/env python3
"""
server.py -- AI Agent 提示词收集 Web 服务。

功能:
  * 复用 collect_prompts.py 的收集器, 在 Web 上按 天/周/月/全部 刷新收集
  * 查看收集结果 (按天分组, 含 agent / 会话 / 工作目录)
  * 调用 LLM (OpenAI 兼容 chat/completions 接口) 生成用户提示词的英文对照,
    翻译结果持久化记录到 data/translations.json, 调用日志追加到 data/translate_log.jsonl

启动:
  uv run server.py                 # 默认 127.0.0.1:8030
  PORT=9000 uv run server.py       # 自定义端口

环境变量:
  PORT             监听端口 (默认 8030)
  TRANSLATE_URL    翻译接口地址 (默认 http://172.25.92.88:18080/v1/chat/completions)
  TRANSLATE_MODEL  指定模型名; 缺省时自动从 /v1/models 获取第一个模型
  TRANSLATE_TIMEOUT 翻译请求超时秒数 (默认 180)
  TRANSLATE_API_KEY API 密钥 (可选, 用于需要认证的接口)
  TRANSLATE_MAX_TOKENS 单次请求最大 token 数 (默认 4096)
  TRANSLATE_TEMPERATURE 翻译温度 (默认 0.1)
  HOST             监听地址 (默认 127.0.0.1)

配置文件:
  config.json 项目根目录下的 JSON 配置文件, 包含翻译 API、服务器和 TTS 设置。
  环境变量优先级高于配置文件; 若 config.json 不存在则使用内置默认值。
  可参考 config.example.json 创建自己的 config.json。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from collect_prompts import (
    ClaudeCodeCollector,
    CodexCollector,
    Prompt,
    _parse_range,
    generate_report,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
TRANSLATIONS_FILE = DATA_DIR / "translations.json"
TRANSLATE_LOG = DATA_DIR / "translate_log.jsonl"

# Load config.json (env vars override file values)
_CONFIG_FILE = BASE_DIR / "config.json"
_cfg: dict[str, Any] = {}
if _CONFIG_FILE.exists():
    with _CONFIG_FILE.open(encoding="utf-8") as _f:
        _cfg = json.load(_f)
_tcfg = _cfg.get("translate", {})
_scfg = _cfg.get("server", {})
_ttscfg = _cfg.get("tts", {})

TRANSLATE_URL = os.environ.get(
    "TRANSLATE_URL", _tcfg.get("url", "http://172.25.92.88:18080/v1/chat/completions")
)
# OpenAI 兼容 API 的 base (由 completions 地址反推, 用于 /v1/models 等)
API_BASE = TRANSLATE_URL.rsplit("/chat/completions", 1)[0]
MODEL_OVERRIDE = os.environ.get("TRANSLATE_MODEL", _tcfg.get("model", ""))
TRANSLATE_API_KEY = os.environ.get("TRANSLATE_API_KEY", _tcfg.get("api_key", ""))
TRANSLATE_TIMEOUT = float(os.environ.get("TRANSLATE_TIMEOUT", _tcfg.get("timeout", 180)))
TRANSLATE_MAX_TOKENS = int(os.environ.get("TRANSLATE_MAX_TOKENS", _tcfg.get("max_tokens", 4096)))
TRANSLATE_TEMPERATURE = float(os.environ.get("TRANSLATE_TEMPERATURE", _tcfg.get("temperature", 0.1)))
DEFAULT_PORT = int(os.environ.get("PORT", _scfg.get("port", 8030)))
DEFAULT_HOST = os.environ.get("HOST", _scfg.get("host", "127.0.0.1"))
TTS_PORT = int(os.environ.get("TTS_PORT", _ttscfg.get("port", 8052)))
MATCHA_DIR = BASE_DIR / "Matcha-TTS"
TTS_PY = BASE_DIR / "tts_server.py"
TTS_VENV_PY = MATCHA_DIR / ".venv" / "bin" / "python"
TTS_URL = f"http://127.0.0.1:{TTS_PORT}"

SYSTEM_PROMPT_ZH2EN = _tcfg.get(
    "system_prompt_zh2en",
    "You are a professional translator. Translate the user's text into "
    "natural, fluent English. Output ONLY the translation itself: no "
    "explanations, no quotation marks, no prefixes. Keep code snippets, "
    "file paths, shell commands, URLs and technical identifiers exactly as "
    "they are. If the text is already in English, output it unchanged.",
)

SYSTEM_PROMPT_EN2ZH = _tcfg.get(
    "system_prompt_en2zh",
    "你是一位专业翻译。将用户的文本翻译成自然、流畅的简体中文。"
    "只输出译文本身: 不要解释、不要引号、不要前缀。"
    "保持代码片段、文件路径、shell 命令、URL 和技术标识符原样不变。"
    "如果文本已经是中文, 则原样输出。",
)

app = FastAPI(title="Prompt Collector", version="1.0.0")


def detect_lang(text: str) -> str:
    """简易语言检测: 含较多中文字符返回 'zh', 否则 'en'。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk >= 3 else "en"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 翻译记录存储 (data/translations.json)
# ---------------------------------------------------------------------------

class TranslationStore:
    """以 prompt id 为 key 的翻译记录持久化存储, 带线程锁。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._records = loaded
            except (json.JSONDecodeError, OSError):
                self._records = {}

    def _save(self) -> None:
        """原子写: 先写临时文件再 rename (调用方需持有锁)。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(self._records, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def get(self, pid: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._records.get(pid)
            if not rec:
                return None
            rec = dict(rec)
            # backward compat: 旧记录把译文存在 "en" 字段
            if "translation" not in rec and "en" in rec:
                rec["translation"] = rec["en"]
            return rec

    def put(self, pid: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._records[pid] = record
            self._save()


store = TranslationStore(TRANSLATIONS_FILE)

# 每个 prompt 一把锁, 避免同一提示词被并发重复翻译
_PID_LOCKS: dict[str, threading.Lock] = {}
_PID_LOCKS_GUARD = threading.Lock()


def _pid_lock(pid: str) -> threading.Lock:
    with _PID_LOCKS_GUARD:
        lock = _PID_LOCKS.get(pid)
        if lock is None:
            lock = threading.Lock()
            _PID_LOCKS[pid] = lock
        return lock


def _log_translate(entry: dict[str, Any]) -> None:
    """翻译调用日志, 追加写入 JSONL (成功/失败都记录)。"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"logged_at": _now_iso(), **entry}
        with TRANSLATE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # 日志写失败不影响主流程


# ---------------------------------------------------------------------------
# TTS subprocess manager
# ---------------------------------------------------------------------------

class TTSManager:
    """Manages the Matcha-TTS subprocess lifecycle."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.running:
                return {"ok": True, "status": "already_running"}
            if not TTS_PY.exists():
                raise HTTPException(500, f"TTS server script not found: {TTS_PY}")
            if not TTS_VENV_PY.exists():
                raise HTTPException(500, f"Matcha-TTS venv not found: {TTS_VENV_PY}")
            env = os.environ.copy()
            env["TTS_PORT"] = str(TTS_PORT)
            # Ensure matcha package is importable from the submodule source
            pp = str(MATCHA_DIR)
            env["PYTHONPATH"] = pp + os.pathsep + env.get("PYTHONPATH", "")
            self._proc = subprocess.Popen(
                [str(TTS_VENV_PY), str(TTS_PY)],
                cwd=str(BASE_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        # Wait for the TTS server to become healthy (model download + load)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if not self.running:
                out = ""
                if self._proc and self._proc.stdout:
                    try:
                        out = self._proc.stdout.read()[-2000:]
                    except Exception:
                        pass
                raise HTTPException(500, f"TTS process exited early: {out}")
            try:
                r = httpx.get(f"{TTS_URL}/health", timeout=5)
                if r.status_code == 200:
                    return {"ok": True, "status": "started"}
            except Exception:
                pass
            time.sleep(2)
        raise HTTPException(504, "TTS server did not become healthy within 5 minutes")

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self.running:
                self._proc = None
                return {"ok": True, "status": "not_running"}
            try:
                self._proc.send_signal(signal.SIGTERM)
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            return {"ok": True, "status": "stopped"}

    def synthesize(self, text: str, speed: float | None = None) -> bytes:
        if not self.running:
            raise HTTPException(503, "TTS service is not running")
        try:
            params = {"text": text}
            if speed is not None:
                params["speed"] = speed
            resp = httpx.get(
                f"{TTS_URL}/synthesize",
                params=params,
                timeout=120,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"TTS service unreachable: {exc}") from exc
        if resp.status_code != 200:
            detail = resp.text[:300]
            raise HTTPException(resp.status_code, f"TTS error: {detail}")
        return resp.content

    def health(self) -> dict[str, Any]:
        if not self.running:
            return {"ok": False, "running": False}
        try:
            r = httpx.get(f"{TTS_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return {"ok": True, "running": True, **data}
        except Exception:
            pass
        return {"ok": False, "running": True, "status": "starting"}


tts_manager = TTSManager()

# ---------------------------------------------------------------------------
# 提示词 id / 收集
# ---------------------------------------------------------------------------

def prompt_id(agent: str, session: str, text: str) -> str:
    """稳定的提示词标识: sha256(agent|session|text) 前 16 位。

    同一条提示词无论被扫描多少次, id 不变, 因此翻译记录可以跨会话持久复用。
    """
    h = hashlib.sha256()
    h.update(agent.encode("utf-8"))
    h.update(b"\x00")
    h.update(session.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def collect_filtered(rng: str) -> tuple[list[Prompt], str, list[str]]:
    """运行所有收集器, 去重, 按时间范围过滤。

    Returns:
        (prompts, label, warnings)
    """
    try:
        start, end, label = _parse_range(rng)
    except SystemExit as exc:  # _parse_range 对非法值调用 sys.exit
        raise HTTPException(400, f"Invalid range: {exc.code}") from None

    collectors = [CodexCollector(), ClaudeCodeCollector()]
    all_prompts: list[Prompt] = []
    warnings: list[str] = []
    for c in collectors:
        try:
            all_prompts.extend(c.collect())
        except Exception as exc:  # 单个 Agent 失败不影响整体
            warnings.append(f"{c.name} collection failed: {exc}")

    # 与脚本相同的去重 key: (session, text[:200])
    seen: set = set()
    unique: list[Prompt] = []
    for p in all_prompts:
        key = (p.session, p.text[:200])
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    filtered = [p for p in unique if start <= p.ts < end]
    # 倒序: 最新在前
    filtered.sort(key=lambda p: p.ts, reverse=True)
    return filtered, label, warnings


# ---------------------------------------------------------------------------
# LLM 翻译
# ---------------------------------------------------------------------------

_models_lock = threading.Lock()
_models_cache: list[str] | None = None


def _resolve_models() -> list[str]:
    """从 /v1/models 获取模型列表 (成功后缓存; 失败不缓存, 下次重试)。"""
    global _models_cache
    with _models_lock:
        if _models_cache is not None:
            return _models_cache
    _hdrs = {}
    if TRANSLATE_API_KEY:
        _hdrs["Authorization"] = f"Bearer {TRANSLATE_API_KEY}"
    try:
        resp = httpx.get(f"{API_BASE}/models", timeout=10, headers=_hdrs)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data") or data.get("models") or []
        models = [m.get("id") or m.get("name") for m in items if isinstance(m, dict)]
        models = [m for m in models if m]
    except Exception:
        models = []
    with _models_lock:
        if models:
            _models_cache = models
    return models


def _resolve_model() -> str:
    if MODEL_OVERRIDE:
        return MODEL_OVERRIDE
    models = _resolve_models()
    if not models:
        raise HTTPException(
            502,
            f"Cannot discover model from {API_BASE}/models; "
            f"set TRANSLATE_MODEL explicitly.",
        )
    return models[0]


def _call_llm(pid: str, agent: str, ts_iso: str, text: str) -> dict[str, Any]:
    """调用翻译接口并落盘记录, 返回记录 dict。中英文互译: 自动检测源语言, 翻译成对方语言。"""
    src_lang = detect_lang(text)
    tgt_lang = "en" if src_lang == "zh" else "zh"
    system_prompt = SYSTEM_PROMPT_ZH2EN if src_lang == "zh" else SYSTEM_PROMPT_EN2ZH
    model = _resolve_model()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "max_tokens": TRANSLATE_MAX_TOKENS,
        "temperature": TRANSLATE_TEMPERATURE,
    }
    _hdrs = {"Content-Type": "application/json"}
    if TRANSLATE_API_KEY:
        _hdrs["Authorization"] = f"Bearer {TRANSLATE_API_KEY}"
    started = time.monotonic()
    try:
        resp = httpx.post(TRANSLATE_URL, json=payload, timeout=TRANSLATE_TIMEOUT, headers=_hdrs)
    except httpx.HTTPError as exc:
        latency = round((time.monotonic() - started) * 1000, 1)
        _log_translate({
            "ok": False, "prompt_id": pid, "model": model,
            "error": f"request failed: {exc}", "latency_ms": latency,
        })
        raise HTTPException(502, f"Translation service unreachable: {exc}") from exc

    latency = round((time.monotonic() - started) * 1000, 1)

    if resp.status_code != 200:
        _log_translate({
            "ok": False, "prompt_id": pid, "model": model,
            "status_code": resp.status_code,
            "error": resp.text[:500], "latency_ms": latency,
        })
        raise HTTPException(
            502, f"Translation service error {resp.status_code}: {resp.text[:300]}"
        )

    try:
        data = resp.json()
        message = data["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        _log_translate({
            "ok": False, "prompt_id": pid, "model": model,
            "error": f"malformed response: {exc}", "latency_ms": latency,
        })
        raise HTTPException(502, "Malformed translation response") from exc

    # 思考型模型 (如 Qwen3) 把思考过程放在 reasoning_content,
    # 正式回答在 content; content 为空时回退到 reasoning_content。
    translation = (message.get("content") or message.get("reasoning_content") or "").strip()
    if not translation:
        _log_translate({
            "ok": False, "prompt_id": pid, "model": model,
            "error": "empty translation", "latency_ms": latency,
        })
        raise HTTPException(502, "Model returned an empty translation")

    record: dict[str, Any] = {
        "agent": agent,
        "ts": ts_iso,
        "text": text,
        "translation": translation,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "model": model,
        "translated_at": _now_iso(),
        "latency_ms": latency,
        "usage": data.get("usage"),
    }
    store.put(pid, record)
    _log_translate({
        "ok": True, "prompt_id": pid, "agent": agent, "model": model,
        "src_lang": src_lang, "tgt_lang": tgt_lang,
        "text_chars": len(text), "translation_chars": len(translation),
        "latency_ms": latency, "usage": data.get("usage"),
    })
    return record


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """健康检查: 服务状态 + 翻译服务可达性。"""
    models: list[str] = []
    translate_ok = False
    try:
        models = _resolve_models()
        translate_ok = bool(models)
    except Exception:
        pass
    return {
        "ok": True,
        "translate": {
            "url": TRANSLATE_URL,
            "ok": translate_ok,
            "models": models,
        },
        "recorded_translations": len(store._records),
        "tts": tts_manager.health(),
    }


@app.get("/api/prompts")
def api_prompts(
    rng: str = Query("day", alias="range"),
    user_only: bool = Query(True),
) -> dict[str, Any]:
    """刷新收集: 扫描全部 Agent 数据, 按范围过滤, 附带已有翻译记录。"""
    filtered, label, warnings = collect_filtered(rng)

    by_agent: dict[str, int] = {}
    by_day: dict[str, int] = {}
    items: list[dict[str, Any]] = []
    for p in filtered:
        pid = prompt_id(p.agent, p.session, p.text)
        rec = store.get(pid)
        local = p.ts.astimezone()
        by_agent[p.agent] = by_agent.get(p.agent, 0) + 1
        day = local.strftime("%Y-%m-%d")
        by_day[day] = by_day.get(day, 0) + 1
        src_lang = (rec.get("src_lang") if rec else None) or detect_lang(p.text)
        tgt_lang = "en" if src_lang == "zh" else "zh"
        items.append({
            "id": pid,
            "agent": p.agent,
            "ts": local.isoformat(),
            "time": local.strftime("%H:%M:%S"),
            "date": day,
            "session": p.session,
            "cwd": p.cwd,
            "text": p.text,
            "translation": rec.get("translation") if rec else None,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "translation_meta": (
                {
                    "model": rec.get("model"),
                    "translated_at": rec.get("translated_at"),
                    "latency_ms": rec.get("latency_ms"),
                }
                if rec
                else None
            ),
        })

    return {
        "label": label,
        "generated_at": _now_iso(),
        "total": len(items),
        "user_only": user_only,
        "by_agent": dict(sorted(by_agent.items())),
        "by_day": dict(sorted(by_day.items())),
        "translated": sum(1 for i in items if i["translation"]),
        "warnings": warnings,
        "prompts": items,
    }


@app.post("/api/prompts/{pid}/translate")
def api_translate(pid: str, force: bool = Query(False)) -> dict[str, Any]:
    """翻译单条提示词 (已有记录且非 force 时直接返回缓存记录)。"""
    with _pid_lock(pid):
        rec = store.get(pid)
        if rec and not force:
            _log_translate({
                "ok": True, "prompt_id": pid, "cached": True,
                "model": rec.get("model"),
            })
            return {"cached": True, **rec}
        if rec is None:
            # 记录里没有这条提示词 (可能来自新的扫描); 从最近一次收集结果中补元数据
            agent, ts_iso, text = _find_prompt_meta(pid)
            if text is None:
                raise HTTPException(
                    404,
                    "Prompt not found in recent collection; refresh first.",
                )
        else:
            agent, ts_iso, text = rec["agent"], rec["ts"], rec["text"]
        new_rec = _call_llm(pid, agent, ts_iso, text)
        return {"cached": False, **new_rec}


def _find_prompt_meta(pid: str) -> tuple[str, str, str | None]:
    """在当前 'all' 范围内查找 prompt 的元数据 (agent / ts / text)。"""
    filtered, _, _ = collect_filtered("all")
    for p in filtered:
        if prompt_id(p.agent, p.session, p.text) == pid:
            return p.agent, p.ts.astimezone().isoformat(), p.text
    return "", "", None


@app.get("/api/report")
def api_report(rng: str = Query("day", alias="range")) -> PlainTextResponse:
    """与命令行脚本一致的纯文本报告 (可直接下载/复制)。"""
    filtered, label, _ = collect_filtered(rng)
    start, end, _ = _parse_range(rng)
    return PlainTextResponse(generate_report(filtered, start, end, label))


@app.get("/api/translations")
def api_translations() -> dict[str, Any]:
    """已记录的全部翻译 (查看记录用)。"""
    with store._lock:
        records = {k: dict(v) for k, v in store._records.items()}
    return {"total": len(records), "records": records}


# ---------------------------------------------------------------------------
# TTS API
# ---------------------------------------------------------------------------

@app.post("/api/tts/start")
def api_tts_start() -> dict[str, Any]:
    """Start the Matcha-TTS subprocess."""
    return tts_manager.start()


@app.post("/api/tts/stop")
def api_tts_stop() -> dict[str, Any]:
    """Stop the Matcha-TTS subprocess."""
    return tts_manager.stop()


@app.get("/api/tts/status")
def api_tts_status() -> dict[str, Any]:
    """Check TTS service status."""
    return tts_manager.health()


@app.get("/api/tts/synthesize")
def api_tts_synthesize(text: str = Query(...), speed: float = Query(default=1.0, ge=0.1, le=3.0)) -> Response:
    """Synthesize text to WAV audio via Matcha-TTS."""
    wav_bytes = tts_manager.synthesize(text, speed=speed)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache"},
    )


# 静态页面 (在 API 路由之后挂载, 保证 /api/* 优先匹配)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt collector web service")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    import uvicorn

    print(f"Prompt collector web service: http://{args.host}:{args.port}")
    print(f"Translation endpoint: {TRANSLATE_URL}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
