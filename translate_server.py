"""Local translation server using HuggingFace t5_translate_en_ru_zh_small_1024.

A lightweight FastAPI service that wraps the multilingual T5 model
(utrobinmv/t5_translate_en_ru_zh_small_1024) for en/ru/zh translation.

Endpoints:
  GET  /health         -> {"ok": true, "model": "..."} once loaded
  POST /translate      -> {"text": "...", "target_lang": "en|zh|ru"} -> {"translation": "..."}
  GET  /translate?text=...&target_lang=...  (query form)

The port is read from the TRANSLATE_PORT env var (default 8053).
"""

from __future__ import annotations

import os
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = os.environ.get(
    "TRANSLATE_MODEL_NAME", "utrobinmv/t5_translate_en_ru_zh_small_1024"
)
DEFAULT_PORT = int(os.environ.get("TRANSLATE_PORT", "8053"))
DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("TRANSLATE_CPU", "0") != "1" else "cpu"

app = FastAPI(title="Local T5 Translation", version="1.0.0")

_tokenizer: T5Tokenizer | None = None
_model: T5ForConditionalGeneration | None = None


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "en"


def _load_models() -> None:
    global _tokenizer, _model
    _tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    _model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    _model = _model.to(DEVICE)
    _model.eval()
    print(f"[+] Model loaded: {MODEL_NAME} on {DEVICE}")


@app.on_event("startup")
def _startup() -> None:
    _load_models()


@app.get("/health")
def health() -> dict:
    return {"ok": _model is not None, "model": MODEL_NAME, "device": DEVICE}


# 模型最大序列长度 (t5_translate_en_ru_zh_small_1024 = 1024)
# T5 模型对长输入会自行截断/概括输出, 经测试每块 ~200 tokens 翻译最完整
_MAX_MODEL_LEN = 1024
_PROMPT_PREFIX = "translate to {lang}: "
_CHUNK_TOKEN_LIMIT = 200


def _token_count(text: str) -> int:
    """返回 text 的 token 数量 (不含特殊 token)。"""
    if _tokenizer is None:
        return 0
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _split_text(text: str, max_tokens: int) -> list[str]:
    """将长文本按段落/句子/词边界分块, 每块不超过 max_tokens。

    分块策略 (优先级从高到低):
      1. 按段落 (双换行) 拆分
      2. 段落过长则按句子 (. ! ? 换行) 拆分
      3. 句子过长则按 token 数量强制截断
    合并尽量短的相邻块以减少分块数量, 但不超过 max_tokens。
    """
    import re

    def _fits(t: str) -> bool:
        return _token_count(t) <= max_tokens

    if _fits(text):
        return [text]

    chunks: list[str] = []
    # 1. 按段落拆分
    paragraphs = re.split(r'(?<=\n)\n+', text)
    for para in paragraphs:
        if _fits(para):
            chunks.append(para)
        else:
            # 2. 段落过长, 按句子拆分
            sentences = re.split(r'(?<=[.!?。！？])\s+', para)
            # 也按单个换行拆分
            if len(sentences) == 1:
                sentences = para.split('\n')
            for sent in sentences:
                if _fits(sent):
                    chunks.append(sent)
                else:
                    # 3. 句子过长, 按 token 强制截断
                    words = sent.split(' ')
                    current = ""
                    for word in words:
                        candidate = current + (" " if current else "") + word
                        if _token_count(candidate) <= max_tokens:
                            current = candidate
                        else:
                            if current:
                                chunks.append(current)
                            current = word
                    if current:
                        chunks.append(current)

    # 合并相邻的短块以减少请求次数
    merged: list[str] = []
    for chunk in chunks:
        if merged and _token_count(merged[-1] + " " + chunk) <= max_tokens:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged if merged else [text]


def _do_translate(text: str, target_lang: str) -> str:
    """翻译文本, 自动分块处理超长输入。

    T5 模型对长输入 (超过 ~200 tokens) 会自行截断或概括输出,
    导致翻译不完整。此函数将文本按段落/句子边界分块, 逐块翻译后拼接,
    确保超长提示词也能完整翻译。
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")

    prompt_prefix = _PROMPT_PREFIX.format(lang=target_lang)
    chunks = _split_text(text, _CHUNK_TOKEN_LIMIT)
    results: list[str] = []
    for chunk in chunks:
        chunk_prompt = prompt_prefix + chunk
        inputs = _tokenizer(chunk_prompt, return_tensors="pt", truncation=True, max_length=_MAX_MODEL_LEN)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output = _model.generate(**inputs, max_new_tokens=512, num_beams=4)
        translated = _tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(translated)

    return " ".join(results)


@app.post("/translate")
def translate_post(req: TranslateRequest) -> dict:
    text = req.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "empty text"})
    try:
        result = _do_translate(text, req.target_lang)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return {"text": text, "target_lang": req.target_lang, "translation": result}


@app.get("/translate")
def translate_get(
    text: str = Query(...),
    target_lang: str = Query("en"),
) -> dict:
    text = text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "empty text"})
    try:
        result = _do_translate(text, target_lang)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return {"text": text, "target_lang": target_lang, "translation": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT, log_level="info")
