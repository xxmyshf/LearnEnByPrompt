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


def _do_translate(text: str, target_lang: str) -> str:
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")
    prompt = f"translate to {target_lang}: {text}"
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        output = _model.generate(**inputs, max_new_tokens=512, num_beams=4)
    return _tokenizer.decode(output[0], skip_special_tokens=True)


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
