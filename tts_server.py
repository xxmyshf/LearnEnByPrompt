"""Lightweight HTTP server wrapping Matcha-TTS inference.

Endpoints:
  GET /health              -> 200 {"ok": true} once the model is loaded
  GET /synthesize?text=... -> audio/wav bytes

The port is read from the TTS_PORT environment variable (default 8052).
"""
from __future__ import annotations

import torch
import io
import os
import tempfile
from argparse import Namespace
from pathlib import Path

# Patch torch.load to default weights_only=False for legacy checkpoint compatibility
# (Matcha-TTS checkpoints predate PyTorch 2.6+ weights_only=True default)
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs or kwargs["weights_only"] is None:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import Response

from matcha.cli import (
    MATCHA_URLS,
    VOCODER_URLS,
    assert_model_downloaded,
    get_device,
    load_matcha,
    load_vocoder,
    process_text,
    to_waveform,
)
from matcha.utils.utils import get_user_data_dir

app = FastAPI()

# ---------------------------------------------------------------------------
# Model setup (single-speaker LJ Speech by default)
# ---------------------------------------------------------------------------
_data_dir = get_user_data_dir()
_model_name = os.environ.get("MATCHA_MODEL", "matcha_ljspeech")
_vocoder_name = os.environ.get("MATCHA_VOCODER", "hifigan_T2_v1")
_spk = None if _model_name == "matcha_ljspeech" else int(os.environ.get("MATCHA_SPK", "0"))
_speaking_rate = float(os.environ.get("MATCHA_SPEAKING_RATE", "0.95"))
_temperature = float(os.environ.get("MATCHA_TEMPERATURE", "0.667"))
_steps = int(os.environ.get("MATCHA_STEPS", "10"))

_model = None
_vocoder = None
_denoiser = None
_device = None


def _load_models() -> None:
    global _model, _vocoder, _denoiser, _device

    args = Namespace(cpu=os.environ.get("MATCHA_CPU", "0") == "1")
    _device = get_device(args)

    ckpt_path = _data_dir / f"{_model_name}.ckpt"
    vocoder_path = _data_dir / f"{_vocoder_name}"
    assert_model_downloaded(ckpt_path, MATCHA_URLS[_model_name])
    assert_model_downloaded(vocoder_path, VOCODER_URLS[_vocoder_name])

    _model = load_matcha(_model_name, ckpt_path, _device)
    _vocoder, _denoiser = load_vocoder(_vocoder_name, vocoder_path, _device)


@app.on_event("startup")
def _startup() -> None:
    _load_models()


@app.get("/health")
def health() -> dict:
    return {"ok": _model is not None}


@app.get("/synthesize")
def synthesize(text: str = Query(...), speed: float = Query(default=None, ge=0.1, le=3.0)) -> Response:
    if _model is None:
        return Response(status_code=503, content="model not loaded")

    text = text.strip()
    if not text:
        return Response(status_code=400, content="empty text")

    # speed=1.0 = normal; >1 slower; <1 faster. Maps to length_scale.
    rate = _speaking_rate if speed is None else speed * _speaking_rate
    tp = process_text(0, text, _device)
    with torch.no_grad():
        output = _model.synthesise(
            tp["x"],
            tp["x_lengths"],
            n_timesteps=_steps,
            temperature=_temperature,
            spks=torch.tensor([_spk], device=_device) if _spk is not None else None,
            length_scale=rate,
        )
        # Clone mel to detach from inference_mode context (PyTorch 2.6+ compat)
        mel = output["mel"].clone().detach()
        waveform = to_waveform(mel, _vocoder, _denoiser)

    wav_io = io.BytesIO()
    sf.write(wav_io, waveform.numpy(), 22050, format="WAV", subtype="PCM_16")
    return Response(content=wav_io.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("TTS_PORT", "8052"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
