"""voice-loop: push-to-talk voice assistant on petsi.

mic (I2S) -> whisper.cpp -> Claude -> piper -> NS4168 speaker
Everything runs locally except the Claude call.
"""
import asyncio
import json
import os
import subprocess
import threading
import time
import wave
from pathlib import Path

import anthropic
import numpy as np
from piper import PiperVoice
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

CAPTURE_DEV = os.getenv("CAPTURE_DEV", "hw:2,0")   # raw card; softvol is playback-only
WHISPER_BIN = ROOT / "whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = Path(os.getenv("WHISPER_MODEL", ROOT / "whisper.cpp/models/ggml-base.en.bin"))
PIPER_VOICE = Path(os.getenv("PIPER_VOICE", ROOT / "voices/en_US-lessac-medium.onnx"))
VENV_PY = ROOT / ".venv/bin/python"
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)

RATE, CHANNELS, WIDTH = 48000, 2, 4
CHUNK_FRAMES = 2400          # 50 ms
VISUAL_GAIN = 6.0            # meter only; does not touch the audio

SYSTEM = (
    "You are a voice assistant. Your reply is read aloud by a text-to-speech engine, "
    "so answer in two or three short sentences of plain spoken English. No markdown, "
    "no lists, no code blocks, no emoji, no stage directions. If you genuinely need "
    "more room to answer well, take it, but never pad."
)

state = {"recording": False, "level": 0.0, "status": "idle",
         "transcript": "", "reply": "", "timings": {}}

_buf = bytearray()
_buf_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_thread: threading.Thread | None = None

client = anthropic.Anthropic()
# Loaded once at startup. Shelling out to `python -m piper` per reply re-read the
# 63 MB ONNX model every time, which cost ~2.4 s of the round trip.
voice = PiperVoice.load(str(PIPER_VOICE))
app = FastAPI()


def _capture_loop() -> None:
    """Read raw frames from arecord; feed both the level meter and the buffer."""
    global _proc
    _proc = subprocess.Popen(
        ["arecord", "-D", CAPTURE_DEV, "-f", "S32_LE", "-r", str(RATE),
         "-c", str(CHANNELS), "-t", "raw", "-q"],
        stdout=subprocess.PIPE,
    )
    nbytes = CHUNK_FRAMES * CHANNELS * WIDTH
    while state["recording"]:
        data = _proc.stdout.read(nbytes)
        if not data:
            break
        with _buf_lock:
            _buf.extend(data)
        frames = np.frombuffer(data, dtype="<i4")
        left = frames[0::2].astype(np.float64) / 2**31   # mic sits in the left slot
        if left.size:
            state["level"] = min(1.0, float(np.sqrt(np.mean(left**2))) * VISUAL_GAIN)
    state["level"] = 0.0


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


def _pipeline() -> dict:
    """Blocking: wav -> 16k mono -> whisper -> Claude -> piper -> speaker."""
    stamp = int(time.time())
    t = {}
    with _buf_lock:
        pcm = bytes(_buf)

    if len(pcm) < RATE * CHANNELS * WIDTH * 0.3:      # under ~0.3 s
        state["status"] = "idle"
        return {"error": "That was too short to hear. Hold the button while you talk."}

    raw_wav = WORK / f"{stamp}_raw.wav"
    with wave.open(str(raw_wav), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(WIDTH)
        w.setframerate(RATE)
        w.writeframes(pcm)

    # Take the LEFT channel explicitly. Plain -ac 1 averages L+R, and since the
    # right slot is silence that would quietly cost 6 dB of signal.
    state["status"] = "converting"
    mono = WORK / f"{stamp}_16k.wav"
    s = time.time()
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_wav),
          "-af", "pan=mono|c0=c0", "-ar", "16000", "-ac", "1", str(mono)])
    t["convert"] = round(time.time() - s, 2)

    state["status"] = "transcribing"
    s = time.time()
    out = _run([str(WHISPER_BIN), "-m", str(WHISPER_MODEL), "-f", str(mono),
                "-t", "4", "-nt"])
    transcript = " ".join(out.stdout.split()).strip()
    t["whisper"] = round(time.time() - s, 2)
    state["transcript"] = transcript

    if not transcript or transcript in ("[BLANK_AUDIO]", "(blank audio)"):
        state["status"] = "idle"
        return {"transcript": "", "reply": "", "timings": t,
                "error": "Didn't catch any speech in that."}

    state["status"] = "thinking"
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "No ANTHROPIC_API_KEY. Put it in ~/voice-loop/.env, then: "
            "sudo systemctl restart voice-loop")
    s = time.time()
    resp = client.beta.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},     # spoken reply: latency beats depth
        system=SYSTEM,
        messages=[{"role": "user", "content": transcript}],
    )
    if resp.stop_reason == "refusal":
        reply = "Sorry, I can't help with that one."
    else:
        reply = "".join(b.text for b in resp.content if b.type == "text").strip()
    t["claude"] = round(time.time() - s, 2)
    state["reply"] = reply

    state["status"] = "speaking"
    s = time.time()
    spoken = WORK / f"{stamp}_reply.wav"
    with wave.open(str(spoken), "wb") as wf:
        voice.synthesize_wav(reply, wf)
    t["piper"] = round(time.time() - s, 2)
    # No -D: goes through `default`, which is where the softvol volume lives.
    subprocess.run(["aplay", "-q", str(spoken)], check=False)

    t["total"] = round(sum(v for k, v in t.items() if k != "total"), 2)
    state["timings"] = t
    state["status"] = "idle"
    return {"transcript": transcript, "reply": reply, "timings": t}


@app.post("/start")
async def start():
    global _thread
    if state["recording"]:
        return JSONResponse({"error": "already recording"}, status_code=409)
    with _buf_lock:
        _buf.clear()
    state.update(recording=True, status="listening", transcript="", reply="", timings={})
    _thread = threading.Thread(target=_capture_loop, daemon=True)
    _thread.start()
    return {"ok": True}


@app.post("/stop")
async def stop():
    if not state["recording"]:
        return JSONResponse({"error": "not recording"}, status_code=409)
    state["recording"] = False
    if _proc and _proc.poll() is None:
        _proc.terminate()                 # unblocks the thread's pending read
    if _thread:
        _thread.join(timeout=3)
    state["status"] = "converting"
    try:
        return await asyncio.to_thread(_pipeline)
    except subprocess.CalledProcessError as e:
        state["status"] = "idle"
        detail = (e.stderr or "").strip().splitlines()[-1:] or [str(e)]
        return JSONResponse({"transcript": state["transcript"],
                             "error": f"{Path(e.cmd[0]).name} failed: {detail[0]}"},
                            status_code=500)
    except anthropic.APIStatusError as e:
        state["status"] = "idle"
        hint = " Check the key in ~/voice-loop/.env." if e.status_code == 401 else ""
        return JSONResponse({"transcript": state["transcript"],
                             "error": f"Claude API {e.status_code}.{hint}"}, status_code=500)
    except Exception as e:
        state["status"] = "idle"
        msg = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
        return JSONResponse({"transcript": state["transcript"], "error": msg},
                            status_code=500)


@app.get("/events")
async def events(request: Request):
    async def gen():
        try:
            while True:
                # Without this the generator never returns, and a single open tab
                # makes `systemctl restart` hang waiting for the connection.
                if await request.is_disconnected():
                    return
                yield "data: " + json.dumps({
                    "recording": state["recording"],
                    "level": round(state["level"], 4),
                    "status": state["status"],
                }) + "\n\n"
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/")
async def index():
    return FileResponse(ROOT / "static/index.html")
