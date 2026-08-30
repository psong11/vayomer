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
import httpx
import numpy as np
from piper import PiperVoice
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

CAPTURE_DEV = os.getenv("CAPTURE_DEV", "hw:2,0")   # raw card; softvol is playback-only
WHISPER_BIN = ROOT / "whisper.cpp/build/bin/whisper-cli"
WHISPER_DIR = ROOT / "whisper.cpp/models"
VOICE_DIR = ROOT / "voices"
WHISPER_MODEL = Path(os.getenv("WHISPER_MODEL", WHISPER_DIR / "ggml-base.en.bin"))
PIPER_VOICE = Path(os.getenv("PIPER_VOICE", VOICE_DIR / "en_US-lessac-medium.onnx"))

# Live, switchable from the dashboard. Applies to the next turn, and survives
# a restart so a service bounce doesn't silently reset your picks.
CONFIG_FILE = ROOT / "work/config.json"
config = {"whisper": WHISPER_MODEL.name,
          "voice": f"piper:{PIPER_VOICE.stem}",
          "claude": os.getenv("CLAUDE_MODEL", "claude-opus-5")}
try:
    config.update({k: v for k, v in json.loads(CONFIG_FILE.read_text()).items()
                   if k in config})
except Exception:
    pass


def save_config() -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
    except OSError:
        pass
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
app = FastAPI()

# Loaded once and cached per voice. Shelling out to `python -m piper` per reply
# re-read the 63 MB ONNX model every time, which cost ~2.4 s of the round trip.
_voice_cache: dict[str, PiperVoice] = {}
_claude_models: list[dict] = []


def get_voice(stem: str) -> PiperVoice:
    if stem not in _voice_cache:
        _voice_cache[stem] = PiperVoice.load(str(VOICE_DIR / f"{stem}.onnx"))
    return _voice_cache[stem]


def list_whisper() -> list[dict]:
    out = []
    for f in sorted(WHISPER_DIR.glob("ggml-*.bin")):
        if f.name.startswith("ggml-") and "for-tests" not in f.name:
            out.append({"id": f.name,
                        "label": f.stem.replace("ggml-", ""),
                        "mb": round(f.stat().st_size / 1e6)})
    return out


ELEVEN_URL = "https://api.elevenlabs.io"
_eleven_cache: list[dict] = []


def list_voices() -> list[dict]:
    local = [{"id": f"piper:{f.stem}", "label": f"{f.stem}  (local)",
              "mb": round(f.stat().st_size / 1e6)}
             for f in sorted(VOICE_DIR.glob("*.onnx"))]
    return local + list_eleven()


def list_eleven() -> list[dict]:
    """ElevenLabs voices, if a key is present. Cached; empty list is a valid answer."""
    global _eleven_cache
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        return []
    if not _eleven_cache:
        try:
            r = httpx.get(f"{ELEVEN_URL}/v2/voices", headers={"xi-api-key": key},
                          params={"page_size": 30}, timeout=15)
            r.raise_for_status()
            _eleven_cache = [{"id": f"11l:{v['voice_id']}",
                              "label": f"{v.get('name', v['voice_id'])}  (ElevenLabs)"}
                             for v in r.json().get("voices", [])]
        except Exception:
            return []
    return _eleven_cache


def synthesize(text: str, voice_id: str, out_path: Path) -> None:
    """Render `text` to a wav at `out_path` using whichever provider the id names."""
    if voice_id.startswith("11l:"):
        key = os.getenv("ELEVENLABS_API_KEY")
        if not key:
            raise RuntimeError("No ELEVENLABS_API_KEY in ~/voice-loop/.env")
        r = httpx.post(
            f"{ELEVEN_URL}/v1/text-to-speech/{voice_id.split(':', 1)[1]}",
            headers={"xi-api-key": key, "Content-Type": "application/json"},
            params={"output_format": "pcm_24000"},
            json={"text": text,
                  "model_id": os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")},
            timeout=60,
        )
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:180]}")
        # pcm_24000 is raw signed 16-bit mono LE, so just put a wav header on it
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(r.content)
    else:
        stem = voice_id.split(":", 1)[1] if ":" in voice_id else voice_id
        with wave.open(str(out_path), "wb") as wf:
            get_voice(stem).synthesize_wav(text, wf)


def list_claude() -> list[dict]:
    global _claude_models
    if not _claude_models:
        try:
            _claude_models = [{"id": m.id, "label": m.display_name}
                              for m in client.models.list(limit=20)]
        except Exception:
            _claude_models = [{"id": "claude-opus-5", "label": "Claude Opus 5"}]
    return _claude_models


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
    out = _run([str(WHISPER_BIN), "-m", str(WHISPER_DIR / config["whisper"]),
                "-f", str(mono), "-t", "4", "-nt"])
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
        model=config["claude"],
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
    synthesize(reply, config["voice"], spoken)
    t["tts"] = round(time.time() - s, 2)
    # No -D: goes through `default`, which is where the softvol volume lives.
    subprocess.run(["aplay", "-q", str(spoken)], check=False)

    t["total"] = round(sum(v for k, v in t.items() if k != "total"), 2)
    state["timings"] = t
    state["status"] = "idle"
    return {"transcript": transcript, "reply": reply, "timings": t,
            "used": dict(config)}


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


@app.on_event("startup")
async def warm() -> None:
    """Preload the configured local voice off the request path, so the first
    reply after a restart doesn't pay the ONNX load (~2-3 s for a -high voice)."""
    if config["voice"].startswith("piper:"):
        threading.Thread(
            target=lambda: get_voice(config["voice"].split(":", 1)[1]),
            daemon=True).start()


@app.get("/models")
async def models():
    return {"whisper": list_whisper(), "voices": list_voices(),
            "claude": list_claude(), "active": config}


@app.post("/config")
async def set_config(body: dict):
    for k in ("whisper", "voice", "claude"):
        if k in body and body[k]:
            config[k] = body[k]
    if body.get("voice", "").startswith("piper:"):
        try:
            get_voice(config["voice"].split(":", 1)[1])   # warm it for the next turn
        except Exception as e:
            return JSONResponse({"error": f"could not load voice: {e}"}, status_code=400)
    save_config()
    return {"active": config}


@app.get("/")
async def index():
    return FileResponse(ROOT / "static/index.html")
