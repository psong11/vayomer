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
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

CAPTURE_DEV = os.getenv("CAPTURE_DEV", "hw:2,0")   # raw card; softvol is playback-only
WHISPER_BIN = ROOT / "whisper.cpp/build/bin/whisper-cli"
# whisper-cli has an absolute RUNPATH baked in at build time, so it stops finding
# libwhisper.so if the project directory is ever moved. Point it at its own folder.
WHISPER_ENV = {**os.environ, "LD_LIBRARY_PATH": str(ROOT / "whisper.cpp/build/bin")}
WHISPER_DIR = ROOT / "whisper.cpp/models"
VOICE_DIR = ROOT / "voices"
WHISPER_MODEL = Path(os.getenv("WHISPER_MODEL", WHISPER_DIR / "ggml-base.en.bin"))
PIPER_VOICE = Path(os.getenv("PIPER_VOICE", VOICE_DIR / "en_US-lessac-medium.onnx"))

# Live, switchable from the dashboard. Applies to the next turn, and survives
# a restart so a service bounce doesn't silently reset your picks.
CONFIG_FILE = ROOT / "work/config.json"
config = {"whisper": WHISPER_MODEL.name,
          "voice": f"piper:{PIPER_VOICE.stem}",
          "claude": os.getenv("CLAUDE_MODEL", "claude-opus-5"),
          "input": "petsi",     # "petsi" = I2S mic, "browser" = the viewing device
          "output": "petsi"}    # "petsi" = NS4168 speaker, "browser" = the viewing device
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

# The pipeline as the UI draws it. Each stage carries what happened there, so the
# dashboard is a live window into the backend rather than a spinner.
STAGES = ["capture", "convert", "whisper", "claude", "tts", "speaker"]

state = {"recording": False, "level": 0.0, "status": "idle",
         "transcript": "", "reply": "", "timings": {}, "stages": {}}


def reset_stages() -> None:
    state["stages"] = {k: {"state": "pending", "t": None, "info": None}
                       for k in STAGES}


def stage(name: str, st: str, info: str | None = None,
          t: float | None = None) -> None:
    sg = state["stages"].setdefault(
        name, {"state": "pending", "t": None, "info": None})
    sg["state"] = st
    if info is not None:
        sg["info"] = info
    if t is not None:
        sg["t"] = round(t, 2)


reset_stages()

_buf = bytearray()
_buf_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_thread: threading.Thread | None = None

client = anthropic.Anthropic()
app = FastAPI()

# Loaded once and cached per voice. Shelling out to `python -m piper` per reply
# re-read the 63 MB ONNX model every time, which cost ~2.4 s of the round trip.
_voice_cache: dict[str, PiperVoice] = {}
_voice_lock = threading.Lock()
_claude_models: list[dict] = []


def get_voice(stem: str) -> PiperVoice:
    # Locked: the startup warmer and a request can ask for the same voice at once,
    # and without this they each start a separate 114 MB ONNX load and fight for CPU.
    with _voice_lock:
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
_eleven_error: str | None = None


def list_voices() -> list[dict]:
    local = [{"id": f"piper:{f.stem}", "label": f"{f.stem}  (local)",
              "mb": round(f.stat().st_size / 1e6)}
             for f in sorted(VOICE_DIR.glob("*.onnx"))]
    return local + list_eleven()


def list_eleven() -> list[dict]:
    """ElevenLabs voices, if a key is present. Records why it failed rather than
    returning a bare empty list -- a silent [] here looks identical to 'no key'."""
    global _eleven_cache, _eleven_error
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        _eleven_error = None
        return []
    if not _eleven_cache:
        try:
            r = httpx.get(f"{ELEVEN_URL}/v2/voices", headers={"xi-api-key": key},
                          params={"page_size": 30}, timeout=15)
            if r.status_code != 200:
                try:
                    why = r.json()["detail"]["message"]
                except Exception:
                    why = r.text[:140]
                _eleven_error = f"ElevenLabs voices unavailable ({r.status_code}): {why}"
                return []
            _eleven_cache = [{"id": f"11l:{v['voice_id']}",
                              "label": f"{v.get('name', v['voice_id'])}  (ElevenLabs)"}
                             for v in r.json().get("voices", [])]
            _eleven_error = None
        except Exception as e:
            _eleven_error = f"ElevenLabs voices unavailable: {type(e).__name__}: {e}"
            return []
    return _eleven_cache


def voice_label(voice_id: str) -> str:
    """Friendly name for the flow diagram: an ElevenLabs id means nothing to a reader."""
    for v in list_voices():
        if v["id"] == voice_id:
            name = v["label"].split("  (")[0]
            return name.split(" - ")[0].strip()[:20]
    return voice_id.split(":", 1)[-1][:20]


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


# Not every model accepts the same request options: `fallbacks` is Opus 5 / Fable 5
# only, and `effort` is rejected by Haiku 4.5 and older. Rather than hardcode a table
# that goes stale as models ship, start optimistic, drop whatever the API rejects,
# and remember the answer per model.
_model_caps: dict[str, dict] = {}


def ask_claude(model: str, transcript: str):
    caps = _model_caps.setdefault(model, {"fallbacks": True, "effort": True})
    for _ in range(3):
        kw: dict = {}
        if caps["fallbacks"]:
            kw["betas"] = ["server-side-fallback-2026-07-01"]
            kw["fallbacks"] = "default"
        if caps["effort"]:
            kw["output_config"] = {"effort": "low"}  # spoken reply: latency beats depth
        try:
            return client.beta.messages.create(
                model=model, max_tokens=1024, system=SYSTEM,
                messages=[{"role": "user", "content": transcript}], **kw)
        except anthropic.BadRequestError as e:
            msg = str(e)
            if "fallbacks" in msg and caps["fallbacks"]:
                caps["fallbacks"] = False
                continue
            if "effort" in msg and caps["effort"]:
                caps["effort"] = False
                continue
            raise
    raise RuntimeError(f"{model} rejected the request options")


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


def _pipeline(src: Path | None = None) -> dict:
    """Blocking: audio -> 16k mono -> whisper -> Claude -> TTS -> speaker.

    `src` is a file uploaded from the browser's microphone; when it is None the
    audio comes from the I2S mic buffer instead.
    """
    stamp = int(time.time())
    t = {}

    if src is None:
        with _buf_lock:
            pcm = bytes(_buf)
        if len(pcm) < RATE * CHANNELS * WIDTH * 0.3:      # under ~0.3 s
            state["status"] = "idle"
            return {"error": "That was too short to hear. Hold the button while you talk."}
        src = WORK / f"{stamp}_raw.wav"
        with wave.open(str(src), "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(WIDTH)
            w.setframerate(RATE)
            w.writeframes(pcm)
        # The I2S mic sits in the LEFT slot and the right is silence, so take the
        # left channel explicitly -- plain -ac 1 would average in the silence and
        # quietly cost 6 dB. Browser audio is ordinary, so it needs no such fix.
        chan = ["-af", "pan=mono|c0=c0"]
    else:
        chan = []

    stage("convert", "active")
    state["status"] = "converting"
    mono = WORK / f"{stamp}_16k.wav"
    s = time.time()
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
          *chan, "-ar", "16000", "-ac", "1", str(mono)])
    t["convert"] = round(time.time() - s, 2)
    stage("convert", "done", "48k stereo \u2192 16k mono", t["convert"])

    stage("whisper", "active", config["whisper"].replace("ggml-", "").replace(".bin", ""))
    state["status"] = "transcribing"
    s = time.time()
    out = _run([str(WHISPER_BIN), "-m", str(WHISPER_DIR / config["whisper"]),
                "-f", str(mono), "-t", "4", "-nt"], env=WHISPER_ENV)
    transcript = " ".join(out.stdout.split()).strip()
    t["whisper"] = round(time.time() - s, 2)
    stage("whisper", "done", f"{len(transcript.split())} words", t["whisper"])
    state["transcript"] = transcript

    if not transcript or transcript in ("[BLANK_AUDIO]", "(blank audio)"):
        state["status"] = "idle"
        return {"transcript": "", "reply": "", "timings": t,
                "error": "Didn't catch any speech in that."}

    stage("claude", "active", config["claude"])
    state["status"] = "thinking"
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "No ANTHROPIC_API_KEY. Put it in ~/voice-loop/.env, then: "
            "sudo systemctl restart voice-loop")
    s = time.time()
    resp = ask_claude(config["claude"], transcript)
    if resp.stop_reason == "refusal":
        reply = "Sorry, I can't help with that one."
    else:
        reply = "".join(b.text for b in resp.content if b.type == "text").strip()
    t["claude"] = round(time.time() - s, 2)
    stage("claude", "done", f"{len(reply)} chars back", t["claude"])
    state["reply"] = reply

    vlabel = voice_label(config["voice"])
    stage("tts", "active", vlabel)
    state["status"] = "speaking"
    s = time.time()
    spoken = WORK / f"{stamp}_reply.wav"
    synthesize(reply, config["voice"], spoken)
    t["tts"] = round(time.time() - s, 2)
    with wave.open(str(spoken), "rb") as _w:
        spoken_secs = _w.getnframes() / _w.getframerate()
    stage("tts", "done", vlabel, t["tts"])

    audio_url = None
    if config["output"] == "browser":
        # Hand the wav back and let the page play it; nothing goes to the amp.
        stage("speaker", "done", f"{spoken_secs:.1f}s \u2192 browser", 0.0)
        audio_url = f"/reply/{spoken.name}"
    else:
        stage("speaker", "active", f"{spoken_secs:.1f}s out")
        state["status"] = "playing"
        s = time.time()
        # No -D: goes through `default`, which is where the softvol volume lives.
        subprocess.run(["aplay", "-q", str(spoken)], check=False)
        t["play"] = round(time.time() - s, 2)
        stage("speaker", "done", f"{spoken_secs:.1f}s out", t["play"])

    # `play` is how long the reply takes to speak aloud, not latency -- keep it
    # visible per-stage but out of the total, which is time-to-first-sound.
    t["total"] = round(sum(v for k, v in t.items() if k not in ("total", "play")), 2)
    state["timings"] = t
    state["status"] = "idle"
    return {"transcript": transcript, "reply": reply, "timings": t,
            "used": dict(config), "audio": audio_url}


@app.post("/start")
async def start():
    global _thread
    if state["recording"]:
        return JSONResponse({"error": "already recording"}, status_code=409)
    with _buf_lock:
        _buf.clear()
    reset_stages()
    stage("capture", "active", "mic open")
    state.update(recording=True, status="listening", transcript="", reply="", timings={})
    _thread = threading.Thread(target=_capture_loop, daemon=True)
    _thread.start()
    return {"ok": True}


async def _finish(src: Path | None = None):
    """Run the pipeline off the event loop, turning any failure into a clean message."""
    try:
        return await asyncio.to_thread(_pipeline, src)
    except subprocess.CalledProcessError as e:
        _fail_active()
        detail = (e.stderr or "").strip().splitlines()[-1:] or [str(e)]
        return JSONResponse({"transcript": state["transcript"],
                             "error": f"{Path(e.cmd[0]).name} failed: {detail[0]}"},
                            status_code=500)
    except anthropic.APIStatusError as e:
        _fail_active()
        try:
            detail = e.response.json()["error"]["message"]
        except Exception:
            detail = str(e)[:200]
        hint = " Check the key in ~/vayomer/.env." if e.status_code == 401 else ""
        return JSONResponse({"transcript": state["transcript"],
                             "error": f"Claude API {e.status_code}: {detail}{hint}"},
                            status_code=500)
    except Exception as e:
        _fail_active()
        msg = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
        return JSONResponse({"transcript": state["transcript"], "error": msg},
                            status_code=500)


def _fail_active() -> None:
    for v in state["stages"].values():
        if v["state"] == "active":
            v["state"] = "failed"
    state["status"] = "idle"


@app.post("/stop")
async def stop():
    """Finish a turn captured on petsi's own microphone."""
    if not state["recording"]:
        return JSONResponse({"error": "not recording"}, status_code=409)
    state["recording"] = False
    if _proc and _proc.poll() is None:
        _proc.terminate()                 # unblocks the thread's pending read
    if _thread:
        _thread.join(timeout=3)
    secs = len(_buf) / (RATE * CHANNELS * WIDTH)
    stage("capture", "done", f"{secs:.1f}s \u00b7 petsi mic", secs)
    state["status"] = "converting"
    return await _finish()


@app.post("/turn")
async def turn(audio: UploadFile = File(...)):
    """Finish a turn recorded by the browser and uploaded as one blob."""
    reset_stages()
    data = await audio.read()
    up = WORK / f"{int(time.time())}_upload"
    up.write_bytes(data)
    stage("capture", "done", f"{len(data)/1024:.0f} KB \u00b7 browser mic", None)
    state["status"] = "converting"
    return await _finish(up)


@app.get("/reply/{name}")
async def reply_audio(name: str):
    """Serve a synthesized reply so the browser can play it."""
    f = (WORK / name).resolve()
    if f.parent != WORK.resolve() or not f.is_file():   # no path traversal
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f, media_type="audio/wav")


@app.get("/events")
async def events(request: Request):
    async def gen():
        try:
            while True:
                # Without this the generator never returns, and a single open tab
                # makes `systemctl restart` hang waiting for the connection.
                if await request.is_disconnected():
                    return
                remote_tts = config["voice"].startswith("11l:")
                stages = {
                    k: {**v, "remote": k == "claude" or (k == "tts" and remote_tts)}
                    for k, v in state["stages"].items()
                }
                yield "data: " + json.dumps({
                    "recording": state["recording"],
                    "level": round(state["level"], 4),
                    "status": state["status"],
                    "stages": stages,
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
    voices = list_voices()
    return {"whisper": list_whisper(), "voices": voices,
            "claude": list_claude(), "active": config, "notice": _eleven_error}


@app.post("/config")
async def set_config(body: dict):
    for k in ("whisper", "voice", "claude", "input", "output"):
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
