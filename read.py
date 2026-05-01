"""
read.py — Edge TTS Player
Pitch-preserving speed changes via WSOLA time-stretch.
Speed changes run in a background thread — UI never freezes.

Usage:
    python read.py                        # open with empty text area
    python read.py --clipboard            # pre-fill from clipboard and auto-speak
    python read.py --text "hello world"
    python read.py --file notes.txt
"""

import argparse
import asyncio
import contextlib
import json
import subprocess
import sys
import os
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import sounddevice as sd


# ── Constants ────────────────────────────────────────────────────────────────

VOICES = [
    ("Guy (US)",       "en-US-GuyNeural"),
    ("Aria (US)",      "en-US-AriaNeural"),
    ("Jenny (US)",     "en-US-JennyNeural"),
    ("Andrew (US)",    "en-US-AndrewMultilingualNeural"),
    ("Ava (US)",       "en-US-AvaMultilingualNeural"),
    ("Ryan (UK)",      "en-GB-RyanNeural"),
    ("Sonia (UK)",     "en-GB-SoniaNeural"),
    ("Natasha (AU)",   "en-AU-NatashaNeural"),
]
DEFAULT_VOICE_IDX = 0

SPEED_STEPS  = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
SPEED_LABELS = ["0.5x", "0.6x", "0.7x", "0.8x", "0.9x", "1.0x",
                "1.1x", "1.2x", "1.4x", "1.6x", "1.8x", "2.0x", "2.5x"]
DEFAULT_SPEED = 6  # 1.1x

REWIND_SEC = 10

# Quality presets: (display label, edge-tts format string, sample rate)
# The format string is monkey-patched into edge-tts at synthesis time because
# edge-tts v7.x hardcodes the output format internally. This is fragile; if
# the patch fails the app falls back to Standard quality automatically.
QUALITY_PRESETS = [
    ("Standard",  "audio-24khz-48kbitrate-mono-mp3", 24_000),
    ("High",      "audio-24khz-96kbitrate-mono-mp3", 24_000),
    ("Studio",    "audio-48khz-96kbitrate-mono-mp3", 48_000),
]
QUALITY_LABELS      = [q[0] for q in QUALITY_PRESETS]
DEFAULT_QUALITY_IDX = 0
_TTS_DEFAULT_FORMAT = "audio-24khz-48kbitrate-mono-mp3"

C = dict(
    base="#1e1e2e", mantle="#181825", surface="#313244", overlay="#45475a",
    text="#cdd6f4", subtext="#a6adc8", muted="#6c7086",
    accent="#89b4fa", green="#a6e3a1", red="#f38ba8", yellow="#f9e2af",
)

_CbStop   = sd.CallbackStop
_PREFS    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_prefs.json")
_VOICES_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_voices.json")


def _load_prefs() -> dict:
    try:
        with open(_PREFS, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_prefs(data: dict):
    try:
        with open(_PREFS, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass


def _load_voice_catalog() -> list[dict]:
    """Return cached voice list, or empty if not yet fetched."""
    try:
        with open(_VOICES_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _fetch_and_cache_voices():
    """Background thread: fetch full edge-tts voice list and persist to disk."""
    try:
        import edge_tts
        loop = asyncio.new_event_loop()
        voices = loop.run_until_complete(edge_tts.list_voices())
        loop.close()
        with open(_VOICES_CACHE, "w", encoding="utf-8") as f:
            json.dump(voices, f)
        return voices
    except Exception:
        return []


# ── Quality monkey-patch ────────────────────────────────────────────────────

@contextlib.contextmanager
def _patch_tts_format(fmt: str):
    """Temporarily override the edge-tts output format by patching aiohttp's
    ClientSession to use a custom WebSocket response class that rewrites the
    speech.config message on the wire. Falls back to Standard if patching fails."""
    if fmt == _TTS_DEFAULT_FORMAT:
        yield
        return
    try:
        import aiohttp

        class _PatchedWS(aiohttp.ClientWebSocketResponse):
            async def send_str(self, data: str, compress=None) -> None:
                data = data.replace(
                    f'"outputFormat":"{_TTS_DEFAULT_FORMAT}"',
                    f'"outputFormat":"{fmt}"',
                )
                return await super().send_str(data, compress=compress)

        _orig_init = aiohttp.ClientSession.__init__

        def _patched_init(self_s, *args, **kwargs):
            kwargs.setdefault("ws_response_class", _PatchedWS)
            _orig_init(self_s, *args, **kwargs)

        aiohttp.ClientSession.__init__ = _patched_init
        try:
            yield
        finally:
            aiohttp.ClientSession.__init__ = _orig_init
    except Exception:
        yield  # silently fall back to default format


# ── WSOLA time-stretch ───────────────────────────────────────────────────────

def _wsola(audio: np.ndarray, speed: float, sr: int = 24_000) -> np.ndarray:
    """
    Waveform Similarity Overlap-Add.
    Changes playback speed while preserving pitch.

    Key parameters for natural-sounding speech:
    - win=1024 (~43ms): enough frequency resolution to resolve voiced speech
    - search=256 (~11ms): covers full pitch periods for voices 85-250 Hz
    - Normalized cross-correlation: avoids bias toward high-energy frames
    """
    if abs(speed - 1.0) < 0.01:
        return audio.copy()

    win     = max(1024, int(sr * 0.043))   # ~43 ms; scales with sample rate
    if len(audio) < win:
        return audio.copy()
    syn_hop = win // 2                     # 50% overlap → Hann window sums to 1
    ana_hop = int(syn_hop * speed)
    search  = max(256, int(sr * 0.011))    # ~11 ms search window

    # Precompute cumulative squared energy for O(1) per-candidate norm lookup
    sq_cum = np.zeros(len(audio) + 1, dtype=np.float64)
    np.cumsum(audio.astype(np.float64) ** 2, out=sq_cum[1:])

    out_len = int(len(audio) / speed) + win
    output  = np.zeros(out_len, dtype=np.float32)
    hann    = np.hanning(win).astype(np.float32)

    ana = 0
    syn = 0

    while ana + win <= len(audio) and syn + win <= out_len:
        output[syn:syn + win] += audio[ana:ana + win] * hann
        syn += syn_hop
        next_ana = ana + ana_hop

        lo = max(0, next_ana - search)
        hi = min(len(audio) - win, next_ana + search)
        if hi > lo and syn + win <= out_len:
            ref   = output[syn:syn + win]
            e_ref = float(np.dot(ref, ref))
            block = audio[lo:hi + win]
            if len(block) >= win:
                cands = np.lib.stride_tricks.sliding_window_view(block, win)
                n     = len(cands)
                dot   = cands @ ref                         # raw cross-correlation
                if e_ref > 1e-10:
                    # Normalized correlation — unbiased toward high-energy frames
                    idx    = np.arange(lo, lo + n)
                    e_cand = np.maximum(sq_cum[idx + win] - sq_cum[idx], 1e-10)
                    corr   = dot / np.sqrt(e_cand * e_ref)
                else:
                    corr = dot                              # silence: raw is fine
                ana = lo + int(np.argmax(corr))
            else:
                ana = next_ana
        else:
            ana = next_ana

    return output[:int(len(audio) / speed)]


# ── Audio engine ──────────────────────────────────────────────────────────────

class AudioEngine:
    """
    Pitch-preserving variable-speed playback.
    The callback reads _active (a WSOLA-stretched copy of _original).
    Speed changes trigger a background WSOLA thread; a generation counter
    discards stale results when the user clicks fast.
    """

    def __init__(self, sample_rate: int = 24_000):
        self.sr = sample_rate
        self._original: np.ndarray | None = None
        self._active:   np.ndarray | None = None   # what the callback reads
        self._pos     = 0          # sample index in _active
        self._speed   = 1.0
        self._volume  = 1.0
        self._stream  = None
        self._lock    = threading.Lock()
        self._state   = "idle"     # idle | playing | paused | done
        self._gen     = 0          # stretch generation (stale cancel)
        self.on_done: callable = None
        self.duration = 0.0

    def _bump_gen(self) -> int:
        with self._lock:
            self._gen += 1
            return self._gen

    # -- State -----------------------------------------------------------------
    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._state == "playing"
    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._state == "paused"
    @property
    def is_idle(self) -> bool:
        with self._lock:
            return self._state in ("idle", "done")

    @property
    def position(self) -> float:
        """Position in seconds of original audio."""
        with self._lock:
            if self._active is None or len(self._active) == 0:
                return 0.0
            return (self._pos / len(self._active)) * self.duration

    # -- Load ------------------------------------------------------------------
    def load_file(self, path: str):
        import miniaudio
        decoded = miniaudio.decode_file(
            path,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=self.sr,
        )
        self._original = np.frombuffer(decoded.samples, dtype=np.float32).copy()
        self.duration  = len(self._original) / self.sr
        self._active   = self._original
        self._pos      = 0
        self._state    = "idle"

    # -- Callback (audio thread) -----------------------------------------------
    def _callback(self, outdata, frames, _t, _s):
        with self._lock:
            active = self._active
            if active is None:
                outdata[:] = 0
                raise _CbStop
            avail = len(active) - self._pos
            if avail <= 0:
                outdata[:] = 0
                raise _CbStop
            n = min(frames, avail)
            outdata[:n, 0] = active[self._pos:self._pos + n] * self._volume
            if n < frames:
                outdata[n:] = 0
            self._pos += n
            if n < frames:
                raise _CbStop

    def _finished(self):
        # Only fire on_done when the audio reached its natural end.
        # Pause/stop/swap all change _state before closing the stream,
        # so this correctly ignores those cases.
        if self._state != "playing":
            return
        self._state = "done"
        if self.on_done:
            self.on_done()

    # -- Stream ----------------------------------------------------------------
    def _open_stream(self):
        self._stream = sd.OutputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            callback=self._callback, finished_callback=self._finished,
        )
        self._stream.start()

    def _close_stream(self):
        s = self._stream
        if s is not None:
            self._stream = None
            try:
                s.stop()
                s.close()
            except sd.PortAudioError:
                pass

    # -- Controls --------------------------------------------------------------
    def play(self, start_sec: float = 0.0, speed: float | None = None,
             on_ready: callable = None):
        """Start playback. WSOLA runs in a background thread to avoid UI freeze."""
        with self._lock:
            self._state = "idle"
        self._close_stream()
        if speed is not None:
            self._speed = speed
        gen = self._bump_gen()
        spd = self._speed
        sec = start_sec

        sr = self.sr

        def _worker():
            if abs(spd - 1.0) < 0.01:
                stretched = self._original
            else:
                stretched = _wsola(self._original, spd, sr)
            with self._lock:
                if gen != self._gen:
                    return  # superseded; bail before touching stream
                self._active = stretched
                self._pos = int((sec / max(self.duration, 1e-9)) * len(stretched))
                self._state = "playing"
            self._open_stream()
            if on_ready:
                try:
                    on_ready()
                except tk.TclError:
                    pass  # widget destroyed before callback fired

        threading.Thread(target=_worker, daemon=True).start()

    def restart(self):
        """Replay from the beginning using the existing stretched array."""
        self._close_stream()
        with self._lock:
            self._pos = 0
        self._state = "playing"
        self._open_stream()

    def pause(self):
        if self._state != "playing":
            return
        self._state = "paused"
        self._close_stream()

    def resume(self):
        if self._state != "paused":
            return
        self._state = "playing"
        self._open_stream()

    def toggle(self):
        if self.is_playing:
            self.pause()
        elif self.is_paused:
            self.resume()

    def seek(self, sec: float):
        target = max(0.0, min(sec, self.duration))
        was_playing = self.is_playing
        if was_playing:
            with self._lock:
                self._state = "idle"   # prevent _finished from firing on_done during seek
            self._close_stream()
        with self._lock:
            if self._active is not None and len(self._active) > 0:
                self._pos = int((target / max(self.duration, 1e-9)) * len(self._active))
            else:
                self._pos = 0
        if was_playing:
            with self._lock:
                self._state = "playing"
            self._open_stream()

    def set_speed(self, speed: float, on_ready: callable = None):
        """
        Change playback speed in background (pitch-preserving WSOLA).
        Audio continues at old speed until the stretch is ready, then hot-swaps.
        """
        if self._original is None:
            self._speed = speed
            return

        self._speed = speed
        gen = self._bump_gen()
        sr = self.sr

        def _worker():
            if abs(speed - 1.0) < 0.01:
                stretched = self._original
            else:
                stretched = _wsola(self._original, speed, sr)

            with self._lock:
                if gen != self._gen:
                    return  # superseded; bail before touching stream
                was_playing = self._state == "playing"
                frac = (self._pos / len(self._active)) if (self._active is not None and len(self._active) > 0) else 0.0
                if was_playing:
                    self._state = "idle"   # prevent _finished from firing

            if was_playing:
                self._close_stream()

            with self._lock:
                if gen != self._gen:
                    return  # guard against a concurrent stop() between close and swap
                self._active = stretched
                self._pos = max(0, min(int(frac * len(stretched)), len(stretched) - 1))
                if was_playing:
                    self._state = "playing"

            if was_playing:
                self._open_stream()
            if on_ready:
                try:
                    on_ready()
                except tk.TclError:
                    pass  # widget destroyed before callback fired

        threading.Thread(target=_worker, daemon=True).start()

    def set_volume(self, vol: float):
        self._volume = max(0.0, min(vol, 1.0))

    def stop(self):
        with self._lock:
            self._state = "idle"
        self._bump_gen()        # cancel any pending WSOLA worker
        self._close_stream()
        with self._lock:
            self._pos = 0


# ── TTS fetch ─────────────────────────────────────────────────────────────────

def _decode_bytes(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = data.decode(enc).strip()
            if text:
                return text
        except (UnicodeDecodeError, ValueError):
            continue
    return data.decode("utf-8", errors="replace").strip()


def get_clipboard_text() -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; Get-Clipboard"],
        capture_output=True,
    )
    return _decode_bytes(result.stdout)


def fetch_audio(text: str, voice: str, audio_format: str = _TTS_DEFAULT_FORMAT,
                pitch_hz: int = 0) -> str:
    import edge_tts
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    loop = asyncio.new_event_loop()
    try:
        async def _run():
            pitch_str = f"{pitch_hz:+d}Hz"
            comm = edge_tts.Communicate(text, voice, rate="+0%", pitch=pitch_str)
            with open(path, "wb") as f:
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])

        with _patch_tts_format(audio_format):
            loop.run_until_complete(_run())
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    finally:
        loop.close()
    return path


# ── GUI ───────────────────────────────────────────────────────────────────────

class TTSPlayer:

    def __init__(self, initial_text: str, voice_idx: int | None, speed_idx: int | None,
                 use_prefs: bool = True):
        prefs = _load_prefs() if use_prefs else {}
        self.voice_idx   = voice_idx if voice_idx is not None else prefs.get("voice_idx", DEFAULT_VOICE_IDX)
        self.speed_idx   = speed_idx if speed_idx is not None else prefs.get("speed_idx", DEFAULT_SPEED)
        self.quality_idx = prefs.get("quality_idx", DEFAULT_QUALITY_IDX)
        self._prefs      = prefs
        _sr = QUALITY_PRESETS[self.quality_idx][2]
        self.engine      = AudioEngine(sample_rate=_sr)
        self.loading     = False
        self._fetch_gen  = 0
        self._tick_id    = None
        self._seek_lock  = False
        self._resume_sec = 0.0
        self._last_speak_text  = ""
        self._last_audio_bytes = None

        # Kick off background voice catalog fetch
        threading.Thread(target=self._load_catalog_async, daemon=True).start()

        # Restore last_text when no new text is provided
        if not initial_text and prefs.get("last_text"):
            initial_text = prefs["last_text"]

        self._build_gui(initial_text)
        self.engine.on_done = lambda: self.root.after(0, self._on_playback_done)

        if initial_text.strip():
            self.root.after(100, self._speak)

    # ── GUI layout ────────────────────────────────────────────────────────────
    def _build_gui(self, initial_text: str):
        self.root = tk.Tk()
        self.root.title("TTS Player")
        self.root.configure(bg=C["base"])
        self.root.minsize(520, 480)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        style = ttk.Style()
        style.theme_use("clam")
        for w in ("TScale", "Horizontal.TScale"):
            style.configure(w, background=C["base"], troughcolor=C["surface"],
                            sliderlength=14, sliderthickness=14)
        style.configure("TScrollbar", background=C["surface"],
                        troughcolor=C["mantle"], arrowcolor=C["muted"])
        style.configure("TCombobox", fieldbackground=C["surface"],
                        background=C["surface"], foreground=C["text"],
                        arrowcolor=C["accent"])

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = tk.Frame(self.root, bg=C["base"])
        main.grid(row=0, column=0, sticky="nsew", padx=12, pady=10)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        # Row 0: Header
        hdr = tk.Frame(main, bg=C["base"])
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        hdr.columnconfigure(1, weight=1)
        tk.Label(hdr, text="TTS Player", bg=C["base"], fg=C["text"],
                 font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w")
        self.pin_on = False
        self.pin_btn = tk.Button(
            hdr, text="📌", command=self._toggle_pin,
            bg=C["base"], fg=C["muted"], activebackground=C["base"],
            activeforeground=C["accent"], relief="flat",
            font=("Segoe UI", 11), cursor="hand2", bd=0,
        )
        self.pin_btn.grid(row=0, column=2, sticky="e")

        # Row 1: Voice + Quality
        bar = tk.Frame(main, bg=C["base"])
        bar.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        tk.Label(bar, text="Voice:", bg=C["base"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        self.voice_var = tk.StringVar(value=VOICES[self.voice_idx][0])
        self._all_voice_labels = [v[0] for v in VOICES]  # starts with favorites; extended by catalog
        self._all_voice_ids    = [v[1] for v in VOICES]
        self.voice_cb = ttk.Combobox(bar, textvariable=self.voice_var,
                                     values=self._all_voice_labels,
                                     width=26)
        self.voice_cb.pack(side="left")
        self.voice_cb.current(self.voice_idx)
        self.voice_cb.bind("<<ComboboxSelected>>", self._on_voice_changed)
        self.voice_cb.bind("<KeyRelease>",          self._on_voice_filter)
        tk.Label(bar, text="Quality:", bg=C["base"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 4))
        self.quality_var = tk.StringVar(value=QUALITY_LABELS[self.quality_idx])
        self.quality_cb = ttk.Combobox(bar, textvariable=self.quality_var,
                                       values=QUALITY_LABELS,
                                       state="readonly", width=10)
        self.quality_cb.pack(side="left")
        self.quality_cb.current(self.quality_idx)
        self.quality_cb.bind("<<ComboboxSelected>>", self._on_quality_changed)
        tk.Label(bar, text="(↑quality = ↑CPU)", bg=C["base"], fg=C["muted"],
                 font=("Segoe UI", 8)).pack(side="left", padx=(4, 0))
        tk.Label(bar, text="Pitch:", bg=C["base"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 4))
        self.pitch_var = tk.IntVar(value=0)
        ttk.Scale(bar, from_=-50, to=50, orient="horizontal",
                  variable=self.pitch_var, length=70,
                  command=lambda _: None).pack(side="left")
        self.pitch_lbl = tk.Label(bar, textvariable=tk.StringVar(), bg=C["base"],
                                  fg=C["subtext"], font=("Segoe UI", 8), width=5)
        self.pitch_lbl.pack(side="left")
        self.pitch_var.trace_add("write", self._on_pitch_changed)

        # Row 2: Text area
        txt_frame = tk.Frame(main, bg=C["surface"])
        txt_frame.grid(row=2, column=0, sticky="nsew", pady=(4, 4))
        txt_frame.columnconfigure(0, weight=1)
        txt_frame.rowconfigure(0, weight=1)
        self.text_area = tk.Text(
            txt_frame, bg=C["mantle"], fg=C["text"],
            insertbackground=C["accent"], selectbackground=C["overlay"],
            selectforeground=C["text"], font=("Consolas", 10),
            wrap="word", relief="flat", padx=10, pady=8, undo=True,
        )
        self.text_area.grid(row=0, column=0, sticky="nsew")
        if initial_text:
            self.text_area.insert("1.0", initial_text)
        sb = ttk.Scrollbar(txt_frame, orient="vertical", command=self.text_area.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.text_area.configure(yscrollcommand=sb.set)

        # Row 3: Action bar
        act = tk.Frame(main, bg=C["base"])
        act.grid(row=3, column=0, sticky="ew", pady=(2, 6))
        act.columnconfigure(1, weight=1)
        self.char_lbl = tk.Label(act, bg=C["base"], fg=C["muted"], font=("Segoe UI", 9))
        self.char_lbl.grid(row=0, column=0, sticky="w")
        btn_bar = tk.Frame(act, bg=C["base"])
        btn_bar.grid(row=0, column=1, sticky="e")
        self._btn(btn_bar, "📋 Paste", self._paste_clipboard,
                  fg=C["subtext"]).pack(side="left", padx=(0, 4))
        self._btn(btn_bar, "🗑 Clear", self._clear_text,
                  fg=C["subtext"]).pack(side="left", padx=(0, 4))
        self.speak_btn = self._btn(btn_bar, "▶  Speak", self._speak,
                                   fg=C["base"], bg=C["accent"], active_bg=C["text"])
        self.speak_btn.pack(side="left")
        self._btn(btn_bar, "💾 Save", self._save_audio,
                  fg=C["subtext"]).pack(side="left", padx=(4, 0))

        # Row 4: Divider
        tk.Frame(main, bg=C["overlay"], height=1).grid(row=4, column=0, sticky="ew")

        # Row 5: Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(main, textvariable=self.status_var, bg=C["base"],
                 fg=C["accent"], font=("Segoe UI", 9), anchor="w"
                 ).grid(row=5, column=0, sticky="ew", pady=(6, 0))

        # Row 6: Progress + time
        prog = tk.Frame(main, bg=C["base"])
        prog.grid(row=6, column=0, sticky="ew", pady=(2, 0))
        prog.columnconfigure(0, weight=1)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Scale(prog, from_=0, to=100, orient="horizontal",
                                  variable=self.progress_var, command=self._on_seek)
        self.progress.grid(row=0, column=0, sticky="ew")
        self.progress.state(["disabled"])
        self.time_var = tk.StringVar(value="0:00 / 0:00")
        tk.Label(prog, textvariable=self.time_var, bg=C["base"],
                 fg=C["subtext"], font=("Segoe UI", 9), width=13,
                 anchor="e").grid(row=0, column=1, padx=(6, 0))

        # Row 7: Transport
        transport = tk.Frame(main, bg=C["base"])
        transport.grid(row=7, column=0, pady=(6, 2))
        self.rew_btn  = self._btn(transport, "⏮ 10s", self._rewind, width=5)
        self.play_btn = self._btn(transport, "⏸", self._toggle_play,
                                  font=("Segoe UI", 15), width=3)
        self.fwd_btn  = self._btn(transport, "10s ⏭", self._forward, width=5)
        self.stop_btn = self._btn(transport, "⏹", self._stop,
                                  font=("Segoe UI", 13), width=3, fg=C["red"])
        for w in (self.rew_btn, self.play_btn, self.fwd_btn, self.stop_btn):
            w.pack(side="left", padx=4)

        # Row 8: Speed + Volume
        ctrl = tk.Frame(main, bg=C["base"])
        ctrl.grid(row=8, column=0, pady=(2, 4))
        tk.Label(ctrl, text="Speed:", bg=C["base"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        self._btn(ctrl, "−", self._slower, width=2).pack(side="left", padx=1)
        self.speed_lbl = tk.Label(ctrl, text=SPEED_LABELS[self.speed_idx],
                                  bg=C["base"], fg=C["accent"],
                                  font=("Segoe UI", 10, "bold"), width=4)
        self.speed_lbl.pack(side="left", padx=2)
        self._btn(ctrl, "+", self._faster, width=2).pack(side="left", padx=1)
        tk.Label(ctrl, text="", bg=C["base"], width=3).pack(side="left")
        tk.Label(ctrl, text="Vol:", bg=C["base"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        self.vol_var = tk.DoubleVar(value=100)
        ttk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                  variable=self.vol_var, length=90,
                  command=self._on_volume).pack(side="left")

        # Row 9: Hints
        tk.Label(main,
                 text="Space: play/pause   ←/→: ±10s   Ctrl+Enter: speak   "
                      "Ctrl+Shift+V: paste & speak   Ctrl+↑/↓: speed   [/]: volume",
                 bg=C["base"], fg=C["muted"], font=("Segoe UI", 8)
                 ).grid(row=9, column=0, sticky="w", pady=(4, 0))

        # Bindings
        self.root.bind("<space>",               self._kb_play)
        self.root.bind("<Left>",                self._kb_rewind)
        self.root.bind("<Right>",               self._kb_forward)
        self.root.bind("<Control-Return>",      lambda e: self._speak())
        self.root.bind("<Control-v>",           self._kb_paste)
        self.root.bind("<Control-Shift-V>",     lambda e: self._paste_and_speak())
        self.root.bind("<Control-Shift-KeyPress-V>", lambda e: self._paste_and_speak())
        self.root.bind("<Escape>",              lambda e: self._stop())
        self.root.bind("<Control-Up>",          lambda e: self._faster() or "break")
        self.root.bind("<Control-Down>",        lambda e: self._slower() or "break")
        self.root.bind("<bracketleft>",         self._kb_vol_down)
        self.root.bind("<bracketright>",        self._kb_vol_up)
        self.text_area.bind("<<Modified>>",     self._on_text_changed)
        self.text_area.bind("<Button-3>",       self._show_context_menu)

        self._update_char_count()
        self._set_transport("disabled")

        # Restore persisted prefs
        geom = self._prefs.get("geometry")
        if geom:
            try:
                self.root.geometry(geom)
            except tk.TclError:
                pass
        if self._prefs.get("pin_on"):
            self.pin_on = True
            self.root.attributes("-topmost", True)
            self.pin_btn.config(fg=C["accent"])
        vol = self._prefs.get("volume")
        if vol is not None:
            self.vol_var.set(vol)
            self.engine.set_volume(vol / 100)
        pitch = self._prefs.get("pitch_hz")
        if pitch is not None:
            self.pitch_var.set(pitch)

    def _btn(self, parent, text, cmd, fg=None, bg=None,
             active_bg=None, font=("Segoe UI", 10), width=6):
        return tk.Button(
            parent, text=text, command=cmd,
            bg=bg or C["surface"], fg=fg or C["text"],
            activebackground=active_bg or C["accent"],
            activeforeground=C["base"], relief="flat",
            font=font, width=width, cursor="hand2", padx=4, pady=3,
        )

    # ── Text ──────────────────────────────────────────────────────────────────
    def _on_voice_changed(self, _=None):
        """Re-generate audio with the new voice, resuming from current position."""
        idx = self.voice_cb.current()
        if idx >= 0:
            self.voice_idx = min(idx, len(VOICES) - 1)  # only update builtin index for prefs
        if self.engine.duration > 0 and not self.loading:
            self._resume_sec = self.engine.position if self.engine.is_playing or self.engine.is_paused else 0.0
            self._speak()

    # ── Voice catalog ─────────────────────────────────────────────────────────
    def _load_catalog_async(self):
        voices = _load_voice_catalog()
        if not voices:
            voices = _fetch_and_cache_voices()
        if voices:
            self.root.after(0, lambda: self._on_catalog_loaded(voices))

    def _on_catalog_loaded(self, voices: list[dict]):
        favorites = [v[0] for v in VOICES]
        fav_ids   = [v[1] for v in VOICES]
        extra_labels, extra_ids = [], []
        seen = set(fav_ids)
        for v in voices:
            vid = v.get("ShortName", "")
            if vid not in seen:
                seen.add(vid)
                name = v.get("FriendlyName", vid)
                extra_labels.append(name)
                extra_ids.append(vid)
        self._all_voice_labels = favorites + extra_labels
        self._all_voice_ids    = fav_ids + extra_ids
        self.voice_cb["values"] = self._all_voice_labels

    def _on_voice_filter(self, _=None):
        typed = self.voice_var.get().lower()
        if not typed:
            self.voice_cb["values"] = self._all_voice_labels
            return
        filtered = [lbl for lbl in self._all_voice_labels if typed in lbl.lower()]
        self.voice_cb["values"] = filtered if filtered else self._all_voice_labels

    def _on_pitch_changed(self, *_):
        v = self.pitch_var.get()
        self.pitch_lbl.config(text=f"{v:+d}Hz")

    def _on_quality_changed(self, _=None):
        idx = self.quality_cb.current()
        if idx < 0:
            return
        self.quality_idx = idx
        new_sr = QUALITY_PRESETS[idx][2]
        if self.engine.is_idle and self.engine.duration == 0:
            self.engine.sr = new_sr
        elif not self.loading:
            self.status_var.set(f"Quality → {QUALITY_LABELS[idx]}  (applies on next Speak)")

    def _on_text_changed(self, _=None):
        self.text_area.edit_modified(False)
        self._update_char_count()

    def _update_char_count(self):
        n = len(self.text_area.get("1.0", "end-1c"))
        self.char_lbl.config(text=f"{n:,} chars")

    def _clear_text(self):
        self.text_area.delete("1.0", "end")
        self._update_char_count()

    # ── Paste ─────────────────────────────────────────────────────────────────
    def _paste_clipboard(self):
        try:
            text = self.root.clipboard_get()
        except tk.TclError:
            text = ""
        if text:
            self.text_area.delete("1.0", "end")
            self.text_area.insert("1.0", text)
            self._update_char_count()
            self.status_var.set(f"Pasted {len(text):,} chars")

    def _paste_and_speak(self):
        self._paste_clipboard()
        self._speak()

    def _kb_paste(self, event):
        if self.root.focus_get() is not self.text_area:
            self._paste_clipboard()
            return "break"

    # ── Pin ───────────────────────────────────────────────────────────────────
    def _toggle_pin(self):
        self.pin_on = not self.pin_on
        self.root.attributes("-topmost", self.pin_on)
        self.pin_btn.config(fg=C["accent"] if self.pin_on else C["muted"])

    # ── Speak / load ──────────────────────────────────────────────────────────
    def _speak(self, text_override: str | None = None):
        text = text_override or self.text_area.get("1.0", "end-1c").strip()
        if not text:
            self.status_var.set("Nothing to speak")
            return

        # If already loading, cancel the in-flight request and restart
        self._fetch_gen += 1
        gen = self._fetch_gen
        self._last_speak_text = text  # for retry

        self.engine.stop()
        self._cancel_tick()
        self.loading = True
        self._set_transport("disabled")
        self.play_btn.config(text="...")
        self.status_var.set("Generating audio ...")
        if hasattr(self, "_retry_btn"):
            self._retry_btn.pack_forget()

        idx = self.voice_cb.current()
        if 0 <= idx < len(self._all_voice_ids):
            voice_id = self._all_voice_ids[idx]
        else:
            # Free-typed name: search full catalog labels
            typed = self.voice_var.get()
            try:
                match_idx = self._all_voice_labels.index(typed)
                voice_id = self._all_voice_ids[match_idx]
            except ValueError:
                voice_id = VOICES[self.voice_idx][1]
        speed = SPEED_STEPS[self.speed_idx]
        audio_fmt, new_sr = QUALITY_PRESETS[self.quality_idx][1], QUALITY_PRESETS[self.quality_idx][2]
        self.engine.sr = new_sr  # update SR before WSOLA workers are spawned
        pitch_hz = int(getattr(self, "pitch_var", None) and self.pitch_var.get() or 0)

        def _worker():
            try:
                path = fetch_audio(text, voice_id, audio_fmt, pitch_hz)
                if gen != self._fetch_gen:
                    try: os.unlink(path)
                    except OSError: pass
                    return  # stale — a newer request is in flight
                self.root.after(0, lambda: self._on_ready(path, speed, gen))
            except Exception as exc:
                if gen == self._fetch_gen:
                    self.root.after(0, lambda: self._on_error(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_ready(self, path: str, speed: float, gen: int = -1):
        if gen != -1 and gen != self._fetch_gen:
            try: os.unlink(path)
            except OSError: pass
            return  # superseded by a newer request
        try:
            self.engine.load_file(path)
            with open(path, "rb") as f:
                self._last_audio_bytes = f.read()
        except Exception as exc:
            self._last_audio_bytes = None
            self._on_error(str(exc))
            return
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        self.engine.set_volume(self.vol_var.get() / 100)
        self.status_var.set("Preparing audio ...")

        resume_sec = self._resume_sec
        self._resume_sec = 0.0

        def _on_play_started():
            self.root.after(0, self._on_play_started)

        self.engine.play(start_sec=resume_sec, speed=speed, on_ready=_on_play_started)

    def _on_play_started(self):
        self.loading = False
        self._set_transport("normal")
        self._sync_ui()
        self._start_tick()

    def _on_error(self, exc: Exception | str):
        self.loading = False
        self.play_btn.config(text="▶")
        msg = str(exc)
        try:
            import aiohttp
            if isinstance(exc, (aiohttp.ClientConnectorError, OSError)):
                msg = "No internet — edge-tts requires an online connection"
        except ImportError:
            pass
        try:
            import edge_tts.exceptions
            if isinstance(exc, edge_tts.exceptions.NoAudioReceived):
                msg = "Empty audio response — try a different voice or text"
        except (ImportError, AttributeError):
            pass
        self.status_var.set(f"Error: {msg}")
        if not hasattr(self, "_retry_btn"):
            self._retry_btn = self._btn(
                self.root, "↻ Retry", self._retry, fg=C["yellow"], width=7,
            )
        self._retry_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

    def _retry(self):
        if hasattr(self, "_retry_btn"):
            self._retry_btn.place_forget()
        text = getattr(self, "_last_speak_text", None)
        if text:
            self._speak(text_override=text)

    def _save_audio(self):
        audio = getattr(self, "_last_audio_bytes", None)
        if not audio:
            self.status_var.set("No audio to save — speak something first")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 audio", "*.mp3"), ("All files", "*.*")],
            title="Save audio",
        )
        if path:
            try:
                with open(path, "wb") as f:
                    f.write(audio)
                self.status_var.set(f"Saved → {os.path.basename(path)}")
            except OSError as e:
                self.status_var.set(f"Save failed: {e}")

    def _on_playback_done(self):
        self._cancel_tick()
        self.play_btn.config(text="▶")
        self.status_var.set("Done")
        self._seek_lock = True
        self.progress_var.set(100)
        self._seek_lock = False
        d = self.engine.duration
        self.time_var.set(f"{_fmt(d)} / {_fmt(d)}")

    # ── Transport ─────────────────────────────────────────────────────────────
    def _set_transport(self, state: str):
        for w in (self.rew_btn, self.play_btn, self.fwd_btn, self.stop_btn):
            w.config(state=state)
        self.progress.state(["!disabled"] if state == "normal" else ["disabled"])

    def _sync_ui(self):
        if self.engine.is_playing:
            self.play_btn.config(text="⏸")
            self.status_var.set(f"Playing  {SPEED_LABELS[self.speed_idx]}")
        elif self.engine.is_paused:
            self.play_btn.config(text="▶")
            self.status_var.set("Paused")

    def _toggle_play(self):
        if self.loading:
            return
        if self.engine.is_idle and self.engine.duration > 0:
            # Replay from beginning — reuse existing stretched array
            self.engine.restart()
            self._start_tick()
        else:
            self.engine.toggle()
            if self.engine.is_playing:
                self._start_tick()
        self._sync_ui()

    def _stop(self):
        self.engine.stop()
        self._cancel_tick()
        self.play_btn.config(text="▶")
        self.status_var.set("Stopped")
        self.progress_var.set(0)
        self.time_var.set(f"0:00 / {_fmt(self.engine.duration)}")

    def _rewind(self):
        if self.engine.duration > 0:
            self.engine.seek(self.engine.position - REWIND_SEC)

    def _forward(self):
        if self.engine.duration > 0:
            self.engine.seek(self.engine.position + REWIND_SEC)

    def _on_seek(self, val):
        if self._seek_lock or self.loading or self.engine.duration == 0:
            return
        self.engine.seek(float(val) / 100 * self.engine.duration)

    # ── Speed (background WSOLA — never blocks UI) ────────────────────────────
    def _slower(self):
        if self.speed_idx > 0:
            self.speed_idx -= 1
            self._apply_speed()

    def _faster(self):
        if self.speed_idx < len(SPEED_STEPS) - 1:
            self.speed_idx += 1
            self._apply_speed()

    def _apply_speed(self):
        self.speed_lbl.config(text=SPEED_LABELS[self.speed_idx])
        if self.engine.duration > 0 and not self.engine.is_idle:
            self.status_var.set(f"Adjusting to {SPEED_LABELS[self.speed_idx]} ...")
            self.engine.set_speed(
                SPEED_STEPS[self.speed_idx],
                on_ready=lambda: self.root.after(0, self._sync_ui),
            )

    # ── Volume ────────────────────────────────────────────────────────────────
    def _on_volume(self, _=None):
        self.engine.set_volume(self.vol_var.get() / 100)

    # ── Tick ──────────────────────────────────────────────────────────────────
    def _start_tick(self):
        self._cancel_tick()
        self._tick()

    def _cancel_tick(self):
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
            self._tick_id = None

    def _tick(self):
        pos = self.engine.position
        dur = self.engine.duration
        if dur > 0:
            self._seek_lock = True
            self.progress_var.set(pos / dur * 100)
            self._seek_lock = False
            self.time_var.set(f"{_fmt(pos)} / {_fmt(dur)}")
        if self.engine.is_playing:
            self._tick_id = self.root.after(200, self._tick)
        else:
            self._tick_id = None

    # ── Keyboard ──────────────────────────────────────────────────────────────
    def _in_text(self) -> bool:
        return self.root.focus_get() is self.text_area

    def _kb_play(self, e):
        if not self._in_text():
            self._toggle_play()
            return "break"

    def _kb_rewind(self, e):
        if not self._in_text():
            self._rewind()
            return "break"

    def _kb_forward(self, e):
        if not self._in_text():
            self._forward()
            return "break"

    def _kb_vol_down(self, e):
        if not self._in_text():
            self.vol_var.set(max(0, self.vol_var.get() - 5))
            self._on_volume()
            return "break"

    def _kb_vol_up(self, e):
        if not self._in_text():
            self.vol_var.set(min(100, self.vol_var.get() + 5))
            self._on_volume()
            return "break"

    # ── Context menu ──────────────────────────────────────────────────────────
    def _show_context_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0, bg=C["surface"], fg=C["text"],
                       activebackground=C["accent"], activeforeground=C["base"])
        menu.add_command(label="Cut",        command=lambda: self.text_area.event_generate("<<Cut>>"))
        menu.add_command(label="Copy",       command=lambda: self.text_area.event_generate("<<Copy>>"))
        menu.add_command(label="Paste",      command=lambda: self.text_area.event_generate("<<Paste>>"))
        menu.add_command(label="Select All", command=lambda: self.text_area.tag_add("sel", "1.0", "end"))
        menu.add_separator()
        menu.add_command(label="Speak Selection", command=self._speak_selection)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _speak_selection(self):
        try:
            text = self.text_area.get("sel.first", "sel.last").strip()
        except tk.TclError:
            return
        if text:
            self._speak(text_override=text)

    # ── Close ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        last_text = self.text_area.get("1.0", "end-1c")
        _save_prefs({
            "speed_idx":   self.speed_idx,
            "voice_idx":   self.voice_idx,
            "quality_idx": self.quality_idx,
            "pitch_hz":    self.pitch_var.get(),
            "geometry":    self.root.geometry(),
            "pin_on":      self.pin_on,
            "volume":      self.vol_var.get(),
            "last_text":   last_text[:50_000],
        })
        self._cancel_tick()
        self.engine.on_done = None  # prevent callbacks on destroyed widget
        self.engine.stop()          # also increments _gen to cancel WSOLA threads
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def _fmt(s: float) -> str:
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def main():
    parser = argparse.ArgumentParser(description="Edge TTS Player")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--clipboard", action="store_true")
    src.add_argument("--text", type=str)
    src.add_argument("--file", type=str)
    parser.add_argument("--voice", type=int, default=None)
    parser.add_argument("--speed", type=int, default=None)
    args = parser.parse_args()

    empty_clipboard = False
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, "rb") as f:
                text = _decode_bytes(f.read())
        except FileNotFoundError:
            parser.error(f"File not found: {args.file}")
    elif args.clipboard:
        text = get_clipboard_text() or ""
        if not text:
            empty_clipboard = True
    else:
        text = ""

    player = TTSPlayer(text, args.voice, args.speed)
    if empty_clipboard:
        player.status_var.set("Clipboard was empty")
    player.run()


if __name__ == "__main__":
    main()
