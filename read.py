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
import json
import subprocess
import sys
import os
import tempfile
import threading
import tkinter as tk
from tkinter import ttk

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

C = dict(
    base="#1e1e2e", mantle="#181825", surface="#313244", overlay="#45475a",
    text="#cdd6f4", subtext="#a6adc8", muted="#6c7086",
    accent="#89b4fa", green="#a6e3a1", red="#f38ba8", yellow="#f9e2af",
)

_CbStop   = sd.CallbackStop
_PREFS    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_prefs.json")


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


# ── WSOLA time-stretch ───────────────────────────────────────────────────────

def _wsola(audio: np.ndarray, speed: float) -> np.ndarray:
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

    win     = 1024              # ~43 ms at 24 kHz
    if len(audio) < win:
        return audio.copy()
    syn_hop = win // 2          # 50% overlap → Hann window sums to 1
    ana_hop = int(syn_hop * speed)
    search  = 256               # ±256 samples — covers speech pitch periods

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

    SR = 24_000

    def __init__(self):
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
            sample_rate=self.SR,
        )
        self._original = np.frombuffer(decoded.samples, dtype=np.float32).copy()
        self.duration  = len(self._original) / self.SR
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
            samplerate=self.SR, channels=1, dtype="float32",
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

        def _worker():
            if abs(spd - 1.0) < 0.01:
                stretched = self._original
            else:
                stretched = _wsola(self._original, spd)
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
            self._close_stream()
        with self._lock:
            if self._active is not None and len(self._active) > 0:
                self._pos = int((target / max(self.duration, 1e-9)) * len(self._active))
            else:
                self._pos = 0
        if was_playing:
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

        def _worker():
            if abs(speed - 1.0) < 0.01:
                stretched = self._original
            else:
                stretched = _wsola(self._original, speed)

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


def fetch_audio(text: str, voice: str) -> str:
    import edge_tts
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    loop = asyncio.new_event_loop()
    try:
        async def _run():
            comm = edge_tts.Communicate(text, voice, rate="+0%")
            with open(path, "wb") as f:
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        f.write(chunk["data"])

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
        self.voice_idx  = voice_idx if voice_idx is not None else prefs.get("voice_idx", DEFAULT_VOICE_IDX)
        self.speed_idx  = speed_idx if speed_idx is not None else prefs.get("speed_idx", DEFAULT_SPEED)
        self.engine     = AudioEngine()
        self.loading     = False
        self._fetch_gen  = 0
        self._tick_id    = None
        self._seek_lock  = False
        self._resume_sec = 0.0

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

        # Row 1: Voice
        bar = tk.Frame(main, bg=C["base"])
        bar.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        tk.Label(bar, text="Voice:", bg=C["base"], fg=C["subtext"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        self.voice_var = tk.StringVar(value=VOICES[self.voice_idx][0])
        self.voice_cb = ttk.Combobox(bar, textvariable=self.voice_var,
                                     values=[v[0] for v in VOICES],
                                     state="readonly", width=22)
        self.voice_cb.pack(side="left")
        self.voice_cb.current(self.voice_idx)
        self.voice_cb.bind("<<ComboboxSelected>>", self._on_voice_changed)

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
                 text="Space: play/pause   ←/→: ±10s   Ctrl+Enter: speak   Ctrl+Shift+V: paste & speak",
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
        self.text_area.bind("<<Modified>>",     self._on_text_changed)

        self._update_char_count()
        self._set_transport("disabled")

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
            self.voice_idx = idx
        if self.engine.duration > 0 and not self.loading:
            self._resume_sec = self.engine.position if self.engine.is_playing or self.engine.is_paused else 0.0
            self._speak()

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
        text = get_clipboard_text()
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
    def _speak(self):
        text = self.text_area.get("1.0", "end-1c").strip()
        if not text:
            self.status_var.set("Nothing to speak")
            return

        # If already loading, cancel the in-flight request and restart
        self._fetch_gen += 1
        gen = self._fetch_gen

        self.engine.stop()
        self._cancel_tick()
        self.loading = True
        self._set_transport("disabled")
        self.play_btn.config(text="...")
        self.status_var.set("Generating audio ...")

        idx = self.voice_cb.current()
        voice_id = VOICES[idx][1] if idx >= 0 else VOICES[self.voice_idx][1]
        speed = SPEED_STEPS[self.speed_idx]

        def _worker():
            try:
                path = fetch_audio(text, voice_id)
                if gen != self._fetch_gen:
                    try: os.unlink(path)
                    except OSError: pass
                    return  # stale — a newer request is in flight
                self.root.after(0, lambda: self._on_ready(path, speed, gen))
            except Exception as exc:
                if gen == self._fetch_gen:
                    self.root.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_ready(self, path: str, speed: float, gen: int = -1):
        if gen != -1 and gen != self._fetch_gen:
            try: os.unlink(path)
            except OSError: pass
            return  # superseded by a newer request
        try:
            self.engine.load_file(path)
        except Exception as exc:
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

    def _on_error(self, msg: str):
        self.loading = False
        self.play_btn.config(text="▶")
        self.status_var.set(f"Error: {msg}")

    def _on_playback_done(self):
        self._cancel_tick()
        self.play_btn.config(text="▶")
        self.status_var.set("Done")
        self.progress_var.set(100)
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

    # ── Close ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        _save_prefs({"speed_idx": self.speed_idx, "voice_idx": self.voice_idx})
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
