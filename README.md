# TTS Player

A lightweight desktop text-to-speech player powered by Microsoft Edge TTS voices. Paste any text, pick a voice, and listen — with pitch-preserving speed control up to 2.5×.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Platform](https://img.shields.io/badge/platform-Windows-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **8 high-quality neural voices** — US, UK, and Australian English
- **Pitch-preserving speed control** from 0.5× to 2.5× via WSOLA time-stretching
- **Full transport controls** — play, pause, stop, seek, ±10s skip
- **Clipboard integration** — paste and auto-speak in one keystroke
- **Preferences are saved** — voice and speed persist across sessions
- **Always-on-top pin** — keep the player floating over other windows
- Dark Catppuccin-themed UI

---

## Installation

```bash
pip install edge-tts sounddevice miniaudio numpy
```

Then run:

```bash
python read.py
```

---

## Usage

```bash
python read.py                      # open with empty text area
python read.py --clipboard          # pre-fill from clipboard and auto-speak
python read.py --text "hello world" # speak a string directly
python read.py --file notes.txt     # read a text file
python read.py --speed 8            # start at a specific speed index (0–12)
```

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `←` / `→` | Seek ±10 seconds |
| `Ctrl+Enter` | Speak current text |
| `Ctrl+Shift+V` | Paste clipboard and speak |
| `Escape` | Stop |

---

## Speed index reference

| Index | Speed |
|---|---|
| 0 | 0.5× |
| 5 | 1.0× |
| 6 | 1.1× *(default)* |
| 9 | 1.6× |
| 12 | 2.5× |

---

## How it works

Audio is synthesized via [`edge-tts`](https://github.com/rany2/edge-tts) (Microsoft Edge's neural TTS engine, no API key needed). Speed changes use **WSOLA** (Waveform Similarity Overlap-Add), which time-stretches audio without altering pitch. All WSOLA computation runs in a background thread so the UI never freezes.

---

## Requirements

- Python 3.10+
- Windows (uses PowerShell for clipboard access and the Windows audio stack)
- `edge-tts`, `sounddevice`, `miniaudio`, `numpy`
