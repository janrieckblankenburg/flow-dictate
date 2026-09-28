#!/usr/bin/env python3
"""
Flow Dictate — hold-to-talk dictation for macOS.

Hold the fn/Globe key, speak, release: the transcript is cleaned up and
pasted into whatever app has focus.

Hands-free: hold fn, tap SPACE — recording locks on. Tap fn again to
stop and transcribe. Esc cancels either mode. Pressing any other key
while holding fn discards the recording, so fn-shortcuts keep working.

A pill overlay at the top of the screen (below the camera) shows a
pulsing dot while recording and a spinner while transcribing. The menu
bar icon mirrors the state (🎙 idle / 🟠 recording / ⏳ busy). The menu
offers recent dictations (click to copy) and a Settings window with
history, dictionary editing and the app log.

Speech-to-text runs locally (mlx-whisper on Apple Silicon) — audio never
leaves the machine. The cleanup pass uses the Claude API (text only);
set ANTHROPIC_API_KEY in the environment or a local .env file. Without
a key, the raw transcript is pasted unchanged.
"""

import collections
import fcntl
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from pynput import keyboard

# ── Config ────────────────────────────────────────────────────────────

# Hotkey is the fn/Globe key: hold to talk, or hold + tap Space for
# hands-free (tap fn again to stop). Requires System Settings → Keyboard →
# "Press 🌐 key to: Do Nothing" so macOS doesn't also react to it.
KEYCODE_FN = 63

# Two weight formats of the same turbo model. Measured on this M1 over TTS
# clips (German/English, clean and at ~10 dB SNR): identical speed (1.69s vs
# 1.68s per 21s clip), 1.3 points more word error for 4-bit — inside the noise
# of that test — and 1.15 GB less resident GPU memory. Since the model stays
# loaded all day in the menu bar, 4-bit is the default; switch in the menu if
# accuracy ever looks worse on real speech.
WHISPER_MODELS = {
    "efficient": ("mlx-community/whisper-large-v3-turbo-q4", "Efficient — 4-bit, ~0.5 GB"),
    "accurate": ("mlx-community/whisper-large-v3-turbo", "Accurate — 16-bit, ~1.6 GB"),
}
MODEL_PROFILE = "efficient"
WHISPER_MODEL = WHISPER_MODELS[MODEL_PROFILE][0]
LANGUAGE = "de"              # fixed language ≈ 2x faster than None (auto-detect costs a
                             # full extra encoder pass per dictation, measured 3.1s → 1.7s)
SAMPLE_RATE = 16_000
MIN_SECONDS = 0.4            # ignore accidental taps
RESTORE_CLIPBOARD = True     # put previous clipboard text back after pasting
CLEANUP_MODEL = "claude-haiku-4-5-20251001"
APP_DIR = Path(__file__).resolve().parent
HISTORY_FILE = APP_DIR / "history.jsonl"
LOG_FILE = APP_DIR / "flow-dictate.log"
HISTORY_MENU_ITEMS = 10
VOCAB_FILE = APP_DIR / "vocabulary.txt"        # one term per line, biases Whisper spelling
REPLACE_FILE = APP_DIR / "replacements.txt"    # "wrong => right" per line, hard post-fix
CONFIG_FILE = APP_DIR / "config.json"          # persisted settings (e.g. language)
LOCK_FILE = APP_DIR / ".instance.lock"         # single-instance guard
LOG_MAX_BYTES = 2 * 1024 * 1024                # rotate past this, keep one older log

KEYCODE_SPACE = 49
KEYCODE_ESC = 53

# Streaming: recordings longer than this are transcribed in background
# chunks WHILE you speak, so stopping only has to process the last bit.
#
# Chunks are cut at real speech pauses, never on a timer. Whisper punctuates
# whatever it is handed as if it were a complete utterance, so a cut through
# the middle of a sentence produces a bogus sentence end — that was the source
# of the chopped "So basically we have the. Reliability loop." transcripts.
# Cutting at a pause makes that period correct instead of wrong.
#
# Chunks also run right up to Whisper's native 30s window, because a call
# costs the same whatever it contains: measured 1.47s for 2s of audio and
# 1.50s for 28s. Chunking every 5s therefore did ~5x the GPU work for worse
# text — it saturated the GPU while speaking, which is what starved the
# machine for memory and let macOS kill the event tap ("fn suddenly dead").
STREAM_MIN_SECONDS = 12      # never cut before this — short sentences stay whole
# Tuned by measurement, and it does not pay to shrink this. Dropping it to 20s
# to bound the leftover tail made things worse on both counts: which pause gets
# picked depends on where the speaker's pauses happen to fall, so the tail came
# out *longer* (11s vs 6.6s) on the same recording, and the smaller chunks gave
# the model less context — "in ein Make Szenario" degraded to
# "in 1-Mail-Szenario". Wider chunks recognise rare terms better.
STREAM_MAX_SECONDS = 26
# 0.46 rather than 0.45: int(0.45 / 0.02) truncates to 22 frames = 0.44s, so
# the constant was quietly lying about the threshold it produced.
PAUSE_SECONDS = 0.46
STREAM_POLL_SECONDS = 0.4
STREAM_MAX_FAILURES = 2      # give up streaming after this many chunk errors
MAX_RECORD_SECONDS = 20 * 60  # hard cap so a stuck recording can't eat all RAM

# Watchdogs. A dead event tap makes the hotkey do nothing, and a wedged
# transcription swallows what you just said — both must be caught quickly
# rather than left for you to notice.
TAP_CHECK_SECONDS = 3
STUCK_WARN_SECONDS = 20     # say something is wrong
# Deadline for one dictation, scaled by how much audio it has to chew through:
# a 15-minute recording that fell back to the segmented path is legitimately
# slow, and a flat timeout killed it mid-transcription.
STUCK_BASE_SECONDS = 45
STUCK_PER_AUDIO_SECOND = 0.6
MODEL_BUSY_TIMEOUT = 8      # recovery paths refuse to wait longer than this
FORCE_JOB_WAIT_SECONDS = 25  # let a healthy dictation finish before butting in

# Whisper will recite its own prompt back on near-empty audio. Only guard
# against that for recordings too short to contain real dictation.
ECHO_GUARD_SECONDS = 2.5
SHOW_LIVE_TEXT = False       # True: pill shows the transcript live while recording

# MLX keeps freed GPU buffers in a pool and grows it without bound; over a
# session of hundreds of transcriptions that pool was the RAM leak. Cap it and
# flush it whenever the app goes idle.
MLX_CACHE_LIMIT = 384 * 1024 * 1024

# Encoder window sizing. Whisper pads every call to 30s, so a short dictation
# pays for 30s of encoder work; running a shorter window is exact (see
# _enable_short_windows) and ~3x faster. Granularity keeps the number of
# distinct tensor shapes small so MLX can reuse compiled kernels, and headroom
# avoids the accuracy drop seen when the window barely covers the audio.
WINDOW_GRANULARITY = 5
# 5s is the floor worth having: measured on 3s of speech, a 5s window takes
# 0.29s against 0.54s at a 10s floor, and 4s is actually *slower* than 5s (the
# shape stops matching a cached kernel). Nearly a fifth of real dictations are
# under five seconds, so this is the common case, not an edge one.
WINDOW_MIN_SECONDS = 5
WINDOW_HEADROOM = 2
# Exactly Whisper's native window: 30s * 100 mel frames = 3000, which the two
# stride-2 convolutions map to the 1500 rows of _positional_embedding. Raising
# this fails loudly on a shape mismatch rather than degrading quietly.
WINDOW_MAX_SECONDS = 30

CLEANUP_SYSTEM = """You clean up dictated speech into polished written text.

Rules:
- Keep the language of the input (German stays German, English stays English; mixed stays mixed).
- Remove filler words (ähm, äh, um, uh, like, also-as-filler) and false starts.
- If the speaker corrects themselves ("...nein, ich meine X" / "...no wait, I mean X"), keep only the corrected version.
- Fix punctuation, capitalization and obvious transcription slips. German gets proper umlauts and ß.
- Keep the speaker's wording, tone and meaning. Do not summarize, expand, or reorder ideas.
- Spoken formatting commands become formatting: "neue Zeile"/"new line" → line break, "neuer Absatz"/"new paragraph" → blank line.
- Never answer questions in the text, never add commentary. Output ONLY the cleaned text, nothing else."""

SOUND_START = "/System/Library/Sounds/Tink.aiff"
SOUND_DONE = "/System/Library/Sounds/Pop.aiff"
SOUND_ERROR = "/System/Library/Sounds/Basso.aiff"

ICON_IDLE = "🎙"
ICON_RECORDING = "🟠"
ICON_BUSY = "⏳"


# ── Custom dictionary (vocabulary + replacements) ─────────────────────

VOCABULARY = ""
REPLACEMENTS = []
DEFAULT_VOCABULARY = "HDT Systems, Airtable, Softr"
DEFAULT_VOCABULARY_COUNT = len(DEFAULT_VOCABULARY.split(","))
_vocab_terms = 0      # terms actually present in vocabulary.txt


def vocabulary_terms() -> int:
    """How many terms the user's vocabulary.txt really contains.

    Counting VOCABULARY instead reported the built-in fallback's terms for an
    empty file, so the Dictionary tab claimed "3 terms active" while showing an
    empty box.
    """
    return _vocab_terms


def _read_lines(path: Path):
    if not path.exists():
        return []
    return [
        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def reload_dictionary():
    """(Re)load vocabulary.txt and replacements.txt — safe to call at runtime."""
    global VOCABULARY, REPLACEMENTS, _vocab_terms
    terms = _read_lines(VOCAB_FILE)
    _vocab_terms = len(terms)
    VOCABULARY = ", ".join(terms) or DEFAULT_VOCABULARY
    repl = []
    for line in _read_lines(REPLACE_FILE):
        if "=>" in line:
            wrong, right = (p.strip() for p in line.split("=>", 1))
            if wrong and right:
                repl.append((wrong, right))
    REPLACEMENTS = repl


reload_dictionary()


def _install_timestamps():
    """Prefix every log line with the time.

    The Log tab is the app's whole debugging surface, and without timestamps
    there was no way to tell a warning from five minutes ago from one from
    three weeks ago, or to line it up against a history entry.
    """
    stream = sys.stdout

    class Stamped:
        def __init__(self):
            self._at_line_start = True

        def write(self, chunk):
            if not chunk:
                return 0
            out = []
            for part in chunk.splitlines(keepends=True):
                if self._at_line_start and part.strip():
                    out.append(datetime.now().strftime("%m-%d %H:%M:%S "))
                out.append(part)
                self._at_line_start = part.endswith(("\n", "\r"))
            stream.write("".join(out))
            return len(chunk)

        def flush(self):
            stream.flush()

        def isatty(self):
            return getattr(stream, "isatty", lambda: False)()

    sys.stdout = Stamped()
    sys.stderr = sys.stdout


def rotate_log(max_bytes: int = LOG_MAX_BYTES):
    """Keep one previous log. Steady-state growth is ~1 MB/year, but a startup
    crash loop writes a full banner per restart — megabytes per day — and the
    Log tab reads the file to show it."""
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > max_bytes:
            LOG_FILE.replace(LOG_FILE.with_suffix(".log.1"))
    except Exception as e:
        print(f"   (could not rotate the log: {e})")


def load_config():
    global LANGUAGE, MODEL_PROFILE, WHISPER_MODEL
    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        LANGUAGE = cfg.get("language", LANGUAGE)
        profile = cfg.get("model", MODEL_PROFILE)
        if profile in WHISPER_MODELS:
            MODEL_PROFILE = profile
            WHISPER_MODEL = WHISPER_MODELS[profile][0]
    except Exception:
        pass


def save_config():
    CONFIG_FILE.write_text(
        json.dumps({"language": LANGUAGE, "model": MODEL_PROFILE}, indent=2),
        encoding="utf-8",
    )


load_config()


# ── macOS UI (menu bar + overlay) — optional, degrades to terminal-only ──

try:
    import AppKit
    import Quartz
    import rumps
    from Foundation import NSMakeRect, NSOperationQueue
    HAS_UI = True
except ImportError:
    HAS_UI = False

_app = None
overlay = None
_hist_menu = None


def on_main(fn):
    """Run fn on the AppKit main thread (UI must never be touched from workers)."""
    if HAS_UI:
        NSOperationQueue.mainQueue().addOperationWithBlock_(fn)


def set_status(icon: str):
    """Set the menu-bar icon. Must go through the main thread: rumps' title
    setter writes straight into NSStatusItem, and this is called from the
    event-tap callback thread on every recording. A cross-thread AppKit call
    from that thread can stall the tap past its timeout, which is one way the
    fn key "goes dead" — so this is the cause _tap_watchdog was papering over.
    """
    if _app is not None:
        on_main(lambda: setattr(_app, "title", icon))


class Overlay:
    """Floating pill at top-center of the screen, just below the camera notch."""

    W, H = 250, 36

    def __init__(self):
        style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
        self.panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, self.W, self.H), style, AppKit.NSBackingStoreBuffered, False
        )
        self.panel.setLevel_(AppKit.NSStatusWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setHasShadow_(True)
        self.panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        self.panel.setAppearance_(
            AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua)
        )

        root = AppKit.NSView.alloc().initWithFrame_(NSMakeRect(0, 0, self.W, self.H))
        root.setWantsLayer_(True)
        root.layer().setCornerRadius_(self.H / 2)
        root.layer().setBackgroundColor_(
            AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.92).CGColor()
        )
        self.panel.setContentView_(root)

        d = 10
        self.dot = AppKit.NSView.alloc().initWithFrame_(NSMakeRect(15, (self.H - d) / 2, d, d))
        self.dot.setWantsLayer_(True)
        self.dot.layer().setCornerRadius_(d / 2)
        self.dot.layer().setBackgroundColor_(AppKit.NSColor.systemOrangeColor().CGColor())
        root.addSubview_(self.dot)

        s = 16
        self.spinner = AppKit.NSProgressIndicator.alloc().initWithFrame_(
            NSMakeRect(12, (self.H - s) / 2, s, s)
        )
        self.spinner.setStyle_(AppKit.NSProgressIndicatorStyleSpinning)
        self.spinner.setIndeterminate_(True)
        self.spinner.setControlSize_(AppKit.NSControlSizeSmall)
        self.spinner.setDisplayedWhenStopped_(False)
        root.addSubview_(self.spinner)

        self.label = AppKit.NSTextField.labelWithString_("")
        self.label.setFrame_(NSMakeRect(36, (self.H - 17) / 2, self.W - 44, 17))
        self.label.setTextColor_(AppKit.NSColor.whiteColor())
        self.label.setFont_(AppKit.NSFont.systemFontOfSize_(12.5))
        root.addSubview_(self.label)

    def _place(self):
        screen = AppKit.NSScreen.mainScreen()
        sf, vf = screen.frame(), screen.visibleFrame()
        x = sf.origin.x + (sf.size.width - self.W) / 2
        y = vf.origin.y + vf.size.height - self.H - 6  # just below menu bar / camera notch
        self.panel.setFrame_display_(NSMakeRect(x, y, self.W, self.H), True)

    def _pulse(self):
        anim = Quartz.CABasicAnimation.animationWithKeyPath_("opacity")
        anim.setFromValue_(1.0)
        anim.setToValue_(0.25)
        anim.setDuration_(0.55)
        anim.setAutoreverses_(True)
        anim.setRepeatCount_(1e9)
        self.dot.layer().addAnimation_forKey_(anim, "pulse")

    def show_recording(self, locked: bool):
        self.dot.setHidden_(False)
        self.spinner.stopAnimation_(None)
        self.label.setStringValue_("Recording — tap fn to stop" if locked else "Recording…")
        self._pulse()
        self._place()
        self.panel.orderFrontRegardless()

    def show_processing(self):
        self.dot.setHidden_(True)
        self.dot.layer().removeAllAnimations()
        self.spinner.startAnimation_(None)
        self.label.setStringValue_("Transcribing…")
        self._place()
        self.panel.orderFrontRegardless()

    def show_message(self, text: str):
        self.dot.setHidden_(True)
        self.dot.layer().removeAllAnimations()
        self.spinner.stopAnimation_(None)
        self.label.setStringValue_(text)
        self._place()
        self.panel.orderFrontRegardless()

    def set_live_text(self, text: str):
        """Live transcript preview while recording (streaming mode)."""
        self.label.setStringValue_(text)

    def hide(self):
        self.spinner.stopAnimation_(None)
        self.dot.layer().removeAllAnimations()
        self.panel.orderOut_(None)


def ui(method_name: str, *args):
    """Call an Overlay method on the main thread (no-op without UI)."""
    if overlay is not None:
        on_main(lambda: getattr(overlay, method_name)(*args))


# ── History ───────────────────────────────────────────────────────────

history = []


def load_history():
    if not HISTORY_FILE.exists():
        return
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue   # a line torn by a kill mid-write; skip it
        # Any valid JSON parsed before, so a bare number became an int in the
        # list and then raised deep inside the menu builder or the history
        # table. Only well-formed entries get in.
        if not isinstance(entry, dict):
            continue
        if "sekunden" in entry:  # legacy key from early versions
            entry["seconds"] = entry.pop("sekunden")
        if not isinstance(entry.get("text"), str):
            continue
        entry.setdefault("ts", "")
        history.append(entry)


def clear_history():
    history.clear()
    HISTORY_FILE.write_text("", encoding="utf-8")
    refresh_history_menu()


def append_history(text: str, seconds: float):
    entry = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(seconds, 1),
        "text": text,
    }
    history.append(entry)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    refresh_history_menu()
    try:
        import settings_window
        settings_window.refresh_if_open()
    except Exception:
        pass


def copy_to_clipboard(text: str):
    subprocess.run(["pbcopy"], input=text.encode("utf-8"))


def _copy_callback(text: str):
    def cb(_):
        copy_to_clipboard(text)
    return cb


def refresh_history_menu():
    if _app is None or _hist_menu is None:
        return

    def _do():
        try:
            if len(_hist_menu):
                _hist_menu.clear()
        except Exception:
            pass
        entries = history[-HISTORY_MENU_ITEMS:]
        if not entries:
            _hist_menu.add(rumps.MenuItem("(empty)"))
            return
        for e in reversed(entries):
            text = e["text"]
            title = f"{e['ts'][11:16]}  {text[:60]}" + ("…" if len(text) > 60 else "")
            _hist_menu.add(rumps.MenuItem(title, callback=_copy_callback(text)))

    on_main(_do)


# ── Cleanup (Claude) ──────────────────────────────────────────────────

_claude = None
if os.environ.get("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        _claude = anthropic.Anthropic()
    except Exception as e:
        print(f"⚠️  Claude client unavailable ({e}) — pasting raw transcripts.")


def clean_up(text: str) -> str:
    """One Haiku pass over the transcript. Falls back to raw text on any error."""
    if _claude is None or len(text) < 12:
        return text
    try:
        resp = _claude.messages.create(
            model=CLEANUP_MODEL,
            max_tokens=4096,
            system=CLEANUP_SYSTEM + f"\n\nKnown names/terms — spell exactly like this: {VOCABULARY}",
            messages=[{"role": "user", "content": text}],
        )
        cleaned = resp.content[0].text.strip()
        return cleaned or text
    except Exception as e:
        print(f"⚠️  Cleanup failed ({e}) — pasting raw transcript.")
        return text


# ── Audio recording ───────────────────────────────────────────────────

MAX_RECORD_SAMPLES = MAX_RECORD_SECONDS * SAMPLE_RATE


class Recorder:
    """Microphone capture into one growable buffer.

    The buffer doubles as it fills, so appending a whole recording costs O(n)
    in total. The previous version rebuilt the entire recording with
    np.concatenate on every poll — 0.3s apart, so a 3-minute dictation
    reallocated and copied an 11 MB array ~570 times. That allocator churn,
    not the audio itself, was the bulk of the memory problem.
    """

    INITIAL_SAMPLES = 60 * SAMPLE_RATE

    def __init__(self):
        self._buf = np.zeros(self.INITIAL_SAMPLES, dtype=np.float32)
        self._n = 0                      # samples committed to _buf
        # A deque, not a list: the audio thread appends while the drainer
        # consumes, and swapping a list out between the callback's attribute
        # load and its append silently dropped that frame. deque.append and
        # popleft are individually atomic, so neither side can lose one.
        self._pending = collections.deque()
        self._lock = threading.Lock()
        self._stream = None
        self.overflowed = False
        self.overruns = 0

    def _on_audio(self, indata, frames, time_info, status):
        # Audio thread: do the cheapest possible thing — stash a copy and go.
        if status:
            # PortAudio telling us it dropped input. Discarding this was the
            # only signal that audio went missing, and lost audio is
            # indistinguishable from the speaker pausing.
            self.overruns += 1
        self._pending.append(indata[:, 0].copy())

    def _drain(self):
        """Fold pending frames into the buffer. Caller must hold self._lock."""
        while True:
            try:
                f = self._pending.popleft()
            except IndexError:
                return
            needed = self._n + f.size
            if needed > self._buf.size:
                size = self._buf.size
                while size < needed and size < MAX_RECORD_SAMPLES:
                    size *= 2
                size = min(size, MAX_RECORD_SAMPLES)
                if size > self._buf.size:
                    grown = np.zeros(size, dtype=np.float32)
                    grown[:self._n] = self._buf[:self._n]
                    self._buf = grown
            room = self._buf.size - self._n
            if room <= 0:
                self.overflowed = True
                self._pending.clear()
                return
            take = min(room, f.size)
            self._buf[self._n:self._n + take] = f[:take]
            self._n += take
            if take < f.size:
                self.overflowed = True
                self._pending.clear()
                return

    def start(self):
        with self._lock:
            self._n = 0
            self._pending.clear()
            self.overflowed = False
            self.overruns = 0
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=self._on_audio,
        )
        try:
            stream.start()
        except Exception:
            # Never leave a constructed-but-unstarted stream behind: it stays
            # registered, keeps calling _on_audio, and then interleaves its
            # frames into every later recording.
            try:
                stream.close()
            except Exception:
                pass
            raise
        self._stream = stream

    def available(self) -> int:
        """How many samples are recorded so far — O(new audio), not O(total)."""
        with self._lock:
            self._drain()
            return self._n

    def view(self, start: int, end: int = None) -> np.ndarray:
        """Copy of just [start:end) — never the whole recording."""
        with self._lock:
            self._drain()
            stop = self._n if end is None else min(end, self._n)
            if start >= stop:
                return np.zeros(0, dtype=np.float32)
            return self._buf[start:stop].copy()

    def snapshot(self) -> np.ndarray:
        """Everything recorded so far, stream left running."""
        return self.view(0)

    def stop(self) -> np.ndarray:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                print(f"⚠️  closing the audio stream failed ({e})")
        with self._lock:
            self._drain()
            return self._buf[:self._n].copy()


# ── Output: paste into the focused app ────────────────────────────────

_kb = keyboard.Controller()


def paste_text(text: str):
    """Put the text on the clipboard and press ⌘V.

    The clipboard restore runs on its own thread. Waiting for it here cost
    0.35s of the perceived latency of every single dictation — about 40% of the
    total once the model got fast — for work the user never sees.

    It also only restores *text*. Reading the clipboard with pbpaste flattens
    an image or file to nothing, so restoring it used to wipe a copied image
    rather than leave it alone; now a non-text clipboard is left untouched.
    """
    old = subprocess.run(["pbpaste"], capture_output=True).stdout
    subprocess.run(["pbcopy"], input=text.encode("utf-8"))
    time.sleep(0.05)  # let the clipboard settle before the synthetic keystroke
    with _kb.pressed(keyboard.Key.cmd):
        _kb.press("v")
        _kb.release("v")
    if RESTORE_CLIPBOARD and old.strip():
        def restore():
            # Long enough that ⌘V has been delivered even into a slow Electron
            # app; too short and the paste lands after the restore and inserts
            # the previous clipboard contents instead of the dictation.
            time.sleep(0.8)
            try:
                subprocess.run(["pbcopy"], input=old, timeout=5)
            except Exception:
                pass
        threading.Thread(target=restore, daemon=True).start()


def play(sound: str):
    subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Pipeline ──────────────────────────────────────────────────────────

import math                # noqa: E402
import mlx.core as mx      # noqa: E402
import mlx.nn as mlx_nn    # noqa: E402
import mlx_whisper         # noqa: E402  (slow import, kept after fast-fail config)
from mlx_whisper import audio as mlx_audio        # noqa: E402
from mlx_whisper import decoding as mlx_decoding  # noqa: E402
from mlx_whisper.transcribe import ModelHolder    # noqa: E402
from mlx_whisper.whisper import AudioEncoder      # noqa: E402

try:
    mx.set_cache_limit(MLX_CACHE_LIMIT)
except Exception as e:      # older mlx builds
    print(f"   (could not cap the MLX buffer cache: {e})")


def _enable_short_windows():
    """Let the encoder run on a window shorter than Whisper's fixed 30s.

    Whisper always pads to 30s, so a 6s dictation costs the same as a 28s one —
    the single biggest source of "why does this take so long". The encoder
    asserts that shape, but its positional embedding is *sinusoidal*: row k
    depends only on k, so the first K rows are exactly the embedding of a
    K-frame window. Slicing it is therefore exact, not an approximation.

    Measured on 7s of German: 1.73s at the stock 30s window, 0.58s at 10s, with
    mean word error over the test set moving 6.1% → 6.8% (inside its noise).
    Confidence does drop with less context (avg_logprob -0.07 → -0.18), which
    is why WINDOW_HEADROOM keeps a couple of seconds of slack.
    """
    def call(self, x):
        # mlx_whisper's language detection decides "are these already encoded?"
        # by comparing against the fixed 1500-frame shape, which a short window
        # fails — so it would encode the features a second time. Encoded
        # features carry n_state channels (1280) where mel carries n_mels (128),
        # so make the encoder itself re-entrant and hand them straight back.
        if x.shape[-1] == self._positional_embedding.shape[-1]:
            return x
        x = mlx_nn.gelu(self.conv1(x))
        x = mlx_nn.gelu(self.conv2(x))
        x = x + self._positional_embedding[: x.shape[1]]
        for block in self.blocks:
            x, _, _ = block(x)
        return self.ln_post(x)

    AudioEncoder.__call__ = call


_enable_short_windows()

_job_lock = threading.Lock()    # one dictation in the pipeline at a time
_model_lock = threading.Lock()  # exactly one call inside the model, ever

# Watchdog state, so a wedged dictation is noticed instead of silently
# swallowing what you just said. Keyed per job rather than one global flag:
# with a single timestamp, two overlapping dictations clobbered each other —
# the first to finish cleared it and the second ran completely unwatched,
# which is precisely the case the watchdog exists for.
_pipeline_lock = threading.Lock()
_pipeline_jobs = {}     # id -> {"since", "deadline", "text"}
_pipeline_next_id = 0


def _pipeline_begin(audio_seconds: float) -> int:
    """Register a dictation. Call *after* the job lock is held, so time spent
    queued behind another dictation is not counted against this one."""
    global _pipeline_next_id
    with _pipeline_lock:
        _pipeline_next_id += 1
        job = _pipeline_next_id
        _pipeline_jobs[job] = {
            "since": time.time(),
            # Long audio on the segmented fallback path is legitimately slow.
            "deadline": STUCK_BASE_SECONDS + audio_seconds * STUCK_PER_AUDIO_SECOND,
            "text": "",
        }
        return job


def _pipeline_text(job: int, text: str):
    """Publish the best text this job has so far, so a watchdog restart can
    save it. Reading _stream_committed instead was useless: it is empty exactly
    when streaming failed, which is the case that makes a dictation slow."""
    with _pipeline_lock:
        if job in _pipeline_jobs:
            _pipeline_jobs[job]["text"] = text


def _pipeline_end(job: int):
    with _pipeline_lock:
        _pipeline_jobs.pop(job, None)


def release_gpu_cache():
    """Hand MLX's pooled GPU buffers back once we go idle."""
    try:
        mx.clear_cache()
    except Exception:
        pass


def _trim_silence(audio: np.ndarray, pad: float = 0.25) -> np.ndarray:
    """Cut leading/trailing silence — noise-only edges trigger hallucinations.

    The threshold comes from the *quiet* part of the recording, not its peak.
    A peak-relative threshold (peak * 0.04) meant a single loud transient — a
    cough, a plosive, the laptop lid closing — raised the bar above quiet
    speech and trimmed real trailing words away. Nothing downstream re-reads
    that audio: the streaming accounting advances over the untrimmed chunk, so
    whatever was trimmed here was simply gone from the transcript.
    """
    if audio.size == 0:
        return audio
    hop = int(FRAME_SECONDS * SAMPLE_RATE)
    rms = _frame_rms(audio, hop)
    if rms.size == 0:
        return audio
    floor = float(np.percentile(rms, 10))
    thresh = max(floor * 2.0, 0.002)
    voiced = np.where(rms > thresh)[0]
    if voiced.size == 0:
        return audio
    a = max(0, voiced[0] * hop - int(pad * SAMPLE_RATE))
    b = min(len(audio), (voiced[-1] + 1) * hop + int(pad * SAMPLE_RATE))
    return audio[a:b]


def build_prompt(context: str = "") -> str:
    """Whisper's initial_prompt sets vocabulary *and* writing style, and the
    style half is easy to get wrong: handing it the bare comma-separated
    vocabulary list taught it that this text has no sentence punctuation, and
    it then returned whole chunks without a single full stop. Wrapping the same
    terms in a punctuated sentence keeps the spelling help and the periods.

    `context` is the transcript so far — already punctuated, so it reinforces
    the style while carrying names across a chunk boundary.
    """
    lang = LANGUAGE or _session_language or "de"
    lead = ("Fachbegriffe: " if lang == "de" else "Terms: ") + VOCABULARY + "."
    context = context.strip()
    return f"{lead} {context}" if context else lead


_detect_lock = threading.Lock()
_session_language = None   # language detected for the dictation in progress


def reset_detected_language():
    global _session_language
    with _detect_lock:
        _session_language = None


def _window_seconds(seconds: float) -> int:
    """Encoder window for this much audio: rounded up, with headroom, capped."""
    want = math.ceil((seconds + WINDOW_HEADROOM) / WINDOW_GRANULARITY) * WINDOW_GRANULARITY
    return int(max(WINDOW_MIN_SECONDS, min(want, WINDOW_MAX_SECONDS)))


def _current_language() -> str:
    """Auto-detect costs a whole extra encoder pass (3.20s vs 1.68s measured),
    and nobody switches language mid-sentence — so detect once per dictation
    and reuse the answer for every later chunk."""
    if LANGUAGE is not None:
        return LANGUAGE
    with _detect_lock:
        return _session_language


def _remember_language(detected: str):
    global _session_language
    if LANGUAGE is None and detected:
        with _detect_lock:
            _session_language = detected


def _transcribe_short(audio: np.ndarray, prompt: str, lang) -> str:
    """One decode over a window sized to the audio — the fast path.

    Everything the app records normally lands here: stream chunks are capped at
    STREAM_MAX_SECONDS and the tail is usually shorter still.
    """
    window = _window_seconds(len(audio) / SAMPLE_RATE)
    frames = window * 100                       # 100 mel frames per second
    model = ModelHolder.get_model(WHISPER_MODEL, mx.float16)
    padding = max(0, window * SAMPLE_RATE - len(audio))
    mel = mlx_audio.log_mel_spectrogram(
        audio, n_mels=model.dims.n_mels, padding=padding
    )[:frames]
    if mel.shape[0] < frames:
        mel = mx.concatenate(
            [mel, mx.zeros((frames - mel.shape[0], mel.shape[1]), mel.dtype)]
        )
    seconds = len(audio) / SAMPLE_RATE
    # Cap tokens relative to audio length: a 2s clip must never decode 400
    # tokens — that's the hallucination loop that used to take 15s+.
    sample_len = min(448, int(min(seconds, 30.0) * 15) + 30)
    result = mlx_decoding.decode(
        model,
        mel,
        mlx_decoding.DecodingOptions(
            language=lang,
            temperature=0.0,            # no retry ladder → faster
            without_timestamps=True,
            fp16=True,
            prompt=prompt,
            sample_len=sample_len,
        ),
    )
    # Same not-speech filter the stock pipeline applies per segment
    if result.no_speech_prob > 0.6 and result.avg_logprob < -1.0:
        return ""
    # Only latch the language from a chunk we actually accepted. Doing it
    # before this filter meant a first chunk of breath or keyboard noise could
    # pin the whole dictation to a misdetected language, with no way back.
    _remember_language(result.language)
    return result.text.strip()


def _transcribe_long(audio: np.ndarray, prompt: str, lang) -> str:
    """Audio past one Whisper window — hand it to the stock segmented pipeline.

    Only reached if streaming was disabled or failed mid-recording, so it is
    the slow-but-correct safety net rather than the normal path.
    """
    seconds = len(audio) / SAMPLE_RATE
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=WHISPER_MODEL,
        initial_prompt=prompt,
        language=lang,
        temperature=0.0,
        condition_on_previous_text=False,
        sample_len=min(448, int(min(seconds, 30.0) * 15) + 30),
    )
    _remember_language(result.get("language"))
    parts = [
        seg["text"] for seg in result["segments"]
        if not (seg.get("no_speech_prob", 0.0) > 0.6 and seg.get("avg_logprob", 0.0) < -1.0)
    ]
    return "".join(parts).strip()


class ModelBusy(Exception):
    """The model lock could not be taken in time — a pass is wedged in there."""


def transcribe(audio: np.ndarray, prompt: str = None,
               lock_timeout: float = None) -> str:
    """Transcribe one piece of audio.

    The model is serialized: two concurrent passes doubled peak GPU memory on a
    machine that was already swapping, and that is what killed the process.

    `lock_timeout` exists for the recovery paths. They must never block here —
    the failure they recover from *is* a pass stuck inside the model holding
    this lock, so waiting for it defeats the entire purpose. With a timeout set,
    a wedged model raises ModelBusy instead of hanging forever.
    """
    audio = _trim_silence(audio)
    if audio.size == 0:
        return ""
    prompt = prompt or build_prompt()
    lang = _current_language()
    fits_one_window = len(audio) / SAMPLE_RATE <= WINDOW_MAX_SECONDS - 1

    if lock_timeout is None:
        got = _model_lock.acquire()
    else:
        got = _model_lock.acquire(timeout=lock_timeout)
    if not got:
        raise ModelBusy(
            f"the model has been busy for over {lock_timeout:.0f}s — "
            "a transcription pass looks wedged"
        )
    try:
        if fits_one_window:
            text = _transcribe_short(audio, prompt, lang)
        else:
            text = _transcribe_long(audio, prompt, lang)
    finally:
        _model_lock.release()

    seconds = len(audio) / SAMPLE_RATE
    if seconds < ECHO_GUARD_SECONDS and _looks_like_prompt_echo(text, prompt):
        print("   (prompt echo on a very short recording — discarded)")
        return ""
    return text


def _looks_like_prompt_echo(text: str, prompt: str) -> bool:
    """Is this the initial_prompt being read back to us?

    On a near-empty recording Whisper sometimes just recites its prompt, and
    confidently enough that the no-speech filter lets it through: three
    accidental sub-second taps pasted the raw vocabulary list. Only consulted
    for very short audio, so a real dictation that happens to name several
    tools is not at risk.
    """
    words = [w for w in re.findall(r"[^\W\d_]+", text.lower()) if len(w) > 1]
    if len(words) < 4:
        return False
    prompt_words = set(re.findall(r"[^\W\d_]+", prompt.lower()))
    if not prompt_words:
        return False
    shared = sum(1 for w in words if w in prompt_words)
    return shared / len(words) >= 0.8


# ── Stitching chunk transcripts back into prose ───────────────────────

_LEADING_DOTS = re.compile(r"^[\s.…]+")


def _stitch(prev: str, nxt: str) -> str:
    """Append one chunk's text to the running transcript.

    Whisper punctuates every chunk as a self-contained utterance, so even a
    well-placed cut leaves artifacts at the seam: a leading "…" where it thinks
    it joined mid-thought, or a full stop where the sentence actually runs on.
    A next chunk that opens lower-case is the reliable tell that the previous
    full stop was invented — German capitalizes nouns, so the reverse case
    can't be detected here and is left to the cleanup pass.
    """
    nxt = _LEADING_DOTS.sub("", nxt.strip()).strip()
    if not prev:
        return nxt
    if not nxt:
        return prev
    prev = re.sub(r"…+\s*$", "", prev.rstrip()).rstrip()
    # Only drop a full stop that ends a *word*. "z.B.", "1.500." and "Punkt 3."
    # all end in a dot too, and stripping theirs corrupted the text
    # ("das z.B." + "eine Lösung" became "das z.B eine Lösung").
    if (prev.endswith(".") and nxt[:1].islower()
            and _ENDS_REAL_SENTENCE.search(prev)):
        prev = prev[:-1].rstrip()
    return f"{prev} {nxt}".strip()


# A sentence-ending dot follows a word of 2+ letters that isn't a known
# abbreviation, and isn't preceded by a digit (ordinals, thousands separators).
_ENDS_REAL_SENTENCE = re.compile(
    r"(?<![\d.])(?<!\bz\.B)(?<!\bbzw)(?<!\busw)(?<!\bevtl)(?<!\bggf)"
    r"(?<!\bca)(?<!\bNr)(?<!\bbspw)(?<!\bd\.h)"
    r"[^\W\d_]{2,}\.$"
)

# Dots that belong to a token rather than ending a sentence: domains, files,
# handles. "Make.com" is in vocabulary.txt, so this fires on the user's own
# terms — and with the cleanup pass off there is nothing downstream to repair it.
_INTRA_TOKEN_DOT = re.compile(r"(?<=[^\W\d_])\.(?=[a-z]{2,4}\b)", re.IGNORECASE)


def normalize_punctuation(text: str) -> str:
    """Tidy the dot soup Whisper leaves on hesitant speech ("Und dann.. ähm...").

    Careful with the "space after punctuation" rule: it used to split
    "Make.com" into "Make. com", and likewise domains, filenames and e-mail
    addresses. Intra-token dots are masked out before that rule runs.
    """
    text = text.replace("…", "...")
    text = re.sub(r"\.{2,}", ".", text)             # hesitation dots → one sentence end
    text = re.sub(r"(?:\s*\.\s*){2,}", ". ", text)  # ". . ." → ". "
    text = re.sub(r"\s+([,!?;:])", r"\1", text)     # no space before punctuation
    text = re.sub(r"\s+\.(?=\s|$)", ".", text)

    # Mask dots inside tokens, apply the spacing rule, then restore them.
    sentinel = "\x00"
    text = _INTRA_TOKEN_DOT.sub(sentinel, text)
    # A closing bracket or quote after punctuation belongs to it, not to the
    # next word — "(wirklich!)" must not become "(wirklich! )".
    text = re.sub(r"([.,!?;:])(?=[^\s\d)\]}»\"'…])", r"\1 ", text)
    text = text.replace(sentinel, ".")

    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def apply_replacements(text: str) -> str:
    """Apply replacements.txt. The right-hand side is inserted literally.

    It used to be passed to re.sub as a *template*, so a backslash or a "\\1"
    typed into the dictionary editor raised re.error on every dictation from
    then on — with an error message pointing nowhere near the cause.
    """
    for wrong, right in REPLACEMENTS:
        try:
            text = re.sub(
                rf"\b{re.escape(wrong)}\b", lambda _m, r=right: r,
                text, flags=re.IGNORECASE,
            )
        except re.error as e:
            print(f"⚠️  replacement {wrong!r} => {right!r} is unusable ({e}) — skipped")
    return text


_REPEAT_RUN = re.compile(
    # the same word 3+ times in a row, separated by commas/semicolons/dots/space
    r"\b([^\W\d_][\w\-]*)((?:[,;.!?\s]+\1\b){2,})", re.IGNORECASE
)
_REPEAT_SENTENCE = re.compile(
    # the same short sentence repeated 3+ times ("Untertitel von X." x20)
    r"([^.!?]{4,80}[.!?])(?:\s*\1){2,}", re.IGNORECASE
)


def collapse_repetitions(text: str) -> str:
    """Collapse the loops Whisper falls into on noisy or near-empty audio.

    Two shapes occur and the old single pattern only caught the first:
    "Sound, Sound, Sound" (comma-separated words) and
    "Untertitel von Stephanie Geiges. Untertitel von …" (repeated sentences,
    the classic German one). The sentence form went through untouched.

    Returns "" only when the text is *nothing but* a loop. The old 25% length
    test also threw away legitimate short speech — "Ja, ja, ja, ja, ja"
    collapses to "Ja", which is 11% of the original, and was discarded.
    """
    collapsed = _REPEAT_SENTENCE.sub(r"\1", text)
    collapsed = _REPEAT_RUN.sub(r"\1", collapsed)
    if collapsed != text:
        # A loop is only "the whole content" if what survives is a single
        # short fragment; anything sentence-shaped is real speech plus a loop.
        words = collapsed.split()
        if len(words) <= 2 and len(text) > 40:
            print("   ⚠️ repetition loop on noisy audio — discarded")
            return ""
        print("   ⚠️ repetition loop detected and collapsed (noisy audio?)")
    return collapsed


def process(audio: np.ndarray, committed: str = "", consumed: int = 0):
    """Transcribe + paste. `committed`/`consumed` come from the streaming
    worker: text already transcribed while recording, and how many samples
    of `audio` it covers — only the remaining tail still needs the model.

    Nothing here is allowed to lose a transcript. Any failure after the model —
    a ⌘V that the Accessibility permission silently refuses, a cleanup call
    that throws — used to end this worker thread with no sound, no message and
    no text: you spoke, and nothing came out. Now the transcript is written to
    the history before delivery is attempted, falls back to the clipboard if
    pasting fails, and every failure is announced.
    """
    with _job_lock:
        seconds = len(audio) / SAMPLE_RATE
        # Registered only now: queueing behind another dictation is not this
        # job's fault and must not count against its deadline.
        job = _pipeline_begin(seconds)
        set_status(ICON_BUSY)
        ui("show_processing")
        # Best text we have at any moment. It starts as whatever streaming
        # already transcribed, so a tail pass that dies cannot take the part
        # you already spoke down with it.
        best = committed
        _pipeline_text(job, best)
        stored = False
        try:
            if not committed and np.abs(audio).max() < 0.01:
                # silence gate: Whisper hallucinates on empty audio
                print("   (silence — skipped; if this keeps happening, check the "
                      "Microphone permission for Flow Dictate)")
                play(SOUND_ERROR)
                ui("show_message", "No audio — check Microphone permission")
                time.sleep(2.5)
                return
            t0 = time.time()
            tail = audio[consumed:]
            tail_text = ""
            if len(tail) / SAMPLE_RATE >= 0.3:
                tail_text = transcribe(tail, prompt=build_prompt(committed[-200:]))
            raw = _stitch(committed, tail_text)
            if not raw:
                print("   (nothing heard)")
                play(SOUND_ERROR)
                ui("show_message", "Nothing heard")
                time.sleep(1.5)
                return
            best = raw
            _pipeline_text(job, best)
            deloop = collapse_repetitions(raw)
            if not deloop:
                # collapse_repetitions rejected the text as pure hallucination.
                # Falling back to `raw` here pasted the loop it had just thrown
                # away — a wall of "Sound, Sound, Sound…" landed in the history.
                play(SOUND_ERROR)
                ui("show_message", "Only noise heard — nothing pasted")
                time.sleep(2.0)
                return
            text = clean_up(normalize_punctuation(apply_replacements(deloop)))
            best = text or deloop
            append_history(best, seconds)   # recorded before we try to deliver it
            stored = True
            paste_text(best)
            play(SOUND_DONE)
            print(f"✓  {seconds:.1f}s audio → {time.time() - t0:.1f}s processing")
            print(f"   {best}")
        except Exception as e:
            print(f"⚠️  dictation failed after the model: {e}")
            _rescue_text(best, stored)
        finally:
            _pipeline_end(job)
            set_status(ICON_IDLE)
            ui("hide")
            release_gpu_cache()  # give the pooled GPU buffers back while idle


def _rescue_text(text: str, already_stored: bool = False):
    """Last resort: get whatever we have to the user rather than dropping it."""
    if not text:
        play(SOUND_ERROR)
        ui("show_message", "Dictation failed — nothing recovered")
        time.sleep(2.5)
        return
    try:
        copy_to_clipboard(text)
        if not already_stored:
            try:
                append_history(text, 0.0)
            except Exception:
                pass   # clipboard already has it; history is the lesser copy
        play(SOUND_ERROR)
        ui("show_message", "Paste failed — text is on the clipboard (⌘V)")
        print(f"   recovered to clipboard: {text[:80]}")
        time.sleep(2.5)
    except Exception as e:
        print(f"⚠️  could not even reach the clipboard: {e}")


# ── Hotkey state machine ──────────────────────────────────────────────

recorder = Recorder()
recording = False
locked = False      # hands-free mode: fn+Space locks, fn stops
cancelled = False
_state_lock = threading.Lock()

# Streaming state: text already transcribed while recording, and how many
# samples of the recording it covers. Guarded by _stream_lock — the worker
# holds it during each chunk transcription, so finish_recording's snapshot
# automatically waits for an in-flight chunk to land.
_stream_lock = threading.Lock()
_stream_committed = ""
_stream_consumed = 0
_rec_generation = 0

# Last stopped recording, kept so Force Transcribe can recover a dictation
# even after the normal pipeline died or wedged.
_last_audio = None


FRAME_SECONDS = 0.02  # RMS analysis frame for pause detection


def _frame_rms(audio: np.ndarray, hop: int) -> np.ndarray:
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((audio[:n * hop].reshape(n, hop) ** 2).mean(axis=1))


def _best_cut(segment: np.ndarray) -> int:
    """Fallback cut for someone who hasn't drawn breath in STREAM_MAX_SECONDS:
    the quietest 50 ms window in the last second, so we at least avoid slicing
    through the middle of a word."""
    win = int(0.05 * SAMPLE_RATE)
    search_start = max(0, len(segment) - SAMPLE_RATE)
    best_idx, best_rms = len(segment), float("inf")
    for start in range(search_start, len(segment) - win, win):
        rms = float(np.sqrt(np.mean(segment[start:start + win] ** 2)))
        if rms < best_rms:
            best_rms, best_idx = rms, start + win // 2
    return best_idx


def _find_pause_cut(segment: np.ndarray) -> int:
    """Index of the latest real speech pause in `segment`, or -1 if the speaker
    is still mid-sentence.

    Cutting here is the whole point of the rewrite: a chunk that ends at a
    pause is a complete utterance, so the full stop Whisper puts on it is
    true. The old fixed 5s cut landed mid-sentence almost every time, which is
    where the 3-word fragment sentences came from.
    """
    hop = int(FRAME_SECONDS * SAMPLE_RATE)
    rms = _frame_rms(segment, hop)
    if rms.size == 0:
        return -1
    peak = float(rms.max())
    if peak <= 1e-6:
        return -1
    # Two independent estimates of "quiet", and we take the STRICTER one.
    #
    # The 20th percentile only approximates the room's noise floor while at
    # least a fifth of the segment is actually silent. For a speaker who does
    # not pause — exactly the case that reaches this code — it is a *speech*
    # level, and taking the larger of the two thresholds then marked half the
    # voiced frames as quiet: unstressed syllables between two emphasised ones
    # read as a sentence boundary and the cut landed mid-word. That produced
    # transcripts like "Using the ray up. Liability loop." out of "using the
    # reliability loop", complete with the invented full stop this whole
    # design exists to avoid.
    floor = float(np.percentile(rms, 20))
    thresh = min(floor * 2.5, peak * 0.06)
    quiet = rms < thresh

    need = max(1, int(PAUSE_SECONDS / FRAME_SECONDS))
    earliest = int(STREAM_MIN_SECONDS / FRAME_SECONDS)
    best = -1
    run_start = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None and i - run_start >= need and run_start >= earliest:
            best = (run_start + i) // 2
        run_start = None
    # A pause still open at the end of the buffer means the speaker is silent
    # right now. That's the ideal cut, but demand extra length so we don't
    # split on a mid-sentence breath.
    if (run_start is not None and run_start >= earliest
            and len(quiet) - run_start >= need * 2):
        best = (run_start + len(quiet)) // 2
    return best * hop if best >= 0 else -1


def _stream_worker(generation: int):
    """While recording: transcribe completed chunks in the background so
    stopping only has to handle the tail.

    Note this no longer holds _stream_lock across the model call. It used to,
    which is why releasing fn could block for seconds and why the app wedged
    when a chunk got stuck in the model. Now the lock is held only to publish
    the result, and a chunk that finishes after recording ended is discarded
    (the final pass re-covers that audio) instead of racing.
    """
    global _stream_committed, _stream_consumed
    max_samples = int(STREAM_MAX_SECONDS * SAMPLE_RATE)
    min_samples = int(STREAM_MIN_SECONDS * SAMPLE_RATE)
    failures = 0

    while recording and generation == _rec_generation:
        time.sleep(STREAM_POLL_SECONDS)
        if not recording or generation != _rec_generation:
            return

        with _stream_lock:
            consumed = _stream_consumed
            prompt_tail = _stream_committed[-200:]
        try:
            total = recorder.available()
        except Exception as e:
            print(f"⚠️  reading the audio buffer failed ({e}) — streaming off")
            return
        # The buffer cap has to actually end the recording. Letting it fill and
        # silently stop accepting audio meant you could keep talking for
        # another ten minutes, pill still pulsing, and get a transcript that
        # just ended at minute twenty.
        if recorder.overflowed:
            print(f"⚠️  hit the {MAX_RECORD_SECONDS // 60}-minute recording limit "
                  "— stopping and transcribing what we have")
            ui("show_message", "Recording limit reached — transcribing")
            threading.Thread(target=finish_recording, daemon=True).start()
            return
        if total - consumed < min_samples:
            continue

        segment = recorder.view(consumed, consumed + max_samples)
        if segment.size < min_samples:
            continue

        forced = segment.size >= max_samples
        cut = _find_pause_cut(segment)
        if cut < 0:
            if not forced:
                continue  # still mid-sentence — let it run to a natural pause
            cut = _best_cut(segment)
            print(f"   [stream] {STREAM_MAX_SECONDS}s without a pause — cutting anyway")

        chunk = segment[:cut]
        if chunk.size == 0:
            continue
        if float(np.abs(chunk).max()) < 0.01:  # silence: skip it, don't transcribe it
            with _stream_lock:
                if generation != _rec_generation:
                    return
                _stream_consumed = consumed + cut
            continue

        try:
            text = transcribe(chunk, prompt=build_prompt(prompt_tail))
            failures = 0
        except Exception as e:
            failures += 1
            print(f"⚠️  stream chunk failed ({e})")
            if failures >= STREAM_MAX_FAILURES:
                print("   streaming disabled for this recording — the final pass covers it")
                return
            continue

        with _stream_lock:
            if generation != _rec_generation:
                return  # recording ended mid-chunk: process() re-does this audio
            _stream_consumed = consumed + cut
            if text:
                _stream_committed = _stitch(_stream_committed, text)
                if SHOW_LIVE_TEXT:
                    ui("set_live_text", "…" + _stream_committed[-38:])
                print(f"   [stream] {chunk.size / SAMPLE_RATE:.0f}s chunk committed"
                      f"{' (forced)' if forced else ''}")


def start_recording():
    global recording, locked, cancelled
    global _stream_committed, _stream_consumed, _rec_generation
    with _state_lock:
        if recording:
            return
        recording, locked, cancelled = True, False, False
        try:
            recorder.start()
        except Exception as e:
            recording = False
            play(SOUND_ERROR)
            print(f"⚠️  Could not start recording: {e}")
            return
        reset_detected_language()  # each dictation detects its own language once
        with _stream_lock:
            _stream_committed, _stream_consumed = "", 0
            _rec_generation += 1
        threading.Thread(
            target=_stream_worker, args=(_rec_generation,), daemon=True
        ).start()
    play(SOUND_START)
    set_status(ICON_RECORDING)
    ui("show_recording", False)
    print("● recording… (Space = hands-free, Esc = cancel)")


def finish_recording():
    """Stop recording and hand the audio to the pipeline.

    Everything that reads or retires streaming state happens inside the one
    _state_lock section, together with recorder.stop(). Splitting it meant a
    second fn press could start a whole new recording in the gap: the new
    recording's stream worker was retired before its first chunk, and this
    recording lost the text it had already streamed and re-transcribed itself
    from scratch through the slow path — concurrently with live capture.
    """
    global recording, locked, _last_audio, _rec_generation
    with _state_lock:
        if not recording:
            return
        recording, locked = False, False
        audio = recorder.stop()
        was_cancelled = cancelled
        short = len(audio) / SAMPLE_RATE < MIN_SECONDS
        # Keep the recovery copy only for audio worth recovering, so an
        # accidental tap can't overwrite a real dictation that failed to land.
        if not was_cancelled and not short:
            _last_audio = audio
        with _stream_lock:
            # Retires any chunk still inside the model: its result is dropped
            # and process() transcribes that audio itself. Costs at most one
            # extra pass and, unlike waiting on the stream lock, can never
            # block the fn key or wedge the app.
            _rec_generation += 1
            committed, consumed = _stream_committed, _stream_consumed
    if was_cancelled:
        set_status(ICON_IDLE)
        ui("hide")
        print("✗ cancelled")
        return
    if short:
        set_status(ICON_IDLE)
        ui("hide")
        return
    threading.Thread(
        target=process, args=(audio, committed, consumed), daemon=True
    ).start()


def force_transcribe() -> str:
    """Hard recovery (Settings → Log tab, or the menu bar item).

    The pipeline can wedge: a stream chunk stuck inside the model holds
    _stream_lock (so releasing fn blocks forever), or a process() thread
    never returns and keeps _job_lock. This path stops whatever is still
    recording (or falls back to the last stopped recording), uses lock
    TIMEOUTS instead of blocking waits, and puts the result on the
    clipboard — no paste, since focus is on the Settings window.
    Returns a status message for the UI.
    """
    global recording, locked, cancelled, _last_audio, _rec_generation

    audio, committed, consumed = None, "", 0
    # Even this lock gets a timeout: start_recording holds it across opening
    # the audio device, which can hang outright when CoreAudio changes devices
    # (AirPods connecting, an interface unplugged). Blocking here would make
    # the recovery path part of the wedge.
    if not _state_lock.acquire(timeout=3):
        return ("The app is busy starting or stopping a recording. "
                "Try again in a moment, or use Restart.")
    try:
        was_recording = recording
        recording, locked, cancelled = False, False, False
        _rec_generation += 1  # makes any stream worker loop exit
        if was_recording or recorder._stream is not None:
            try:
                audio = recorder.stop()
            except Exception as e:
                print(f"⚠️  stopping the audio stream failed ({e})")
                audio = recorder.snapshot()
                # Close it properly — leaving the stream registered means its
                # callback keeps writing into every later recording.
                stale, recorder._stream = recorder._stream, None
                if stale is not None:
                    try:
                        stale.close()
                    except Exception:
                        pass
            if audio is not None and len(audio):
                _last_audio = audio
            if _stream_lock.acquire(timeout=2):
                committed, consumed = _stream_committed, _stream_consumed
                _stream_lock.release()
            else:
                committed, consumed = _stream_committed, _stream_consumed
                print("⚠️  stream lock stuck — recovered committed text without it")
    finally:
        _state_lock.release()

    if audio is None or not len(audio):
        if _last_audio is not None and len(_last_audio):
            audio, committed, consumed = _last_audio, "", 0
            print("↻ force transcribe: re-running the last recording")
        else:
            return "Nothing to transcribe — no active or saved recording."

    set_status(ICON_BUSY)
    ui("show_processing")
    # Wait long enough that a *healthy* dictation finishes first. The old 2s
    # was shorter than a normal transcription, so pressing this during a
    # working dictation reliably double-processed it: two history entries, and
    # process()'s clipboard restore clobbering the text you were told to paste.
    got_job = _job_lock.acquire(timeout=FORCE_JOB_WAIT_SECONDS)
    if not got_job:
        print("⚠️  a transcription job seems stuck — bypassing the job lock")
    try:
        tail = audio[consumed:]
        tail_text = ""
        if len(tail) / SAMPLE_RATE >= 0.3:
            tail_text = transcribe(
                tail, prompt=build_prompt(committed[-200:]),
                lock_timeout=MODEL_BUSY_TIMEOUT,
            )
        raw = _stitch(committed, tail_text)
        if not raw:
            return "Nothing heard in the recording."
        text = clean_up(
            normalize_punctuation(apply_replacements(collapse_repetitions(raw)))
        )
        if not text:
            return "Only noise heard — nothing usable."
        copy_to_clipboard(text)
        append_history(text, len(audio) / SAMPLE_RATE)
        play(SOUND_DONE)
        print(f"✓  force transcribe → clipboard ({len(text)} chars)")
        return f"Transcribed ✓ — copied to clipboard ({len(text)} chars), ⌘V to paste."
    except ModelBusy:
        play(SOUND_ERROR)
        # The streamed text is still worth having even though the tail is lost.
        if committed:
            copy_to_clipboard(committed)
            append_history(committed, len(audio) / SAMPLE_RATE)
            print(f"⚠️  model wedged — saved {len(committed)} streamed chars")
            return (f"The model is wedged. Saved what was transcribed while you "
                    f"spoke ({len(committed)} chars) to the clipboard. Restart to clear it.")
        return ("The model is wedged and nothing was transcribed yet. "
                "Use Restart Flow Dictate.")
    except Exception as e:
        play(SOUND_ERROR)
        print(f"⚠️  force transcribe failed: {e}")
        return f"Force transcribe failed: {e}"
    finally:
        if got_job:
            _job_lock.release()
        set_status(ICON_IDLE)
        ui("hide")
        release_gpu_cache()


def handle_fn(fn_down: bool):
    if fn_down:
        if not recording:
            start_recording()
        elif locked:
            finish_recording()
    else:
        if recording and not locked:
            finish_recording()


# Key events are strictly ordered; handling them on one thread each was not.
# A fast fn tap could deliver the release before the press had been scheduled:
# the release found recording==False and did nothing, then the press started a
# recording that nothing would ever stop — pill pulsing, next fn press inert.
# One worker draining a queue keeps the handling in the order the keys arrived.
_hotkey_queue = queue.Queue()


def _hotkey_worker():
    while True:
        action = _hotkey_queue.get()
        try:
            action()
        except Exception as e:
            print(f"⚠️  hotkey handling failed: {e}")


def _post_hotkey(action):
    _hotkey_queue.put(action)


def on_press(key):
    # Space (lock) and Esc (cancel) are handled in darwin_intercept, which
    # also swallows them. Here we only catch "some other key while fn is
    # held" → it's a shortcut (fn+arrow etc.), not dictation: discard.
    #
    # Queued rather than run inline: this is the event-tap callback thread, and
    # cancelling tears the audio device down. Doing that here could overrun the
    # tap's timeout, which macOS answers by disabling the tap — the fn key
    # going dead, i.e. the very bug _tap_watchdog exists to undo.
    if recording and not locked and key not in (keyboard.Key.space, keyboard.Key.esc):
        _post_hotkey(_cancel_recording)


def _lock_hands_free():
    global locked
    locked = True
    ui("show_recording", True)
    print("● hands-free — tap fn to stop")


def _cancel_recording():
    global cancelled
    cancelled = True
    finish_recording()


_listener = None


def _patch_tap_capture(listener):
    """pynput keeps its CGEventTap in a local var — capture it so the
    watchdog can re-enable the tap when macOS disables it under load."""
    orig = listener._create_event_tap

    def patched():
        tap = orig()
        listener._flow_tap = tap
        return tap

    listener._create_event_tap = patched


def _tap_watchdog():
    """macOS silently disables event taps when the process stalls (e.g. GPU
    spikes during transcription) — that's the 'fn suddenly dead' bug.
    Re-enable it whenever that happens.

    Polls often: while the tap is dead the hotkey does nothing at all, so every
    second of delay is a second of the app looking broken.
    """
    while True:
        time.sleep(TAP_CHECK_SECONDS)
        tap = getattr(_listener, "_flow_tap", None)
        if tap is None:
            continue
        try:
            if not Quartz.CGEventTapIsEnabled(tap):
                Quartz.CGEventTapEnable(tap, True)
                print("⚠️  event tap was disabled by macOS — re-enabled (fn hotkey restored)")
        except Exception as e:
            print(f"   tap watchdog error: {e}")


def _pipeline_watchdog():
    """Make a wedged dictation impossible to sit through unnoticed.

    A transcription that never returns is the failure that costs trust: you
    spoke, the spinner turns, and nothing ever arrives. A pass inside the model
    cannot be interrupted from Python, so instead we bound how long the silence
    can last — say so out loud first, and if the job blows past a deadline
    scaled to its own audio length, save its best text and let the LaunchAgent
    restart us into a working state.
    """
    announced = set()
    while True:
        time.sleep(2)
        now = time.time()
        with _pipeline_lock:
            jobs = [(j, dict(v)) for j, v in _pipeline_jobs.items()]
        announced.intersection_update({j for j, _ in jobs})
        for job, info in jobs:
            stuck_for = now - info["since"]
            if stuck_for > info["deadline"]:
                _recover_wedged(job, info, stuck_for)
                return
            if stuck_for > STUCK_WARN_SECONDS and job not in announced:
                announced.add(job)
                print(f"⚠️  this dictation has been processing for {stuck_for:.0f}s "
                      f"(deadline {info['deadline']:.0f}s) — if it does not land, "
                      "the app restarts itself.")
                play(SOUND_ERROR)
                ui("show_message", "Taking unusually long…")


def _recover_wedged(job: int, info: dict, stuck_for: float):
    """Save what the wedged job managed to transcribe, then restart."""
    partial = info.get("text") or ""
    print(f"⚠️  dictation wedged for {stuck_for:.0f}s — restarting the app")
    if partial:
        try:
            copy_to_clipboard(partial)
            append_history(partial, 0.0)
            print(f"   saved {len(partial)} chars to the clipboard first")
        except Exception as e:
            print(f"   could not save the partial transcript: {e}")
        ui("show_message", "Stuck — restarting, text on clipboard")
    else:
        ui("show_message", "Stuck — restarting")
    play(SOUND_ERROR)
    time.sleep(1.0)
    if not restart_app():
        # Not launchd-managed, so nothing will bring us back. Staying alive
        # with a working hotkey beats exiting into nothing.
        print("   could not restart via launchd — staying up; use Quit and relaunch")
        _pipeline_end(job)


def darwin_intercept(event_type, event):
    """Drive the fn hotkey and swallow the Space/Esc keystrokes we consume.

    State changes happen HERE (not only in on_press) so behavior does not
    depend on whether suppressed events still reach the pynput handlers.
    """
    try:
        if event_type == Quartz.kCGEventTapDisabledByTimeout:
            tap = getattr(_listener, "_flow_tap", None)
            if tap is not None:
                Quartz.CGEventTapEnable(tap, True)
                print("⚠️  event tap timeout — re-enabled immediately")
        elif event_type == Quartz.kCGEventFlagsChanged:
            keycode = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            if keycode == KEYCODE_FN:
                fn_down = bool(
                    Quartz.CGEventGetFlags(event) & Quartz.kCGEventFlagMaskSecondaryFn
                )
                # Keep the event-tap callback fast, and keep press/release in
                # the order they arrived — see _hotkey_worker.
                _post_hotkey(lambda d=fn_down: handle_fn(d))
        elif event_type == Quartz.kCGEventKeyDown and recording:
            keycode = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            if keycode == KEYCODE_SPACE and not locked:
                _post_hotkey(_lock_hands_free)
                return None
            if keycode == KEYCODE_ESC:
                _post_hotkey(_cancel_recording)
                return None
    except Exception:
        pass
    return event


# ── Main ──────────────────────────────────────────────────────────────

def restart_app() -> bool:
    """Restart via the LaunchAgent. Returns whether launchctl accepted it.

    Started by hand (run.sh, or with the agent booted out) launchctl exits
    non-zero and nothing brings the app back — so callers must know, instead of
    firing a Popen into the void and reporting success.
    """
    try:
        done = subprocess.run(
            ["launchctl", "kickstart", "-k",
             f"gui/{os.getuid()}/com.hdt.flow-dictate"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        print(f"⚠️  restart failed to launch: {e}")
        return False
    if done.returncode != 0:
        print(f"⚠️  launchctl refused the restart ({done.stderr.strip() or done.returncode})"
              " — not running under the LaunchAgent?")
        return False
    return True


_singleton_lock = None  # kept open for the process lifetime


def acquire_singleton():
    """Take an exclusive lock so exactly one instance runs.

    Returns (won, reason). The reason distinguishes "another instance holds it"
    from "the lock file could not be created at all" — conflating them meant a
    read-only app folder made the app exit claiming it was already running.

    This replaces a pgrep-based duplicate check that could deadlock: when
    launchd started two instances at once, each saw the other, waited out its
    grace period, and both exited — and because they exited 0, KeepAlive
    (SuccessfulExit=false) did not restart either. The app was simply gone
    until someone noticed. A kernel lock has no such tie: exactly one process
    gets it, and it is released automatically however the holder dies, so
    there is no stale lock to clean up after a crash or SIGKILL.
    """
    global _singleton_lock
    try:
        # "a+", not "w": opening for write truncated the file before the lock
        # was taken, so a losing second instance blanked the winner's recorded
        # PID on its way out and the file was useless for diagnostics.
        handle = open(LOCK_FILE, "a+")
    except OSError as e:
        return False, f"cannot open {LOCK_FILE.name} ({e}) — is the app folder writable?"
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False, "Flow Dictate is already running — this second instance exits."
    except Exception as e:
        print(f"⚠️  could not take the instance lock ({e}) — starting anyway")
        return True, ""
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    _singleton_lock = handle
    return True, ""


def switch_model(profile: str):
    """Swap the Whisper weights at runtime.

    mlx_whisper's ModelHolder keeps exactly one model and drops the old one on
    a path change, so the only extra step is flushing MLX's buffer pool —
    otherwise the previous model's freed memory stays parked in the cache.
    """
    global MODEL_PROFILE, WHISPER_MODEL
    if profile not in WHISPER_MODELS or profile == MODEL_PROFILE:
        return True
    previous = MODEL_PROFILE
    repo, label = WHISPER_MODELS[profile]
    with _model_lock:
        MODEL_PROFILE, WHISPER_MODEL = profile, repo
    release_gpu_cache()
    print(f"Model → {label} ({repo})")
    ui("show_message", "Loading model…")
    try:
        transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # load + warm up
    except Exception as e:
        # Persisting the choice before proving it loads was a way to brick the
        # app: config.json named a model whose weights could not be fetched,
        # the unguarded warm-up in main() then threw, and KeepAlive restarted
        # into the same failure forever. Only a model that actually loaded gets
        # written to disk.
        print(f"⚠️  loading {repo} failed: {e} — staying on {previous}")
        with _model_lock:
            MODEL_PROFILE, WHISPER_MODEL = previous, WHISPER_MODELS[previous][0]
        ui("show_message", "Model failed to load")
        time.sleep(2.0)
        return False
    finally:
        ui("hide")
    save_config()
    print(f"   ready ({mx.get_active_memory() / 1e9:.2f} GB resident)")
    return True


def _warm_up_model() -> bool:
    """Load the configured model, falling back to the other profile.

    An unguarded warm-up here was the second half of the boot-loop trap: a
    model that cannot be fetched threw straight out of main(), the process
    exited non-zero, and KeepAlive restarted it into the same failure every ten
    seconds until config.json was edited by hand.
    """
    global MODEL_PROFILE, WHISPER_MODEL
    for profile in [MODEL_PROFILE] + [p for p in WHISPER_MODELS if p != MODEL_PROFILE]:
        MODEL_PROFILE, WHISPER_MODEL = profile, WHISPER_MODELS[profile][0]
        try:
            transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except Exception as e:
            print(f"⚠️  could not load {WHISPER_MODEL}: {e}")
            release_gpu_cache()
            continue
        release_gpu_cache()
        print(f"   {mx.get_active_memory() / 1e9:.2f} GB resident"
              f"{'' if profile == MODEL_PROFILE else f' (fell back to {profile})'}")
        save_config()
        return True
    print("⚠️  no Whisper model could be loaded — check the network and disk space.")
    print("   Starting anyway: the menu bar and Settings stay reachable so you can")
    print("   retry from the Model menu. Exiting here would just hide the problem.")
    return False


def _open_settings(_):
    import settings_window
    settings_window.open_settings(sys.modules[__name__])


def _force_transcribe_menu(_):
    # never block the AppKit main thread with model work
    threading.Thread(target=force_transcribe, daemon=True).start()


def main():
    global _app, overlay, _hist_menu
    rotate_log()
    _install_timestamps()
    ok, why = acquire_singleton()
    if not ok:
        print(f"Not starting: {why}")
        return
    print("Flow Dictate")
    print(f"  STT:        {WHISPER_MODEL} ({WHISPER_MODELS[MODEL_PROFILE][1]}, local)")
    print(f"  Chunking:   at speech pauses, {STREAM_MIN_SECONDS}-{STREAM_MAX_SECONDS}s")
    print(f"  Cleanup:    {CLEANUP_MODEL if _claude else 'OFF — no ANTHROPIC_API_KEY, pasting raw'}")
    print(f"  History:    {HISTORY_FILE}")
    terms = vocabulary_terms()
    print(f"  Dictionary: {terms or DEFAULT_VOCABULARY_COUNT} terms"
          f"{'' if terms else ' (built-in defaults)'}, "
          f"{len(REPLACEMENTS)} replacements")
    print()
    load_history()
    print("Loading Whisper model (first run downloads the weights)…")
    _warm_up_model()
    print()
    print("Ready. Hold fn/🌐 to talk — or fn+Space for hands-free, fn stops.")

    global _listener
    listener = keyboard.Listener(on_press=on_press, darwin_intercept=darwin_intercept)
    _patch_tap_capture(listener)
    _listener = listener
    listener.start()
    threading.Thread(target=_hotkey_worker, daemon=True).start()
    threading.Thread(target=_tap_watchdog, daemon=True).start()
    threading.Thread(target=_pipeline_watchdog, daemon=True).start()

    if HAS_UI:
        # Menu-bar-only app: never show a Dock icon, even with windows open.
        nsapp = AppKit.NSApplication.sharedApplication()
        nsapp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        for icns in (
            # /Applications is where the bundle actually lives; the old
            # ~/Applications path has never existed, so the icon only worked
            # because the APP_DIR fallback happens to be there too.
            Path("/Applications/Flow Dictate.app/Contents/Resources/AppIcon.icns"),
            Path.home() / "Applications/Flow Dictate.app/Contents/Resources/AppIcon.icns",
            APP_DIR / "AppIcon.icns",
        ):
            if icns.exists():
                img = AppKit.NSImage.alloc().initWithContentsOfFile_(str(icns))
                if img:
                    nsapp.setApplicationIconImage_(img)
                break
        overlay = Overlay()
        _app = rumps.App("Flow Dictate", title=ICON_IDLE, quit_button="Quit Flow Dictate")
        _hist_menu = rumps.MenuItem("History (click to copy)")
        settings_item = rumps.MenuItem("Settings…", callback=_open_settings)
        force_item = rumps.MenuItem("Force Transcribe (if stuck)", callback=_force_transcribe_menu)

        lang_menu = rumps.MenuItem("Language")
        lang_items = []

        def _lang_callback(code):
            def cb(sender):
                global LANGUAGE
                LANGUAGE = code
                save_config()
                for it in lang_items:
                    it.state = 0
                sender.state = 1
                print(f"Language set to: {code or 'auto-detect'}")
            return cb

        for title, code in (
            ("German", "de"),
            ("English", "en"),
            ("Auto-detect (slower)", None),
        ):
            mi = rumps.MenuItem(title, callback=_lang_callback(code))
            mi.state = 1 if code == LANGUAGE else 0
            lang_items.append(mi)
            lang_menu.add(mi)

        model_menu = rumps.MenuItem("Model")
        model_items = []

        def _model_callback(profile):
            def cb(sender):
                # Tick only after the model actually loaded. Ticking eagerly
                # left the menu claiming a model that had failed to load, and
                # switch_model's own "already on this profile" guard then made
                # the item unclickable — permanently wrong, log-only.
                def run():
                    if switch_model(profile):
                        def mark():
                            for it in model_items:
                                it.state = 1 if it is sender else 0
                        on_main(mark)
                threading.Thread(target=run, daemon=True).start()
            return cb

        for profile, (_, label) in WHISPER_MODELS.items():
            mi = rumps.MenuItem(label, callback=_model_callback(profile))
            mi.state = 1 if profile == MODEL_PROFILE else 0
            model_items.append(mi)
            model_menu.add(mi)

        _app.menu = [settings_item, force_item, lang_menu, model_menu, _hist_menu]
        refresh_history_menu()
        _app.run()
    else:
        print("(tip: pip install rumps pyobjc for menu bar icon + overlay)")
        listener.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
