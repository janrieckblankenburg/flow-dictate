# Flow Dictate

Hold-to-talk dictation for macOS — a self-built Wispr Flow replacement.

**Hold the fn/🌐 key → speak → release.** The audio is transcribed locally
(Whisper on Apple Silicon), cleaned up by Claude Haiku (filler words out,
punctuation in) and pasted into whatever app has focus. **Esc** while
recording cancels. fn-shortcuts (fn+arrows etc.) keep working — pressing
any other key while holding fn discards the recording.

**Hands-free:** hold fn, tap **Space** → recording locks on, you can release
fn. Tap **fn** again to stop and transcribe.

**UI:** pill overlay at top-center (pulsing dot = recording, spinner =
transcribing), menu bar icon 🎙/🟠/⏳ with **History** (click an entry to
copy it) and a **Settings…** window (history table with search, dictionary
editor with live apply, log viewer with restart button). All dictations are
also stored in `history.jsonl`.

Privacy: audio never leaves the Mac. Only the finished text is (optionally)
sent to the Claude API for cleanup. Without `ANTHROPIC_API_KEY`, the raw
transcript is pasted unchanged.

## Architecture

- `/Applications/Flow Dictate.app` — thin app bundle: a launcher script plus a
  renamed copy of the Python runtime, so macOS shows "Flow Dictate" in
  permission dialogs, Login Items and Activity Monitor.
- `~/Applications/flow-dictate/` — the actual code, venv, config and data.
- `~/Library/LaunchAgents/com.hdt.flow-dictate.plist` — starts the app at
  login, restarts it on crash, writes `flow-dictate.log`.
  Control: `launchctl bootout/bootstrap gui/$(id -u) [plist]`,
  restart: `launchctl kickstart -k gui/$(id -u)/com.hdt.flow-dictate`.

**Rebuild the bundle with `./install.sh`, never by hand.** It discovers the
Python version instead of hardcoding it, puts the executable where CFBundle
expects it, and signs with an explicit identifier. Doing any of that wrong has
non-obvious consequences — see *Code signing and the permission trap* below.

## Setup

```zsh
./setup.sh          # venv + dependencies (one-time)
./install.sh        # build the app bundle + install the LaunchAgent
```

`ANTHROPIC_API_KEY` for the cleanup pass (optional, recommended): put
`ANTHROPIC_API_KEY=sk-ant-…` into `flow-dictate/.env` (chmod 600) — the
launcher loads it; a shell export does not reach launchd.

## macOS permissions (one-time, for "Flow Dictate")

System Settings → Privacy & Security:

1. **Microphone** — prompted automatically on first dictation
2. **Input Monitoring** — for the global fn hotkey
3. **Accessibility** — to send ⌘V into other apps

If entries for "python" remain from the development phase, they can be
removed. Also required: System Settings → Keyboard → "Press 🌐 key to:
**Do Nothing**" (already set via `defaults write com.apple.HIToolbox
AppleFnUsageType -int 0`).

## Configuration

Top of `dictate.py`:

| Setting | Default | Meaning |
|---|---|---|
| `KEYCODE_FN` | 63 (fn/🌐) | push-to-talk key |
| `MODEL_PROFILE` | `efficient` | `efficient` = turbo 4-bit (~0.5 GB), `accurate` = turbo 16-bit (~1.6 GB). Switchable in the menu bar under **Model** |
| `LANGUAGE` | auto | `"de"`/`"en"` skips detection; on auto it is detected once per dictation, not per chunk |
| `STREAM_MIN/MAX_SECONDS` | 12 / 26 | chunk boundaries while speaking — see below |
| `PAUSE_SECONDS` | 0.45 | silence that counts as a sentence boundary |
| `MLX_CACHE_LIMIT` | 384 MB | ceiling on MLX's pooled GPU buffers |
| `CLEANUP_MODEL` | Haiku 4.5 | cleanup LLM |
| `RESTORE_CLIPBOARD` | `true` | restore previous clipboard after pasting |

## How chunking works (and why it matters)

Long dictations are transcribed in background chunks while you speak, so
releasing fn only has to process the tail. Two properties of Whisper decide
how those chunks have to be cut, and both were originally got wrong:

- **A call costs the same whatever it contains.** Whisper pads every call to a
  30s window: measured 1.47s for 2s of audio and 1.50s for 28s. Cutting every
  5s therefore did ~5x the GPU work of cutting every 26s, saturating the GPU
  for the whole time you spoke — which starved the machine for memory and let
  macOS silently kill the event tap (the "fn key suddenly dead" bug).
- **Whisper punctuates each chunk as a complete utterance.** A cut through the
  middle of a sentence gets a full stop or an ellipsis that is simply wrong,
  which is where transcripts like `we have the. Reliability loop.` and
  `Airtable Schema... ...für den Kunden` came from.

So chunks are cut at **real speech pauses** (≥ `PAUSE_SECONDS` of silence),
never on a timer, and only between `STREAM_MIN_SECONDS` and
`STREAM_MAX_SECONDS`. A speaker who never pauses gets one forced cut at the
quietest point every `STREAM_MAX_SECONDS`. `_stitch()` then repairs the seam:
a chunk opening in lower case proves the previous full stop was invented.

One more trap: `initial_prompt` sets **style as well as vocabulary**. Passing
the bare comma-separated `vocabulary.txt` list taught Whisper that this text
has no sentence punctuation, and it returned whole chunks without a single
full stop. `build_prompt()` wraps the same terms in a punctuated sentence,
which keeps the spelling help *and* the periods.

## Where the latency went

Releasing fn used to take ~3.2s on a 35s dictation; it now takes ~0.7s. Three
separate pieces of waste, none of them the model being slow:

| Fix | Effect |
|---|---|
| Cut at pauses, up to 26s, instead of every 5s | 8 → 2 model passes on a 35s dictation |
| Detect the language once per dictation, not per chunk | auto-detect costs a full extra encoder pass: 3.20s vs 1.68s |
| Size the encoder window to the audio (`_enable_short_windows`) | 1.73s → 0.58s for 7s of speech |

The window trick is the biggest single win and deserves the detail: Whisper's
encoder is hard-wired to a 30s window, so a 6-second dictation did 30 seconds
of encoder work. Its positional embedding is *sinusoidal* — row k is a function
of k alone — so the first K rows are exactly the embedding for a K-frame
window, and slicing it is arithmetically identical, not an approximation. Cost
is a small confidence drop with less context (`avg_logprob` -0.07 at 30s vs
-0.18 at 10s) and mean word error moving 6.1% → 6.8%, which is inside the noise
of that test set. `WINDOW_HEADROOM` keeps slack because a window that only just
covers the audio measurably degrades.

`whisper-medium-q4` and `whisper-small-q4` were both tried as "just go smaller"
alternatives: medium is slower *and* worse than turbo (13.5% WER, 0.68s),
small halves the time but doubles the errors (13.0%, 0.27s), and base is
unusable at 38.7%. turbo-q4 at 6.8% / 0.56s is the right point on that curve.

## Never losing a dictation

Speaking and getting nothing back is the failure that makes the app untrustworthy,
so the pipeline is built so that it cannot happen quietly:

- The transcript is written to `history.jsonl` **before** delivery is attempted.
- If ⌘V fails (Accessibility permission revoked, secure input field), the text
  goes to the clipboard and the pill says so.
- If a stage throws, the best text available at that point is rescued — a tail
  pass that dies cannot take the part already transcribed while you spoke.
- `_pipeline_watchdog` bounds how long silence can last: it announces a
  dictation still running after `STUCK_WARN_SECONDS`, and if it is still wedged
  at `STUCK_RESTART_SECONDS` it saves the partial transcript to the clipboard
  and restarts via the LaunchAgent. A pass inside the model cannot be
  interrupted from Python, so a bounded restart is the only real guarantee.
- `_tap_watchdog` re-enables the event tap every `TAP_CHECK_SECONDS`; while the
  tap is dead the hotkey does nothing at all, so it is polled often.

Custom dictionary: `vocabulary.txt` (one term per line → Whisper bias +
cleanup spelling) and `replacements.txt` (`wrong => right`, hard whole-word
replacement, e.g. `Softer => Softr`). Editable in the Settings window —
changes apply live.

## Known limitations

- Clipboard restore only preserves text (images in the clipboard are lost)
- Secure input fields (passwords) block synthetic ⌘V — macOS by design
- The first transcription after a restart is slower (model warm-up)
- Rare technical terms still need `replacements.txt`; the prompt only biases
  spelling, it cannot force it. The Haiku cleanup pass fixes these from
  context, but it is off without an API key.
- Recordings are capped at `MAX_RECORD_SECONDS` (20 min) so a stuck recording
  cannot fill memory

## Code signing and the permission trap

The app is ad-hoc signed, and an ad-hoc signature has no stable identity — its
designated requirement is a hash of the binary. So **every rebuild of the
bundle revokes Microphone, Input Monitoring and Accessibility**, because macOS
sees a different program. The failure mode is the worst kind: fn does nothing,
or recording captures pure silence, with nothing pointing at the rebuild as the
cause.

To make permissions survive rebuilds, create a self-signed certificate named
**Flow Dictate Signing** (Keychain Access → Certificate Assistant → Create a
Certificate, type *Code Signing*), then re-run `./install.sh`. It picks the
identity up automatically and the bundle keeps one identity from then on.

Related, and why `install.sh` puts the executable directly in `Contents/MacOS/`:
the old hand-built bundle exec'd `Contents/MacOS/runtime/Flow Dictate`, one
level deeper than CFBundle looks for `Contents/Info.plist`. The running process
therefore had **no bundle identity at all** (`CFBundleIdentifier` came back
NULL), which meant `NSMicrophoneUsageDescription` never reached the permission
prompt, `LSUIElement` and `CFBundleIconFile` did nothing — the app compensated
by setting its activation policy and icon by hand at runtime — and the
LaunchAgent's `AssociatedBundleIdentifiers` could not attribute the process for
Login Items.

## Single instance

Only one instance may run, enforced by an exclusive `flock` on
`.instance.lock`. This replaced a `pgrep`-based check that could deadlock:
when launchd started two instances at once, each saw the other, waited out its
grace period, and both exited — and since they exited 0, `KeepAlive
(SuccessfulExit=false)` restarted neither, so the app was simply gone. A
kernel lock has no such tie, and it is released however the holder dies, so
there is no stale lock to clean up after a crash.
