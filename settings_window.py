"""
Flow Dictate — Settings window (native AppKit).

Three tabs:
  History     — searchable table of all dictations, click/copy, clear
  Dictionary  — edit vocabulary.txt and replacements.txt, apply live
  Log         — tail of the app log, force-transcribe recovery, restart button

Opened from the menu bar via dictate.py. All UI runs on the main thread.
Every control is laid out against the tab view's REAL content size
(queried at build time), never against assumed dimensions.
"""

import subprocess
import threading

import AppKit
import objc
from Foundation import NSMakeRect, NSObject, NSOperationQueue

# Keep a strong reference — AppKit does not retain button targets, and it does
# not retain an NSTableView's dataSource either. This global is the only thing
# keeping the controller alive, so never clear it while the window can still be
# shown: the window survives being closed (setReleasedWhenClosed_(False)), and
# a freed data source means AppKit calls into deallocated memory on the next
# redraw — a hard crash, not an exception.
_controller = None


def _on_main(fn):
    NSOperationQueue.mainQueue().addOperationWithBlock_(fn)


def _button(title, target, action, frame):
    btn = AppKit.NSButton.alloc().initWithFrame_(frame)
    btn.setTitle_(title)
    btn.setBezelStyle_(AppKit.NSBezelStyleRounded)
    btn.setTarget_(target)
    btn.setAction_(action)
    return btn


def _label(text, frame, small=False):
    lbl = AppKit.NSTextField.labelWithString_(text)
    lbl.setFrame_(frame)
    if small:
        lbl.setTextColor_(AppKit.NSColor.secondaryLabelColor())
        lbl.setFont_(AppKit.NSFont.systemFontOfSize_(11))
    return lbl


def _text_scroll(frame, editable=True):
    scroll = AppKit.NSScrollView.alloc().initWithFrame_(frame)
    scroll.setHasVerticalScroller_(True)
    scroll.setAutohidesScrollers_(True)
    scroll.setBorderType_(AppKit.NSBezelBorder)
    tv = AppKit.NSTextView.alloc().initWithFrame_(
        NSMakeRect(0, 0, frame.size.width - 4, frame.size.height - 4)
    )
    tv.setEditable_(editable)
    tv.setRichText_(False)
    tv.setAutomaticQuoteSubstitutionEnabled_(False)
    tv.setAutomaticDashSubstitutionEnabled_(False)
    tv.setAutomaticTextReplacementEnabled_(False)
    tv.setFont_(AppKit.NSFont.monospacedSystemFontOfSize_weight_(12, 0.0))
    tv.setVerticallyResizable_(True)
    tv.setHorizontallyResizable_(False)
    tv.setMaxSize_((1e7, 1e7))
    tv.textContainer().setWidthTracksTextView_(True)
    tv.setAutoresizingMask_(AppKit.NSViewWidthSizable)
    scroll.setDocumentView_(tv)
    return scroll, tv


class SettingsController(NSObject):
    """Owns the window and wires every control. `dictate` is the app module."""

    def initWithModule_(self, module):
        self = objc.super(SettingsController, self).init()
        if self is None:
            return None
        self.dictate = module
        self.filtered = []
        self._build_window()
        return self

    # ── Window & tabs ────────────────────────────────────────────────

    def _build_window(self):
        W, H = 780, 520
        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskMiniaturizable
        )
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), style, AppKit.NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Flow Dictate")
        self.window.setReleasedWhenClosed_(False)
        self.window.center()

        self.tabs = AppKit.NSTabView.alloc().initWithFrame_(
            NSMakeRect(10, 10, W - 20, H - 20)
        )
        self.window.contentView().addSubview_(self.tabs)

        # Real usable area inside the tab view (below the segmented control)
        content = self.tabs.contentRect()
        w, h = int(content.size.width), int(content.size.height)

        for title, builder in (
            ("History", self._build_history_tab),
            ("Dictionary", self._build_dictionary_tab),
            ("Log", self._build_log_tab),
        ):
            item = AppKit.NSTabViewItem.alloc().initWithIdentifier_(title)
            item.setLabel_(title)
            view = AppKit.NSView.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
            builder(view, w, h)
            item.setView_(view)
            self.tabs.addTabViewItem_(item)

    # ── History tab ──────────────────────────────────────────────────

    def _build_history_tab(self, view, w, h):
        self.search = AppKit.NSSearchField.alloc().initWithFrame_(
            NSMakeRect(12, h - 38, w - 24, 28)
        )
        self.search.setPlaceholderString_("Search dictations…")
        self.search.setTarget_(self)
        self.search.setAction_("searchChanged:")
        self.search.setSendsSearchStringImmediately_(True)
        view.addSubview_(self.search)

        scroll = AppKit.NSScrollView.alloc().initWithFrame_(
            NSMakeRect(12, 48, w - 24, h - 96)
        )
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(AppKit.NSBezelBorder)

        self.table = AppKit.NSTableView.alloc().initWithFrame_(
            NSMakeRect(0, 0, w - 24, h - 96)
        )
        text_col_width = (w - 24) - 150 - 50 - 24
        for ident, title, width in (
            ("ts", "Time", 150),
            ("seconds", "Sec", 50),
            ("text", "Text", text_col_width),
        ):
            col = AppKit.NSTableColumn.alloc().initWithIdentifier_(ident)
            col.headerCell().setStringValue_(title)
            col.setWidth_(width)
            self.table.addTableColumn_(col)
        self.table.setDataSource_(self)
        self.table.setUsesAlternatingRowBackgroundColors_(True)
        self.table.setTarget_(self)
        self.table.setDoubleAction_("copySelected:")
        self.table.sizeLastColumnToFit()
        scroll.setDocumentView_(self.table)
        view.addSubview_(scroll)

        view.addSubview_(_button("Copy Selected", self, "copySelected:", NSMakeRect(12, 10, 130, 30)))
        view.addSubview_(_button("Clear History", self, "clearHistory:", NSMakeRect(150, 10, 130, 30)))
        self.hint = _label("Double-click a row to copy it.", NSMakeRect(296, 16, w - 310, 18), small=True)
        view.addSubview_(self.hint)

        self._refilter()

    # NSTableView data source

    def numberOfRowsInTableView_(self, table):
        return len(self.filtered)

    def tableView_objectValueForTableColumn_row_(self, table, column, row):
        if row >= len(self.filtered):
            return ""
        entry = self.filtered[row]
        return str(entry.get(str(column.identifier()), ""))

    def searchChanged_(self, sender):
        self._refilter()

    def _refilter(self):
        query = str(self.search.stringValue()).lower() if hasattr(self, "search") else ""
        entries = list(reversed(self.dictate.history))
        self.filtered = (
            [e for e in entries if query in e.get("text", "").lower()] if query else entries
        )
        self.table.reloadData()

    def copySelected_(self, sender):
        row = self.table.selectedRow()
        if 0 <= row < len(self.filtered):
            self.dictate.copy_to_clipboard(self.filtered[row]["text"])
            self.hint.setStringValue_("Copied ✓")

    def clearHistory_(self, sender):
        alert = AppKit.NSAlert.alloc().init()
        alert.setMessageText_("Clear the entire dictation history?")
        alert.setInformativeText_("This deletes all entries from history.jsonl.")
        alert.addButtonWithTitle_("Clear")
        alert.addButtonWithTitle_("Cancel")
        if alert.runModal() == AppKit.NSAlertFirstButtonReturn:
            self.dictate.clear_history()
            self._refilter()

    # ── Dictionary tab ───────────────────────────────────────────────

    def _build_dictionary_tab(self, view, w, h):
        half = (w - 36) / 2

        view.addSubview_(_label("Vocabulary (one term per line)", NSMakeRect(12, h - 30, half, 18)))
        view.addSubview_(_label("Replacements (wrong => right)", NSMakeRect(24 + half, h - 30, half, 18)))

        vocab_scroll, self.vocab_tv = _text_scroll(NSMakeRect(12, 48, half, h - 88))
        view.addSubview_(vocab_scroll)
        repl_scroll, self.repl_tv = _text_scroll(NSMakeRect(24 + half, 48, half, h - 88))
        view.addSubview_(repl_scroll)

        view.addSubview_(_button("Save & Apply", self, "saveDictionary:", NSMakeRect(12, 10, 130, 30)))
        self.dict_hint = _label(
            "Changes apply immediately to the next dictation — no restart needed.",
            NSMakeRect(150, 16, w - 165, 18), small=True,
        )
        view.addSubview_(self.dict_hint)

        self._load_dictionary_files()

    def _load_dictionary_files(self):
        for path, tv in (
            (self.dictate.VOCAB_FILE, self.vocab_tv),
            (self.dictate.REPLACE_FILE, self.repl_tv),
        ):
            try:
                tv.setString_(path.read_text(encoding="utf-8") if path.exists() else "")
            except Exception as e:
                tv.setString_(f"(could not read {path.name}: {e})")

    def saveDictionary_(self, sender):
        """Write both dictionary files, then report what is actually active.

        Both writes are guarded together: an unguarded second write left
        vocabulary saved, replacements not, reload never called — and because
        the exception escaped into PyObjC's action dispatch, the hint kept its
        old text and the user saw neither an error nor a confirmation.
        """
        try:
            self.dictate.VOCAB_FILE.write_text(
                str(self.vocab_tv.string()), encoding="utf-8")
            self.dictate.REPLACE_FILE.write_text(
                str(self.repl_tv.string()), encoding="utf-8")
        except Exception as e:
            self.dict_hint.setStringValue_(f"Could not save: {e}")
            return
        try:
            self.dictate.reload_dictionary()
        except Exception as e:
            self.dict_hint.setStringValue_(f"Saved, but reloading failed: {e}")
            return
        # Count what reload_dictionary actually produced, and say so when it
        # fell back to its built-in defaults — the old count was derived from
        # VOCABULARY and so reported "3 terms" for an empty vocabulary box,
        # silently biasing Whisper toward terms no longer shown in the editor.
        terms = self.dictate.vocabulary_terms()
        repls = len(self.dictate.REPLACEMENTS)
        note = "" if terms else "  (empty — using built-in defaults)"
        shown = terms or self.dictate.DEFAULT_VOCABULARY_COUNT
        self.dict_hint.setStringValue_(
            f"Applied ✓ — {shown} terms, {repls} replacements active.{note}")

    # ── Log tab ──────────────────────────────────────────────────────

    def _build_log_tab(self, view, w, h):
        log_scroll, self.log_tv = _text_scroll(NSMakeRect(12, 86, w - 24, h - 98), editable=False)
        view.addSubview_(log_scroll)

        # Recovery row: hard-stop a stuck recording/transcription and get the text
        view.addSubview_(_button("Force Transcribe Now", self, "forceTranscribe:", NSMakeRect(12, 46, 180, 30)))
        self.log_hint = _label(
            "Stuck while recording or transcribing? This force-stops, transcribes and copies the text.",
            NSMakeRect(200, 52, w - 214, 18), small=True,
        )
        view.addSubview_(self.log_hint)

        view.addSubview_(_button("Refresh", self, "refreshLog:", NSMakeRect(12, 10, 100, 30)))
        view.addSubview_(_button("Open App Folder", self, "openFolder:", NSMakeRect(120, 10, 150, 30)))
        view.addSubview_(_button("Restart Flow Dictate", self, "restartApp:", NSMakeRect(278, 10, 180, 30)))

        self.refreshLog_(None)

    LOG_TAIL_BYTES = 256 * 1024

    def refreshLog_(self, sender):
        """Show the tail of the log. Reads only the last chunk: this runs on
        the main thread every time Settings opens, and during a restart loop
        the log grows by megabytes a day — reading it whole would beachball the
        app exactly when the Log tab is what you need."""
        path = self.dictate.LOG_FILE
        try:
            if not path.exists():
                self.log_tv.setString_("(no log file found)")
                return
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > self.LOG_TAIL_BYTES:
                    f.seek(size - self.LOG_TAIL_BYTES)
                    f.readline()          # drop the partial first line
                blob = f.read()
            lines = blob.decode("utf-8", errors="replace").splitlines()[-400:]
            self.log_tv.setString_("\n".join(lines))
            self.log_tv.scrollToEndOfDocument_(None)
        except Exception as e:
            self.log_tv.setString_(f"(could not read the log: {e})")

    def openFolder_(self, sender):
        try:
            subprocess.run(["open", str(self.dictate.APP_DIR)], timeout=10)
        except Exception as e:
            self.log_hint.setStringValue_(f"Could not open the folder: {e}")

    def forceTranscribe_(self, sender):
        self.log_hint.setStringValue_("Transcribing…")

        def work():
            # Always report back. Without the guard, a throw before
            # force_transcribe's own try left the hint reading "Transcribing…"
            # forever, with no way to retry that gave any feedback.
            try:
                msg = self.dictate.force_transcribe()
            except Exception as e:
                msg = f"Force transcribe failed: {e}"

            def show():
                self.log_hint.setStringValue_(msg)
                self._refilter()

            _on_main(show)

        threading.Thread(target=work, daemon=True).start()

    def restartApp_(self, sender):
        """Restart, and say so if it didn't happen. Started outside launchd,
        launchctl exits non-zero and this button was a silent no-op."""
        self.log_hint.setStringValue_("Restarting…")

        def work():
            try:
                ok = self.dictate.restart_app()
            except Exception as e:
                ok, err = False, str(e)
            else:
                err = "launchctl refused it — not running under the LaunchAgent?"
            if not ok:
                _on_main(lambda: self.log_hint.setStringValue_(
                    f"Restart failed: {err} Quit from the menu bar and relaunch."))

        threading.Thread(target=work, daemon=True).start()

    # ── Public ───────────────────────────────────────────────────────

    def show(self):
        self._refilter()
        self.refreshLog_(None)
        self.window.makeKeyAndOrderFront_(None)
        AppKit.NSApp.activateIgnoringOtherApps_(True)


def open_settings(dictate_module):
    """Open (or focus) the settings window. Called from the menu bar."""
    global _controller

    def _do():
        global _controller
        if _controller is None:
            _controller = SettingsController.alloc().initWithModule_(dictate_module)
        _controller.show()

    _on_main(_do)


def refresh_if_open():
    """Reload the history table after a new dictation, if the window exists."""
    if _controller is not None:
        _on_main(lambda: _controller._refilter())
