#!/usr/bin/env python3
"""A control socket for the editor, so a problem can be reproduced instead of
described.

    eclipse-aligner gui CLIP --serve

Off unless asked for, bound to 127.0.0.1 and to nothing else. It exists to let
somebody helping with a bug drive the same window the reporter is looking at:
press the keys, read the state, take the picture. A screenshot answers in one
round trip what a paragraph cannot -- what the window actually shows.

    curl localhost:8765/state
    curl -X POST localhost:8765/key -d '{"key": "Right", "mods": ["shift"]}'
    curl -X POST localhost:8765/select -d '{"from": 12, "to": 40}'
    curl -X POST localhost:8765/call -d '{"name": "interpolate_moon"}'
    curl -o shot.png localhost:8765/shot

Qt objects belong to the thread that made them, so nothing here touches a
widget. Every request drops a callable on a queue and waits; a timer inside the
GUI thread drains it, runs it there, and hands the answer back. The wait has a
timeout, so a hung window returns an error rather than a hung client.
"""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import alignment

# What `/call` is allowed to reach. A whitelist rather than getattr on anything:
# this is a debugging aid on a socket, and the smallest useful surface is the
# right one.
CALLABLE = (
    "go", "next_untrusted", "cycle_mode", "cycle_ref", "pin_ref",
    "toggle_curve", "toggle_editing", "toggle_skip",
    "interpolate", "interpolate_moon", "measure_moon_rate",
    "apply_track", "extrapolate_moon",
    "copy_from_ref", "size_from_frame", "revert", "save", "nudge",
    "show_frame", "fit_view", "undo", "redo",
)

KEYS = {}          # filled in on first use, name -> Qt key


def _qt():
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


def _key_table():
    """Name to key code: "right", "n", "space", "pagedown".

    Built from `Qt.Key.__members__` and not from `dir(Qt)`. The names are on
    the enum, not loose on the namespace -- scanning the namespace finds none
    of the 471 of them and every key comes back unknown.
    """
    if KEYS:
        return KEYS
    QtCore, _, _ = _qt()
    for name, value in QtCore.Qt.Key.__members__.items():
        KEYS[name[4:].lower()] = value
    return KEYS


class _Bridge:
    """Runs callables on the GUI thread and hands back what they return."""

    def __init__(self, win):
        QtCore, _, _ = _qt()
        self.win = win
        self.jobs = queue.Queue()
        self.timer = QtCore.QTimer(win)
        self.timer.timeout.connect(self._drain)
        self.timer.start(20)

    def _drain(self):
        while True:
            try:
                fn, box, done = self.jobs.get_nowait()
            except queue.Empty:
                return
            try:
                box.append(("ok", fn()))
            except Exception as e:                       # noqa: BLE001
                box.append(("error", "%s: %s" % (type(e).__name__, e)))
            done.set()

    def run(self, fn, timeout=30.0):
        # A modal dialog runs its own loop and the window stops answering.
        # Saying which one is open beats timing out with no explanation --
        # that is the difference between "the app is broken" and "something
        # is waiting for a button".
        _, _, QtWidgets = _qt()
        modal = QtWidgets.QApplication.activeModalWidget()
        if modal is not None:
            return "error", ("a dialog is waiting: %r. Answer it in the "
                             "window, or POST /dismiss"
                             % (modal.windowTitle() or modal.__class__.__name__))
        box, done = [], threading.Event()
        self.jobs.put((fn, box, done))
        if not done.wait(timeout):
            return "error", "timed out waiting for the window"
        return box[0]


def _state(win):
    f = win.frames[win.i]
    rows = sorted(x.row() for x in win.list.selectedIndexes())
    return {
        "frame": win.i + 1, "frames": len(win.frames), "file": f["file"],
        "source": f["source"], "agreement": f["agreement"],
        "skip": bool(f.get("skip")),
        "sun": None if f["cx"] is None else {"cx": f["cx"], "cy": f["cy"]},
        # The stored offset and the position it works out to: a client
        # scripting the editor wants the second, a client checking the
        # document wants the first, and deriving either from the other is
        # arithmetic nobody should have to repeat.
        "moon": f.get("moon"),
        "moon_at": (lambda xy: None if xy is None
                    else {"cx": xy[0], "cy": xy[1]})(alignment.moon_xy(f)),
        "size": {b: alignment.size_of(win.doc, b) for b in ("sun", "moon")},
        "mode": win.mode, "editing": win.editing,
        "ref": win.ref, "ref_pin": win.ref_pin,
        "ref_frame": (None if win.ref_index() is None
                      else win.frames[win.ref_index()]["file"]),
        # Not a setting any more: the arrows move one photosite, which is
        # 2 px on a mosaic and 1 on anything demosaiced.
        "grid_px": win.grid, "dirty": win.dirty,
        "selection": [r + 1 for r in rows],
        "track": win.track,
        "status": win.msg_label.text(),
        "info": win.frame_label.text(),
    }


def _shot(win):
    QtCore, _, _ = _qt()
    pix = win.grab()
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.WriteOnly)
    pix.save(buf, "PNG")
    return bytes(buf.data())


def _press(win, key, mods):
    QtCore, QtGui, QtWidgets = _qt()
    table = _key_table()
    code = table.get(str(key).lower())
    if code is None:
        raise ValueError("unknown key %r" % key)
    flags = QtCore.Qt.NoModifier
    for m in mods or []:
        flags |= {"shift": QtCore.Qt.ShiftModifier,
                  "ctrl": QtCore.Qt.ControlModifier,
                  "alt": QtCore.Qt.AltModifier,
                  "meta": QtCore.Qt.MetaModifier}[m.lower()]
    target = QtWidgets.QApplication.focusWidget() or win
    for kind in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease):
        QtWidgets.QApplication.sendEvent(target, QtGui.QKeyEvent(kind, code, flags))
    return _state(win)


def _select(win, rows):
    win.list.clearSelection()
    for r in rows:
        if 0 <= r - 1 < len(win.frames):
            win.list.item(r - 1).setSelected(True)
    if rows:
        win.show_frame(rows[0] - 1)
    return _state(win)


def _call(win, name, args):
    if name not in CALLABLE:
        raise ValueError("%s is not on the list of things this may call" % name)
    if name == "fit_view":
        win.view.fit()
        return _state(win)
    getattr(win, name)(*(args or []))
    return _state(win)


def serve(win, port=8765):
    """Start the control socket. Returns the port it is listening on."""
    bridge = _Bridge(win)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):        # the terminal is the user's
            pass

        def _send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(
                body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _reply(self, fn, kind="application/json"):
            how, what = bridge.run(fn)
            if how == "error":
                return self._send(500, {"error": what})
            self._send(200, what, kind)

        def do_GET(self):
            if self.path.startswith("/shot"):
                return self._reply(lambda: _shot(bridge.win), "image/png")
            if self.path.startswith("/state"):
                return self._reply(lambda: _state(bridge.win))
            self._send(404, {"error": "try /state, /shot, /key, /select, /call"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError as e:
                return self._send(400, {"error": "bad JSON: %s" % e})
            w = bridge.win
            if self.path.startswith("/key"):
                return self._reply(
                    lambda: _press(w, body.get("key"), body.get("mods")))
            if self.path.startswith("/select"):
                rows = body.get("rows")
                if rows is None and "from" in body:
                    rows = list(range(int(body["from"]), int(body["to"]) + 1))
                return self._reply(lambda: _select(w, rows or []))
            if self.path.startswith("/dismiss"):
                _, _, QtWidgets = _qt()
                m = QtWidgets.QApplication.activeModalWidget()
                if m is None:
                    return self._send(200, {"dismissed": False})
                m.reject()
                return self._send(200, {"dismissed": True,
                                        "was": m.windowTitle()})
            if self.path.startswith("/call"):
                return self._reply(
                    lambda: _call(w, body.get("name"), body.get("args")))
            self._send(404, {"error": "try /key, /select, /call"})

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]
