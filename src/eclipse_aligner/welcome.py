#!/usr/bin/env python3
"""The window that opens when there is no clip to open yet.

An icon in the Dock is launched with nothing to go on, and a bare file chooser
is a poor answer to that: it says nothing about what the program wants, and it
forgets every clip the moment it closes. This says what a clip is -- a folder
of frames -- takes one dropped on it, and remembers the last few.

It is deliberately the only thing on screen. The editor needs a clip before it
can draw anything at all, so there is no half-open state worth showing.
"""

import json
import os

from . import alignment
from . import image_io as io

RECENT = 8


def _store():
    """Where the recent list lives, per platform, via Qt's own answer."""
    from .editor import _widgets
    QtCore, _, _ = _widgets()
    d = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.AppDataLocation)
    return os.path.join(d, "recent.json")


def recent():
    """The clips opened before, newest first, minus the ones now gone."""
    try:
        with open(_store()) as fh:
            paths = json.load(fh)
    except (OSError, ValueError):
        return []
    return [p for p in paths if os.path.isdir(p)][:RECENT]


def remember(path):
    """Put a clip at the top of the recent list."""
    path = os.path.abspath(path)
    paths = [p for p in recent() if p != path]
    paths.insert(0, path)
    p = _store()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            json.dump(paths[:RECENT], fh, indent=1)
    except OSError:
        pass          # a list of shortcuts is not worth failing a launch over


def forget(path):
    """Take a clip off the recent list, and touch nothing else.

    The list is a convenience, so removing from it is a convenience too: no
    dialog, no confirmation, and nothing on disk -- not a frame, not the
    alignment document, not the journal. The row goes and that is all.

    Read first, then open for writing. `open(store, "w")` truncates the file
    the moment it is evaluated, so doing both in one statement would have
    `recent()` read back an empty list and removing one row would remove them
    all. Its sibling learnt that the hard way; here the two are simply kept
    apart, as they already are in `remember`.
    """
    path = os.path.abspath(path)
    keep = [q for q in recent() if q != path]
    try:
        with open(_store(), "w") as fh:
            json.dump(keep, fh, indent=1)
    except OSError:
        pass          # a list of shortcuts is not worth failing over


def _count(path):
    """What to say about a clip without opening it.

    The document if there is one, since that is the useful number -- how much
    of it is already measured -- and a plain frame count if there is not.
    """
    doc = alignment.document_path(path)
    if os.path.isfile(doc):
        try:
            d = alignment.load(doc)
            n = len(d["frames"])
            done = sum(1 for f in d["frames"]
                       if f["source"] != alignment.NONE)
            return "%d frames · %d measured" % (n, done)
        except (ValueError, KeyError, OSError):
            pass
    try:
        n = len(io.frames(path))
    except (ValueError, OSError):
        return "not a clip"
    return "%d frames · not measured yet" % n if n else "no frames here"


def ask():
    """Show the window and return the clip chosen, or None if it was closed."""
    from .editor import (_widgets, _dark, _app, app_icon, _identity,
                         signature, sized, place_centred, keep_size)
    from . import __version__
    QtCore, QtGui, QtWidgets = _widgets()
    app = _app()
    _dark(app)
    app.setWindowIcon(app_icon())

    chosen = []

    class Drop(QtWidgets.QFrame):
        """The target. It lights up under a folder and says why it will not."""

        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setObjectName("drop")
            self.setMinimumHeight(150)

        def _folder(self, e):
            urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
            for u in urls:
                p = u.toLocalFile()
                if os.path.isdir(p) or p.endswith(".json"):
                    return p
            return None

        def dragEnterEvent(self, e):
            if self._folder(e):
                self.setProperty("over", True)
                self.setStyleSheet("")      # re-polish
                e.acceptProposedAction()

        def dragLeaveEvent(self, e):
            self.setProperty("over", False)
            self.setStyleSheet("")

        def dropEvent(self, e):
            self.setProperty("over", False)
            self.setStyleSheet("")
            p = self._folder(e)
            if p:
                take(p)

    def take(path):
        """Accept a clip, or say what is wrong with it and stay open."""
        path = path[:-len(os.path.basename(path))] \
            if path.endswith(".json") else path
        path = path.rstrip(os.sep) or os.sep
        try:
            frames = io.frames(path)
        except (ValueError, OSError) as exc:
            return say(str(exc))
        if not frames:
            return say("No EXR, DNG, NEF, CR2, CR3 or JPEG frames in %s"
                       % os.path.basename(path))
        chosen.append(path)
        win.close()

    def say(text):
        problem.setText(text)
        problem.show()

    def browse():
        p = QtWidgets.QFileDialog.getExistingDirectory(
            win, "Open a clip — the folder of frames")
        if p:
            take(p)

    win = QtWidgets.QWidget()
    win.setWindowTitle("eclipse-aligner")
    win.setMinimumWidth(560)
    sized(win, "welcome", (640, 620))
    v = QtWidgets.QVBoxLayout(win)
    # The bottom margin is small because the signature sits on it, and a
    # signature belongs in the corner: the editor's is 8 px off the
    # bottom of its own panel, and these two windows are supposed to look
    # like the same program. The air the footer needs above it is its own
    # top margin, below, so it does not push the corner up.
    v.setContentsMargins(30, 26, 30, 10)
    v.setSpacing(16)

    head = QtWidgets.QLabel(
        '<div style="font-size:20px"><b>eclipse-aligner</b>'
        '<span style="color:#5a5a60"> %s</span></div>'
        '<div style="color:#8a8a90; margin-top:4px">'
        'Align a sequence of frames, preserving raw format.'
        '</div>' % __version__)
    head.setTextFormat(QtCore.Qt.RichText)

    # The icon beside the name, which is what an opening screen is for: the
    # first thing it has to say is which program this is, and the icon says it
    # faster than the word does. It is the same art the Dock and the title bar
    # carry, so the window and its icon introduce each other.
    #
    # Asked for in **points**, not pixels. `QIcon.pixmap` already applies the
    # screen's ratio and stamps it on what it returns, so on a Retina display
    # `pixmap(56)` is 112 pixels that draw as 56 points, sharp. Multiplying by
    # the ratio first applies it twice: the pixmap came back twice the size it
    # should be, the label clipped the middle of it, and the icon looked
    # magnified and cropped -- which is what it was.
    ICON = 56
    badge = QtWidgets.QLabel()
    badge.setPixmap(app_icon().pixmap(QtCore.QSize(ICON, ICON)))
    badge.setFixedSize(ICON, ICON)
    top = QtWidgets.QWidget()
    th = QtWidgets.QHBoxLayout(top)
    th.setContentsMargins(0, 0, 0, 0)
    th.setSpacing(16)
    th.addWidget(badge, 0, QtCore.Qt.AlignVCenter)
    th.addWidget(head, 1, QtCore.Qt.AlignVCenter)
    v.addWidget(top)

    drop = Drop()
    dv = QtWidgets.QVBoxLayout(drop)
    dv.setSpacing(12)
    dv.addStretch(1)
    hint = QtWidgets.QLabel("Drop a clip folder here")
    hint.setAlignment(QtCore.Qt.AlignCenter)
    hint.setStyleSheet("color:#c8c8cc; font-size:15px;")
    dv.addWidget(hint)
    sub = QtWidgets.QLabel("a folder of EXR, DNG, NEF, CR2, CR3 or JPEG frames")
    sub.setAlignment(QtCore.Qt.AlignCenter)
    sub.setStyleSheet("color:#5a5a60;")
    dv.addWidget(sub)
    pick = QtWidgets.QPushButton("Choose a folder…")
    pick.clicked.connect(browse)
    row = QtWidgets.QHBoxLayout()
    row.addStretch(1)
    row.addWidget(pick)
    row.addStretch(1)
    dv.addLayout(row)
    dv.addStretch(1)
    v.addWidget(drop)

    problem = QtWidgets.QLabel()
    problem.setWordWrap(True)
    problem.setStyleSheet("color:#ff6b60;")
    problem.hide()
    v.addWidget(problem)

    rows = []

    def _tidy():
        """When the last row goes, the heading goes with it.

        A `RECENT` with nothing under it is a heading that has outlived its
        list.
        """
        if not any(r.isVisible() for r in rows):
            lab.hide()

    seen = recent()
    lab = QtWidgets.QLabel("RECENT")
    if seen:
        lab.setStyleSheet("color:#5a5a60; font-size:9px; letter-spacing:1.4px;")
        v.addWidget(lab)
        for p in seen:
            # Two left-aligned lines rather than a button: a QToolButton
            # centres multi-line text whatever the stylesheet says, and a list
            # of paths that does not share a left edge is a list you have to
            # read one entry at a time.
            row = QtWidgets.QFrame()
            row.setObjectName("recent")
            row.setCursor(QtCore.Qt.PointingHandCursor)
            rh = QtWidgets.QHBoxLayout(row)
            rh.setContentsMargins(10, 5, 10, 6)
            rv = QtWidgets.QVBoxLayout()
            rv.setSpacing(1)
            name = QtWidgets.QLabel(os.path.basename(p))
            name.setStyleSheet("color:#e8e8ea; font-weight:600;")
            where = QtWidgets.QLabel("%s  ·  %s" % (_count(p),
                                                    os.path.dirname(p)))
            where.setStyleSheet("color:#70737c; font-size:11px;")
            rv.addWidget(name)
            rv.addWidget(where)
            rh.addLayout(rv, 1)
            # A button rather than the row's own handler, because a button
            # eats the press: clicking `remove` must not also open the clip
            # it is removing.
            drop = QtWidgets.QPushButton("remove")
            drop.setCursor(QtCore.Qt.PointingHandCursor)
            drop.setToolTip("Take this clip off the list. Nothing on disk is "
                            "touched -- not a frame, not the alignment "
                            "document, not the journal.")
            drop.setStyleSheet(
                "QPushButton{background:transparent; color:#5a5a60; border:0;"
                " padding:2px 6px; font-size:11px;}"
                "QPushButton:hover{color:#ff9f6b;}")
            drop.clicked.connect(
                lambda _c=False, q=p, r=row: (forget(q), r.hide(), _tidy()))
            rh.addWidget(drop, 0, QtCore.Qt.AlignTop)
            # The two lines answer the pointer; the row behind them does too,
            # so the gaps between them are not dead.
            name.mousePressEvent = lambda _e, q=p: take(q)
            where.mousePressEvent = lambda _e, q=p: take(q)
            row.mousePressEvent = lambda _e, q=p: take(q)
            v.addWidget(row)
            rows.append(row)

    # The same line the editor carries in the corner under its grid: what
    # this is, and where it lives. Both windows sign the same way because it
    # is the same claim about the same program, and it is built in one place
    # so they cannot drift apart.
    foot = QtWidgets.QHBoxLayout()
    foot.setContentsMargins(0, 14, 0, 0)
    foot.addStretch(1)
    foot.addWidget(signature(win))
    v.addStretch(1)
    v.addLayout(foot)

    win.setStyleSheet(
        "#drop { border: 1.5px dashed #3a3d45; border-radius: 6px;"
        "        background: #17181c; }"
        "#drop[over=\"true\"] { border-color: #ffd400; background: #1d1e23; }"
        "#recent { border-radius: 4px; }"
        "#recent:hover { background: #22242a; }")
    place_centred(win)
    win.show()
    app.exec()
    # Not in a close handler: this is a plain widget and not a QMainWindow, so
    # there is nothing to override. After `exec` returns the window is down
    # but the object is still referenced here, which is all `keep_size` needs.
    keep_size(win, "welcome")
    return chosen[0] if chosen else None
