#!/usr/bin/env python3
"""The editor window: a Qt application, not a plot pretending to be one.

The frame is a `QGraphicsView`, so panning and zooming are the toolkit's own --
smooth, about the cursor, with no rectangle to drag and no redraw to wait for --
and the parts that are lists really are lists, with the scrollbar and the
selection that a list is supposed to have.

What is on screen:

    the frame        red/green against a reference, or difference, or plain
    the crosshair    where the subject is supposed to end up
    the cyan curve   the conic that was fitted, anchored to the recorded centre
    the strip        every frame in the clip, one tick, coloured by `source`
    the list         the frames by name, scrollable and clickable
    the fields       the numbers behind all of it, editable

The curve is a template rather than a readout: it stays on the crosshair while
the disc slides underneath, so seating it on the limb *is* placing the centre.
That is what makes a frame nothing could measure -- a sun deformed on the
horizon, the moment a filter comes off -- a job for a person's thirty seconds
rather than another heuristic.
"""

import collections
import copy
import json
import inspect
import os
import sys
import threading

import numpy as np

from . import alignment
from . import journal
from .detect import _principal
from . import image_io as io

# There is no step to choose any more. The arrows move **one photosite** -- 2
# px on a mosaic, 1 px on anything already demosaiced -- and that is the
# smallest move that costs no interpolation, which makes every other rung on
# the old ladder either the same thing or a multiple of it.
#
# The ladder went because it was a setting with one right answer. Below a
# photosite the shift resamples the frame, and resampling a mosaic costs half
# its sharpness (42 % at half a photosite, measured) for a fraction of a pixel
# nobody can see on a 2300 px disc -- and `apply` rounds it away regardless.
# Above one, `ctrl` still takes ten at a time, which is the only reason the
# coarse rungs existed.

SRC_COLOUR = {alignment.DETECTED: "#5ac8fa", alignment.HELD: "#ffb340",
              alignment.MANUAL: "#ff6ac1", alignment.NONE: "#ff5f56"}
# The sun is yellow and the moon is cold, which needs no explaining. Both are
# drawn over a casing, so neither depends on the background to be seen.
SUN = "#ffd400"
MOON = "#00e5ff"
# The reference frame had cyan, and the moon has taken it. Colours mean one
# thing each or they mean nothing: this one is only ever chrome -- a mark in the
# list, a line on the strip, a name in the status bar -- and never sits on the
# frame beside a circle.
REF = "#a78bfa"
TARGET = "#00ff88"


class Cache:
    """Frames' luma kept in memory so paging is instant.

    Budgeted by **memory, not frame count**: at display resolution a frame is
    1.5 MB from a DNG and 6 MB from an EXR, so a fixed count would either waste
    memory on one or thrash on the other.

    Reading is the whole cost -- measured on this material, 0.05 s for a DNG and
    0.45 s for an EXR, where the compositing takes about 0.01 s -- so the
    neighbours are fetched in a background thread while a person looks at the
    current frame. By the time the next one is asked for it is usually already
    here. Two threads can race to load the same frame; that wastes a read and
    nothing else, which is cheaper than the locking it would take to prevent.
    """

    def __init__(self, paths, step, budget_mb=400, sc=None):
        self.paths, self.step = paths, step
        # Every frame ends up on **one** grid, whatever its format. A mosaic's
        # luma already arrives at half size and an EXR's does not, so a clip
        # holding both would otherwise have two display scales and every
        # drawing -- the curve, the crosshair, the drag -- would need to know
        # which frame it was on. Reading each file at the step that lands it
        # on the coarsest scale present costs nothing and keeps `sc` a single
        # number for the whole window.
        self.sc = sc or step * max(io.luma_scale(p) for p in paths)
        self.budget = budget_mb * 1e6
        self._d = collections.OrderedDict()
        self._lock = threading.Lock()

    def _read(self, i):
        """Normalised, and **not** yet curved.

        The display curve moved out of here the day the view got knobs: a
        frame kept with the curve baked in has to be read again from disk
        every time one of them moves, and reading is the whole cost -- 0.05 s
        for a DNG, 0.45 s for an EXR, against 0.01 s to compose. Kept
        normalised, a knob is a repaint.
        """
        p = self.paths[i]
        a = io.read_luma(p, max(1, int(round(self.sc / io.luma_scale(p)))))
        hi = float(np.percentile(a, 99.9))
        return np.clip(a / (hi if hi > 0 else 1.0), 0, 1)

    def _store(self, i, a):
        with self._lock:
            self._d[i] = a
            self._d.move_to_end(i)
            used = sum(x.nbytes for x in self._d.values())
            while used > self.budget and len(self._d) > 1:
                used -= self._d.popitem(last=False)[1].nbytes

    def __call__(self, i):
        with self._lock:
            if i in self._d:
                self._d.move_to_end(i)
                return self._d[i]
        a = self._read(i)
        self._store(i, a)
        return a

    def prefetch(self, idx):
        """Pull these in behind the back of the interface, in the order given."""
        todo = [i for i in idx if 0 <= i < len(self.paths)]
        if not todo:
            return

        def work():
            for i in todo:
                with self._lock:
                    have = i in self._d
                if have:
                    continue
                try:
                    self._store(i, self._read(i))
                except Exception:
                    # A prefetch is a guess about what will be wanted next. If
                    # it fails -- an unreadable frame, or the interpreter
                    # shutting down underneath it as the window closes -- the
                    # frame is simply read on demand later. Never let a guess
                    # print a traceback, let alone take anything down.
                    return

        threading.Thread(target=work, daemon=True).start()


def _base(y, radius, eps=2e-3):
    """The picture without its fine structure, with the edges left standing.

    A guided filter, self-guided: four box filters and some arithmetic, which
    on a display-sized frame is milliseconds. A plain Gaussian would be
    simpler and wrong: blurring *across* the lunar limb puts a dark ring
    around the moon when the detail is added back, and a ring around the moon
    is an artefact this window exists to hunt, not to draw. `eps` is what
    counts as an edge.

    Ported from hdrmerge-timelapser, which reached it the same way.
    """
    from scipy.ndimage import uniform_filter
    r = max(3, int(radius))
    mean = uniform_filter(y, size=r, mode="nearest")
    mean_sq = uniform_filter(y * y, size=r, mode="nearest")
    var = np.clip(mean_sq - mean * mean, 0, None)
    a = var / (var + eps)
    b = mean - a * mean
    return uniform_filter(a, size=r, mode="nearest") * y + \
        uniform_filter(b, size=r, mode="nearest")


def tone(y, ev=0.0, detail=0.0):
    """A normalised frame, as a screen should show it.

    Two knobs, and both are **about the screen and not about the data**:
    nothing here reaches the document, the measurement or the output.

    **Exposure** is a gain before the curve, for a corona that is four stops
    under the limb.

    **Local contrast** splits the picture into a *base* -- the slow falloff
    from the limb outwards, which is most of the range and none of the
    information -- and the *detail* left over, the streamers and prominences
    and the grain. It then flattens the base and amplifies the detail. One
    knob moves both, because half of the gesture is useless on its own:
    measured by hdrmerge-timelapser on a totality merge, over the outer
    corona, in display levels out of 255 --

        flattening the base alone   26 -> 40 levels, local contrast unchanged
        raising the detail alone    26 levels unchanged, contrast 0.71 -> 2.23
        the two together            40 levels and 2.02

    -- so alone, one gives a grey picture with no structure and the other
    gives structure too dark to read. The base is flattened a little more
    than the detail is raised, which keeps the limb from going flat while the
    corona comes up.

    The split is done **after** the display curve, on what the eye is going to
    see, and the clip to white is done **after that** -- which is the order
    hdrmerge-timelapser uses and the one thing worth being careful about.
    Clipping first, as this did for an afternoon, leaves the base flat and the
    detail exactly zero everywhere the frame is blown, so local contrast had
    nothing to lift in the very region you raised the exposure to see. Gamma
    of a number above one is still a number above one; it comes back inside
    the range at the end, and the split gets to see the structure on the way
    through.

    At `ev=0, detail=0` this is exactly the curve the editor always had, to
    the last bit, so the plain view is not quietly a new picture.
    """
    if ev:
        y = y * (2.0 ** ev)
    y = np.clip(y, 0, None) ** (1 / 2.2)
    if detail:
        # A fifth of the short side. hdrmerge-timelapser uses 48 px on a
        # 1.5 MP preview and this lands on 50 for the same frame, but ours is
        # whatever the display step leaves -- a fixed 48 would be a fifth of
        # one frame and a twentieth of another.
        r = max(3, int(min(y.shape[:2]) / 20))
        base = _base(y, r)
        fine = y - base
        anchor = float(np.median(base))
        compress = 0.85 * detail
        y = (anchor + (base - anchor) * (1.0 - 0.85 * compress)
             + fine * (1.0 + 4.0 * detail))
    return np.clip(y, 0, 1)


def edge_map(a, sigma=1.2):
    """Gradient magnitude, scaled to itself: a limb becomes a thin line.

    What makes it worth having is that it does not care how bright the subject
    is. Near totality the disc is clipped flat -- its 99.9th percentile sits at
    the top of the scale -- so an overlay of two frames is two white masses
    with a sliver showing at one edge. Their gradients are two thin lines, and
    two lines either sit on each other or they do not; the gap between them is
    the error, in pixels, readable.
    """
    from scipy import ndimage
    # Its own curve, since the cache hands over linear values now: the
    # gradient of a linear frame is the bright disc and nothing else.
    g = ndimage.gaussian_filter(
        np.clip(a, 0, 1).astype(np.float32) ** (1 / 2.2), sigma)
    gy, gx = np.gradient(g)
    m = np.hypot(gx, gy)
    # Only the strong edges. Scaling by the top of the range alone drew every
    # gradient there was, and in a short exposure of a corona the grain is a
    # gradient from corner to corner: the overlay filled with speckle and hid
    # the one line that matters. The floor is the 96th percentile, so what is
    # left is the brightest 4 % of the edges -- the limb, and little else.
    # hdrmerge-timelapser arrived at the same two numbers on the same sky.
    floor, top = (float(x) for x in np.percentile(m, (96.0, 99.8)))
    return np.clip((m - floor) / max(top - floor, 1e-6), 0, 1) ** 0.8


def _compose(cur, other, mode):
    """The two frames as one RGB image: red is the frame being edited.

    A colour fringe is read far faster than a difference image -- aligned, the
    two fuse to grey; out by a pixel, the limb edges red on one side and cyan
    on the other. Under `edges` the same thing happens to two thin lines
    instead of two masses, which is easier to judge and says how far apart.
    """
    if other is None or mode == "plain":
        rgb = np.dstack([cur, cur, cur])
    else:
        h = min(cur.shape[0], other.shape[0])
        w = min(cur.shape[1], other.shape[1])
        cur, other = cur[:h, :w], other[:h, :w]
        if mode == "diff":
            d = np.abs(cur - other)
            d = d / (d.max() or 1.0)
            rgb = np.dstack([d, d, d])
        else:
            rgb = np.dstack([cur, other, other])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _conic_points(shape, n=361):
    """The fitted conic as offsets from its centre, in full-frame pixels.

    Traced parametrically from the stored geometry with the same rotation the
    fit used, so the drawing cannot drift from the numbers by a sign.
    """
    a = shape.get("a", shape.get("r"))
    b = shape.get("b", shape.get("r"))
    ang = np.radians(shape.get("angle", 0.0))
    t = np.linspace(0, 2 * np.pi, n)
    dx = a * np.cos(t) * np.cos(ang) - b * np.sin(t) * np.sin(ang)
    dy = a * np.cos(t) * np.sin(ang) + b * np.sin(t) * np.cos(ang)
    return dx, dy


def _widgets():
    """Qt, imported here so that `detect` and `apply` never load it."""
    from PySide6 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


# Tacked onto the homepage link so the site can tell the visit came from here,
# and which version. Standard UTM keys, which is what analytics already reads.
_CAME_FROM = "utm_source=eclipse-aligner&utm_medium=app&utm_content=%s"


APP_NAME = "eclipse-aligner"


def _app():
    """The QApplication, named **before** it exists.

    macOS builds the application menu when the platform integration comes up,
    and that happens inside QApplication's own constructor. A name set
    afterwards is too late: the menu keeps whatever Qt derived from `argv[0]`,
    which under the launcher -- `python -m eclipse_aligner` -- is `python3.13`.
    So the name is set on QCoreApplication first and passed as `argv[0]` as
    well, and the menu says what the program is called.

    The `.app` was never affected: a bundle's `CFBundleName` wins over all of
    this. Running from a clone is where it showed.

    Qt also derives where per-user files live from the name, so without it the
    recent list landed in a folder called "PySideApp". The organisation is
    deliberately left unset -- setting it would move that folder.
    """
    QtCore, QtGui, QtWidgets = _widgets()
    app = QtWidgets.QApplication.instance()
    if app is None:
        QtCore.QCoreApplication.setApplicationName(APP_NAME)
        app = QtWidgets.QApplication([APP_NAME])
    return app


def _dark(app):
    """A dark window, because the subject is a bright disc on a black sky."""
    app.setApplicationName(APP_NAME)
    QtCore, QtGui, QtWidgets = _widgets()
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    bg, mid, fg = QtGui.QColor("#1c1c1e"), QtGui.QColor("#2c2c2e"), QtGui.QColor("#e8e8ea")
    p.setColor(QtGui.QPalette.Window, bg)
    p.setColor(QtGui.QPalette.Base, QtGui.QColor("#141416"))
    p.setColor(QtGui.QPalette.AlternateBase, mid)
    p.setColor(QtGui.QPalette.Button, mid)
    p.setColor(QtGui.QPalette.Text, fg)
    p.setColor(QtGui.QPalette.WindowText, fg)
    p.setColor(QtGui.QPalette.ButtonText, fg)
    p.setColor(QtGui.QPalette.ToolTipBase, mid)
    p.setColor(QtGui.QPalette.ToolTipText, fg)
    p.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#0a84ff"))
    p.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(p)


def _make_view():
    """The frame, as a graphics view: the toolkit does the panning and zooming."""
    QtCore, QtGui, QtWidgets = _widgets()

    class FrameView(QtWidgets.QGraphicsView):
        # Scene-space delta of a shift-drag, and the two ends of one. Split in
        # three so the window can take a single undo step and write a single
        # journal line for a gesture that arrives as a hundred small moves.
        dragged = QtCore.Signal(float, float)
        drag_began = QtCore.Signal()
        drag_ended = QtCore.Signal()

        def __init__(self):
            scene = QtWidgets.QGraphicsScene()
            super().__init__(scene)
            self._scene = scene       # the view does not own it; hold a ref
            self.setRenderHint(QtGui.QPainter.Antialiasing)
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.setTransformationAnchor(
                QtWidgets.QGraphicsView.AnchorUnderMouse)
            self.setBackgroundBrush(QtGui.QColor("#0a0a0b"))
            self.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)
            self._buf = None
            self._from = None      # where a shift-drag last was
            self.pix = self.scene().addPixmap(QtGui.QPixmap())
            pen = QtGui.QPen(QtGui.QColor(TARGET), 0)
            self.hline = self.scene().addLine(0, 0, 0, 0, pen)
            self.vline = self.scene().addLine(0, 0, 0, 0, pen)
            # Each curve is drawn twice: a dark casing first, the colour on
            # top. No single colour reads on both a saturated white disc and a
            # black sky, and both are in the same frame -- amber vanished over
            # the disc. The map-maker's answer is not a better colour but an
            # outline under it.
            self.curve_bg = self.scene().addPath(QtGui.QPainterPath(),
                                                 self._casing())
            self.curve = self.scene().addPath(QtGui.QPainterPath(),
                                              self._stroke(SUN))
            self.moon_bg = self.scene().addPath(QtGui.QPainterPath(),
                                                self._casing())
            self.moon = self.scene().addPath(QtGui.QPainterPath(),
                                             self._stroke(MOON))

        def set_image(self, rgb):
            h, w, _ = rgb.shape
            self._buf = np.ascontiguousarray(rgb)   # QImage does not copy
            img = QtGui.QImage(self._buf.data, w, h, 3 * w,
                               QtGui.QImage.Format_RGB888)
            self.pix.setPixmap(QtGui.QPixmap.fromImage(img))
            self.scene().setSceneRect(0, 0, w, h)
            self.hline.setLine(0, 0, w, 0)
            self.vline.setLine(0, 0, 0, h)

        def set_target(self, x, y):
            r = self.scene().sceneRect()
            self.hline.setLine(0, y, r.width(), y)
            self.vline.setLine(x, 0, x, r.height())

        @staticmethod
        def _casing():
            p = QtGui.QPen(QtGui.QColor(0, 0, 0, 190), 3.4)
            p.setCosmetic(True)          # constant on screen, whatever the zoom
            return p

        @staticmethod
        def _stroke(colour, measured=True):
            c = QtGui.QColor(colour)
            c.setAlpha(255 if measured else 130)
            p = QtGui.QPen(c, 1.6)
            p.setCosmetic(True)
            p.setStyle(QtCore.Qt.DashLine if measured else QtCore.Qt.DotLine)
            return p

        def _path(self, xs, ys):
            path = QtGui.QPainterPath()
            if xs is not None:
                path.moveTo(float(xs[0]), float(ys[0]))
                for x, y in zip(xs[1:], ys[1:]):
                    path.lineTo(float(x), float(y))
            return path

        def set_curve(self, xs, ys):
            path = self._path(xs, ys)
            self.curve_bg.setPath(path)
            self.curve.setPath(path)

        def set_moon(self, xs, ys, measured=True):
            """Faint and dotted when nobody measured it, so it cannot pass."""
            self.moon.setPen(self._stroke(MOON, measured))
            path = self._path(xs, ys)
            casing = self._casing()
            casing.setColor(QtGui.QColor(0, 0, 0, 190 if measured else 120))
            self.moon_bg.setPen(casing)
            self.moon_bg.setPath(path)
            self.moon.setPath(path)

        def fit(self):
            self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

        def wheelEvent(self, e):
            s = 1.15 ** (e.angleDelta().y() / 120.0)
            self.scale(s, s)

        # ------------------------------------------------- shift and drag
        # The arrows with a mouse. A quarter of a pixel is a key press, but
        # the first placing of a circle on a frame the detector lost is
        # centimetres of hand movement, and doing that in steps is the one
        # part of this editor that felt like work.
        #
        # Shift is what tells it apart from panning, which is the plain drag
        # and stays exactly where it was. Held, the hand grabs the subject
        # instead of the canvas.

        def mousePressEvent(self, e):
            if (e.button() == QtCore.Qt.LeftButton
                    and e.modifiers() & QtCore.Qt.ShiftModifier):
                self._from = self.mapToScene(e.position().toPoint())
                # The view must not also pan, and NoDrag is the only way to
                # tell it so: ScrollHandDrag eats the move events.
                self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
                self.setCursor(QtCore.Qt.SizeAllCursor)
                self.drag_began.emit()
                e.accept()
                return
            super().mousePressEvent(e)

        def mouseMoveEvent(self, e):
            if getattr(self, "_from", None) is not None:
                at = self.mapToScene(e.position().toPoint())
                d = at - self._from
                # Measured from the last position rather than the first, so
                # the frame that has already moved does not drag the anchor
                # along with it and double every step.
                self._from = at
                if d.x() or d.y():
                    self.dragged.emit(d.x(), d.y())
                e.accept()
                return
            super().mouseMoveEvent(e)

        def mouseReleaseEvent(self, e):
            if getattr(self, "_from", None) is not None:
                self._from = None
                self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
                self.unsetCursor()
                self.drag_ended.emit()
                e.accept()
                return
            super().mouseReleaseEvent(e)

        def keyPressEvent(self, e):
            """Let the navigation keys through to the window.

            A graphics view scrolls itself with the arrows, and having the
            focus it would swallow them -- so zoomed in, the one moment the
            arrows matter most, they would pan instead of nudging the frame.
            Ignoring them here sends them up to the window, which nudges.
            Panning is dragging; the arrows belong to the subject.
            """
            if e.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                           QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                           QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown,
                           QtCore.Qt.Key_Home, QtCore.Qt.Key_End,
                           QtCore.Qt.Key_Space):
                e.ignore()
                return
            super().keyPressEvent(e)

        def keyReleaseEvent(self, e):
            if e.key() == QtCore.Qt.Key_Space:
                e.ignore()          # the window wants the release, not just the press
                return
            super().keyReleaseEvent(e)

    return FrameView


def _make_strip():
    """Every frame as one tick. A list cannot do this at 500 frames; a row can."""
    QtCore, QtGui, QtWidgets = _widgets()

    class Strip(QtWidgets.QWidget):
        picked = QtCore.Signal(int)

        def __init__(self, frames):
            super().__init__()
            self.frames = frames
            self.cur = 0
            self.ref = None
            self.setFixedHeight(18)
            self.setFocusPolicy(QtCore.Qt.NoFocus)
            self.setToolTip("every frame in the clip; click to go there")

        def paintEvent(self, _):
            p = QtGui.QPainter(self)
            w, h, n = self.width(), self.height(), len(self.frames)
            if not n:
                return
            for k, f in enumerate(self.frames):
                x0, x1 = k * w / n, (k + 1) * w / n
                p.fillRect(QtCore.QRectF(x0, 0, max(1.0, x1 - x0), h),
                           QtGui.QColor("#3a3a3f" if f.get("skip")
                                        else SRC_COLOUR.get(f["source"], "#888")))
            if self.ref is not None and self.ref != self.cur:
                p.setPen(QtGui.QPen(QtGui.QColor(REF), 2))
                x = (self.ref + 0.5) * w / n
                p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            x = (self.cur + 0.5) * w / n
            p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))

        def mousePressEvent(self, e):
            n = len(self.frames)
            self.picked.emit(max(0, min(n - 1, int(e.position().x() / self.width() * n))))

    return Strip


class _Cancelled(Exception):
    """Raised out of the progress hook to stop a run in flight."""


def _estimated_size(doc, kept_n, bits=16):
    """Bytes the output will take, or 0 when it cannot be known.

    Worth saying out loud before starting: this writer stores a DNG mosaic
    uncompressed, so the result is two to three times the original and a long
    clip is tens of gigabytes.
    """
    w, h = doc["frame_size"]
    # Per frame, since a clip can hold both: a mosaic is one value a
    # photosite, an EXR three channels at `bits`. Counting the whole clip as
    # whichever format came first was out by a factor of six on the twenty
    # frames that were the other one.
    per = {"dng": w * h * 2, "raw": w * h * 2,
           "exr": w * h * 3 * (bits // 8)}
    keep = set(alignment.kept(doc))
    total = 0
    for n, f in enumerate(doc["frames"]):
        if n not in keep:
            continue
        total += per.get(io.kind(f["file"]) or "", 0)
    return total


def _make_list():
    """The list of frames, with the arrows it deserves once it can hold focus.

    Its own key handling, rather than the one a list comes with, for two
    reasons found by trying it: moving with the arrows natively drags the
    *selection* along, which would destroy a range prepared for interpolating;
    and left and right would fall through to the window and nudge the subject,
    so the same two keys would edit while the other two navigated.

    With the focus here the arrows belong to the list. Nothing else changes:
    the letter shortcuts are the window's and keep working, as does escape,
    which hands the focus back to the frame.
    """
    QtCore, QtGui, QtWidgets = _widgets()

    class FrameList(QtWidgets.QListWidget):
        stepped = QtCore.Signal(int)

        def keyPressEvent(self, e):
            """The arrows walk the film; with shift they pick a run.

            Up and Down are taken over here so they step over the frames
            marked skip, the way `n` and `p` do -- the list's own arrows would
            stop on rows that are not in the output. But holding shift means
            "extend the selection", which is the one thing this list should
            never reinvent: Qt's own handling moves the current row and drags
            the anchor behind it, and `show_frame` already sets the current
            row with NoUpdate so navigating cannot wipe a selection.

            Without this, shift and the arrows did exactly what the arrows do
            -- the modifier was read nowhere -- and the only way to pick a run
            from the keyboard was not to.
            """
            k = e.key()
            if e.modifiers() & QtCore.Qt.ShiftModifier and k in (
                    QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                    QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown,
                    QtCore.Qt.Key_Home, QtCore.Qt.Key_End):
                super().keyPressEvent(e)
                self.stepped.emit(0)      # say where we are, move nothing
                return
            if k in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                self.stepped.emit(-1 if k == QtCore.Qt.Key_Up else 1)
                e.accept()
                return
            if k in (QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown):
                self.stepped.emit(-20 if k == QtCore.Qt.Key_PageUp else 20)
                e.accept()
                return
            if k in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                self.stepped.emit(0)   # nowhere to go, but say where we are
                e.accept()
                return
            super().keyPressEvent(e)

        def keyboardSearch(self, _):
            """No type-ahead: every letter here is a command elsewhere."""

    return FrameList


def app_icon():
    """The window's icon, every size drawn for its size.

    Handing Qt all of them rather than one to scale is the point of drawing
    them separately: it picks the one made for the slot it is filling.

    On macOS the *Dock* icon comes from an .app bundle, not from here, so
    running as a script it stays Python's. This governs the window, the
    dialogs, and everything outside macOS.
    """
    QtCore, QtGui, QtWidgets = _widgets()
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
    ic = QtGui.QIcon()
    for s in (16, 32, 64, 128, 256, 512):
        p = os.path.join(here, "icon-%d.png" % s)
        if os.path.exists(p):
            ic.addFile(p, QtCore.QSize(s, s))
    return ic


def _identity():
    from .cli import identity
    return identity()


# --------------------------------------------------- where windows go and how
# big --------------------------------------------------------------------
#
# The window store is a **convenience**, and it is kept well away from the
# alignment document: nothing here changes a measurement or a pixel, and a
# missing or corrupt file must cost nothing. Every read falls back to the
# built-in size and every failed write is dropped, because a program that
# cannot start because it could not remember how big it was is worse than one
# that opens the wrong size.
#
# It lives beside `recent.json`, in the platform's own place for such things,
# and it is keyed by **window** -- `editor`, `welcome` -- never by folder. How
# big you like the editor is a fact about you, not about a clip.

def _sizes_store():
    QtCore, _, _ = _widgets()
    d = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.AppDataLocation)
    return os.path.join(d, "windows.json")


def _sizes():
    try:
        with open(_sizes_store()) as fh:
            got = json.load(fh)
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def sized(win, name, default):
    """Open a window at whatever size it was left at, if that still fits.

    Only the size is kept, **never the position**: a remembered position is a
    window that opens off the edge of a screen that is no longer plugged in,
    and where a window goes is answered better by `place_centred`. The size is
    clamped to the screen it is about to open on, so one sized on a large
    display and reopened on a laptop is merely large rather than unreachable.
    """
    QtCore, QtGui, _ = _widgets()
    got = _sizes().get(name)
    w, h = (got if isinstance(got, list) and len(got) == 2 else default)
    screen = (QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
              or QtGui.QGuiApplication.primaryScreen())
    if screen is not None:
        r = screen.availableGeometry()
        w, h = min(int(w), r.width()), min(int(h), r.height())
    try:
        win.resize(int(w), int(h))
    except (TypeError, ValueError):
        win.resize(*default)


def keep_size(win, name):
    """Write down the size a window was left at. Called on the way out.

    On the way out and not from a resize handler: the size that matters is the
    one it was left at, and watching resizes would write the file on every
    pixel of a drag.
    """
    if win is None or win.width() < 200 or win.height() < 150:
        return                       # minimised, or already torn down
    sizes = _sizes()
    sizes[name] = [win.width(), win.height()]
    try:
        os.makedirs(os.path.dirname(_sizes_store()), exist_ok=True)
        with open(_sizes_store(), "w") as fh:
            json.dump(sizes, fh, indent=1)
    except OSError:
        pass


# What a title bar adds to a window, learnt from the first one shown. Qt only
# knows it once the native window exists, and placing before showing is the
# only way to place without a visible jump -- so the first window pays for the
# measurement and every one after it is put in the right place first time.
_FRAME = [0, 0]


def place_centred(win, on=None):
    """Put a window in the middle of the screen it is going to appear on.

    Every window in the program, and the small ones most of all: Qt puts a
    dialog over its parent, which on a wide screen with the editor at one end
    leaves the question you have to answer off in a corner. Centred, what is
    being asked is where the eye already is.

    The screen is the parent's when there is one, and otherwise the one holding
    the pointer -- with two displays that is the one being worked on, which the
    primary screen need not be.

    **It places twice, and that is not belt and braces.** Before a window is
    shown Qt does not know the height of its title bar, and a dialog with a
    parent has not yet been moved over that parent by Qt itself. Measured here
    on a 1728x994 desktop, placing only before `show()` left a plain widget
    14 px low, a parentless dialog 14 px low and a dialog with a parent 42 px
    low. Correcting after `show()`, when the frame is real, lands all three on
    (+0, +0). So: place from what is known, then correct on the next turn of
    the event loop -- which for a modal dialog is the first turn of its own.

    Not `centre`, which in the editor already means the centre of a disc.
    """
    QtCore, QtGui, _ = _widgets()

    screen = None
    if on is not None:
        handle = on.window().windowHandle()
        screen = handle.screen() if handle is not None else None
    if screen is None:
        screen = (QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
                  or QtGui.QGuiApplication.primaryScreen())
    if screen is None:
        return

    def place(learn=False):
        try:
            r = screen.availableGeometry()
            if win.isVisible():
                g = win.frameGeometry()
                if learn and g.height() > win.height():
                    _FRAME[:] = [g.width() - win.width(),
                                 g.height() - win.height()]
                w, h = g.width(), g.height()
            else:
                # A window nobody has resized still measures Qt's default
                # 640x480 -- and `adjustSize` would throw away a size that
                # *was* set, so it is only for the untouched case. One that
                # was resized still grows at show time to whatever its layout
                # needs, which is why the size to place by is the one expanded
                # to the layout's minimum.
                if not win.testAttribute(QtCore.Qt.WA_Resized):
                    win.adjustSize()
                # `width()` and `height()` rather than `size()`: the editor
                # has a `size()` of its own -- the clip's radius -- and it
                # shadows the widget's. The same collision as `centre`, in a
                # second name, and it fails as an AttributeError on a dict
                # rather than as anything that reads like a layout problem.
                size = QtCore.QSize(win.width(), win.height()).expandedTo(
                    win.minimumSizeHint())
                w = size.width() + _FRAME[0]
                h = size.height() + _FRAME[1]
            win.move(r.x() + (r.width() - w) // 2,
                     r.y() + (r.height() - h) // 2)
        except RuntimeError:
            pass                       # the window went away before its turn

    place()
    QtCore.QTimer.singleShot(0, lambda: place(learn=True))


def say_box(parent, title, text, buttons=None, default=None, icon=None):
    """A message box, centred, in place of the static conveniences.

    `QMessageBox.information` and its siblings build the box themselves and
    show it in one call, which leaves nowhere to put it. Built here instead,
    with the same arguments in the same order, and the answer returned the
    same way.
    """
    QtCore, QtGui, QtWidgets = _widgets()
    b = QtWidgets.QMessageBox(parent)
    b.setWindowTitle(title)
    b.setText(text)
    if icon is not None:
        b.setIcon(icon)
    b.setStandardButtons(buttons if buttons is not None
                         else QtWidgets.QMessageBox.Ok)
    if default is not None:
        b.setDefaultButton(default)
    place_centred(b, parent)
    return b.exec()


def about_box(parent=None):
    """What this is, who wrote it, and where it lives.

    A free function rather than a method, because the opening window wants it
    too and it has nothing to do with a document.
    """
    from . import __version__
    QtCore, QtGui, QtWidgets = _widgets()
    b = QtWidgets.QMessageBox(parent)
    b.setTextFormat(QtCore.Qt.RichText)
    who, site = _identity()
    b.setText(
        '<div style="font-size:15px"><b>eclipse-aligner</b> %s</div>'
        '<div style="color:#8a8a90; margin-top:6px">Align a sequence '
        'of frames,<br>preserving raw format.</div>'
        '<div style="margin-top:12px">%s<br>'
        '<a style="color:#5ac8fa" href="https://www.%s">www.%s</a></div>'
        % (__version__, who, site, site))
    b.setIconPixmap(app_icon().pixmap(QtCore.QSize(64, 64)))
    b.setStandardButtons(QtWidgets.QMessageBox.Ok)
    place_centred(b, parent)
    b.exec()


def signature(parent=None):
    """The name, the version and the address, as one clickable line.

    Both windows carry it, in the same corner and the same words, because it
    is the same claim about the same program. Built here rather than twice so
    they cannot drift apart -- and so the address only has to be got right
    once.

    Two links in one line: the name opens About, the address opens the
    browser. Handled by hand rather than by `openExternalLinks`, which would
    try to open `#about` as a URL.

    The author is in the About box and not here: three things in a corner
    nobody is reading is two things too many, and the name is the one that
    identifies it.

    The address carries where the click came from. A desktop app opening a
    link sends no Referer header -- there is no page it came from -- so the
    only way the site can tell is in the URL itself. What is shown stays the
    plain address.
    """
    from . import __version__
    QtCore, QtGui, QtWidgets = _widgets()
    site = _identity()[1]
    lab = QtWidgets.QLabel()
    lab.setTextFormat(QtCore.Qt.RichText)
    lab.setCursor(QtCore.Qt.PointingHandCursor)
    lab.setContentsMargins(0, 0, 0, 0)
    lab.setToolTip("what this is, and where it lives")
    lab.linkActivated.connect(
        lambda u: about_box(parent) if u == "#about"
        else QtGui.QDesktopServices.openUrl(QtCore.QUrl(u)))
    # Two links, so two colours: whichever one the pointer is on lights up
    # and the other does not. One placeholder lit both -- the name and the
    # address are separate targets, and the address was answering for the
    # name, which is the sort of thing that makes a person doubt where the
    # click will go.
    # One `%` over the whole thing, and the version passed in rather than
    # spliced with `+`: `%` binds tighter than `+`, so the version's `+` used
    # to cut the string in two and only the tail was formatted. Harmless while
    # the head held no placeholder, and a TypeError the moment it did.
    html = (
        '<a href="#about" style="color:%%s; text-decoration:none">'
        '<b>eclipse-aligner</b>'
        '<span style="color:#4a4c53"> %s</span></a>'
        '<span style="color:#4a4c53"> \u00b7 </span>'
        '<a href="https://www.%s/?%s" '
        'style="color:%%s; text-decoration:none">www.%s</a>'
    ) % (__version__, site, _CAME_FROM % __version__, site)
    DIM, LIT = ("#8a8a90", "#5a5a60"), ("#e8e8ea", "#9ee0ff")

    def paint(hovered=""):
        lab.setText(html % (LIT[0] if hovered == "#about" else DIM[0],
                            LIT[1] if hovered.startswith("http") else DIM[1]))
    paint()
    lab.linkHovered.connect(paint)
    return lab


# The command sheet is written with the macOS glyphs -- the platform the tool
# grew up on -- and Qt maps the actual *bindings* (declared `Ctrl+...`) to those
# glyphs on macOS by itself. But the hand-drawn key labels are strings, and a
# string does not know what platform it is on: on Windows and Linux the command
# glyph would sit there lying. So off macOS command (⌘) becomes Ctrl. The
# shift glyph (⇧) is kept -- an up-arrow reads the same on any keyboard, and
# it is the one Apple symbol that is not Apple's.
_MAC = sys.platform == "darwin"
_MODS = ("⌘", "⇧")   # the modifier glyphs, in the order a run may hold them


def _make_editor():
    QtCore, QtGui, QtWidgets = _widgets()
    FrameView, Strip, FrameList = _make_view(), _make_strip(), _make_list()

    class _Cmd(QtWidgets.QLabel):
        """One command's name, clickable, and dimmed when it cannot be run.

        A label rather than a button: sixteen buttons in a grid is sixteen
        frames and sixteen hover boxes, and what this has to look like is a
        printed list. It still answers the pointer, because a thing that does
        something should say so under the cursor.
        """

        clicked = QtCore.Signal()

        def __init__(self, text, clickable=True):
            super().__init__(text)
            self._on = True
            self._base, self._scope = text, 1
            # A gesture has a row too, and no click: there is no way to press
            # shift+arrow with a mouse, and a label that lights up under the
            # pointer and then does nothing is a worse lie than a plain one.
            self._clickable = clickable
            if clickable:
                self.setCursor(QtCore.Qt.PointingHandCursor)
            self._paint()

        def setScope(self, n):
            """How many frames this row will land on, when it is more than one.

            The grid's columns say what a command **reaches**, and for most
            rows that is fixed. Three of them are not: the arrows, `remeasure`
            and `skip` widen to the selection when the frame on screen is part
            of it. Rather than duplicate them in the `RANGE` column -- one
            command in two places, each half true -- the row says its own
            reach, at the moment it changes.

            It also answers the opposite case without a word: a selection that
            is *not* in force lights nothing up.
            """
            if n == self._scope:
                return
            self._scope = n
            self.setText(self._base if n <= 1
                         else "%s  \u00d7%d" % (self._base, n))

        def setEnabled(self, on):
            self._on = bool(on)
            super().setEnabled(True)      # keep the tooltip working
            self.setCursor(QtCore.Qt.PointingHandCursor
                           if (on and self._clickable)
                           else QtCore.Qt.ArrowCursor)
            self._paint(False)

        def isEnabled(self):
            return self._on

        def _paint(self, over=False):
            self.setStyleSheet("color: %s;" % (
                "#ffffff" if (self._on and over and self._clickable) else
                "#c8c8cc" if self._on else "#4a4c53"))

        def enterEvent(self, e):
            self._paint(True)

        def leaveEvent(self, e):
            self._paint(False)

        def mousePressEvent(self, e):
            if self._on and self._clickable:
                self.clicked.emit()

    class Editor(QtWidgets.QMainWindow):
        # Everything a hand can do that changes what is on screen or in the
        # document. Wrapped in one place rather than logged inside each method:
        # a call that arrives from the control socket goes through the same
        # attribute, so it is recorded too, and a new action is one name here
        # rather than a line nobody remembers to add.
        WATCHED = (
            "go", "next_untrusted", "nudge", "copy_from_ref",
            "toggle_skip", "interpolate", "interpolate_moon",
            "measure_moon_rate", "apply_track", "extrapolate_moon",
            "size_from_frame", "revert", "redetect", "process",
            "undo", "redo",
            "save", "cycle_mode", "cycle_ref", "pin_ref", "toggle_curve",
            "toggle_editing", "_set_peek",
            "about")

        def __init__(self, doc, path, step=4, log=None, retain=None):
            super().__init__()
            self.doc, self.path, self.step = doc, path, step
            self.frames = doc["frames"]
            self.paths = alignment.paths(doc)
            # The coarsest scale present, so one number serves the window.
            self.sc = step * max(io.luma_scale(p) for p in self.paths)
            self.cache = Cache(self.paths, step, sc=self.sc)
            # Plain to start with, not the overlay. Placing a frame by hand
            # is done by looking at the frame and blinking to its neighbour
            # with the space bar, and an overlay in the way makes both halves
            # harder to read: the picture is tinted and the neighbour is
            # already on top of it. `x` still cycles to the overlay, the
            # difference and the edges, which are for judging an alignment
            # rather than making one.
            self.i, self.mode, self.ref = 0, "plain", "prev"
            self.ref_pin = None
            self.show_curve, self.dirty, self._syncing = True, False, False
            self._marked = None
            self._ref_marked = None
            # One photosite, not one pixel. On a mosaic the four sub-planes
            # sit on a grid of spacing 2, so a move of 2 lands on whole
            # sub-plane pixels and carries the same numbers to a new place;
            # anything else blends neighbours. The ladder still has the finer
            # rungs -- the subject does not sit on even pixels and sub-pixel
            # work is the point of this tool -- but the rung you start on is
            # the free one.
            # The coarsest grid present: a step that is whole photosites on
            # a mosaic is also whole pixels on an EXR, so one ladder is safe
            # for every frame of a mixed clip. The other way round it would
            # not be -- a 1 px step is half a photosite.
            self.grid = float(max(io.bayer_step(p) for p in self.paths))

            self.step_measure = 2
            self._shot = {}
            self._edges = {}
            # How the frame is shown, and nothing else: neither of these
            # reaches the document, the measurement or the output.
            self.ev = 0.0
            self.detail = 0.0
            self._peek = False
            self.editing = "align"
            self.track = None
            self._track_end = False
            # Both on to start with, and the dialog remembers what you
            # last chose. They are what a run of this editor almost always
            # wants: the output is a sequence for something else to read, and
            # a gap stops that reader at the first missing number; and a
            # destination holding the previous attempt's frames is the
            # likeliest way to end up with two clips mixed in one folder.
            # Neither can reach an original -- `apply` has no in-place mode --
            # so the cost of having them wrong is a re-run, and the cost of
            # not noticing them is a bad sequence.
            self.renumber = True
            self.clean = True
            from . import __version__ as _ver
            self._depth, self._note = 0, None
            self._past, self._future = [], []
            self.log = journal.Journal(log, {
                "version": _ver, "document": path,
                "frames": len(self.frames), "target": list(doc["target"]),
                "clip": doc.get("input_dir")}, retain) if log else None
            # Wrapped before the widgets are built, not after: a button
            # connected to `self.save` captures the attribute as it stands at
            # connect time, so wrapping afterwards leaves every such button
            # calling straight past the journal.
            self._watch()
            self._build()
            self.show_frame(0, fit=True)
            self.view.setFocus()

        def _watch(self):
            """Route every watched action through the journal.

            The wrapper records the call's arguments *by name* -- `nudge` comes
            out as dx/dy rather than a pair of numbers whose order a reader has
            to look up -- and the frame's geometry afterwards, which is what
            says whether the action did anything.

            Installed whether or not there is a journal, because undo hangs off
            the same hook. Tying it to the log meant `--no-log` quietly took
            undo with it, which is the kind of coupling that only shows up when
            somebody turns the other thing off.
            """
            for name in self.WATCHED:
                fn = getattr(self, name)
                def wrap(fn=fn, name=name):
                    def call(*a, **k):
                        try:
                            b = inspect.signature(fn).bind(*a, **k)
                            b.apply_defaults()
                            args = dict(b.arguments)
                        except TypeError:
                            args = {"args": list(a)}
                        # Remembered before the call and dropped again if
                        # nothing moved. A cancelled dialog would otherwise
                        # leave a step in the history that undoes nothing,
                        # which is worse than no history at all.
                        keep = (self._remember(
                            name.lstrip("_").replace("_", " "))
                            if not self._depth and name in self.UNDOES
                            else None)
                        # Logged after the call, so the geometry recorded is
                        # the one the action left behind -- and with the depth
                        # raised, so the repaint it causes does not write its
                        # own line first and put the effect above the cause.
                        self._depth += 1
                        try:
                            return fn(*a, **k)
                        finally:
                            self._depth -= 1
                            if keep is not None:
                                self._settle(keep)
                            note, self._note = self._note, None
                            try:
                                self._log(name.lstrip("_"), note=note, **args)
                            except Exception as exc:      # never break an edit
                                print("journal: %s" % exc, file=sys.stderr)
                    return call
                setattr(self, name, wrap())

        def _remember(self, label):
            """Put the document on the undo stack. Returns the entry pushed.

            A whole copy rather than an inverse operation per command. Measured
            on a 525-frame document: 1.8 ms and 85 kB a step, 4.3 MB for the
            fifty kept. At that price, hand-written inverses buy nothing and
            cost the one bug an undo must not have -- an inverse that does not
            quite invert.
            """
            entry = (label, copy.deepcopy(self.frames),
                     copy.deepcopy(self.doc.get("size")))
            self._past.append(entry)
            del self._past[:-self.HISTORY]
            self._future.clear()
            return entry

        def _settle(self, entry):
            """Drop a remembered step that turned out to change nothing."""
            if (self._past and self._past[-1] is entry
                    and entry[1] == self.frames
                    and entry[2] == self.doc.get("size")):
                self._past.pop()

        def _travel(self, take, keep, verb, done):
            if not take:
                return self._say("nothing to %s" % verb, 2000)
            label, frames, size = take.pop()
            keep.append((label, copy.deepcopy(self.frames),
                         copy.deepcopy(self.doc.get("size"))))
            self.doc["size"] = size or {}
            self.doc["frames"] = frames
            self._adopt(frames)
            self._say("%s: %s" % (done, label), 3000)

        def undo(self):
            """Take back the last thing that changed the document."""
            self._travel(self._past, self._future, "undo", "undone")

        def redo(self):
            """Put back the last thing undone."""
            self._travel(self._future, self._past, "redo", "redone")

        def _log(self, do, **detail):
            """One event, with where the session was when it happened."""
            if getattr(self, "log", None) is None:
                return
            if not hasattr(self, "list"):
                # A signal firing while the widgets are still being built is
                # the toolbar settling, not something a hand did.
                return
            rows = sorted(x.row() for x in self.list.selectedIndexes())
            f = self.frames[self.i]
            m = f.get("moon")
            sel = None
            if len(rows) > 1:
                # The ends and the count, not the hundred indices in between:
                # a selection in this editor is always a run picked in the list.
                sel = [rows[0], rows[-1], len(rows)]
            # Merged, not passed as two sets of keywords: an action whose
            # own argument shares a name with the context -- `_set_step(px)`
            # was one, before the step became a fact rather than a setting --
            # would collide and raise inside the slot, and the action would
            # silently stop being recorded. Its own names win.
            rec = {"i": self.i, "file": f["file"], "sel": sel,
                   "body": self.editing,
                   "mode": self.mode, "px": self.grid,
                   "dirty": self.dirty or None,
                   "skip": True if f.get("skip") else None,
                   "sun": [f["cx"], f["cy"], f["source"]],
                   "moon": (list(alignment.moon_xy(f)) + [m["source"]]
                            if m and f["cx"] is not None else None),
                   "size": alignment.size_of(self.doc, self._body()) or None}
            rec.update(detail)
            self.log.event(do, **rec)

        # -------------------------------------------------------------- build
        @staticmethod
        def _key_runs(key):
            """Split a key label into (is_modifier, text) runs, translated.

            The sheet is written in macOS glyphs; a run is either a block of
            modifier glyphs and whatever they modify, or a plain stretch left
            alone. On macOS the glyphs are the right answer, so the split is the
            old one: the first glyph is the dim modifier, the rest is the bright
            letter, and `\u21e7\u2318Z` reads `\u21e7` + `\u2318Z` exactly as before.

            Off macOS only command is renamed -- `\u2318`\u2192`Ctrl+` -- and Ctrl comes
            first, the way a Windows keyboard names it: `\u21e7\u2318Z`\u2192`Ctrl+\u21e7Z`. The
            shift glyph stays a glyph and stays pressed against its letter, with
            no `+`, exactly as `\u21e7S` does; a modifier with a space or nothing
            after it gets no dangling `+` either, so `\u21e7 drag`\u2192`\u21e7 drag` and
            `\u2318` alone would be a bare `Ctrl`. The dim/bright split rides along,
            so the prefix is dim and the letter the eye hunts for is bright.
            """
            if _MAC:
                for m in ("\u21e7", "\u2318"):
                    if key.startswith(m):
                        return [(True, m), (False, key[1:])]
                return [(False, key)]
            runs, i, n = [], 0, len(key)
            while i < n:
                if key[i] in _MODS:
                    mods = []
                    while i < n and key[i] in _MODS:
                        mods.append(key[i])
                        i += 1
                    modifies = i < n and key[i] != " "   # a letter follows
                    text = ""
                    if "\u2318" in mods:
                        # Ctrl leads; it takes a `+` only if a shift glyph or a
                        # letter follows for it to bind to.
                        text = "Ctrl+" if ("\u21e7" in mods or modifies) else "Ctrl"
                    if "\u21e7" in mods:
                        text += "\u21e7"          # the glyph is kept, pressed on
                    runs.append((True, text))
                else:
                    j = i
                    while j < n and key[j] not in _MODS:
                        j += 1
                    runs.append((False, key[i:j]))
                    i = j
            return runs

        @staticmethod
        def _key_plain(key):
            """The key label as it is shown -- glyphs on macOS, names off it.

            What the column has to be wide enough for: `Ctrl+D` is wider than
            `\u2318D`, so measuring the raw glyph would clip the translated text.
            """
            return "".join(t for _, t in Editor._key_runs(key))

        @staticmethod
        def _key_html(key):
            """The shown label, coloured: modifiers dim, the letter bright."""
            return "".join(
                '<span style="color:%s">%s</span>'
                % ("#9a7a12" if mod else "#ffd400", t)
                for mod, t in Editor._key_runs(key))

        def _cell(self, grid, r, key, text, slot, tip):
            """One command in the grid: its key, its name, and what it does.

            The key is right-aligned inside its column so the **letter** falls
            on one vertical whatever hangs in front of it -- `p`, `\u21e7R` and
            `\u2318D` all line up -- and the names start on one edge because of it.
            The modifier is dimmer than the letter: it is a prefix, and the
            letter is what the eye is hunting for. `_key_html` does the glyph
            translation, so this reads the same on every platform.
            """
            k = QtWidgets.QLabel(self._key_html(key))
            k.setTextFormat(QtCore.Qt.RichText)
            k.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            d = _Cmd(text, slot is not None)
            if slot is not None:
                d.clicked.connect(slot)
            for w in (k, d):
                w.setToolTip(tip)
            grid.addWidget(k, r, 0)
            grid.addWidget(d, r, 1)
            return d

        def _group(self, into, title, items):
            """A column of the command grid, headed by what its entries reach.

            The heading is not decoration: the four of them are `navigation`,
            `frame`, `range` and `all`, and they widen left to right. An entry
            does not have to say what it acts on because the column above it
            does -- which is how `remeasure` and `radius from this frame` stop
            needing a sentence each.
            """
            box = QtWidgets.QWidget()
            g = QtWidgets.QGridLayout(box)
            g.setContentsMargins(0, 0, 0, 0)
            g.setHorizontalSpacing(9)
            g.setVerticalSpacing(1)
            h = QtWidgets.QLabel(title.upper())
            h.setStyleSheet("color:#5a5a60; font-size:9px;"
                            "letter-spacing:1.4px; padding-bottom:2px;")
            g.addWidget(h, 0, 0, 1, 2)
            out = {}
            for r, (key, text, name, tip) in enumerate(items, start=1):
                slot = getattr(self, name) if isinstance(name, str) else name
                out[text] = self._cell(g, r, key, text, slot, tip)
            g.setRowStretch(len(items) + 1, 1)
            into.addWidget(box)
            return out

        def _commands(self):
            """The command grid, with the credit tucked into the space it leaves.

            It replaces a toolbar of sixteen buttons whose labels ran 289
            characters and did not fit. Grouped into columns they fit, they say
            what each key does instead of only naming it, and the four modes
            that were pretending to be commands have gone up to the read line
            where they belong.

            The credit sits in the gap to the right of the last column rather
            than on a line of its own beneath everything -- the room is already
            there, and a row added under the grid for three words that never
            change is a row spent on the least important thing on screen.
            """
            panel = QtWidgets.QWidget()
            panel.setObjectName("commands")
            ch = QtWidgets.QHBoxLayout(panel)
            ch.setContentsMargins(12, 8, 12, 8)
            ch.setSpacing(30)
            self.cmd = {}
            for title, items in self.COMMANDS:
                self.cmd.update(self._group(ch, title, items))
            ch.addStretch(1)

            # Transient messages in the same gap, on its left. They used to
            # take over the status bar and leave it empty when they expired;
            # here nothing else is competing for the room.
            self.msg_label = QtWidgets.QLabel()
            self.msg_label.setStyleSheet("color: #8a8a90;")
            self.msg_label.setWordWrap(True)
            ch.addWidget(self.msg_label, 0, QtCore.Qt.AlignBottom)
            ch.addSpacing(18)

            # Two links in one line: the name opens About, the address opens
            # the browser. Handled here rather than by openExternalLinks,
            about = QtGui.QAction("About eclipse-aligner", self)
            about.setMenuRole(QtGui.QAction.AboutRole)
            about.triggered.connect(self.about)
            self.addAction(about)
            self._sig = signature(self)
            ch.addWidget(self._sig, 0, QtCore.Qt.AlignBottom)
            return panel

        # What changes the document, and so is worth being able to take back.
        # Everything else the journal watches moves the view or the selection,
        # and there is nothing to undo about having looked somewhere.
        #
        # `redetect` is not here: it replaces the frames from a worker thread
        # long after its call returned, so it remembers for itself, at the
        # moment the results land.
        UNDOES = frozenset((
            "nudge", "copy_from_ref", "toggle_skip", "interpolate",
            "interpolate_moon", "apply_track", "extrapolate_moon",
            "size_from_frame", "revert", "_fields_changed"))
        HISTORY = 50

        # Every command, in four columns that widen left to right:
        # where you are, this frame, this range, the whole clip. The column
        # heading finishes each entry's sentence, so `remeasure` does not have
        # to say "these frames" and `radius from this frame` does not have to
        # say it is the clip's.
        COMMANDS = (
            ("view", (
                ("x", "blend", "cycle_mode",
                 "overlay / difference / edges / plain"),
                ("o", "outline", "toggle_curve",
                 "show or hide the fitted curve"),
                ("f", "fit the frame", lambda s: s.view.fit(),
                 "show the whole frame, undoing any zoom"),
                ("g", "reference", "cycle_ref",
                 "compare against the previous frame, the next, the first, or "
                 "nothing"),
                ("\u21e7G", "pin it", "pin_ref",
                 "hold this frame as the reference while you move around"),
                ("space", "blink, held", lambda s: s._set_peek(not s._peek),
                 "show the reference instead, for as long as the space bar is "
                 "held"))),
            ("navigation", (
                ("p", "previous", lambda s: s.go(-1),
                 "previous frame in the film, stepping over skipped"),
                ("n", "next", lambda s: s.go(1),
                 "next frame in the film, stepping over skipped"),
                ("\u21e7P \u21e7N", "include skipped",
                 lambda s: s.go(1, every=True),
                 "step to the neighbouring frame even when it is marked skip"),
                ("j", "next untrusted", "next_untrusted",
                 "jump to the next held or unmeasured frame"))),
            ("frame", (
                ("m", "move alignment / moon", "toggle_editing",
                 "which of the frame's two numbers the arrows, the drag and "
                 "the fields write: where the picture sits, or where the moon "
                 "sits on it"),
                ("\u2190 \u2191 \u2193 \u2192", "move what m selects", None,
                 "one step; ctrl for ten, alt for one photosite. the step is "
                 "the box at the top right. shift and drag on the frame does "
                 "the same with the mouse"),
                ("c", "alignment from prev", "copy_from_ref",
                 "the previous frame's position only, leaving this frame's "
                 "moon exactly where it is"),
                ("\u21e7C", "alignment from next",
                 lambda s: s.copy_from_ref(forward=True),
                 "the next frame's position only, leaving this frame's moon "
                 "exactly where it is"),
                ("v", "all from prev",
                 lambda s: s.copy_from_ref(whole=True),
                 "the previous frame's whole record: sun, moon, and so the "
                 "same transformation"),
                ("\u21e7V", "all from next",
                 lambda s: s.copy_from_ref(forward=True, whole=True),
                 "the next frame's whole record: sun, moon, and so the same "
                 "transformation"),
                ("r", "remeasure", "revert",
                 "measure the selected frames again, throwing away what is on "
                 "them now"),
                ("\u232b", "skip", "toggle_skip",
                 "leave these frames out of the output; press again to bring "
                 "them back"))),
            ("range", (
                ("i", "interpolate", "interpolate",
                 "walk the alignment in a straight line between the ends of "
                 "the selection; each moon comes along keeping its offset"),
                ("\u21e7I", "interpolate moon", "interpolate_moon",
                 "walk the moon within the pair instead, leaving every sun "
                 "where it is"),
                ("e", "extrapolate moon\u2026", "extrapolate_moon",
                 "carry the measured rate out from one end of the selection"),
                ("\u21e7E", "measure moon rate", "measure_moon_rate",
                 "measure how fast the moon moves, over a stretch that is "
                 "already right"))),
            ("all", (
                ("\u2318Z", "undo", "undo", "take back the last change"),
                ("\u21e7\u2318Z", "redo", "redo", "put back the last undone"),
                ("s", "sun radius from frame",
                 lambda w: w.size_from_frame("sun"),
                 "measure this frame again and make its sun radius the "
                 "clip's"),
                ("\u21e7S", "moon radius from frame",
                 lambda w: w.size_from_frame("moon"),
                 "the same for the moon, which is best measured on a "
                 "different frame: deep across the disc, not barely on it"),
                ("\u2318D", "detect all\u2026", "redetect",
                 "measure every frame again from scratch"),
                ("\u2318R", "align all\u2026", "process",
                 "write the aligned frames, without leaving the editor"),
                ("\u2318S", "save", "save", "write the document"))),
        )

        # The keys are the window's and work wherever the focus is. Kept apart
        # from the grid because they do not match it one for one: `include
        # skipped` is one row standing for two keys, and the four view modes
        # have no row at all -- they are read in the line above the frame.
        KEYS = (
            ("p", lambda s: s.go(-1)), ("n", lambda s: s.go(1)),
            ("Shift+P", lambda s: s.go(-1, every=True)),
            ("Shift+N", lambda s: s.go(1, every=True)),
            ("j", "next_untrusted"),
            ("c", "copy_from_ref"),
            ("Shift+C", lambda s: s.copy_from_ref(forward=True)),
            ("v", lambda s: s.copy_from_ref(whole=True)),
            ("Shift+V", lambda s: s.copy_from_ref(forward=True, whole=True)),
            ("r", "revert"), ("Del", "toggle_skip"),
            ("i", "interpolate"), ("Shift+I", "interpolate_moon"),
            ("e", "extrapolate_moon"), ("Shift+E", "measure_moon_rate"),
            ("s", lambda w: w.size_from_frame("sun")),
            ("Shift+S", lambda w: w.size_from_frame("moon")),
            ("Ctrl+D", "redetect"), ("Ctrl+R", "process"), ("Ctrl+S", "save"),
            ("Ctrl+Z", "undo"), ("Ctrl+Shift+Z", "redo"),
            ("x", "cycle_mode"), ("g", "cycle_ref"), ("Shift+G", "pin_ref"),
            ("o", "toggle_curve"),
            ("f", lambda s: s.view.fit()), ("m", "toggle_editing"),
        )

        def _build(self):
            for key, what in self.KEYS:
                a = QtGui.QAction(key, self)
                a.setShortcut(QtGui.QKeySequence(key))
                a.triggered.connect(
                    (lambda _=False, w=what: w(self)) if callable(what)
                    else getattr(self, what))
                self.addAction(a)


            self.view = FrameView()
            # A floating word on the frame, not in the status bar: the hand is
            # on the arrows and the eye is on the image, and that is where a
            # surprise has to be answered.
            self.notice = QtWidgets.QLabel(self.view)
            self.notice.setStyleSheet(
                "background: rgba(18,18,24,0.92); color: #e8e8ea;"
                "border: 1px solid #0a84ff; border-radius: 7px;"
                "padding: 9px 16px; font-size: 12px;")
            self.notice.setTextFormat(QtCore.Qt.RichText)
            self.notice.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            self.notice.hide()
            self._notice_timer = QtCore.QTimer(self)
            self._notice_timer.setSingleShot(True)
            self._notice_timer.timeout.connect(self.notice.hide)
            self.view.drag_began.connect(self._drag_began)
            self.view.dragged.connect(self._dragged)
            self.view.drag_ended.connect(self._drag_ended)
            self._drag = None

            self.strip = Strip(self.frames)
            self.strip.picked.connect(lambda k: self.show_frame(k))

            self.list = FrameList()
            self.list.setFocusPolicy(QtCore.Qt.ClickFocus)
            self.list.stepped.connect(self._list_stepped)
            self.list.setSelectionMode(
                QtWidgets.QAbstractItemView.ExtendedSelection)
            self.list.setAlternatingRowColors(True)
            self.list.setUniformItemSizes(True)
            f = QtGui.QFont("Menlo")
            f.setStyleHint(QtGui.QFont.Monospace)
            f.setPointSize(11)
            self.list.setFont(f)
            for _ in self.frames:
                self.list.addItem(QtWidgets.QListWidgetItem())
            for k in range(len(self.frames)):
                self._refresh_item(k)
            self.list.currentRowChanged.connect(
                lambda k: None if self._syncing or k < 0 else self.show_frame(k))
            # And separately on the selection, because Qt emits
            # `currentRowChanged` *before* it applies the selection the same
            # click is making. Everything that reads the selection -- which
            # rows an edit lands on, which commands are possible -- saw the
            # previous one, so `measure moon rate` stayed grey with six frames
            # picked. Dragging happened to look right only because the next
            # move redrew with the previous move's selection in place.
            self.list.itemSelectionChanged.connect(
                lambda: None if self._syncing else self._status())

            left = QtWidgets.QWidget()
            lv = QtWidgets.QVBoxLayout(left)
            lv.setContentsMargins(0, 0, 0, 0)
            lv.setSpacing(4)
            lv.addWidget(self.view, 1)
            lv.addWidget(self.strip)

            split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            split.addWidget(left)
            split.addWidget(self.list)
            split.setStretchFactor(0, 1)
            split.setSizes([1100, 260])

            # What you read, above the frame, in your line of sight from the
            # subject. What you operate goes below it. That is the whole rule.
            self.frame_label = QtWidgets.QLabel()
            self.frame_label.setTextFormat(QtCore.Qt.RichText)
            self.shot_label = QtWidgets.QLabel()
            self.shot_label.setTextFormat(QtCore.Qt.RichText)
            # One read line, ordered by what it is about: **which frame ·
            # what the keys will do · what it is being compared with · how it
            # was shot**. It was split in two for a while, with the program's
            # name in the top-left corner and the settings dropped to the
            # fields row, on the argument that the two halves change at
            # different rates. Read side by side the single line was the
            # better one: the name is on the title bar and in the corner of
            # the grid already, and the four things above the frame are one
            # sentence about the frame.
            read = QtWidgets.QWidget()
            read.setObjectName("readline")
            rd = QtWidgets.QHBoxLayout(read)
            rd.setContentsMargins(10, 4, 10, 4)
            rd.addWidget(self.frame_label)
            rd.addStretch(1)
            rd.addWidget(self.shot_label)

            self.fields = {}
            row = QtWidgets.QWidget()
            rh = QtWidgets.QHBoxLayout(row)
            rh.setContentsMargins(10, 3, 10, 4)
            for name, label, lo, hi, dec in (
                    ("cx", "centre x", -1e5, 1e5, 3),
                    ("cy", "y", -1e5, 1e5, 3),
                    ("a", "radius", 1.0, 1e5, 3),
                    ("b", "minor", 1.0, 1e5, 3),
                    ("angle", "angle", -180.0, 180.0, 2)):
                lab = QtWidgets.QLabel(label)
                box = QtWidgets.QDoubleSpinBox()
                box.setRange(lo, hi)
                box.setDecimals(dec)
                box.setSingleStep(1.0 if name != "angle" else 0.5)
                if name != "angle":
                    box.setSingleStep(self.grid)
                box.setKeyboardTracking(False)
                box.setFocusPolicy(QtCore.Qt.ClickFocus)
                # The fields are the mouse path to what the arrow keys do, so
                # they say so: an operation with no visible control is an
                # operation nobody finds.
                box.setToolTip(
                    "centre, in full-frame pixels.  arrows move one step, "
                    "ctrl+arrows ten, alt+arrows one photosite"
                    if name in ("cx", "cy") else
                    "the size of both bodies, for the whole clip")
                lab.setToolTip(box.toolTip())
                box.valueChanged.connect(self._fields_changed)
                box.editingFinished.connect(self.view.setFocus)
                self.fields[name] = (lab, box)
                rh.addWidget(lab)
                rh.addWidget(box)
            # Two knobs on how the frame is *shown*, next to the numbers
            # that say where it *is*. They sit after the fields and before
            # the step because that is the order of the sentence: this is the
            # subject, this is how I am looking at it, this is how far a key
            # moves it.
            rh.addSpacing(18)
            self.ev_slider, self.ev_label = self._knob(
                rh, "exposure", -80, 80, 0, 120, self._set_ev, "0.0 EV",
                "how bright the frame is drawn, in stops.  it changes "
                "nothing but the picture on screen")
            rh.addSpacing(14)
            self.detail_slider, self.detail_label = self._knob(
                rh, "local contrast", 0, 100, 0, 120, self._set_detail, "0 %",
                "flattens the slow falloff and lifts the fine structure, "
                "which is how a corona becomes visible.  the picture only")
            reset = QtWidgets.QLabel("reset")
            reset.setStyleSheet("color:#5a5a60;")
            reset.setCursor(QtCore.Qt.PointingHandCursor)
            reset.setToolTip("back to the plain view")
            reset.mousePressEvent = lambda _e: self._reset_tone()
            rh.addWidget(reset)

            rh.addStretch(1)

            central = QtWidgets.QWidget()
            cv = QtWidgets.QVBoxLayout(central)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(0)
            cv.addWidget(read)
            cv.addWidget(row)
            cv.addWidget(split, 1)
            cv.addWidget(self._commands())
            self.setCentralWidget(central)
            self.statusBar().hide()
            self._msg_timer = QtCore.QTimer(self)
            self._msg_timer.setSingleShot(True)
            self._msg_timer.timeout.connect(lambda: self.msg_label.setText(""))

            # A mode you cannot see is a mode you will blame the program for.
            # Two marks, since one is never enough: a rim on whichever panel
            # holds the focus, and words in the status bar for what the arrows
            # do right now.
            self.setStyleSheet(
                "#readline { background: #1a1b20; "
                "border-bottom: 1px solid #26282e; }"
                "#commands { background: #1a1b20; "
                "border-top: 1px solid #26282e; }"

                "QGraphicsView { border: 2px solid transparent; }"
                "QGraphicsView:focus { border: 2px solid #0a84ff; }"
                "QListWidget { border: 2px solid transparent; }"
                "QListWidget:focus { border: 2px solid #0a84ff; }")
            self._on_focus = lambda *_: self._show_mode()
            QtWidgets.QApplication.instance().focusChanged.connect(
                self._on_focus)
            # The space bar has to work wherever the focus is, like the letter
            # shortcuts do -- and it cannot be an action shortcut, because an
            # action fires on the press and knows nothing of the release. A
            # list, a combo box and a spin box all swallow space for their own
            # purposes, so it is caught before they see it.
            QtWidgets.QApplication.instance().installEventFilter(self)
            self._show_mode()
            # The size it was left at last time, on the screen the pointer is
            # on, in the middle of it.
            sized(self, "editor", (1440, 900))
            place_centred(self)

        # ------------------------------------------------------------ helpers
        def centre(self, i):
            f = self.frames[i]
            return None if f["cx"] is None else np.array([f["cx"], f["cy"]])

        def _walk(self, i, d):
            """The next frame in direction `d` that is actually in the film.

            A frame marked skip is not in the output, so comparing against it
            or stopping on it is a wasted press. Returns None when there is
            nothing that way.
            """
            k = i + d
            while 0 <= k < len(self.frames):
                if not self.frames[k].get("skip"):
                    return k
                k += d
            return None

        def ref_index(self):
            """Which frame is being compared against, if any.

            A pin outranks the mode: it is the answer to "hold *that* one
            still", which the relative modes cannot express -- and the frame
            worth holding still is usually the last good one before a bad run.
            """
            if self.ref_pin is not None:
                return min(self.ref_pin, len(self.frames) - 1)
            if self.ref == "first":
                k = 0 if not self.frames[0].get("skip") else self._walk(0, 1)
                return k
            if self.ref == "prev":
                return self._walk(self.i, -1)
            if self.ref == "next":
                return self._walk(self.i, 1)
            return None

        def anchor_of(self, i):
            """The point this frame is lined up on: **the sun**, always.

            One rule, no cases, and the three editing modes fall out of it
            instead of being three rules of their own. The drawing is shifted
            so this frame's sun lands on the crosshair, which is also exactly
            what `apply` will write, so the view is a preview and not a
            separate convention.

            What each mode then does, without the view knowing which mode it
            is in:

            - **sun** -- the sun stays on the crosshair, so the picture slides
              under it, and the moon's circle rides along with the picture it
              belongs to. You are aiming the frame.
            - **moon** -- the sun is untouched, so nothing shifts: the moon's
              circle is the only thing that moves. You are placing the moon on
              a still picture.
            - **both** -- the two move together by the same amount, the shift
              cancels the sun's own move, and both circles stay put while the
              picture travels beneath them. You are moving the picture under a
              pair whose relationship is already right.

            The earlier version anchored on whichever body you were *not*
            editing, on the theory that placing a circle needs something still
            to place it against. It read as confusing in the hand: the
            crosshair jumped bodies when you pressed `m`, and in moon mode the
            sun's circle was the thing seen moving even though the sun was not
            what you were editing. Where the crosshair sits should not depend
            on a mode.
            """
            return self.centre(i)

        def _shift(self, a, i, like):
            from scipy.ndimage import shift as ndshift
            c = self.anchor_of(i)
            if c is None or like is None:
                return a
            d = (like - c) / self.sc
            return ndshift(a, (d[1], d[0]), order=1, mode="constant", cval=0.0)

        def shifted(self, i, like):
            """Frame i, moved so its recorded centre lands on `like`.

            Under `edges` the gradient is taken **before** the shift, once per
            frame and kept: a translation and a gradient commute, so the answer
            is the same and the 4-13 ms it costs is not paid again on every
            nudge. The tone knobs do not reach it, and should not: the edge
            map normalises to its own percentiles, so exposure moves nothing
            in it, and lifting local contrast in a picture made of gradients
            would be amplifying an amplification.
            """
            if self.mode == "edges":
                if i not in self._edges:
                    self._edges[i] = edge_map(self.cache(i))
                return self._shift(self._edges[i], i, like)
            # Curved here, at paint time, so a knob costs a repaint and not a
            # read. With both knobs at rest this is the plain gamma the cache
            # used to hold, unchanged.
            return self._shift(tone(self.cache(i), self.ev, self.detail),
                               i, like)

        # ------------------------------------------------------------- paint
        def show_frame(self, k, fit=False):
            was = self.i
            self.i = max(0, min(len(self.frames) - 1, k))
            tgt = np.array(self.doc["target"], float)
            onto = tgt
            j = self.ref_index()
            peek = self._peek and j is not None and j != self.i
            # Held down, the space bar swaps in the reference on its own: the
            # blink comparator, which is the oldest tool there is for this and
            # still the best one. Both frames are drawn shifted onto the same
            # target, so what jumps between them is the misalignment and
            # nothing else.
            cur = self.shifted(j if peek else self.i, onto)
            other = None if peek or j is None or j == self.i \
                else self.shifted(j, onto)
            self.view.set_image(_compose(cur, other, self.mode))
            self.view.set_target(tgt[0] / self.sc, tgt[1] / self.sc)
            self._paint_curve(tgt)
            self._paint_moon(tgt)
            if fit:
                self.view.fit()
            self._syncing = True
            # NoUpdate: moving the current frame must not wipe a selection the
            # user is in the middle of making -- clicking one already selects
            # it, and interpolating needs the range to survive the navigation
            # that selecting it causes.
            self.list.setCurrentRow(
                self.i, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
            self.list.scrollToItem(self.list.currentItem())
            self._syncing = False
            old = [k for k in (self._marked, self._ref_marked)
                   if k is not None]
            self._marked, self._ref_marked = self.i, j
            for k in set(old + [self.i] + ([j] if j is not None else [])):
                if 0 <= k < len(self.frames):
                    self._refresh_item(k)
            self.strip.cur, self.strip.ref = self.i, j
            self.strip.update()
            self._sync_fields()
            self._status()
            self.cache.prefetch([self.i + 1, self.i - 1, self.i + 2, self.i - 2])
            # Only when it actually moved. Every edit repaints, and a repaint
            # is not something the hand did: logging it would bury the actions
            # under their own consequences.
            if self.i != was and not self._depth:
                self._log("frame", was=was)

        def _paint_curve(self, tgt=None):
            """One rule for both views: the curve goes where the recorded
            centre is being shown.

            In the corrected preview the frame is drawn shifted onto the
            target, so the curve sits on the crosshair and the disc slides
            underneath. With the correction off the frame is drawn as shot, so
            the curve sits on the centre itself and moves when the centre does
            -- which is the view to fit a curve to a limb by hand.
            """
            if tgt is None:
                tgt = np.array(self.doc["target"], float)
            sh = alignment.size_of(self.doc, "sun")
            if not (self.show_curve and sh and
                    self.frames[self.i]["cx"] is not None):
                self.view.set_curve(None, None)
                return
            base = self.anchor_of(self.i)
            off = (tgt - base) if base is not None else np.zeros(2)
            c = self.centre(self.i)
            if c is None:
                self.view.set_curve(None, None)
                return
            c = c + off
            dx, dy = _conic_points(sh)
            self.view.set_curve((c[0] + dx) / self.sc, (c[1] + dy) / self.sc)

        def _paint_moon(self, tgt=None):
            """The occulting body, where it actually sits on the frame.

            Not anchored to the crosshair like the sun's curve: the moon is part
            of the picture, so it travels with it. Its distance from the cross
            is the moon-to-sun offset, which is the quantity that moves in a
            straight line -- and seeing it as a gap makes a frame that breaks
            the line obvious.
            """
            f = self.frames[self.i]
            m = f.get("moon")
            if not (self.show_curve and m and f["cx"] is not None):
                self.view.set_moon(None, None)
                return
            if tgt is None:
                tgt = np.array(self.doc["target"], float)
            base = self.anchor_of(self.i)
            off = (tgt - base) if base is not None else np.zeros(2)
            t = np.linspace(0, 2 * np.pi, 241)
            r = _principal(self.size("moon"))
            mx, my = alignment.moon_xy(f)
            xs = (mx + off[0] + r * np.cos(t)) / self.sc
            ys = (my + off[1] + r * np.sin(t)) / self.sc
            self.view.set_moon(xs, ys,
                               m.get("source") == alignment.DETECTED)

        def _shot_of(self, i):
            """The frame's shooting data, looked up once and kept.

            Cheap enough to read every time -- 0.0008 s -- but it never changes
            for a given file, so it is read once.
            """
            if i not in self._shot:
                self._shot[i] = io.shot_info(self.paths[i])
            return self._shot[i]

        def _list_stepped(self, d):
            if d:
                self.go(d, every=True)
            self._notice(
                '<b style="color:#0a84ff">NAVIGATE</b>&nbsp;&nbsp;the arrows '
                'are walking the list<br>'
                '<span style="color:#8a8a90">esc, or a click on the frame, '
                'hands them back to the subject</span>')

        def _notice(self, html, ms=2600):
            """A word on the frame itself, gone in a moment."""
            self.notice.setText(html)
            self.notice.adjustSize()
            r = self.view.rect()
            self.notice.move((r.width() - self.notice.width()) // 2,
                             max(12, int(r.height() * 0.08)))
            self.notice.show()
            self.notice.raise_()
            self._notice_timer.start(ms)

        def _say(self, text, ms=3000):
            # The status line is where an action says what it did -- "walked
            # the moon across 98 frames", "9 of 11 already had one". Those
            # sentences are the outcome in the only form a person reads, and a
            # screencast wants them as its captions.
            # Said *by* an action in flight: kept and handed to that action's
            # own line rather than written above it, so one action is one
            # event carrying its own outcome.
            if getattr(self, "_depth", 0):
                self._note = text
            else:
                self._log("said", text=text)
            """A passing notice, in its own corner so it hides nothing."""
            self.msg_label.setText(text)
            self._msg_timer.start(ms)

        def _show_mode(self):
            """Focus moved. The read line says which mode is on, so redraw it."""
            self._status()

        def _avail(self):
            """Grey out what cannot be done from where the session is standing.

            Dimmed and not hidden. A button that vanishes takes its own
            explanation with it -- you cannot ask why it is not there -- and it
            drags the grid sideways under the pointer. Dim keeps the place,
            keeps the tooltip, and answers the question.
            """
            rows = sorted(x.row() for x in self.list.selectedIndexes())
            run = len(rows)
            on = self._targets()
            back = self._walk(on[0], -1)
            ahead = self._walk(on[-1], 1)
            back = None if back is None or back in on else back
            ahead = None if ahead is None or ahead in on else ahead
            for text, ok in (
                    ("extrapolate moon…", bool(self.track)),
                    ("measure moon rate", run >= 2),
                    ("interpolate", run >= 3),
                    ("interpolate moon", run >= 3),
                    # The neighbour of the *run*, which is what a copy takes
                    # from -- so at the top of a selection `from prev` is grey
                    # even though the shown frame has a neighbour behind it.
                    ("alignment from prev", back is not None),
                    ("alignment from next", ahead is not None),
                    ("all from prev", back is not None),
                    ("all from next", ahead is not None),
                    ("undo", bool(self._past)),
                    ("redo", bool(self._future))):
                self.cmd[text].setEnabled(ok)
            # The three rows the selection widens, and only while it is in
            # force -- `_targets()` returns one frame when the frame on screen
            # is not part of the selection.
            n = len(self._targets())
            for text in ("move what m selects", "remeasure", "skip",
                         "alignment from prev", "alignment from next",
                         "all from prev", "all from next"):
                self.cmd[text].setScope(n)
            # The row says what it will take back, so you can tell at a glance
            # whether the thing you regret is the thing on top.
            for text, stack in (("undo", self._past), ("redo", self._future)):
                self.cmd[text].setText(
                    "%s %s" % (text, stack[-1][0]) if stack else text)

        def _body(self):
            """Which body's numbers the fields and the size box speak for.

            `both` moves the pair and has no size of its own, so it reads as
            the sun -- the fields still have to show something, and the sun is
            what the document is mostly about.
            """
            return "moon" if self.editing == "moon" else "sun"

        def _told_size(self):
            """The clip's size for this body, in words.

            Shown here and nowhere per-frame, because that is now the truth: it
            is one number for the whole clip, and putting it beside the frame's
            own numbers is what made people think it was one of them.
            """
            sz = alignment.size_of(self.doc, self._body())
            return ", ".join("%s %.1f" % (k, v) for k, v in sz.items()) \
                or "no size"

        def _status(self):
            """The one line above the frame: everything you read, in order.

            **Which frame · what the keys will do · what it is being compared
            with · how it was shot.** The modes are values and are shown with
            the key that changes them, because a mode you can read but cannot
            find the switch for is half an answer.

            It was two lines for a while, split by how often each half
            changes, with the program's name in the corner. That fixed a real
            annoyance -- the filename shifting sideways when a reference name
            three words away grew by a letter -- and cost more than it fixed:
            two rows to scan, and a name that never changes taking the corner
            your eye lands on first.
            """
            f = self.frames[self.i]
            g = lambda c, t: '<span style="color:%s">%s</span>' % (c, t)
            # Same glyph translation as the command sheet, so a key named in the
            # read line reads right off macOS too. Only pinned-ref uses one today
            # (⇧G, kept as is), but a ⌘ added here would resolve to Ctrl.
            key = lambda k: g("#5a5a60", self._key_plain(k))
            bar = g("#3a3a42", "\u2502")

            bits = [g("#8a8a90", "%d/%d" % (self.i + 1, len(self.frames))),
                    '<b style="color:#e8e8ea">%s</b>' % f["file"],
                    '<span style="color:%s;font-weight:bold">%s</span>'
                    % (SRC_COLOUR.get(f["source"], "#ccc"),
                       f["source"].upper())]
            if f.get("skip"):
                bits.append('<span style="color:#ff453a;font-weight:bold">'
                            'SKIPPED</span>')
            if f["agreement"] is not None:
                bits.append(g("#8a8a90", "%.0f%%" % (100 * f["agreement"])))

            navigating = self.list.hasFocus()
            if not navigating and hasattr(self, "notice"):
                self.notice.hide()
            mode = bits
            mode.append(bar)
            if navigating:
                mode.append('<b style="color:#0a84ff">NAVIGATE</b>')
                mode.append(g("#8a8a90", "arrows change frame"))
                mode.append(key("esc to the frame"))
            else:
                mode.append('<b style="color:#3ddc97">EDIT</b>')
                mode.append(g("#8a8a90", "arrows move the"))
                mode.append('<b style="color:%s">%s</b>'
                            % (MOON if self.editing == "moon" else SUN,
                               self.editing.upper()))
                mode.append(key("m"))
                mode.append(g("#8a8a90", self._told_size()))
            # Two different things and they must not look alike: a selection
            # that will take the next edit, and one that will not because the
            # frame on screen is outside it. The second used to say nothing at
            # all, which is the wrong silence -- it is exactly where what you
            # expect and what happens come apart.
            picked = len(sorted(x.row() for x in self.list.selectedIndexes()))
            n = len(self._targets())
            if n > 1:
                mode.append(g("#ffb340", "%d selected" % n))
            elif picked > 1:
                mode.append(g("#8a6a20", "%d selected" % picked)
                            + "&nbsp;" + g("#5a5a60", "not this frame"))

            j = self.ref_index()
            peeking = self._peek and j is not None and j != self.i
            mode.append(bar)
            for text, k in ((self.mode, "x"),
                            ("outline" if self.show_curve
                             else "no outline", "o")):
                mode.append(g("#8a8a90", text) + "&nbsp;" + key(k))
            if j is None or j == self.i:
                mode.append(g("#5a5a60", "no ref") + "&nbsp;" + key("g"))
            elif peeking:
                mode.append('<b style="color:%s">showing ref %s</b>'
                            % (REF, self.frames[j]["file"]))
            else:
                mode.append(g("#5a5a60", "ref") + "&nbsp;"
                            + g(REF, self.frames[j]["file"]) + "&nbsp;"
                            + key("pinned \u21e7G" if self.ref_pin is not None
                                  else "g"))

            shot = self._shot_of(self.i)
            told = [shot.get("exposure"),
                    "f/%.1f" % shot["aperture"] if shot.get("aperture") else None,
                    "ISO %d" % shot["iso"] if shot.get("iso") else None,
                    "%g mm" % shot["focal"] if shot.get("focal") else None,
                    shot["when"].split()[-1] if shot.get("when") else None]
            told = [t for t in told if t]
            tail = (g("#8a8a90", g("#5a5a60", " \u00b7 ").join(told))
                    if told else "")

            self.frame_label.setText("&nbsp;&nbsp; ".join(bits))
            self.frame_label.setToolTip(
                "\n".join(x for x in (shot.get("camera"),
                                       shot.get("when", "").replace(":", "-", 2))
                           if x) or "")
            self.shot_label.setText(tail)

            self._avail()
            self.setWindowTitle("eclipse-aligner — %s%s"
                                % (os.path.basename(self.doc["input_dir"]),
                                   "  •" if self.dirty else ""))

        def _sync_fields(self):
            """The numbers of whichever body the arrows are on.

            One row of fields for two bodies, because they are the same three
            numbers and two rows would ask which is which. What answers that is
            the colour and the words: the moon's fields are amber, the same
            amber its circle is drawn in, and they say `moon x` rather than
            `centre x`.
            """
            f = self.frames[self.i]
            moon = self.editing == "moon"
            src = (f.get("moon") or {}) if moon else {}
            # Position from the frame, size from the clip: they are two
            # different things now and the fields say so. The moon's boxes
            # show where it *is*, not the offset the document keeps -- a
            # number you read off the screen has to be one you can compare
            # with what you see.
            sz = self.size()
            at = alignment.moon_xy(f) if moon else None
            cx = at[0] if at else (None if moon else f["cx"])
            cy = at[1] if at else (None if moon else f["cy"])
            circle = moon or "r" in sz
            colour = MOON if moon else SUN

            self._syncing = True
            self.fields["cx"][1].setValue(cx if cx is not None else 0.0)
            self.fields["cy"][1].setValue(cy if cy is not None else 0.0)
            self.fields["a"][1].setValue(sz.get("r", sz.get("a", 1.0)))
            self.fields["b"][1].setValue(sz.get("b", 1.0))
            self.fields["angle"][1].setValue(src.get("angle", 0.0))
            self.fields["cx"][0].setText("moon x" if moon else "centre x")
            self.fields["a"][0].setText("radius" if circle else "major")
            for name in ("cx", "cy", "a", "b", "angle"):
                lab, box = self.fields[name]
                lab.setStyleSheet("color: %s;" % colour)
                # The size always exists now -- it is the clip's, not this
                # frame's -- so its field is always there to be typed in. The
                # rule that hid it still asked whether *this frame* had a
                # shape, which nothing has had since the size moved.
                has = True if name in ("a", "b", "angle") else cx is not None
                lab.setEnabled(has)
                box.setEnabled(has)
                shown = name in ("cx", "cy", "a") or not circle
                lab.setVisible(shown)
                box.setVisible(shown)
            self._syncing = False

        def _refresh_item(self, i):
            """One row's looks: coloured by source, struck out if skipped, and
            a slab of background on the frame being shown.

            The current row has to be painted by hand. It is set with NoUpdate
            so that navigating cannot wipe a selection, and Qt draws nothing for
            a current-but-unselected row in a list that never takes focus -- so
            without this the one row that matters most is the only one carrying
            no mark at all.
            """
            it, f = self.list.item(i), self.frames[i]
            skip = bool(f.get("skip"))
            here = i == self.i
            is_ref = i == self._ref_marked and not here
            # A caret in the text as well as the background, because a selected
            # row is painted by the palette and the background loses: the one
            # mark that survives every state is the one made of characters.
            # The reference is drawn cyan in the overlay, so it is cyan here:
            # the mark and the thing it marks look the same.
            it.setText("%s %4d  %s" % ("\u25b8" if here else
                                       "\u00b7" if is_ref else " ",
                                       i + 1, f["file"]))
            it.setForeground(QtGui.QColor(
                "#6b6b70" if skip else SRC_COLOUR.get(f["source"], "#ccc")))
            it.setBackground(QtGui.QColor("#39394a") if here else
                             QtGui.QColor("#2b2440") if is_ref else
                             QtGui.QBrush())
            font = it.font()
            font.setStrikeOut(skip)
            font.setBold(here)
            it.setFont(font)
            self.strip.update()

        # ------------------------------------------------------------ editing
        def _targets(self):
            """Which frames an edit lands on: the selection, or just this one.

            Only when the frame on screen is part of the selection. Otherwise a
            selection left over from something else would take the edit away to
            frames nobody is looking at, and the one in front of you would sit
            still -- which is the wrong way round for a mistake to happen.
            """
            rows = sorted(x.row() for x in self.list.selectedIndexes())
            return rows if len(rows) > 1 and self.i in rows else [self.i]

        def _touch(self, k=None):
            self.frames[self.i if k is None else k]["source"] = alignment.MANUAL
            self.dirty = True
            self._refresh_item(self.i if k is None else k)

        def _fields_changed(self):
            if self._syncing:
                return
            self._log("field", body=self.editing)
            f = self.frames[self.i]
            # The radius box edits the clip's size, the centre boxes edit this
            # frame. Typing a radius here is one of the two ways to set it from
            # a frame you trust; the other is typing it in the radius box.
            sz = self.size()
            if "r" in sz:
                sz["r"] = round(self.fields["a"][1].value(), 3)
            else:
                sz["a"] = round(self.fields["a"][1].value(), 3)
                sz["b"] = round(self.fields["b"][1].value(), 3)
                sz["angle"] = round(self.fields["angle"][1].value(), 3)
            self.doc["size"][self._body()] = sz
            if self.editing == "moon":
                if f.get("moon"):
                    alignment.set_moon_xy(
                        f, (self.fields["cx"][1].value(),
                            self.fields["cy"][1].value()), alignment.MANUAL)
                self.dirty = True
                self.show_frame(self.i)
                return
            # The alignment, and only that: the offset is the other mode's
            # to write, so the moon rides along and the pair keeps whatever
            # relationship it had.
            f["cx"] = round(self.fields["cx"][1].value(), 3)
            f["cy"] = round(self.fields["cy"][1].value(), 3)
            self._touch()
            self.show_frame(self.i)

        def nudge(self, dx, dy):
            """One step of whichever of the frame's two numbers `m` selects.

            Watched, so a key press is one undo step and one journal line.
            The drag calls `_move` underneath instead, and wraps a whole
            gesture in one of each.
            """
            return self._move(dx, dy)

        def _move(self, dx, dy):
            """The move itself. **The thing under your hand follows.**

            One rule for both modes, and it is the only one a hand can hold:
            in `align` the picture goes where the arrow points, in `moon` the
            moon does. That the alignment's own number goes the other way --
            the frame is drawn shifted by `target - centre`, so a smaller
            centre slides the picture right -- is arithmetic the hand should
            never have to know.

            The signs disagreed for one afternoon, while there were three
            modes: `sun` moved the picture with the arrow and `both` moved it
            against, because one was named for the body and the other for the
            pair. Naming the mode after the thing that visibly moves settled
            it.
            """
            if self.editing == "align":
                # The alignment: where the picture has to sit for the sun to
                # land on the crosshair. In this document that is simply
                # `cx, cy`, and the moon comes along because it is kept as an
                # offset from it -- so the pair holds still on screen and the
                # picture travels underneath, which is the gesture.
                for k in self._targets():
                    f = self.frames[k]
                    if f["cx"] is None:
                        f["cx"], f["cy"] = self.doc["target"]
                    f["cx"] = round(f["cx"] - dx, 3)
                    f["cy"] = round(f["cy"] - dy, 3)
                    self._touch(k)
                return self.show_frame(self.i)
            # The offset. The arrow points where the moon goes, the same as
            # in the other mode it points where the picture goes: on screen
            # both read as "the thing under your hand follows the key".
            for k in self._targets():
                m = self._moon_of(k, make=True)
                if m is None:
                    continue
                m["dx"] = round(m["dx"] + dx, 3)
                m["dy"] = round(m["dy"] + dy, 3)
                m["source"] = alignment.MANUAL
            self.dirty = True
            self.show_frame(self.i)

        # ---------------------------------------------------------- drag
        # A gesture is a hundred move events and has to leave one undo step
        # and one journal line, so the drag does by hand what `_watch` does
        # for a key: remember once at the start, settle and log once at the
        # end, and call `_move` in between rather than `nudge`.

        def _drag_began(self):
            self._drag = [self._remember("drag"), 0.0, 0.0]

        def _dragged(self, sx, sy):
            """One step of a drag, rounded onto the grid.

            The pointer moves continuously and the frame may not: what is
            accumulated is the raw travel, and what is applied is the
            difference between where the *rounded* total was and where it is
            now. So the frame follows the hand in whole photosites without
            drifting behind it -- rounding each little delta on its own would
            throw away everything under half a step and the frame would lag
            further behind the further you dragged.
            """
            if self._drag is None:
                return
            # Scene units are the downsampled image; the document speaks in
            # full-frame pixels.
            was = (round(self._drag[1] / self.grid),
                   round(self._drag[2] / self.grid))
            self._drag[1] += sx * self.sc
            self._drag[2] += sy * self.sc
            now = (round(self._drag[1] / self.grid),
                   round(self._drag[2] / self.grid))
            dx = (now[0] - was[0]) * self.grid
            dy = (now[1] - was[1]) * self.grid
            if dx or dy:
                self._move(dx, dy)

        def _drag_ended(self):
            if self._drag is None:
                return
            keep, dx, dy = self._drag
            # What the frame actually took, not what the hand travelled.
            dx = round(dx / self.grid) * self.grid
            dy = round(dy / self.grid) * self.grid
            self._drag = None
            # `_settle` drops the step again if nothing actually moved, so a
            # shift-click that never travelled leaves no history behind.
            self._settle(keep)
            self._log("drag", dx=round(dx, 3), dy=round(dy, 3))
            if dx or dy:
                n = len(self._targets())
                self._say("moved the %s %.1f, %.1f px%s"
                          % ("moon" if self.editing == "moon"
                             else "alignment", dx, dy,
                             "" if n == 1 else " on %d frames" % n), 2500)

        # ---------------------------------------------------------- tone
        # Nothing under this heading touches the document. It is what the eye
        # needs to judge a corona, and the corona is four stops under the limb
        # it has to be judged against.

        def _knob(self, into, name, lo, hi, at, width, slot, text, tip):
            """A named slider with its value beside it, added to a row."""
            QtCore, QtGui, QtWidgets = _widgets()
            lab = QtWidgets.QLabel(name)
            bar = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            bar.setRange(lo, hi)
            bar.setValue(at)
            bar.setFixedWidth(width)
            # The slider must not keep the keys: it steals the arrows, and the
            # arrows are the most used thing in this window.
            bar.setFocusPolicy(QtCore.Qt.NoFocus)
            bar.valueChanged.connect(slot)
            val = QtWidgets.QLabel(text)
            val.setStyleSheet("color:#8a8a90;")
            val.setMinimumWidth(52)
            for w in (lab, bar, val):
                w.setToolTip(tip)
                into.addWidget(w)
            return bar, val

        def _set_ev(self, tenths):
            self.ev = int(tenths) / 10.0
            self.ev_label.setText("%+.1f EV" % self.ev if self.ev
                                  else "0.0 EV")
            self.show_frame(self.i)

        def _set_detail(self, percent):
            self.detail = int(percent) / 100.0
            self.detail_label.setText("%d %%" % int(percent))
            self.show_frame(self.i)

        def _reset_tone(self):
            """Back to the plain view: no gain, no compression, no lift."""
            self.ev_slider.setValue(0)
            self.detail_slider.setValue(0)
            self._say("showing it plain", 1800)

        def size(self, body=None):
            """The clip's shape for a body, made if the document has none.

            One place, because every drawing, every readout and every resize
            wants the same number and they must not be able to disagree.
            """
            body = body or self._body()
            sz = alignment.size_of(self.doc, body)
            if not sz:
                w, h = self.doc["frame_size"]
                sun = alignment.size_of(self.doc, "sun")
                sz = ({"r": round(_principal(sun) * 1.03, 3)} if sun and
                      body == "moon" else {"r": round(min(w, h) / 6.0, 3)})
                self.doc.setdefault("size", {})[body] = sz
            return sz

        def copy_from_ref(self, forward=False, whole=False):
            """Take a neighbour's record: just the alignment, or all of it.

            Between consecutive frames the subject barely moves and the disc
            does not change size at all, so the neighbour's answer is the best
            first guess there is -- better than anything a person can place by
            eye on a frame with no limb to place it against. It is what `held`
            does automatically for a frame the detector refused; this is the
            same move, by hand, for a frame it got wrong.

            **The neighbour is the run's, not the shown frame's.** With
            15-200 selected, `c` gives every one of them the alignment of 14
            and `shift+c` gives every one of them 201 -- whichever frame
            happens to be on screen inside the run. Taking the neighbour of
            the shown frame instead meant that with a run selected the answer
            depended on where you happened to be standing in it, and that
            twelve frames in, `c` copied from a frame that was itself inside
            the selection and about to be overwritten.

            That is also what a run is *for* here: a stretch the detector lost
            has good frames on both sides, and the question is which of the
            two to carry across it. One frame selected is the same rule with a
            run of one.

            The source is the frame before the run, or the one after it with
            `forward`. Full stop: it follows nothing else.

            It used to follow a pinned frame, on the argument that pinning is an
            explicit "hold that one". In use that turned out to be a button
            that lies -- it says `copy ◀` and takes from somewhere else -- and a
            control whose label is only true sometimes is worse than one
            function short. Pinning is about what to compare against by eye;
            copying is an edit, and its sensible source is a neighbour.

            **`c` copies the alignment alone**: `cx, cy` and nothing else.
            The moon is an offset from this frame's own sun, so not touching
            it means exactly that -- it travels with the alignment and does
            not move a pixel on screen. It is the plain key because it is the
            common case: the moon is often the part that is already right,
            and the frame it sits on is the part the detector lost.

            **`v` copies the whole record**: the moon comes across at the
            neighbour's own coordinates, so the frame ends up carrying the same
            geometry and therefore the same transformation. That is the only
            reading of "copy everything" that does not need a footnote.

            The two exist because the moon is often the part that is
            *already right* -- interpolated, extrapolated, or placed by hand
            over a run where both limbs were visible -- while the frame it
            sits on is the part the detector lost. Copying everything
            there would throw away the work and put the neighbour's moon on
            this frame's sky.

            Between them the middle option is gone on purpose. It shifted the
            moon by however far the sun had moved, keeping this frame's own
            offset: the right move for an arrow, and here a pair that matched
            neither its neighbour nor itself.

            The `agreement` does not come across either way. It scores how well
            a curve fitted *that* frame's limb, and this frame's limb was never
            looked at.
            """
            rows = self._targets()
            j = self._walk(rows[-1], 1) if forward else self._walk(rows[0], -1)
            if j is None:
                j = -1
            if not (0 <= j < len(self.frames)) or j in rows:
                self._say("no reference frame to copy from", 2500)
                return
            src = self.frames[j]
            if src["cx"] is None:
                self._say("%s has no position to copy" % src["file"], 2500)
                return
            lost = 0
            for k in rows:
                f = self.frames[k]
                f["cx"], f["cy"] = src["cx"], src["cy"]
                f["agreement"] = None
                if whole:
                    if src.get("moon"):
                        f["moon"] = dict(src["moon"],
                                         source=alignment.MANUAL)
                    else:
                        lost += f.pop("moon", None) is not None
                # Otherwise nothing else. The moon is an offset from this
                # frame's own sun, so leaving it alone is the whole of "do not
                # touch the moon" -- it travels with the alignment and does not
                # move a pixel on screen. It briefly held its absolute position
                # instead, which wrote both of the frame's numbers and showed
                # up as the moon sliding on a copy that was supposed to be
                # about the picture.
                self._touch(k)
            said = ("sun and moon" if whole and src.get("moon")
                    else "the alignment")
            self.show_frame(self.i)
            self._say("copied %s from %s onto %d frame%s%s"
                      % (said, src["file"], len(rows),
                         "" if len(rows) == 1 else "s",
                         "; %d lost a moon it had, since %s has none"
                         % (lost, src["file"]) if lost else ""), 3000)

        def toggle_skip(self):
            """Leave these frames out of the output, or put them back.

            Nothing is deleted and nothing is even written: the original is
            untouched as always and the record keeps its measurement, so the
            decision is reversible by pressing this again. It is a choice about
            the film -- a frame where the filter came off, a gust, a cloud --
            not about the shoot.
            """
            rows = self._targets()
            on = not all(self.frames[k].get("skip") for k in rows)
            for k in rows:
                if on:
                    self.frames[k]["skip"] = True
                else:
                    self.frames[k].pop("skip", None)
                self._refresh_item(k)
            self.dirty = True
            self.show_frame(self.i)
            self._say(
                "%d frame%s %s the output"
                % (len(rows), "" if len(rows) == 1 else "s",
                   "left out of" if on else "back in"), 3000)

        def _run(self):
            """The selected run, or None with a word about why not.

            Every range operation wants the same thing: at least three frames,
            two ends and something between them. Asked once here rather than
            three times in three slightly different sentences.
            """
            rows = sorted(x.row() for x in self.list.selectedIndexes())
            if len(rows) < 3:
                self._say("select at least three frames: the two ends and "
                          "something between them", 3000)
                return None
            return rows

        def _measured_moon(self, f):
            """A moon somebody stands behind: fitted, or placed by hand.

            A hand-placed moon is not a guess. It is the one a person put where
            they could see it belonged, on a frame the detector could not
            manage -- which is exactly the frame at the edge of totality that
            ends up being an anchor. Refusing it left the two ends of a
            hundred-frame run looking empty.
            """
            m = f.get("moon")
            return m if m and m.get("source") in (
                alignment.DETECTED, alignment.MANUAL) else None

        def interpolate(self):
            """Walk the whole frame from one end of the selection to the other.

            **It writes the alignment and nothing else.** A frame holds two
            independent things -- where the picture has to sit for the sun to
            land on the crosshair, and where the moon sits relative to that
            sun -- and since the document keeps the second as an offset, the
            moon comes along with no arithmetic at all. The loop below writes
            `cx, cy` and stops.

            Writing the two separately is what makes them **commute**, and
            that is the point. `i` sets the alignment, `shift+i` sets the
            offsets, and pressing them in either order lands on the same
            document to 9e-13 px -- both bodies stepping evenly, sigma 0.00
            each. Alone, each does its half: `i` leaves the offsets' own
            wobble (sigma 3.32 over frames 30-70 of a real document),
            `shift+i` leaves the alignment's (sigma 8.19).

            An earlier version drew a straight line through the moon's
            absolute position too. It reached the same place, but only in one
            order, and only because it was silently doing `shift+i`'s job as
            well. Order mattering between two commands that touch different
            quantities is a sign the quantities were wrong, not the order.

            A run the detector lost sits between two it did not, and the
            subject crossed that gap the way it crosses every other:
            steadily. Copying one neighbour's answer onto all of them gives a
            staircase instead, and a staircase is visible in a timelapse.
            """
            rows = self._run()
            if rows is None:
                return
            i0, i1 = rows[0], rows[-1]
            a, b = self.frames[i0], self.frames[i1]
            if a["cx"] is None or b["cx"] is None:
                self._say("both ends of the selection need a sun", 3000)
                return
            p0 = np.array([a["cx"], a["cy"]], float)
            p1 = np.array([b["cx"], b["cy"]], float)
            n = len(rows) - 2
            span = float(i1 - i0)
            for k in rows[1:-1]:
                t = (k - i0) / span
                f = self.frames[k]
                p = p0 + (p1 - p0) * t
                f["cx"], f["cy"] = [round(float(v), 3) for v in p]
                f["agreement"] = None
                f["source"] = alignment.MANUAL
                self._refresh_item(k)
            self.dirty = True
            self.show_frame(self.i)
            self._say("%d frames aimed between %s and %s, %.2f px a frame; "
                      "every offset untouched"
                      % (n, a["file"], b["file"],
                         float(np.hypot(*(p1 - p0))) / span), 5000)

        def _interpolate_moon_track(self, rows):
            """Walk the moon's offset from one end of the selection to the
            other.

            Worth having because the moon's own track is a thing a person can
            judge on screen -- it should crawl, evenly -- while the sun's is
            hidden under the mount's wandering. Straightening the offsets and
            straightening the alignment are two separate jobs, and this is the
            first of them.

            **The offset is what travels in a straight line**; where the moon
            *sits* does not, because the mount wanders underneath it. Filled
            on runs where both bodies really were detected, so the answer is
            known, a straight line through the moon's absolute position
            missed by 61-69 px on average and 178 px at worst, while the
            offset line missed by 2.7 and 3.7 px on two of the three runs.
            Since the document keeps the offset, that better answer is also
            the simpler code: interpolate what is stored.

            It never touches a sun. Aiming the whole frame is `i`, and the two
            commute.
            """
            i0, i1 = rows[0], rows[-1]
            ma = self.frames[i0].get("moon")
            mb = self.frames[i1].get("moon")
            if not (ma and mb):
                say_box(
                    self, "No moon at the ends",
                    "Both ends of the selection need a moon to walk between.")
                return
            off0 = np.array([ma["dx"], ma["dy"]], float)
            off1 = np.array([mb["dx"], mb["dy"]], float)
            walked = 0
            for k in rows[1:-1]:
                f = self.frames[k]
                if f["cx"] is None:
                    continue        # no sun, so no offset can mean anything
                off = off0 + (off1 - off0) * ((k - i0) / float(i1 - i0))
                f["moon"] = {"dx": round(float(off[0]), 3),
                             "dy": round(float(off[1]), 3),
                             "source": alignment.MANUAL}
                walked += 1
                self._refresh_item(k)
            self.dirty = True
            self.show_frame(self.i)
            n = len(rows) - 2
            missed = ("" if walked == n else
                      ", and %d had no sun to sit against" % (n - walked))
            self._say("walked the moon across %d frame%s; the suns are "
                      "untouched%s"
                      % (walked, "" if walked == 1 else "s", missed), 6000)

        def _moon_track(self, rows):
            """Least squares through the moon-to-sun offset of `rows`.

            Returns (fx, fy, rms, n) where fx(i) gives the offset at frame i,
            or None when there is not enough to fit.

            The *offset* and not the moon's own place in the frame. Fitting a
            line to where the moon sits would be fitting the mount's wandering,
            which is the thing this whole tool exists to remove -- measured on
            this clip, a straight line through 100 frames of it leaves 63 px of
            residual in x and 2012 px in y. The offset, over the same material,
            stays on its line to 0.9 px. It is also what the document holds,
            so this reads the numbers rather than deriving them.
            """
            good = [k for k in rows
                    if self.frames[k]["cx"] is not None
                    and (self.frames[k].get("moon") or {}).get("source")
                    in (alignment.DETECTED, alignment.MANUAL)]
            if len(good) < 2:
                return None
            t = np.array(good, float)
            off = np.array([[self.frames[k]["moon"]["dx"],
                             self.frames[k]["moon"]["dy"]] for k in good])
            A = np.c_[t, np.ones_like(t)]
            fits, res = [], []
            for axis in (0, 1):
                m, b = np.linalg.lstsq(A, off[:, axis], rcond=None)[0]
                fits.append((m, b))
                res.append(off[:, axis] - (m * t + b))
            rms = float(np.hypot(*[r.std() for r in res]))
            return fits, rms, len(good)

        def measure_moon_rate(self):
            """Measure how fast the moon moves, and keep it.

            Split from applying it on purpose. A line needs two things and they
            are best measured in different places: the **rate** wants a long
            stretch, where noise averages out, and the **starting point** wants
            a frame right next to the run being fixed, where nothing has had
            time to drift. Taking both from one selection makes the rate as
            local as the anchor, which is the worse of the two.
            """
            rows = sorted(x.row() for x in self.list.selectedIndexes())
            if len(rows) < 2:
                self._say("select a stretch whose moon is already right", 3000)
                return
            fit = self._moon_track(rows)
            if fit is None:
                say_box(
                    self, "Nothing to fit",
                    "At least two of the selected frames need a moon that was "
                    "measured or placed by hand, and a position for the sun.")
                return
            fits, rms, n = fit
            speed = float(np.hypot(fits[0][0], fits[1][0]))
            self.track = {"mx": fits[0][0], "my": fits[1][0],
                          "rms": rms, "n": n,
                          "from": "%d-%d" % (rows[0] + 1, rows[-1] + 1)}
            self.cmd["measure moon rate"].setText(
                "moon rate %.2f px/f" % speed)
            self._say("track: %.2f px per frame, %.1f px rms over %d frames "
                      "(%s)" % (speed, rms, n, self.track["from"]), 8000)

        def _track_plan(self, from_end, moving):
            """What carrying the track would do, worked out without doing it.

            Split from the doing so the chooser can show it changing as the
            choice changes: the anchor and the body between them decide which
            frames can be written at all, and asking two questions without
            saying what either one costs is just moving the guess to the user.

            Returns (plan, problem); exactly one of the two is None.
            """
            rows = sorted(x.row() for x in self.list.selectedIndexes())
            if len(rows) < 2:
                return None, ("Select the frames to fill first, including one "
                              "good one at the end you mean to anchor on.")
            anchor = rows[-1] if from_end else rows[0]
            a = self.frames[anchor]
            ma = a.get("moon") or {}
            if a["cx"] is None or ma.get("source") not in (
                    alignment.DETECTED, alignment.MANUAL):
                return None, ("%s is the %s frame of the selection, so it is "
                              "the anchor, and it needs a sun and a moon of "
                              "its own."
                              % (a["file"], "last" if from_end else "first"))
            if moving == "sun":
                # the sun follows from each frame's own measured moon
                targets = [k for k in rows if k != anchor
                           and (self.frames[k].get("moon") or {}).get("source")
                           in (alignment.DETECTED, alignment.MANUAL)]
                need = "a moon"
            else:
                # and the moon follows from each frame's own sun
                targets = [k for k in rows if k != anchor
                           and self.frames[k]["cx"] is not None]
                need = "a sun"
            if not targets:
                return None, ("None of the other selected frames has %s to go "
                              "on, so there is nothing to place the %s from."
                              % (need, moving))
            measured = sum(
                (self.frames[k]["source"] == alignment.DETECTED)
                if moving == "sun"
                else ((self.frames[k].get("moon") or {}).get("source")
                      == alignment.DETECTED)
                for k in targets)
            return {"anchor": anchor, "targets": targets, "measured": measured,
                    "reach": max(abs(k - anchor) for k in targets),
                    "off0": np.array([ma["dx"], ma["dy"]], float)}, None

        def _track_words(self, plan, from_end, moving):
            """The sentence under the chooser: what this choice would do."""
            a = self.frames[plan["anchor"]]
            mx, my = self.track["mx"], self.track["my"]
            return ("Anchor on %s, the %s of the selection, and place the %s "
                    "on %d frames from it at %.2f px a frame.\n\nThe track was "
                    "measured over %d frames (%s) and held its line to %.1f "
                    "px. The farthest frame here is %d away from the anchor.%s"
                    % (a["file"], "last" if from_end else "first", moving,
                       len(plan["targets"]), float(np.hypot(mx, my)),
                       self.track["n"], self.track["from"], self.track["rms"],
                       plan["reach"],
                       "\n\n%d of them have a %s the detector measured, which "
                       "this will replace."
                       % (plan["measured"], moving) if plan["measured"] else ""))

        def extrapolate_moon(self):
            """Ask which end and which body, then carry the track.

            One key rather than two, and the body asked for rather than taken
            from the `m` toggle. Four ways to run it were four buttons that all
            said "sun", and the difference between them -- which end holds
            still -- was in a modifier nobody could see. Asked outright, with
            the consequence of each answer written underneath, it is one
            question with the answer in view.
            """
            if not self.track:
                say_box(
                    self, "No track captured",
                    "Select a stretch whose moon is already right and press "
                    "shift+T first. That measures how fast the moon moves "
                    "against the sun, which is what gets carried; this then "
                    "starts it from one end of a selection and fills the rest.")
                return
            d = QtWidgets.QDialog(self)
            d.setWindowTitle("Extrapolate the moon")
            place_centred(d, self)
            lay = QtWidgets.QVBoxLayout(d)
            grid = QtWidgets.QGridLayout()
            lay.addLayout(grid)

            def row(n, label, a_text, b_text, a_on):
                grid.addWidget(QtWidgets.QLabel(label), n, 0)
                ga, gb = (QtWidgets.QRadioButton(a_text),
                          QtWidgets.QRadioButton(b_text))
                (ga if a_on else gb).setChecked(True)
                grid.addWidget(ga, n, 1)
                grid.addWidget(gb, n, 2)
                g = QtWidgets.QButtonGroup(d)
                g.addButton(ga)
                g.addButton(gb)
                return ga, gb

            first, last = row(0, "Anchor on", "the &first frame",
                              "the &last frame", not self._track_end)

            words = QtWidgets.QLabel()
            words.setWordWrap(True)
            words.setMinimumWidth(430)
            # Room for the longest of the four answers, kept whether or not
            # this one needs it. The text changes as the radios change, and a
            # dialog that resizes under the pointer moves the button you were
            # about to press.
            words.setMinimumHeight(132)
            words.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
            lay.addSpacing(6)
            lay.addWidget(words)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Extrapolate")
            lay.addWidget(bb)
            bb.accepted.connect(d.accept)
            bb.rejected.connect(d.reject)

            def refresh():
                plan, why = self._track_plan(last.isChecked(), "moon")
                words.setText(why if plan is None else self._track_words(
                    plan, last.isChecked(), "moon"))
                bb.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(
                    plan is not None)

            for b in (first, last):
                b.toggled.connect(refresh)
            # The end that can actually take it, chosen for you: often only one
            # of the two has a measured moon to hand over, and then the whole
            # dialog is one Enter.
            if self._track_plan(not self._track_end, "moon")[0] is None:
                (first if self._track_end else last).setChecked(True)
            refresh()
            if d.exec() != QtWidgets.QDialog.Accepted:
                return
            # Remembered for next time, because a session tends to walk out of
            # bad runs in one direction and then the other.
            self._track_end = last.isChecked()
            self.apply_track(last.isChecked(), "moon", confirm=False)

        def apply_track(self, from_end, moving="moon", confirm=True):
            """Carry the captured track across the selection.

            The anchor keeps its own position and hands over its offset;
            everything else in the selection is written from it. Anchored on
            the selection's first frame, or its last: walking out of a bad run
            forwards, the frame before it is the one still worth trusting, and
            walking in from the far side it is the one after.

            `confirm` is off when the chooser already asked -- it showed the
            same numbers and got the same answer, and asking twice for one
            action is how a person learns to click through without reading.
            """
            if not self.track:
                say_box(
                    self, "No track captured",
                    "Select a stretch whose moon is already right and press "
                    "shift+T first. That measures how fast the moon moves "
                    "against the sun, which is what gets carried; this then "
                    "starts it from one end of a selection and fills the rest.")
                return
            moving = moving or "moon"
            plan, why = self._track_plan(from_end, moving)
            if plan is None:
                self._say(why, 5000)
                return
            if confirm and say_box(
                    self, "Carry the track",
                    self._track_words(plan, from_end, moving),
                    QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Ok) != QtWidgets.QMessageBox.Ok:
                return
            anchor, off0 = plan["anchor"], plan["off0"]
            mx, my = self.track["mx"], self.track["my"]
            for k in plan["targets"]:
                f = self.frames[k]
                d = k - anchor
                ox, oy = off0[0] + mx * d, off0[1] + my * d
                if moving == "sun":
                    # The moon is the measurement here and must not move: put
                    # the sun where the predicted offset puts it, then restate
                    # the moon at the same pixels, which stores that very
                    # offset.
                    at = alignment.moon_xy(f)
                    f["cx"] = round(at[0] - ox, 3)
                    f["cy"] = round(at[1] - oy, 3)
                    f["source"] = alignment.MANUAL
                    alignment.set_moon_xy(f, at, alignment.MANUAL)
                else:
                    f["moon"] = {"dx": round(float(ox), 3),
                                 "dy": round(float(oy), 3),
                                 "source": alignment.MANUAL}
                self._refresh_item(k)
            self.dirty = True
            self.show_frame(self.i)
            self._say("placed the %s on %d frames from %s"
                      % (moving, len(plan["targets"]),
                         self.frames[anchor]["file"]), 6000)

        def interpolate_moon(self):
            """Walk the moon from one end of the selection to the other.

            It walks **the offset**, which is what the document holds and
            what actually travels in a straight line -- where the moon sits
            does not, because the mount wanders underneath it.

            It never touches a sun. Aiming the whole frame is `i`, and the two
            commute.
            """
            rows = self._run()
            if rows is None:
                return
            return self._interpolate_moon_track(rows)

        def size_from_frame(self, body=None):
            """Take the clip's size for this body from the frame on screen.

            The size belongs to the clip -- neither body changes apparent size
            over an afternoon -- but *which frame knows it best* is a judgement
            no score makes reliably. Both scores saturate: on a 30-frame sample
            the sun's agreement hit 1.000 on 7 frames and the moon's consensus
            on 19 of 20, whose radii spanned 6.4%. So detection picks the
            middle of the tie and this is how a person overrules it.

            One body at a time, and each with its own key, because **the best
            frame for the sun is not the best frame for the moon**: the sun's
            limb is fully drawn when little of it is covered, the moon's when it
            is deep across the disc. Same event, an hour apart, so asking for
            them on one key would mean flipping a toggle between two frames.

            It measures the frame again rather than reading what is stored,
            which is the whole point -- the stored size is what is being
            replaced. To set it by eye instead, seat the curve on the limb with
            type it in the radius box below the frame.
            """
            from .centre_disc import detect_centre, detect_moon
            body = body or self._body()
            f = self.frames[self.i]
            sc = self.step_measure * io.luma_scale(self.paths[self.i])
            luma = io.read_luma(self.paths[self.i], self.step_measure)
            c, agree, shape = detect_centre(luma, self.doc["method"], 0.15)
            if c is None or not shape:
                self._say("nothing fitted on %s; there is no size to take "
                          "from it" % f["file"], 4000)
                return
            if body == "sun":
                got = {k: round(float(v * sc if k in ("r", "a", "b") else v), 3)
                       for k, v in shape.items()}
                quality = "agreement %.0f%%" % (100 * agree)
            else:
                m = detect_moon(luma, c, shape.get("r", shape.get("a", 0.0)),
                                0.15)
                if m is None:
                    self._say("no moon limb on %s; pick a frame where the "
                              "moon is well across the disc" % f["file"], 4500)
                    return
                got = {"r": round(float(m[1] * sc), 3)}
                quality = "consensus %.0f%%" % (100 * m[2])
            was = alignment.size_of(self.doc, body)
            told = ", ".join("%s %.1f" % (k, v) for k, v in got.items())
            change = ""
            if was and _principal(was):
                change = ", %+.1f px on every frame" % (
                    _principal(got) - _principal(was))
            self.doc.setdefault("size", {})[body] = got
            self.dirty = True
            self.show_frame(self.i)
            # No confirmation. It measures and writes a number that shows up
            # the instant it lands -- in the line above the frame, in the
            # radius field, and in the circle drawn on the limb -- and pressing
            # the key again on a better frame replaces it. Nothing is lost, so
            # asking first is a click spent on nothing.
            #
            # What the dialog was for was the *quality*: taken from a frame
            # deep in the eclipse with 69% agreement, one real measurement came
            # out 131 px small. That belongs in the sentence, not behind a
            # button.
            self._say("%s %s from %s (%s)%s"
                      % (body, told, f["file"], quality, change), 7000)

        def revert(self):
            """Measure the selected frames again, here and now.

            It used to restore what the document said when the window opened,
            which sounds the same and is not: open a document that is already
            all hand-placed -- which is what a day's work produces -- and
            "revert" hands back the hand-placed values. It looked like the key
            had stopped working, and in the only sense that matters it had.

            Now it re-runs the detector on those frames, so a botched stretch
            can be handed back to the machine without re-running the whole
            clip. `skip` survives, being a decision about the film rather than
            a measurement.
            """
            from .centre_disc import detect_centre, detect_moon
            rows = self._targets()
            if len(rows) > 40 and say_box(
                    self, "Measure again",
                    "Measure %d frames again? Reading them takes about %.0f "
                    "seconds, and the window waits."
                    % (len(rows), len(rows) * 0.25),
                    QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                    QtWidgets.QMessageBox.Ok) != QtWidgets.QMessageBox.Ok:
                return
            method = self.doc.get("method", "circle")
            if method == "correlate":
                self._say("a correlate document measures the whole chain at "
                          "once; use ctrl+D", 5000)
                return
            step = self.step_measure
            found = 0
            for k in rows:
                p = self.paths[k]
                sc = step * io.luma_scale(p)
                luma = io.read_luma(p, step)
                c, agree, shape = detect_centre(luma, method, 0.15)
                f = self.frames[k]
                if c is None or agree < 0.35:
                    f["agreement"] = round(float(agree), 4)
                    f["source"] = alignment.HELD if f["cx"] is not None \
                        else alignment.NONE
                    self._refresh_item(k)
                    continue
                found += 1
                r_sun = shape.get("r", shape.get("a", 0.0))
                m = detect_moon(luma, c, r_sun, 0.15)
                f["cx"], f["cy"] = round(c[0] * sc, 3), round(c[1] * sc, 3)
                f["agreement"] = round(float(agree), 4)
                f["source"] = alignment.DETECTED
                if m is not None:
                    # Both were fitted on the same downsampled luma, so the
                    # offset scales the same way the positions do.
                    f["moon"] = {"dx": round((m[0][0] - c[0]) * sc, 3),
                                 "dy": round((m[0][1] - c[1]) * sc, 3),
                                 "source": alignment.DETECTED}
                self._refresh_item(k)
            self.dirty = True
            self.show_frame(self.i)
            self._say("measured %d frames again; %d came back detected"
                      % (len(rows), found), 5000)

        # --------------------------------------------------------- navigation
        def go(self, d, every=False):
            """Move `d` frames along the film, or along the list.

            `n` and `p` walk the film, so they step over what is marked skip:
            those frames are not in the output and stopping on them is a press
            spent on nothing. The list's own arrows walk the list, every row of
            it, because the list is the inventory -- and because otherwise a
            frame you had just excluded would be unreachable by keyboard, with
            no way back.
            """
            if every or self.i >= len(self.frames):
                return self.show_frame(self.i + d)
            k, step = self.i, (1 if d > 0 else -1)
            for _ in range(abs(d)):
                nxt = self._walk(k, step)
                if nxt is None:
                    break
                k = nxt
            self.show_frame(k)

        def next_untrusted(self):
            for n in range(self.i + 1, len(self.frames)):
                if self.frames[n].get("skip"):
                    continue
                if self.frames[n]["source"] in (alignment.HELD, alignment.NONE):
                    self.show_frame(n)
                    return
            self._say("no untrusted frame after this one", 2500)

        def cycle_mode(self):
            self.mode = {"overlay": "diff", "diff": "edges",
                         "edges": "plain", "plain": "overlay"}[self.mode]
            self.show_frame(self.i)

        def cycle_ref(self):
            """Cycle the relative modes, and let go of any pinned frame."""
            self.ref_pin = None
            self.ref = {"prev": "next", "next": "first",
                        "first": "none", "none": "prev"}[self.ref]
            self.show_frame(self.i)

        def pin_ref(self):
            """Hold the current frame as the reference until told otherwise."""
            self.ref_pin = self.i
            self.show_frame(self.i)
            self._say("reference pinned to %s"
                      % self.frames[self.i]["file"], 3000)

        def toggle_editing(self):
            self.editing = "moon" if self.editing == "align" else "align"
            self._show_mode()          # the mode line names what the arrows move
            self.show_frame(self.i)

        def _moon_of(self, k, make=False):
            """This frame's moon, seeded from the nearest one if asked for.

            The nearest frame's **offset**, which is the right seed for the
            same reason the detector carries it: between neighbours the two
            bodies barely move against each other -- 5.7 px a frame over a
            partial phase -- while where the moon sits goes wherever the mount
            went.

            A frame with no sun gets nothing. An offset from nothing is not a
            position, and there is no honest place to put a circle.
            """
            f = self.frames[k]
            m = f.get("moon")
            if m or not make or f["cx"] is None:
                return m
            for d in range(1, len(self.frames)):
                for j in (k - d, k + d):
                    if 0 <= j < len(self.frames) and self.frames[j].get("moon"):
                        near = self.frames[j]["moon"]
                        m = {"dx": near["dx"], "dy": near["dy"],
                             "source": alignment.MANUAL}
                        f["moon"] = m
                        return m
            return None

        def toggle_curve(self):
            self.show_curve = not self.show_curve
            self._paint_curve()
            self._paint_moon()

        def save(self):
            alignment.save(self.path, self.doc["input_dir"], self.doc["format"],
                           self.doc["method"], self.doc["frame_size"],
                           self.doc["target"], self.frames,
                           self.doc.get("size"))
            self.dirty = False
            self._status()
            self._say("saved %s" % os.path.basename(self.path), 2500)

        def about(self):
            about_box(self)

        # ------------------------------------------------------------- detect
        def _detect_page(self):
            """The Detect half of the Process dialog: its words and its options.

            Returns (widget, read), where `read` yields the options chosen. The
            asking is separated from the doing because there are three ways in
            -- the button, ctrl+D and ctrl+R -- and one dialog has to serve
            them all.
            """
            manual = sum(f["source"] == alignment.MANUAL for f in self.frames)
            moons = sum(1 for f in self.frames
                        if (f.get("moon") or {}).get("source")
                        == alignment.MANUAL)
            skipped = len(self.frames) - len(alignment.kept(self.doc))
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(10)
            lost = []
            if manual:
                lost.append("<b>%d hand-placed frame%s</b>"
                            % (manual, "" if manual == 1 else "s"))
            if moons:
                lost.append("%d hand-placed moon%s"
                            % (moons, "" if moons == 1 else "s"))
            head = QtWidgets.QLabel(
                '<div style="color:#8a8a90">Everything in the document is '
                'replaced by a fresh pass over all %d frames.%s'
                '<br>No image is touched either way.</div>'
                % (len(self.frames),
                   "<br>This throws away " + " and ".join(lost) + "."
                   if lost else ""))
            head.setTextFormat(QtCore.Qt.RichText)
            head.setWordWrap(True)
            v.addWidget(head)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("method"))
            how = QtWidgets.QComboBox()
            for m in ("circle", "ellipse", "correlate"):
                how.addItem(m)
            how.setCurrentText(self.doc.get("method", "circle"))
            how.setToolTip("circle: a truly circular disc.  ellipse: a disc "
                           "flattened near the horizon.  correlate: no disc "
                           "at all.")
            row.addWidget(how)
            row.addStretch(1)
            v.addLayout(row)
            unskip = QtWidgets.QCheckBox(
                "also bring back the %d frame%s marked skip"
                % (skipped, "" if skipped == 1 else "s"))
            unskip.setEnabled(bool(skipped))
            unskip.setToolTip("Leaving a frame out is a decision about the "
                              "film, not a measurement, so a re-run keeps it "
                              "unless you say otherwise.")
            v.addWidget(unskip)
            v.addStretch(1)
            return page, (lambda: (how.currentText(), unskip.isChecked()))

        def redetect(self, method=None, unskip=False):
            """Measure the whole clip again, discarding what is here.

            The expensive pass is meant to run once and be corrected by hand,
            which is why nothing else in this window can start it. But a first
            pass made with the wrong method leaves a document not worth
            correcting, and going back to a terminal to say so is a poor answer.

            The asking lives in the Process dialog, which names what is about
            to be lost -- hand-placed frames are the one thing no re-run can
            give back. Called with no method it asks there; called with one it
            gets on with it.
            """
            from . import detect as detect_cmd
            if method is None:
                got = self._ask("Detect", "Detect", *self._detect_page())
                if got is None:
                    return
                return self.redetect(*got)
            keep_skips = {} if unskip else {
                f["file"] for f in self.frames if f.get("skip")}
            paths = alignment.paths(self.doc)
            bar = QtWidgets.QProgressDialog("Measuring...", "Stop", 0,
                                            len(paths), self)
            place_centred(bar, self)
            bar.setWindowModality(QtCore.Qt.WindowModal)
            bar.setMinimumDuration(0)
            bar.setValue(0)
            state = {"n": 0, "done": False, "recs": None, "size": None,
                     "error": None}

            def tick(done, total):
                state["n"] = done
                if state.get("stop"):
                    raise _Cancelled()

            def work():
                try:
                    if method == "correlate":
                        state["recs"] = detect_cmd.detect_correlate(
                            paths, tuple(self.doc["target"]))
                    else:
                        state["recs"], (state["size"], _) = \
                            detect_cmd.detect_disc(
                                paths, method, self.step_measure, 0.15, 0.35,
                                None, None, None, verbose=False, progress=tick,
                                frame_size=tuple(self.doc["frame_size"]))
                except _Cancelled:
                    state["error"] = "stopped; the document is unchanged"
                except Exception as e:                       # noqa: BLE001
                    state["error"] = str(e)
                state["done"] = True

            threading.Thread(target=work, daemon=True).start()
            timer = QtCore.QTimer(self)

            def poll():
                bar.setValue(state["n"])
                if bar.wasCanceled():
                    state["stop"] = True
                if not state["done"]:
                    return
                timer.stop()
                bar.reset()
                if state["error"] or state["recs"] is None:
                    self._say(state["error"] or "detection failed", 8000)
                    return
                for r in state["recs"]:
                    if r["file"] in keep_skips:
                        r["skip"] = True
                self._remember("detect all")
                self.doc["method"] = method
                self.doc["frames"] = state["recs"]
                if state["size"]:
                    self.doc["size"] = state["size"]
                self._adopt(state["recs"])
                self._say("measured %d frames again with %s"
                          % (len(state["recs"]), method), 6000)
            timer.timeout.connect(poll)
            timer.start(100)

        def _adopt(self, recs):
            """Take a freshly measured set of records in place of the old."""
            self.frames = recs
            self.strip.frames = self.frames
            self._marked = self._ref_marked = None
            self.dirty = True
            for k in range(len(self.frames)):
                self._refresh_item(k)
            self.show_frame(min(self.i, len(self.frames) - 1))

        # ------------------------------------------------------------ process
        def _align_page(self):
            """The Align half of the Process dialog: its words and its options."""
            from . import apply as apply_cmd
            keep = alignment.kept(self.doc)
            out_dir = os.path.join(self.doc["input_dir"], apply_cmd.DEFAULT_OUT)
            size = _estimated_size(self.doc, len(keep))
            left = len(self.frames) - len(keep)
            already = len(apply_cmd.images_in(out_dir))
            room = ("" if not size else
                    "about %.1f GB" % (size / 1e9) if size >= 1e9 else
                    "about %.0f MB" % (size / 1e6))
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(10)
            head = QtWidgets.QLabel(
                '<div style="color:#8a8a90">Write %d aligned frames into %s'
                '<br>%s%sThe originals are only read.</div>'
                % (len(keep), out_dir,
                   "%d frame%s marked skip %s left out.<br>"
                   % (left, "" if left == 1 else "s",
                      "is" if left == 1 else "are") if left else "",
                   room + "<br>" if room else ""))
            head.setTextFormat(QtCore.Qt.RichText)
            head.setWordWrap(True)
            v.addWidget(head)
            box = QtWidgets.QCheckBox(
                "number the output as a contiguous sequence (frame_0001...)")
            box.setChecked(self.renumber)
            box.setToolTip(
                "Software that reads an image sequence stops at the first gap, "
                "and leaving a frame out makes one. A sequence.txt beside the "
                "frames says which original each one came from.")
            wipe = QtWidgets.QCheckBox(
                "delete the %d image%s already in the destination first"
                % (already, "" if already == 1 else "s"))
            wipe.setEnabled(bool(already))
            wipe.setChecked(self.clean and bool(already))
            wipe.setToolTip(
                "Without this, only frames this tool recognises as its own are "
                "cleared, which leaves another clip's frames sitting there. "
                "Images only: notes and folders are left alone.")
            v.addWidget(box)
            v.addWidget(wipe)
            v.addStretch(1)
            return page, (lambda: (box.isChecked(), wipe.isChecked()))

        def _ask(self, title, verb, page, read):
            """One dialog around a page of options.

            Returns what `read` says the answers are, or None if it was
            cancelled. It reads them here, before returning, because the
            widgets do not outlive this call: the dialog is a local, so the
            last Python reference to it dies on the way out and Qt takes the
            page's children down with it. Reading afterwards raised
            `Internal C++ object (QComboBox) already deleted` on every Detect.

            The two long passes each get their own again. They shared a chooser
            while they shared a button; with a row each in the grid the chooser
            was asking a question the grid had already answered.
            """
            d = QtWidgets.QDialog(self)
            d.setWindowTitle(title)
            place_centred(d, self)
            v = QtWidgets.QVBoxLayout(d)
            v.setSpacing(10)
            page.setMinimumWidth(470)
            v.addWidget(page)
            bb = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok
                | QtWidgets.QDialogButtonBox.Cancel)
            bb.button(QtWidgets.QDialogButtonBox.Ok).setText(verb)
            bb.accepted.connect(d.accept)
            bb.rejected.connect(d.reject)
            v.addWidget(bb)
            if d.exec() != QtWidgets.QDialog.Accepted:
                return None
            return read()

        def process(self, renumber=None, clean=None):
            """Run `apply` on this document without leaving the editor.

            The three steps stay separate in every way that matters -- this
            still writes only what the document says, into a directory of its
            own, and cannot touch an original. What it removes is the need to
            leave the window, retype the path, and come back.

            It saves first. Processing a document that differs from the one on
            screen would produce frames that match neither.
            """
            from . import apply as apply_cmd
            if renumber is None:
                got = self._ask("Align", "Align", *self._align_page())
                if got is None:
                    return
                return self.process(*got)
            if self.dirty:
                self.save()
            keep = alignment.kept(self.doc)
            out_dir = os.path.join(self.doc["input_dir"], apply_cmd.DEFAULT_OUT)
            self.renumber, self.clean = bool(renumber), bool(clean)

            bar = QtWidgets.QProgressDialog("Writing frames...", "Stop", 0,
                                            len(keep), self)
            place_centred(bar, self)
            bar.setWindowModality(QtCore.Qt.WindowModal)
            bar.setMinimumDuration(0)
            bar.setValue(0)

            state = {"error": None, "done": False, "n": 0}

            def tick(done, total):
                # Called from the worker thread: only touch `state` here.
                state["n"] = done
                if state.get("stop"):
                    raise _Cancelled()

            def work():
                try:
                    apply_cmd.apply_doc(self.doc, out_dir, verbose=False,
                                        progress=tick,
                                        renumber=self.renumber,
                                        clean=self.clean)
                except _Cancelled:
                    state["error"] = "stopped; %d frames were written" % state["n"]
                except Exception as e:                      # noqa: BLE001
                    state["error"] = str(e)
                state["done"] = True

            thread = threading.Thread(target=work, daemon=True)
            thread.start()
            timer = QtCore.QTimer(self)

            def poll():
                bar.setValue(state["n"])
                if bar.wasCanceled():
                    state["stop"] = True
                if state["done"]:
                    timer.stop()
                    bar.reset()
                    if state["error"]:
                        self._say(state["error"], 8000)
                    else:
                        self._say("wrote %d frames to %s/"
                                  % (len(keep), os.path.basename(out_dir)), 8000)
            timer.timeout.connect(poll)
            timer.start(100)

        # -------------------------------------------------------------- input
        def keyPressEvent(self, e):
            d = {QtCore.Qt.Key_Left: (-1, 0), QtCore.Qt.Key_Right: (1, 0),
                 QtCore.Qt.Key_Up: (0, -1), QtCore.Qt.Key_Down: (0, 1)}
            if e.key() in d:
                ux, uy = d[e.key()]
                m = e.modifiers()
                alt = bool(m & QtCore.Qt.AltModifier)
                big = m & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)
                # alt used to be a tenth of a step, which is the one thing
                # this ladder cannot do any more. It is now the smallest move
                # there is -- one photosite -- which is what a person reaching
                # for "finer" actually wants.
                # One photosite, or ten of them with ctrl. `alt` used to
                # mean "finer"; there is nothing finer than a photosite that
                # does not resample the frame, so it means the same as no
                # modifier and is simply harmless.
                d = self.grid * (10.0 if big else 1.0)
                self.nudge(ux * d, uy * d)
                e.accept()
                return
            if e.key() == QtCore.Qt.Key_Space and not e.isAutoRepeat():
                self._set_peek(True)
                e.accept()
                return
            if e.key() == QtCore.Qt.Key_Escape:
                self._log("focus", to="image")
                self.view.setFocus()
                e.accept()
                return
            if e.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                self.toggle_skip()
                e.accept()
                return
            if e.key() in (QtCore.Qt.Key_PageDown, QtCore.Qt.Key_PageUp):
                self.go(20 if e.key() == QtCore.Qt.Key_PageDown else -20)
                e.accept()
                return
            super().keyPressEvent(e)

        def eventFilter(self, obj, e):
            if (e.type() in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease)
                    and e.key() == QtCore.Qt.Key_Space
                    and not e.isAutoRepeat()
                    and self.isActiveWindow()
                    and QtWidgets.QApplication.activeModalWidget() is None):
                self._set_peek(e.type() == QtCore.QEvent.KeyPress)
                return True
            return super().eventFilter(obj, e)

        def _set_peek(self, on):
            j = self.ref_index()
            if on and (j is None or j == self.i):
                self._say("no reference frame to show", 1500)
                return
            self._peek = on
            self.show_frame(self.i)

        def keyReleaseEvent(self, e):
            if e.key() == QtCore.Qt.Key_Space and not e.isAutoRepeat():
                self._set_peek(False)
                e.accept()
                return
            super().keyReleaseEvent(e)

        def closeEvent(self, e):
            # Before anything else, and before the cancel below can send us
            # back: the size is worth keeping whatever the answer turns out to
            # be, and by the time Qt is taking the window apart it is gone.
            keep_size(self, "editor")
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
                # Focus keeps moving while Qt takes the window apart, and a
                # redraw of a half-deleted window is a traceback on the way out.
                app.focusChanged.disconnect(self._on_focus)
            if not self.dirty:
                if self.log is not None:
                    self.log.close()
                return e.accept()
            b = QtWidgets.QMessageBox(self)
            b.setText("The document has unsaved corrections.")
            b.setStandardButtons(QtWidgets.QMessageBox.Save
                                 | QtWidgets.QMessageBox.Discard
                                 | QtWidgets.QMessageBox.Cancel)
            b.setDefaultButton(QtWidgets.QMessageBox.Save)
            place_centred(b, self)
            r = b.exec()
            if r == QtWidgets.QMessageBox.Save:
                self.save()
            if r == QtWidgets.QMessageBox.Cancel:
                return e.ignore()
            if self.log is not None:
                self.log.close("saved" if r == QtWidgets.QMessageBox.Save
                               else "discarded")
            e.accept()

    return Editor


def edit(document, step=4, serve=None, log=None, retain=None):
    """Open the editor on a document and block until its window closes."""
    QtCore, QtGui, QtWidgets = _widgets()
    doc = alignment.load(document)
    missing = [p for p in alignment.paths(doc) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("%d frames named in the document are missing, "
                                "first %s" % (len(missing), missing[0]))
    app = _app()
    _dark(app)
    app.setWindowIcon(app_icon())
    win = _make_editor()(doc, document, step, log, retain)
    win.show()
    if log:
        print("journal: %s" % log, file=sys.stderr)
    if serve is not None:
        from .remote import serve as _serve
        port = _serve(win, serve)
        print("control socket on http://127.0.0.1:%d  "
              "(/state /shot /key /select /call)" % port, file=sys.stderr)
    app.exec()
    return 0
