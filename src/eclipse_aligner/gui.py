#!/usr/bin/env python3
"""Inspect and correct an alignment document by hand.

The detector is right about most frames and wrong about a few, and the few are
predictable: an over-exposed frame with no limb, a sun deformed on the horizon,
the moment a filter comes off and the scene changes entirely. Those are worth a
human's thirty seconds each, not another heuristic.

The frame is shown as a **red/green overlay** -- the frame being edited in red,
a reference in green. Aligned, the two fuse to grey and the subject looks
neutral; misaligned, its edges fringe red on one side and green on the other,
which the eye reads far faster than a difference image.

Over it sits the **conic that was fitted**, in cyan, anchored to the recorded
centre. It is a template rather than a readout: it stays on the crosshair while
the disc slides underneath, so seating it on the limb *is* placing the centre --
which is the only way to place one on a frame nothing could measure.

Nudging or resizing marks a frame `manual`, and `apply` then treats it exactly
like a detected one. Nothing here writes an image; the document is the only
output.

    arrows        move 1 px           ctrl+arrows   move 10 px
    alt+arrows    move 0.1 px

Every command is laid out in the window itself, grouped by what it reaches:
view, navigation, frame, range, all. The line above the frame carries what you
read -- which frame, what the keys will do, what it is compared with, how it
was shot -- with the view modes and the keys that change them.

Where an operation has both bodies, **the plain key is the sun and shift is
the moon**: `i` / `shift+i` to interpolate, `s` / `shift+s` to set the radius.

`ctrl+z` takes back the last thing that changed the document and `shift+ctrl+z`
puts it back; the row says what it will undo. Fifty steps are kept.

Keys with no entry in the grid:

    pgup/pgdn     twenty frames at a time
    [ ]           pixel step for the arrows
    esc           focus back to the frame
    click         a name in the list, or anywhere on the strip
    wheel         zoom about the cursor       drag  pan
    shift+drag    move whatever `m` selects, right on the frame

Select several frames and the arrows move all of them at once, as long as the
frame on screen is one of the selected.
"""

import os
import sys

from . import alignment
from . import journal

HELP = __doc__.split("    arrows")[1]

# What `detect --step` defaults to. The first pass has to measure exactly as
# well as the command would, or the two disagree about the same clip.
MEASURE_STEP = 2


def add_args(ap):
    """The options of `eclipse-aligner gui`."""
    ap.add_argument("document", nargs="?",
                    help="The clip directory, or the document itself. Asked "
                         "for if left out, which is how the window opens when "
                         "it is launched from the Finder with nothing to go "
                         "on.")
    ap.add_argument("--step", type=int, default=4,
                    help="Downsample factor for display (default 4).")
    ap.add_argument("--serve", nargs="?", type=int, const=8765, default=None,
                    metavar="PORT",
                    help="Open a control socket on 127.0.0.1 (default port "
                         "8765) so the window can be driven and photographed "
                         "from outside. For reproducing a problem with "
                         "somebody, not for daily use.")
    ap.add_argument("--log", metavar="FILE",
                    help="Where to record the session's actions. By default "
                         "beside the document, as <document>.log.jsonl.")
    ap.add_argument("--no-log", action="store_true",
                    help="Record nothing. The session leaves no trace but the "
                         "document itself.")
    ap.add_argument("--log-retain", default=journal.DEFAULT_RETAIN,
                    metavar="AGE",
                    help="How long a session stays in the log; older ones are "
                         "dropped when the editor opens (default %s). 1d, "
                         "12h, 2w; a bare number is days. `off` keeps every "
                         "session for ever."
                         % journal.DEFAULT_RETAIN)


def _ask_for_clip():
    """A folder chooser, for a window started with nothing to open.

    Double-clicking an icon passes no arguments, so without this the app would
    have to fail at the one moment it has nothing to say. It returns the
    directory; `document_path` turns it into the document inside it.
    """
    from .editor import _widgets, _dark, _app, app_icon
    _, _, QtWidgets = _widgets()
    app = _app()
    _dark(app)
    app.setWindowIcon(app_icon())
    return QtWidgets.QFileDialog.getExistingDirectory(
        None, "Open a clip — the folder of frames") or None


def _first_pass(clip, doc):
    """Measure a clip that has never been measured, and say so while it runs.

    A folder opened for the first time has no document, and the editor cannot
    show anything without one. Refusing with "run `eclipse-aligner detect`
    first" is a fair sentence in a terminal and no answer at all to somebody
    who dropped a folder on the window, so the first pass just happens.

    It is not a silent one. The pass is minutes long on a real clip, so it
    runs behind a progress bar with a Stop button; stopping leaves no document
    behind, which is the same promise `detect` makes.

    Defaults, and the editor is where they get overruled. Choosing a method
    before seeing a single frame is a question nobody can answer; `ctrl+D`
    asks it again with the frames in view. `MEASURE_STEP` is `detect`'s own
    and not this command's `--step`, which is how coarsely the *display*
    downsamples -- two different numbers that would have quietly measured the
    clip at half the resolution the command line gives it.

    Returns True when there is a document to open.
    """
    import threading
    from . import detect as detect_cmd
    from . import image_io as io
    from .editor import (_widgets, _dark, _app, app_icon, place_centred,
                         say_box)
    QtCore, _, QtWidgets = _widgets()
    app = _app()
    _dark(app)
    app.setWindowIcon(app_icon())

    try:
        paths = io.frames(clip)
    except ValueError as e:
        paths, why = [], str(e)
    else:
        why = "No EXR, DNG, camera raw or JPEG frames in this folder."
    if not paths:
        say_box(None, "Nothing to measure", why,
                icon=QtWidgets.QMessageBox.Warning)
        return False

    w, h = detect_cmd.frame_size(paths[0])
    target = (w / 2.0, h / 2.0)
    bar = QtWidgets.QProgressDialog(
        "Measuring %d frames for the first time…" % len(paths),
        "Stop", 0, len(paths), None)
    bar.setWindowTitle("New clip")
    bar.setWindowModality(QtCore.Qt.ApplicationModal)
    bar.setMinimumDuration(0)
    bar.setValue(0)
    place_centred(bar)
    state = {"n": 0, "done": False, "recs": None, "size": {}, "error": None}

    class _Stop(Exception):
        pass

    def tick(done, total):
        state["n"] = done
        if state.get("stop"):
            raise _Stop()

    def work():
        try:
            state["recs"], (state["size"], _) = detect_cmd.detect_disc(
                paths, "ellipse", MEASURE_STEP, 0.15, 0.35, None, None, None,
                verbose=False, progress=tick, frame_size=(w, h))
        except _Stop:
            state["error"] = ""
        except Exception as e:                               # noqa: BLE001
            state["error"] = str(e)
        state["done"] = True

    threading.Thread(target=work, daemon=True).start()
    while not state["done"]:
        bar.setValue(state["n"])
        if bar.wasCanceled():
            state["stop"] = True
        app.processEvents()
        QtCore.QThread.msleep(40)
    bar.reset()
    if state["recs"] is None:
        if state["error"]:
            say_box(None, "Could not measure", state["error"],
                    icon=QtWidgets.QMessageBox.Warning)
        return False
    alignment.save(doc, clip, io.kind(paths[0]), "ellipse", (w, h), target,
                   state["recs"], state["size"])
    return True


def run(args):
    target = args.document
    if target is None:
        # Launched with nothing to open -- from the Dock, or from a bare
        # `gui`. The welcome window says what a clip is, takes one dropped on
        # it, and remembers the last few.
        from .welcome import ask
        target = ask()
        if target is None:
            return 0
    doc = alignment.document_path(target)
    if not os.path.isfile(doc) and os.path.isdir(target):
        if not _first_pass(target, doc):
            return 1
    log = None if args.no_log else (args.log or journal.default_path(doc))
    try:
        retain = journal.parse_retain(args.log_retain)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    from .welcome import remember
    remember(os.path.dirname(doc) or ".")
    return edit(doc, args.step, args.serve, log, retain)


def edit(document, step=4, serve=None, log=None, retain=None):
    """Open the editor on a document and block until its window closes.

    The window itself lives in `editor`, which imports Qt. Keeping that out of
    this module is what lets `cli` import all three commands at start-up on a
    machine where `detect` and `apply` are all that is installed.
    """
    from .editor import edit as _edit
    print(HELP, file=sys.stderr)
    return _edit(document, step, serve, log, retain)
