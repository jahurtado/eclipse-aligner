"""The file that sits between measuring and moving.

Detection is expensive and imperfect; applying a shift is cheap and
destructive. Putting a plain JSON document between them means the slow pass
runs once, a human can correct the handful of frames the detector got wrong,
and only then does anything get written.

    eclipse-aligner detect CLIP_DIR            # slow, read-only
    eclipse-aligner gui    CLIP_DIR            # fix the few bad ones
    eclipse-aligner apply  CLIP_DIR -o OUT     # writes

The document is written into the clip's own directory as `alignment.json`, so
every command takes the clip and finds it. `-o` overrides where `detect` puts
it, and the other two also accept the document's path directly.

Each frame carries where its subject was found and *how that was arrived at*,
which is the part that matters when a human reads the file:

    detected  the estimator found it and the fit agreed
    held      nothing usable in the frame, so the previous position was carried
    manual    a person placed it
    none      nothing usable and nothing to carry (only before the first hit)

**A frame holds two independent numbers: an alignment and an offset.** The
alignment is `cx, cy` -- where the sun was found, and therefore how far the
picture has to move for the sun to land on `target`. The offset is
`moon.dx, dy` -- where the moon sits *relative to that sun*. They are stored
separately because they are separately true, and because every operation in
the editor writes one or the other: interpolating the alignment and
interpolating the offsets commute, and did not while the moon was kept in
absolute pixels and one command silently dragged the other's answer along.

It follows that a frame with no sun has no moon. An offset from nothing is not
a position, and the editor makes one the moment a sun is placed.

**The size of each body belongs to the clip, not to the frame.** Neither the
sun nor the moon changes apparent size over an afternoon, so `size` carries one
shape for the sun and one radius for the moon and every frame uses them. Frames
hold only where each body *is*. Measured on a real 525-frame document that had
per-frame radii: the moon had converged to a single value across all of them
and the sun to one value on 470 of 489 -- the other 19 were stragglers a manual
"make them all the same" command had never reached, which is that model's
failure mode rather than a measurement.

`agreement` is the fraction of the subject's outline the fitted conic actually
explains. It is what separates a real detection from a confident-looking lie:
an over-exposed frame whose "limb" is the image border scores 0.04 where a
clean disc scores 1.00.
"""

import json
import os

VERSION = 3
DETECTED, HELD, MANUAL, NONE = "detected", "held", "manual", "none"
DEFAULT_NAME = "alignment.json"


def document_path(target):
    """The document, given either its own path or the clip directory holding it.

    The document lives beside the frames it describes, so naming the clip is
    enough to find it. There is one alignment per clip -- the frames do not
    move, so a second opinion about where they are is not a thing worth
    keeping -- which is what makes a fixed name safe rather than lossy.
    """
    if os.path.isdir(target):
        return os.path.join(target, DEFAULT_NAME)
    return target


def size_of(doc, body="sun"):
    """The clip's shape for a body: `{"r"}` or `{"a","b","angle"}`, or `{}`.

    One lookup rather than each caller reaching into the document, because
    every drawing, every readout and every resize wants the same number and
    they must not be able to disagree about where it lives.
    """
    return dict((doc.get("size") or {}).get(body) or {})


def save(path, input_dir, fmt, method, frame_size, target, frames, size=None):
    """Write an alignment document. `frames` is a list of dicts."""
    doc = {
        "version": VERSION,
        "input_dir": os.path.abspath(input_dir),
        "format": fmt,
        "method": method,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],   # w, h
        "target": [float(target[0]), float(target[1])],           # x, y
        "size": size or {},
        "frames": frames,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    return doc


def load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "no alignment document at %s -- run `eclipse-aligner detect` on "
            "the clip first" % path)
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("version") == 1:
        _lift_sizes(doc)
    if doc.get("version") == 2:
        _lift_moons(doc)
    if doc.get("version") != VERSION:
        raise ValueError("%s is version %s, expected %d"
                         % (path, doc.get("version"), VERSION))
    return doc


def _median(values):
    v = sorted(values)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def _lift_sizes(doc):
    """Version 1 kept a radius on every frame. Lift them into one for the clip.

    The median of the frames that were actually fitted, per key, and never the
    hand-held ones: a `held` frame's shape was copied from a neighbour and
    voting with it just weights that neighbour twice.

    Done on load rather than by a migration command, so an old document opens
    and works. It is rewritten in the new shape the next time anything saves,
    and nothing is lost that `detect` could not measure again.
    """
    fr = doc.get("frames") or []
    sun, moon = {}, {}
    for key in ("r", "a", "b", "angle"):
        vals = [f["shape"][key] for f in fr
                if f.get("source") == DETECTED and f.get("shape")
                and key in f["shape"]]
        m = _median(vals)
        if m is not None:
            sun[key] = round(float(m), 3)
    vals = [f["moon"]["r"] for f in fr
            if (f.get("moon") or {}).get("source") in (DETECTED, MANUAL)
            and "r" in f["moon"]]
    m = _median(vals)
    if m is not None:
        moon["r"] = round(float(m), 3)
    doc["size"] = {k: v for k, v in (("sun", sun), ("moon", moon)) if v}
    for f in fr:
        f.pop("shape", None)
        if f.get("moon"):
            f["moon"].pop("r", None)
    # 2, not VERSION: this is one step of the chain, and the next one still
    # has to see a version-2 document to know it has work to do.
    doc["version"] = 2


def _lift_moons(doc):
    """Version 2 kept the moon in absolute pixels. Turn each into an offset.

    Arithmetic only -- `dx = moon.cx - cx` -- so nothing is lost and nothing
    is guessed. A moon on a frame with no sun is dropped: it had no offset to
    become, and the position it held was the frame's centre, which is a
    placeholder rather than a measurement.

    Done on load rather than by a migration command, so an old document opens
    and works. It is rewritten in the new shape the next time anything saves.
    """
    for f in doc.get("frames") or []:
        m = f.get("moon")
        if not m:
            continue
        if f.get("cx") is None or "cx" not in m:
            f.pop("moon", None)
            continue
        m["dx"] = round(m.pop("cx") - f["cx"], 3)
        m["dy"] = round(m.pop("cy") - f["cy"], 3)
    doc["version"] = VERSION


def frame_record(name, centre, agreement, source, moon=None):
    """One frame's line in the document: where each body is, and nothing else.

    No radius here. The size of a body is the clip's, in `size`, because
    neither changes over an afternoon and one good measurement beats each
    frame's own -- see the module docstring for what per-frame radii actually
    did to a real document.

    `moon` is the occulting body when it could be measured: `{dx, dy, source}`,
    full-frame pixels **from this frame's sun**. It is the other half of the
    same outline, and it is what makes a steady correction possible where the
    sun's own limb is too thin to trust -- the offset crosses in a straight
    line, and a hand does not. Kept as an offset because that is the quantity
    that behaves: over one partial phase it moved 5.7 px a frame and held that
    line to 0.9 px rms across 75 frames, while where the moon *sits* wanders
    with the mount underneath it.
    """
    rec = {
        "file": name,
        "cx": None if centre is None else round(float(centre[0]), 3),
        "cy": None if centre is None else round(float(centre[1]), 3),
        "agreement": None if agreement is None else round(float(agreement), 4),
        "source": source,
    }
    if moon and centre is not None:
        rec["moon"] = {k: (v if isinstance(v, str) else round(float(v), 3))
                       for k, v in moon.items()}
    return rec


def moon_xy(f):
    """Where the frame's moon is, in full-frame pixels, or None.

    The document stores the offset; almost everything that draws or measures
    wants the position. One place does the addition so that no caller has to
    remember which of the two it is holding.
    """
    m = f.get("moon")
    if not m or f.get("cx") is None:
        return None
    return (round(f["cx"] + m["dx"], 3), round(f["cy"] + m["dy"], 3))


def set_moon_xy(f, xy, source=None):
    """Put the frame's moon at an absolute position, storing the offset.

    Refuses on a frame with no sun, because then there is nothing to be
    offset from -- see the module docstring.
    """
    if f.get("cx") is None:
        return None
    m = f.setdefault("moon", {})
    m["dx"] = round(float(xy[0]) - f["cx"], 3)
    m["dy"] = round(float(xy[1]) - f["cy"], 3)
    if source is not None:
        m["source"] = source
    return m


def paths(doc):
    """Absolute path of every frame, in order."""
    return [os.path.join(doc["input_dir"], f["file"]) for f in doc["frames"]]


def kept(doc):
    """Indices of the frames that go to the output.

    A frame carrying `skip` is left out of it. Nothing is deleted: the original
    is never touched, the record keeps its measurement, and clearing the flag
    brings it back. It is a decision about the film, not about the shoot.
    """
    return [i for i, f in enumerate(doc["frames"]) if not f.get("skip")]


def shifts(doc):
    """Per-frame (dy, dx) to bring the subject onto the target. NaN where unknown."""
    import numpy as np
    tx, ty = doc["target"]
    out = []
    for f in doc["frames"]:
        if f["cx"] is None:
            out.append((np.nan, np.nan))
        else:
            out.append((ty - f["cy"], tx - f["cx"]))
    return np.array(out)


def summary(doc):
    n = {}
    for f in doc["frames"]:
        n[f["source"]] = n.get(f["source"], 0) + 1
    out = ", ".join("%d %s" % (v, k) for k, v in sorted(n.items()))
    skipped = len(doc["frames"]) - len(kept(doc))
    return out + (", %d skipped" % skipped if skipped else "")
