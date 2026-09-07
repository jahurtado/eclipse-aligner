#!/usr/bin/env python3
"""Measure where the subject is in every frame, and write it down.

Detection is the slow, fallible half of aligning a timelapse; moving pixels is
the fast, destructive half. This does only the first, and its output is a
readable JSON document a person can inspect, correct in the GUI, and only then
hand to `apply`.

Three methods, and which is right is measured, not chosen:

    ellipse    a disc flattened by refraction near the horizon (default)
    circle     a genuinely circular disc: the moon at any phase, a high sun
    correlate  no disc at all: totality, a landscape

Frames are EXR, DNG, NEF, CR2, CR3 or JPEG. A mosaic is measured on one
green plane of its
mosaic, which is half resolution and ample for finding a disc a thousand
pixels across.
"""

import os
import sys

import numpy as np

from . import alignment
from . import image_io as io
from .centre_disc import detect_centre, detect_moon


def frame_size(path):
    """(width, height) of a frame in its own full-resolution pixels."""
    luma = io.read_luma(path, 1)
    sc = io.luma_scale(path)
    return luma.shape[1] * sc, luma.shape[0] * sc


def _carried_offset(last_offset):
    """A moon for a frame that could not have one measured.

    The last measured **offset**, unchanged: between neighbours the two bodies
    barely move against each other -- 5.7 px a frame over a partial phase --
    so the last one is the best guess there is and a person only has to nudge
    it. Carrying the offset rather than the absolute position also means the
    guess follows this frame's own sun instead of the previous frame's.

    With nothing to carry -- nothing measured yet in the clip -- the moon
    starts on top of the sun, which is visibly a placeholder and is the
    honest thing to show when nothing whatever is known.

    Its `source` says `held` or `none`, never `detected`. A circle nobody
    fitted must not be able to pass for one that was: `shift+i` counts only
    measured moons, and the editor draws the rest faintly.
    """
    if last_offset is not None:
        m = dict(last_offset)
        m["source"] = alignment.HELD
        return m
    return {"dx": 0.0, "dy": 0.0, "source": alignment.NONE}


def _principal(shape):
    """The radius a shape is ranked by: a circle's `r`, an ellipse's major."""
    return shape.get("r", shape.get("a", 0.0))


def _pick(cands):
    """The frame a body's size comes from: (shape, name, quality) or None.

    Best score first, and then **the median of everything tied with it**, which
    matters more than it sounds. Both scores saturate: on a 30-frame sample the
    sun's agreement reached 1.000 on 7 frames and the moon's consensus on 19 of
    20, so "the best frame" is usually a large group and taking whichever came
    first is taking whichever was shot first. That pick landed on 1217.0 px
    where the tied group's median was 1189.5 -- the top of a 6.4% spread,
    chosen by nothing.

    So: rank, keep the ties, and among them take the frame whose radius is the
    middle one. Still one real frame, with its name, and not a number no
    exposure ever produced.
    """
    if not cands:
        return None
    best = max(q for q, _, _ in cands)
    tied = [c for c in cands if q_ok(c[0], best)]
    tied.sort(key=lambda c: _principal(c[1]))
    return tied[len(tied) // 2]


def q_ok(q, best):
    """Tied with the best, allowing for a score that is nearly saturated."""
    return q >= best - 0.01


def _clip_size(suns, moons):
    """One shape for the sun and one radius for the moon, each from its **own**
    best frame.

    Not the median of every fit. A body's apparent size does not change over an
    afternoon, but the *estimate* degrades badly as the lit arc thins -- across
    a partial phase the sun read 144.2 px falling to 143.7 and the moon 149.4
    to 147.9, and that 1% is the fit going soft, not the sun shrinking. A
    median over everything mixes the good readings with the degrading ones;
    somewhere in the clip there is one frame that knows the radius better than
    the whole clip does together.

    Two frames, not one, because the two limbs are best drawn at different
    moments: **the sun's when little of it is covered, the moon's when it is
    deep across the disc**, which is the same event an hour apart. Each body is
    ranked by its own score -- `agreement` for the sun, the fraction of the
    discarded arc the circle explains for the moon -- because ranking moons by
    how well the sun fitted picks the wrong frame. `_pick` settles the ties.

    Both are a starting point, not a verdict: the editor can take either from
    whichever frame a person judges best.
    """
    size, from_ = {}, {}
    for body, cands in (("sun", suns), ("moon", moons)):
        got = _pick(cands)
        if got is None:
            continue
        q, shape, name = got
        size[body] = {k: round(float(v), 3) for k, v in shape.items()}
        from_[body] = (name, q, sum(1 for c in cands if q_ok(c[0], q)))
    return size, from_


def detect_disc(paths, method, step, frac, min_agreement, rmin, rmax,
                known=None, verbose=True, progress=None, frame_size=(0, 0)):
    """Per-frame records using the disc estimators.

    `known` maps a frame's name to the record an earlier run left for it. Those
    frames are not measured again -- not re-doing the expensive pass is the
    whole point of keeping a document -- but they still feed the `held` chain,
    so a new frame that follows a hand-corrected one inherits the corrected
    position instead of a stale measurement.
    """
    known = known or {}
    recs = []
    suns, moons = [], []
    last = None
    last_offset = None
    for i, p in enumerate(paths):
        if progress is not None:
            progress(i, len(paths))
        name = os.path.basename(p)
        if name in known:
            r = known[name]
            if r["cx"] is not None:
                last = (r["cx"], r["cy"])
            recs.append(r)
            continue
        sc = step * io.luma_scale(p)
        luma = io.read_luma(p, step)
        c, agree, shape = detect_centre(luma, method, frac,
                                        rmin=rmin, rmax=rmax)
        # The occulting body, from the arc the hull discarded. Free at this
        # point: the outline is already computed and the moon is the half of it
        # that the sun's fit throws away. Measured or not, **every frame gets
        # one**, so there is always a circle to take hold of -- carried from
        # the last one that was measured, and saying so.
        r_sun = shape.get("r", shape.get("a", 0.0)) if shape else 0.0
        # Only from a sun the fit is trusted on. The moon's radius is checked
        # against the sun's, so a bogus sun licenses a bogus moon: on an
        # over-exposed frame whose "limb" is the image border, the two agree
        # with each other and both are wrong.
        trusted = c is not None and r_sun and agree >= min_agreement
        m = detect_moon(luma, c, r_sun, frac) if trusted else None
        if m is not None:
            # The offset, not the position: the document keeps the moon
            # relative to its own frame's sun, and both are in the same
            # downsampled coordinates here.
            last_offset = {"dx": (m[0][0] - c[0]) * sc,
                           "dy": (m[0][1] - c[1]) * sc,
                           "source": alignment.DETECTED}
            moon = dict(last_offset)
            moons.append((m[2], {"r": m[1] * sc}, name))
        else:
            moon = _carried_offset(last_offset)

        if c is not None and agree >= min_agreement:
            last = (c[0] * sc, c[1] * sc)
            # the fit was made on the downsampled luma; the document speaks in
            # full-frame pixels, and only the lengths scale, not the angle
            shape = {k: (v * sc if k in ("r", "a", "b") else v)
                     for k, v in shape.items()}
            # Kept aside, not written on the frame: the clip takes its size
            # from the best of these, and a per-frame copy of a number that
            # belongs to the clip is a copy that can go stale.
            suns.append((agree, shape, name))
            recs.append(alignment.frame_record(name, last, agree,
                                               alignment.DETECTED, moon))
        elif last is not None:
            recs.append(alignment.frame_record(name, last, agree,
                                               alignment.HELD, moon))
        else:
            recs.append(alignment.frame_record(name, None, agree,
                                               alignment.NONE, moon))
        if verbose and (i + 1) % 25 == 0:
            print("    %d/%d" % (i + 1, len(paths)), file=sys.stderr)
    return recs, _clip_size(suns, moons)


def detect_correlate(paths, target):
    """Per-frame records from chained phase correlation.

    There is no disc to find, so the "centre" recorded is the virtual one that
    makes the frame's shift come out right: target minus the measured offset.
    That keeps one document format for every method.
    """
    from . import register_exr
    offs, quality, good = register_exr.measure(paths)
    offs = register_exr.to_reference(offs, "middle")
    recs = []
    for p, (dy, dx), q, g in zip(paths, offs, np.r_[1.0, quality], np.r_[True, good]):
        recs.append(alignment.frame_record(
            os.path.basename(p), (target[0] - dx, target[1] - dy), float(q),
            alignment.DETECTED if g else alignment.HELD))
    return recs


def add_args(ap):
    """The options of `eclipse-aligner detect`."""
    ap.add_argument("input_dir", help="Folder of EXR, DNG, NEF, CR2, CR3 or JPEG frames.")
    ap.add_argument("-o", "--out",
                    help="Where to write the alignment document. By default it "
                         "goes into the clip's own directory as "
                         "alignment.json, so the other commands find it from "
                         "the clip alone.")
    ap.add_argument("--method", default="ellipse",
                    choices=("ellipse", "circle", "correlate"),
                    help="ellipse: disc flattened near the horizon (default). "
                         "circle: a truly circular disc (moon, high sun). "
                         "correlate: no disc -- totality, landscape.")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="Bright-mask threshold as a fraction of the frame's "
                         "99.999th percentile (default 0.15).")
    ap.add_argument("--step", type=int, default=2,
                    help="Downsample factor for measuring (default 2).")
    ap.add_argument("--min-agreement", type=float, default=0.35,
                    help="Below this fraction of the outline explained by the "
                         "fit, the frame is not trusted and holds the previous "
                         "position instead (default 0.35).")
    ap.add_argument("--rmin", type=float, help="Smallest plausible radius, in "
                    "measuring-space px. From a reference frame; bounds the "
                    "search without pinning the radius.")
    ap.add_argument("--rmax", type=float, help="Largest plausible radius.")
    ap.add_argument("--target", nargs=2, type=float, metavar=("X", "Y"),
                    help="Where the subject should end up (default: frame centre).")
    ap.add_argument("--force", action="store_true",
                    help="Measure every frame again, throwing away what the "
                         "document said -- hand corrections included. Without "
                         "it, a re-run only measures frames the document does "
                         "not already have.")
    ap.add_argument("--gui", action="store_true",
                    help="Open the editor on the document once it is written. "
                         "A convenience, not a merge of the two steps: the "
                         "editor still writes no image, and applying stays a "
                         "separate command.")


def _skips(out):
    """Frames an earlier run was told to leave out of the output.

    Carried across every re-run, `--force` included: excluding a frame is a
    decision about the film, not a measurement, so re-measuring has no business
    undoing it. Clearing the flag in the editor is what brings a frame back.
    """
    if not os.path.isfile(out):
        return set()
    try:
        return {f["file"] for f in alignment.load(out)["frames"]
                if f.get("skip")}
    except (ValueError, KeyError):
        return set()


def _already_measured(out, args, fmt, frame_size, target):
    """What an earlier run left in the document, or {} if there is nothing to
    reuse. None means stop: the document exists but describes another job.

    A document carries one `method` and one `target` for all its frames, so
    topping it up with numbers obtained a different way would make it lie about
    itself. Better to say so and let `--force` rewrite it whole.
    """
    if not os.path.isfile(out):
        return {}
    if args.force:
        n = sum(f["source"] == alignment.MANUAL
                for f in alignment.load(out)["frames"])
        if n:
            print("  ! --force: discarding %d hand-corrected frame%s"
                  % (n, "" if n == 1 else "s"), file=sys.stderr)
        return {}
    doc = alignment.load(out)
    for what, was, now in (("method", doc["method"], args.method),
                           ("format", doc["format"], fmt),
                           ("frame size", tuple(doc["frame_size"]),
                            tuple(frame_size)),
                           ("target", tuple(doc["target"]), tuple(target))):
        if was != now:
            print("%s was measured with %s %s, not %s; --force rewrites it"
                  % (out, what, was, now), file=sys.stderr)
            return None
    return {f["file"]: f for f in doc["frames"]}


def run(args):
    out = args.out or os.path.join(args.input_dir, alignment.DEFAULT_NAME)
    try:
        paths = io.frames(args.input_dir)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    if not paths:
        print("no EXR/DNG/JPEG frames in %s" % args.input_dir, file=sys.stderr)
        return 1

    w, h = frame_size(paths[0])
    # A clip may hold more than one format -- single exposures for most of an
    # eclipse and merged EXRs through totality are one film. What it may not
    # hold is more than one frame size, and that is the check worth making:
    # the document speaks in full-frame pixels, so every frame has to agree
    # on how many there are. One open per format, not one per frame.
    kinds = io.kinds_of(paths)
    sizes = {}
    for k in kinds:
        first = next(p for p in paths if io.kind(p) == k)
        sizes[k] = frame_size(first)
    if len(set(sizes.values())) > 1:
        print("frames of different sizes in %s: %s -- a clip has to agree on "
              "how many pixels a frame has"
              % (args.input_dir,
                 ", ".join("%s %dx%d" % (k, s[0], s[1])
                           for k, s in sorted(sizes.items()))), file=sys.stderr)
        return 1
    fmt = io.format_of(paths)
    if fmt == "mixed" and args.method == "correlate":
        # Phase correlation measures the whole chain against one reference and
        # assumes a common measuring scale; a mosaic's luma comes in at half
        # size and an EXR's does not. A clear refusal beats a plausible number.
        print("--method correlate needs one format: this clip holds %s"
              % ", ".join(kinds), file=sys.stderr)
        return 1
    target = tuple(args.target) if args.target else (w / 2.0, h / 2.0)
    print("%d %s frames in %s (%dx%d)"
          % (len(paths), fmt.upper(), args.input_dir, w, h))
    if len(kinds) > 1:
        print("  %s" % ", ".join("%d %s" % (sum(1 for p in paths
                                                if io.kind(p) == k), k.upper())
                                 for k in kinds))

    skips = _skips(out)
    had_size = {}
    if os.path.isfile(out):
        try:
            had_size = alignment.load(out).get("size") or {}
        except (ValueError, KeyError):
            had_size = {}
    known = _already_measured(out, args, fmt, (w, h), target)
    if known is None:
        return 1
    # There used to be a re-measure here for records written before `shape`
    # existed. It has to go with the field: no frame carries a shape any more,
    # so the same test now says every frame is stale and an incremental re-run
    # would measure the whole clip. The size lives in the document and
    # `_lift_sizes` gives an old one its own on load.
    fresh = [p for p in paths if os.path.basename(p) not in known]
    if known:
        print("  %d already measured, %d new" % (len(paths) - len(fresh),
                                                 len(fresh)))
        if not fresh and args.method != "correlate":
            print("  nothing to do; --force measures them all again")
            return 0

    size, from_ = {}, {}
    if args.method == "correlate":
        recs = detect_correlate(paths, target)
        # Phase correlation measures the whole chain against its middle frame,
        # so one new frame changes every number: there is no incremental
        # version of it. Hand corrections are the one thing worth carrying
        # across, because nothing else can reproduce them.
        kept = 0
        for r in recs:
            old = known.get(r["file"])
            if old is not None and old["source"] == alignment.MANUAL:
                r.update(old)
                kept += 1
        if kept:
            print("  re-measured the whole chain, kept %d hand-corrected"
                  % kept)
    else:
        recs, (size, from_) = detect_disc(
            paths, args.method, args.step, args.threshold,
            args.min_agreement, args.rmin, args.rmax, known,
            frame_size=(w, h))
        # An incremental run keeps the size the document already has. One new
        # frame is not better evidence about the clip's radius than the whole
        # clip was, and the number may have been set by hand from the frame a
        # person judged best -- which no re-run should be able to undo by
        # arriving with a single measurement.
        if had_size and not args.force:
            if size != had_size:
                print("  keeping the size already in the document; --force "
                      "measures it again")
            size, from_ = had_size, {}

    for r in recs:
        if r["file"] in skips:
            r["skip"] = True
    if skips:
        print("  %d frame%s stay marked skip" % (len(skips),
                                                 "" if len(skips) == 1 else "s"))

    doc = alignment.save(out, args.input_dir, fmt, args.method,
                         (w, h), target, recs, size)
    print("  %s" % alignment.summary(doc))
    for body in ("sun", "moon"):
        sz = size.get(body)
        if not sz:
            continue
        told = ", ".join("%s %.1f" % (k, v) for k, v in sz.items())
        where = from_.get(body)
        print("  %s size for the clip: %s%s"
              % (body, told,
                 "  (from %s%s)" % (where[0],
                                      ", the middle of %d equally good"
                                      % where[2] if where[2] > 1 else "")
                 if where else ""))
    known = np.array([[f["cx"], f["cy"]] for f in recs if f["cx"] is not None])
    if len(known):
        d = np.hypot(known[:, 0] - target[0], known[:, 1] - target[1])
        print("  subject %.1f px from the target (median), worst %.1f px"
              % (np.median(d), d.max()))
    print("  written to %s" % out)

    if args.gui:
        from . import gui
        return gui.edit(out)
    return 0
