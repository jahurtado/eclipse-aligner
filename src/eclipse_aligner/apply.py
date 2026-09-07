#!/usr/bin/env python3
"""Move the frames according to an alignment document.

The destructive half, kept separate on purpose: by the time this runs, the
positions have been measured, inspected and -- where the detector was wrong --
corrected by hand. It only moves pixels.

**It never writes over the originals.** By default the shifted frames go into a
subdirectory of the clip, `aligned/`, and `-o` sends them somewhere else
entirely. The originals are the only copy of what the sensor saw; a shifted
frame is a derivative that can always be made again, and for JPEG in particular
writing in place would re-encode, which compounds with every run.
"""

import os
import shutil
import sys

import numpy as np

from . import alignment
from . import image_io as io


DEFAULT_OUT = "aligned"
SEQ_PREFIX = "frame_"
SEQ_MAP = "sequence.txt"


def output_names(doc, renumber):
    """What each kept frame is called in the output.

    Left alone, a frame keeps its own name, which is what makes the output
    match the shoot one to one. `renumber` gives it a place in a contiguous
    sequence instead -- `frame_0001`, `frame_0002` -- because software that
    reads a numbered image sequence stops at the first gap, and skipping a
    frame makes gaps by definition. The two wants are opposites, so it is a
    choice and not a default.
    """
    keep = kept_paths(doc)
    # A camera's own raw comes out as a DNG, since that is what this writes,
    # so its name has to come out as one too. Leaving DSC_0087.NEF on a file
    # that is a DNG inside would be a lie the next tool believes.
    def named(p):
        base = os.path.basename(p)
        return os.path.splitext(base)[0] + ".dng" \
            if io.kind(p) == "raw" else base
    if not renumber:
        return [named(p) for p in keep]
    # The extension comes from each frame, not from the first one: a clip that
    # is DNG around a stretch of merged EXRs would otherwise have every EXR
    # renamed `.dng`, which is a lie the next tool believes.
    return ["%s%04d%s" % (SEQ_PREFIX, i + 1,
                          os.path.splitext(named(p))[1].lower())
            for i, p in enumerate(keep)]


def kept_paths(doc):
    return [alignment.paths(doc)[i] for i in alignment.kept(doc)]


def images_in(out_dir):
    """The image files sitting in a directory, whatever wrote them."""
    if not os.path.isdir(out_dir):
        return []
    return sorted(f for f in os.listdir(out_dir)
                  if os.path.isfile(os.path.join(out_dir, f))
                  and os.path.splitext(f)[1].lower() in
                  tuple(e.lower() for e in io.SUPPORTED))


def clean_output(out_dir, verbose=True):
    """Empty the output of images before writing a new set.

    `_reconcile` only removes what this tool can recognise as its own, which
    leaves one hole: point the output at a directory holding *another clip's*
    frames and they are neither in this document nor numbered in our scheme, so
    they survive and join the sequence. This closes it by taking the whole
    directory's word for nothing.

    Images only. A directory is left alone and so is anything that is not a
    frame -- notes, sidecars, whatever a person put there -- because sweeping
    those away would be a bigger promise than the one being asked for.
    """
    gone = images_in(out_dir)
    for f in gone:
        os.remove(os.path.join(out_dir, f))
    seq = os.path.join(out_dir, SEQ_MAP)
    if os.path.isfile(seq):
        os.remove(seq)
    if gone and verbose:
        print("  cleaned %d image%s out of the destination first"
              % (len(gone), "" if len(gone) == 1 else "s"))
    return gone


def snap_shifts(shifts, steps):
    """Round every shift onto the sensor's own grid.

    A mosaic cannot be resampled without cost. Each of the four sub-planes
    sits on a grid of spacing 2, so a **even** shift carries the same
    photosites to a new place and anything else blends neighbours -- and the
    blend is not slight. Measured on a real frame, as the mean gradient of one
    sub-plane against the original:

        moved 2.00 px (fraction 0.00)    103 % of the original sharpness
        moved 0.50 px (fraction 0.25)     58 %
        moved 1.00 px (fraction 0.50)     42 %

    Which is the real objection: the loss depends on the fractional part, and
    every frame has a different one, so a sequence resampled sub-pixel breathes
    between 42 % and 100 % sharp. That reads as focus pumping, and no viewer
    ever reads it as anything else.

    A cubic spline would soften less -- 70 % at half a pixel, measured -- and
    still varies. Landing on the grid is the only answer that does not.

    What it costs is position. On a real 525-frame document, snapping leaves a
    median 0.79 px and at worst 1.36 px uncorrected, and because the leftover
    changes from frame to frame it is a shimmer of about 0.93 px against a
    subject that really travels 11.4 px between neighbours. A pixel of shimmer
    on a 2300 px disc against a sharpness that halves and comes back: the
    numbers pick the shimmer, and so does the eye.

    The editor is **not** snapped, on purpose: it is the instrument you place
    the centre with, and a preview that quantised would make a fine nudge look
    like it did nothing until it crossed a boundary. So what you see can sit
    up to a pixel from what is written, and `apply` says by how much.

    `steps` is **per frame**, because a clip may hold more than one format: a
    mosaic's grid is 2 and an EXR's is 1, and in a sequence that is DNG either
    side of a stretch of merged EXRs the two sit next to each other. A step of
    1 leaves the shift alone, so the EXRs keep the sub-pixel answer they can
    use and the DNGs land on the sensor's grid.
    """
    import numpy as np
    steps = np.asarray(steps, float).reshape(-1, 1)
    snapped = np.round(shifts / steps) * steps
    left = np.hypot(*(shifts - snapped).T)
    return snapped, left


def apply_doc(doc, out_dir, bits=16, quality=95, crop=False, verbose=True,
              progress=None, renumber=False, clean=False, subpixel=False):
    """Write every frame shifted onto the target, into `out_dir`.

    `out_dir` is required and is never the frames' own directory: see the
    module docstring.

    `progress(done, total)` is called before each frame. It is how a caller
    watches a five-minute pass, and raising from it is how a caller stops one:
    the exception travels straight out, leaving the frames written so far
    where they are.
    """
    if os.path.abspath(out_dir) == os.path.abspath(doc["input_dir"]):
        # Belt and braces: this function deletes files from `out_dir`, and the
        # one directory it must never be pointed at is the one holding the
        # originals.
        raise ValueError("out_dir is the frames' own directory: %s" % out_dir)
    keep = alignment.kept(doc)
    paths = kept_paths(doc)
    names = output_names(doc, renumber)
    shifts = alignment.shifts(doc)[keep]
    # On the sensor's grid unless asked otherwise, and only where there is a
    # grid: an EXR or a JPEG has been demosaiced already and has no phase to
    # keep, so its step is 1 and `snap_shifts` hands it back untouched. Asked
    # of each frame, since one clip can hold both.
    steps = [1 if subpixel else io.bayer_step(p) for p in paths]
    shifts, leftover = snap_shifts(shifts, steps)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("%d frames named in the document are not on "
                                "disk, starting with %s"
                                % (len(missing), os.path.basename(missing[0])))
    os.makedirs(out_dir, exist_ok=True)
    if clean:
        clean_output(out_dir, verbose)

    known = np.isfinite(shifts[:, 0])
    if verbose:
        d = np.hypot(*shifts[known].T)
        left = len(doc["frames"]) - len(keep)
        print("  shifting %d frames (%d have no measurement and are copied "
              "through); max %.0f px" % (known.sum(), (~known).sum(),
                                         d.max() if known.any() else 0))
        on_grid = np.array([s > 1 for s in steps])
        ok = np.isfinite(leftover) & known & on_grid
        if ok.any():
            print("  %d on the Bayer grid: nothing is resampled, and a median "
                  "%.2f px is left uncorrected (worst %.2f)"
                  % (ok.sum(), np.median(leftover[ok]), leftover[ok].max()))
        free = known & ~on_grid
        if free.any():
            print("  %d have no mosaic and keep the exact sub-pixel shift"
                  % free.sum())
        if left:
            print("  leaving out %d frame%s marked skip"
                  % (left, "" if left == 1 else "s"))
    for n, (p, (dy, dx)) in enumerate(zip(paths, shifts)):
        if progress is not None:
            progress(n, len(paths))
        dest = os.path.join(out_dir, names[n])
        if not np.isfinite(dy):
            # Nothing measured, so nothing to move -- but a raw still has to
            # change format on the way out, or the folder would hold a mix.
            if io.kind(p) == "raw":
                io.shift_frame(p, dest, 0.0, 0.0, bits, quality)
            else:
                shutil.copyfile(p, dest)
            continue
        io.shift_frame(p, dest, dy, dx, bits, quality)

    if renumber:
        _write_map(out_dir, names, paths, verbose)
    _reconcile(doc, out_dir, names, renumber, verbose)

    if crop:
        _crop_common([os.path.join(out_dir, n) for n in names],
                     shifts, bits, quality, verbose)
    return shifts


def _write_map(out_dir, names, paths, verbose=True):
    """Say which original each numbered frame came from.

    Renumbering buys a sequence a video editor will open and costs the one
    thing the names were carrying: which shot is which. This gives it back, in
    a plain text file next to the frames, where it stays with them.
    """
    with open(os.path.join(out_dir, SEQ_MAP), "w") as fh:
        fh.write("# eclipse-aligner: output frame, then the original it came "
                 "from\n")
        for n, p in zip(names, paths):
            fh.write("%s\t%s\n" % (n, os.path.basename(p)))
    if verbose:
        print("  wrote %s" % SEQ_MAP)


def _reconcile(doc, out_dir, names, renumber, verbose=True):
    """Leave the output holding exactly what this run says it should.

    The state of the output has to be a function of the document and the flags
    and of nothing else. Otherwise every earlier decision lingers there: a frame
    dropped after a previous pass keeps its old file, and changing the naming
    leaves both namings side by side for a video editor to read as two clips.

    It removes only what this tool could have written itself -- a name the
    document lists, or a numbered frame in our own scheme -- and only inside the
    output directory. Anything else in there belongs to somebody else.
    """
    if not os.path.isdir(out_dir):
        return []
    wanted = set(names) | ({SEQ_MAP} if renumber else set())
    ours = {os.path.basename(p) for p in alignment.paths(doc)}
    gone = []
    for f in sorted(os.listdir(out_dir)):
        if f in wanted:
            continue
        stem = f[len(SEQ_PREFIX):].split(".")[0]
        numbered = f.startswith(SEQ_PREFIX) and stem.isdigit()
        if f in ours or numbered or f == SEQ_MAP:
            os.remove(os.path.join(out_dir, f))
            gone.append(f)
    if gone and verbose:
        print("  removed %d file%s the output should no longer hold"
              % (len(gone), "" if len(gone) == 1 else "s"))
    return gone


def _crop_common(paths, shifts, bits, quality, verbose):
    """Trim every frame to the rectangle they all still cover."""
    k = np.isfinite(shifts[:, 0])
    if not k.any():
        return
    dy, dx = shifts[k, 0], shifts[k, 1]
    h, w = io.frame_shape(paths[0])
    top = int(np.ceil(max(0.0, dy.max())))
    bottom = int(np.floor(h + min(0.0, dy.min())))
    left = int(np.ceil(max(0.0, dx.max())))
    right = int(np.floor(w + min(0.0, dx.min())))
    if any(io.kind(p) in ("dng", "raw") for p in paths):
        # The crop origin has to keep the Bayer phase, so it rounds down to
        # even -- and if any frame is a mosaic the whole clip crops evenly,
        # since one rectangle serves them all.
        top, left = (top // 2) * 2, (left // 2) * 2
    if verbose:
        print("  cropping to the common area: %dx%d (from %dx%d)"
              % (right - left, bottom - top, w, h))
    for p in paths:
        io.crop_frame(p, p, top, bottom, left, right, bits, quality)


def add_args(ap):
    """The options of `eclipse-aligner apply`."""
    ap.add_argument("document",
                    help="The clip directory, or the document itself.")
    ap.add_argument("-o", "--out-dir",
                    help="Write the shifted frames here. Default: an `aligned` "
                         "subdirectory of the clip. The originals are never "
                         "written over.")
    ap.add_argument("--bits", type=int, default=16, choices=(16, 32),
                    help="EXR bit depth (default 16).")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality (default 95).")
    ap.add_argument("--crop", action="store_true",
                    help="Trim to the area every frame still covers. Off by "
                         "default: during totality the border is 3e-4 of the "
                         "subject's brightness, so black fill is invisible and "
                         "costs no resolution.")
    ap.add_argument("--renumber", action="store_true",
                    help="Name the output as a contiguous sequence "
                         "(frame_0001...) instead of keeping each frame's own "
                         "name. Skipping a frame leaves a gap in the original "
                         "numbering, and software that reads an image sequence "
                         "stops at the first gap. A sequence.txt beside the "
                         "frames says which original each one came from.")
    ap.add_argument("--clean", action="store_true",
                    help="Delete every image already in the destination before "
                         "writing. Without it, only what this tool recognises "
                         "as its own is cleared, which leaves another clip's "
                         "frames sitting there.")
    ap.add_argument("--subpixel", action="store_true",
                    help="Shift by the exact fractional amount instead of "
                         "landing on the sensor's grid. More accurate in "
                         "position and softer in the picture, by an amount "
                         "that changes from frame to frame -- measured, 42 %% "
                         "of the original sharpness at half a photosite and "
                         "100 %% on the grid, which over a sequence reads as "
                         "focus pumping. Snapping instead leaves a median "
                         "0.79 px uncorrected on a real document. No effect "
                         "on EXR or JPEG, which have no mosaic.")
    ap.add_argument("-n", "--dry-run", action="store_true")


def run(args):
    doc = alignment.load(alignment.document_path(args.document))
    out_dir = args.out_dir or os.path.join(doc["input_dir"], DEFAULT_OUT)
    if os.path.abspath(out_dir) == os.path.abspath(doc["input_dir"]):
        print("that is the originals' own directory; give -o somewhere else",
              file=sys.stderr)
        return 1
    print("%d %s frames from %s" % (len(doc["frames"]), doc["format"].upper(),
                                    doc["input_dir"]))
    print("  %s; method %s" % (alignment.summary(doc), doc["method"]))
    print("  writing to %s" % out_dir)
    if args.dry_run:
        keep = alignment.kept(doc)
        s = alignment.shifts(doc)[keep]
        k = np.isfinite(s[:, 0])
        names = output_names(doc, args.renumber)
        print("  would write %d frames, max %.0f px%s"
              % (len(keep), np.hypot(*s[k].T).max() if k.any() else 0,
                 " as %s ... %s" % (names[0], names[-1]) if args.renumber
                 and names else ""))
        if args.clean:
            n = len(images_in(out_dir))
            print("  would delete the %d image%s already there first"
                  % (n, "" if n == 1 else "s"))
        stale = [f["file"] for f in doc["frames"] if f.get("skip")
                 and os.path.isfile(os.path.join(out_dir, f["file"]))]
        if stale:
            print("  would remove %d from the output, left by a previous pass "
                  "(first %s)" % (len(stale), stale[0]))
        return 0
    apply_doc(doc, out_dir, args.bits, args.quality, args.crop,
              renumber=args.renumber, clean=args.clean,
              subpixel=args.subpixel)
    print("  done")
    return 0
