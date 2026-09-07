#!/usr/bin/env python3
"""Register a scene-linear EXR sequence so the subject stops wandering.

The mount drifts and gets re-centred by hand during a long shoot, so a clip's
frames do not share a framing. Measured on a total eclipse sequence: a steady
1.89 px/s of drift plus 15 discrete re-centrings of up to 1800 px across the
partial phase. Left alone the timelapse swims; the bracket clips also inherit
~80 px of drift across their 40-50 s.

Why this is tractable here even though the eclipse community's verdict is that
no software aligns eclipse frames well: their hard case is the *moon moving
across the sun*, which changes the subject itself and defeats a rigid
transform. Ours is plain camera translation -- measured monotonic, no rotation
or scale -- which phase correlation solves outright. The measurement is
scikit-image's `phase_cross_correlation` (sub-pixel, well tested) rather than
anything hand-rolled; this module is the glue around it.

Two deliberate choices worth knowing:

* **Measure on a log-compressed luminance.** Scene-linear eclipse frames span
  five decades with the subject occupying a fraction of a percent of the frame;
  correlating them raw lets the few brightest pixels dominate. Compressing
  first is what makes the correlation lock (measured peak-to-background rises
  from ~100 to ~3600).

* **Shift with bilinear interpolation, not splines.** scipy's default cubic
  spline rings around a highlight that sits 4 decades above its surroundings,
  which on linear data means negative pixels and a halo. Bilinear is softer and
  cannot overshoot.
"""

import argparse
import glob
import os
import sys

import numpy as np

from . import image_io as io

# Below this agreement (Pearson, after aligning) a measured step is not a
# measurement of drift -- the two frames are not the same scene. Measured on
# this material: good steps sit at 0.93-1.00 including genuine 640 px
# re-centrings, broken ones at 0.03-0.67. 0.80 separates them with room.
MIN_QUALITY = 0.80


def load_luma(path, step=4):
    """Downsampled, log-compressed luminance, for correlation only.

    EXR, DNG, camera raw or JPEG -- image_io decides. Log-compressing first is what makes
    the correlation lock on scene-linear data: those frames span five decades
    with the subject occupying a fraction of a percent of the frame, so
    correlating them raw lets the few brightest pixels dominate (measured
    peak-to-background rises from ~100 to ~3600 once compressed).
    """
    return np.log1p(np.clip(io.read_luma(path, step), 0, None) * 1e3)


def agreement(a, b, s):
    """Pearson correlation between two frames once `b` is shifted onto `a`.

    The `error` phase_cross_correlation returns is useless here -- it reads
    1.000 for good and broken alignments alike on this material. This measures
    the thing we actually care about, on the overlap only, and lands in [0, 1]
    where it can be thresholded meaningfully. Measured: 0.997-1.000 on the
    totality clips, 0.99 even across a genuine 640 px re-centring, and
    0.03-0.67 where the correlation is lying.
    """
    from scipy.ndimage import shift as ndshift
    bb = ndshift(b, s, order=1, mode="constant", cval=0.0)
    m = bb != 0
    if m.sum() < 100:
        return 0.0
    x, y = a[m], bb[m]
    x = x - x.mean()
    y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else 0.0


def measure(paths, step=4, upsample=40, min_quality=MIN_QUALITY):
    """Per-frame (dy, dx) offsets in full-resolution pixels, plus diagnostics.

    Chained: each frame is correlated against its predecessor, which keeps the
    two images as similar as possible (during the partial phase the crescent
    changes shape steadily, so correlating everything against one reference
    would degrade towards the ends of a long clip). The chain's accumulation
    error is what `residual()` exists to check.

    A step whose agreement falls below `min_quality` is not trusted -- it means
    the two frames are not the same scene, as when the solar filter comes off
    mid-clip and the frame goes from a disc on black to a landscape. Rejecting
    on *agreement* rather than on step size is what keeps the real re-centrings
    (640 px, agreement 0.99) while dropping the lies (1850 px, agreement 0.03).
    A rejected step falls back to the median of the accepted ones around it, so
    a steady drift keeps being tracked across the gap instead of collapsing.
    """
    from skimage.registration import phase_cross_correlation
    steps, quality = [], []
    scale = io.luma_scale(paths[0])      # a DNG's luma is already half size
    prev = load_luma(paths[0], step)
    for p in paths[1:]:
        cur = load_luma(p, step)
        s, _, _ = phase_cross_correlation(prev, cur, upsample_factor=upsample)
        steps.append(s * step * scale)
        quality.append(agreement(prev, cur, s))
        prev = cur
    steps = np.array(steps, dtype=float)
    quality = np.array(quality)
    good = quality >= min_quality
    if good.any() and not good.all():
        steps[~good] = np.median(steps[good], axis=0)
    elif not good.any():
        steps[:] = 0.0
    offs = np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
    return offs, quality, good


def to_reference(offs, ref="middle"):
    """Re-base absolute offsets so one frame stays put.

    The middle frame halves the largest displacement, which halves the border
    left without data (and the crop, if one is taken).
    """
    k = len(offs) // 2 if ref == "middle" else 0
    return offs - offs[k]


def apply_shift(path, dy, dx, bits=16):
    """Shift one EXR in place by a sub-pixel amount, filling with black."""
    from scipy.ndimage import shift as ndshift
    io.shift_frame(path, path, dy, dx, bits)


def common_crop(offs, shape):
    """The rectangle every shifted frame still covers: (top, bottom, left, right)."""
    h, w = shape
    dy, dx = offs[:, 0], offs[:, 1]
    top = int(np.ceil(max(0.0, dy.max())))
    bottom = int(np.floor(h + min(0.0, dy.min())))
    left = int(np.ceil(max(0.0, dx.max())))
    right = int(np.floor(w + min(0.0, dx.min())))
    return top, bottom, left, right


def residual(paths, step=4, upsample=40):
    """Re-measure a registered sequence: how far it still drifts, in pixels."""
    offs, _, _ = measure(paths, step, upsample)
    d = np.hypot(np.diff(offs[:, 0]), np.diff(offs[:, 1]))
    span = np.hypot(np.ptp(offs[:, 0]), np.ptp(offs[:, 1]))
    return float(np.median(d)), float(d.max() if len(d) else 0.0), float(span)


def register(paths, ref="middle", crop=False, bits=16, verbose=True):
    """Register an EXR sequence in place. Returns the per-frame offsets used."""
    if len(paths) < 2:
        return np.zeros((len(paths), 2))
    offs, quality, good = measure(paths)
    offs = to_reference(offs, ref)
    span = (np.ptp(offs[:, 1]), np.ptp(offs[:, 0]))
    if verbose:
        print("    drift: x %.1f px, y %.1f px   agreement median %.3f"
              % (span[0], span[1], np.median(quality) if len(quality) else 0.0))
        if not good.all():
            bad = np.nonzero(~good)[0]
            print("    ! %d of %d steps untrusted (agreement < %.2f), filled from "
                  "the neighbours: frames %s"
                  % ((~good).sum(), len(good), MIN_QUALITY,
                     ", ".join(str(i + 1) for i in bad[:12])
                     + (" ..." if len(bad) > 12 else "")))
            runs = np.split(bad, np.nonzero(np.diff(bad) != 1)[0] + 1)
            long = [r for r in runs if len(r) >= 2]
            if long:
                print("    ! %d run(s) of consecutive untrusted steps -- likely a "
                      "scene change, not drift; consider splitting the clip there"
                      % len(long))
    # phase_cross_correlation returns the shift that registers the moving frame
    # onto the reference, so the accumulated offset is applied as measured, not
    # negated. Getting this backwards doubles the drift instead of removing it,
    # which is exactly what the self-check below catches.
    for p, (dy, dx) in zip(paths, offs):
        apply_shift(p, dy, dx, bits)
    if crop:
        a = io.read_image(paths[0])
        t, b, l, r = common_crop(offs, a.shape[:2])
        if verbose:
            print("    cropping to the common area: %dx%d (from %dx%d)"
                  % (r - l, b - t, a.shape[1], a.shape[0]))
        for p in paths:
            a = io.read_image(p)
            io.write_image(a[t:b, l:r], p, bits)
    return offs


def check(paths, offs, verbose=True):
    """Verify the registration actually reduced the drift. Returns the residual.

    Cheap insurance against a sign or convention error: registering must leave
    the sequence more still than it found it. If the residual span is not well
    below the drift that was measured, something is wrong with the transform,
    not with the material.
    """
    med, mx, span = residual(paths)
    before = float(np.hypot(np.ptp(offs[:, 0]), np.ptp(offs[:, 1])))
    ok = span < before * 0.5 or span < 2.0
    if verbose:
        print("    residual: median %.2f px/frame, max %.2f px, span %.2f px "
              "(was %.1f px)%s" % (med, mx, span, before,
                                   "" if ok else "   <-- NOT REDUCED, check the transform"))
    return med, mx, span, ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Register a scene-linear EXR sequence in place.")
    ap.add_argument("input_dir", help="Folder of EXR frames (one clip).")
    ap.add_argument("--ref", default="middle", choices=("middle", "first"),
                    help="Which frame stays put (default middle: halves the "
                         "largest displacement and so the border without data).")
    ap.add_argument("--crop", action="store_true",
                    help="Crop every frame to the area they all still cover. "
                         "Off by default: during totality the border is 3e-4 of "
                         "the subject's brightness, so black fill is invisible "
                         "and costs no resolution.")
    ap.add_argument("--bits", type=int, default=16, choices=(16, 32))
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="Measure and report, write nothing.")
    args = ap.parse_args(argv)

    paths = io.frames(args.input_dir)
    if len(paths) < 2:
        print("need at least 2 frames in %s" % args.input_dir, file=sys.stderr)
        return 1
    print("%d frames in %s" % (len(paths), args.input_dir))
    if args.dry_run:
        offs, q, good = measure(paths)
        offs = to_reference(offs, args.ref)
        print("  drift: x %.1f px, y %.1f px   agreement median %.3f   untrusted %d/%d"
              % (np.ptp(offs[:, 1]), np.ptp(offs[:, 0]), np.median(q), (~good).sum(), len(good)))
        t, b, l, r = common_crop(offs, (4042, 6064))
        print("  common area would be %dx%d" % (r - l, b - t))
        return 0
    offs = register(paths, args.ref, args.crop, args.bits)
    _, _, _, ok = check(paths, offs)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
