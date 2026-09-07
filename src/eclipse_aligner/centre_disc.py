#!/usr/bin/env python3
"""Find the centre of a solar or lunar disc in one frame.

The subject is a disc with a bite taken out of it -- the moon eating the sun in
a partial eclipse, the terminator eating the moon through its phases -- and the
job is to find the disc's centre from whatever arc of its limb is still lit, so
the timelapse stops swimming.

The measurement is always the same three steps:

    1. threshold to a bright mask and take its outline
    2. keep the outline points on the **convex hull**
    3. fit a conic to those and take its centre

Step 2 is what makes it work: in a partial eclipse the occulting moon cuts a
*concave* bite, so the lunar limb lies interior to the hull and drops out by
construction, leaving only solar limb. On a crescent moon the terminator is
concave too and the same thing happens.

Step 3 has two estimators, and which one is right is not a matter of taste:

**circle** (RANSAC) -- for a disc that really is circular. Measured on
synthetic moon phases with a known centre, it is sub-pixel across the whole
cycle (0.1-1.2 px at 5%, 25%, 50%, 75%, 95% illuminated) and recovers the
radius to 0.5%. It is the only one that survives a *gibbous* phase, where the
terminator turns convex and joins the hull: the circle's three degrees of
freedom cannot absorb those points so RANSAC throws them out as outliers
(inliers drop to ~59%, which is exactly the terminator's share of the outline).

**ellipse** -- for a disc that is not circular. Near the horizon atmospheric
refraction flattens the sun measurably (the vertical axis 8% shorter than the
horizontal on the last frames before sunset), and forcing a circle
there puts the centre 257 px low. The ellipse's five degrees of freedom follow
the flattening. The same freedom is its weakness: on a gibbous moon it absorbs
the terminator into a distorted ellipse and the centre lands 65-127 px out.

So: **circle for the moon and for a high sun, ellipse for a sun near the
horizon.** Default is ellipse because that is the case this was built and
validated on.

Not for totality -- there is no disc to find and the corona is diffuse. Use
phase correlation (`register_exr`) there; it gives 1-4 px residuals because
with the moon covering the disc there is no crescent to be misled by.

One deliberate omission: there is **no trajectory model**. An earlier version
smoothed the measured track to suppress what looked like measurement noise; the
noise was real mount motion (autocorrelation +0.76 at lag 1) and smoothing it
was what stopped the correction from following the subject. Each frame is
centred on its own measurement, full stop.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from . import image_io as io


def _hull_mask(P, tol=2.0):
    """The outline points lying on the convex hull, within `tol` px.

    Not just the hull vertices: every outline point close to a hull edge, so a
    smooth arc contributes all of its points rather than a handful of corners.
    """
    try:
        h = ConvexHull(P)
    except Exception:
        return None
    keep = np.zeros(len(P), bool)
    V = np.vstack([P[h.vertices], P[h.vertices][:1]])
    for i in range(len(V) - 1):
        p, q = V[i], V[i + 1]
        d = q - p
        L2 = float(d @ d)
        if L2 < 1e-9:
            continue
        t = np.clip(((P - p) @ d) / L2, 0, 1)
        keep |= np.hypot(*(P - (p + t[:, None] * d)).T) < tol
    return keep


def _hull(P, tol=2.0):
    """The outline points lying on the convex hull, or None if too few."""
    keep = _hull_mask(P, tol)
    if keep is None:
        return None
    return P[keep] if keep.sum() >= 30 else None


def outline_points(luma, frac=0.15):
    """Every point on the bright blob's outline, hull or not.

    Split out because the half the hull *discards* turns out to be worth as
    much as the half it keeps: what bites into the sun is the moon's limb, so
    the points thrown away for being concave are a measurement of the moon.
    """
    a = ndimage.gaussian_filter(luma, 1.0)
    hi = float(np.percentile(a, 99.999))
    if hi <= 0:
        return None
    m = a > hi * frac
    lab, k = ndimage.label(m)
    if k == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, k + 1))
    big = ndimage.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    outline = big ^ ndimage.binary_erosion(big)
    ys, xs = np.nonzero(outline)
    if len(xs) < 30:
        return None
    return np.c_[xs, ys].astype(float)


def detect_moon(luma, sun_centre, sun_r, frac=0.15, tol=3.0):
    """The occulting body, fitted to the arc the hull threw away.

    Returns (centre, r, quality), where quality is the fraction of the
    discarded arc the circle explains. It is the moon's **own** score and not
    the sun's `agreement`: the two limbs are best drawn on different frames --
    the sun's when little of it is covered, the moon's when it is deep across
    the disc -- so ranking moons by how well the sun fitted picks the wrong
    frame.

    During a partial eclipse the moon's limb is the concave bite, so it is
    exactly the set of outline points that are *not* on the convex hull. The
    two bodies come out of one outline, one from each half of it.

    Checked rather than trusted, because a fit to leftovers is easy to fool:
    the radius has to be 0.9 to 1.2 times the sun's -- during a total eclipse
    the moon is a few per cent the larger, or it could not cover it -- and the
    consensus has to hold. Measured across a partial phase the
    ratio came out 1.029 to 1.036 and the two centres closed on each other by
    5.7 px a frame in a straight line, to within 0.9 px over 75 frames.

    None during totality: there is no solar limb to be bitten, so the leftovers
    are corona. There the disc the ordinary detector finds *is* the moon.
    """
    P = outline_points(luma, frac)
    if P is None:
        return None
    keep = _hull_mask(P)
    if keep is None or keep.sum() < 30:
        return None
    R = P[~keep]
    if len(R) < 20:
        return None
    c, r, inl = _ransac_circle(R)
    if c is None or inl.mean() < 0.5:
        return None
    if not (0.9 <= r / max(sun_r, 1e-9) <= 1.2):
        return None
    if np.hypot(*(c - np.asarray(sun_centre, float))) > 3.0 * sun_r:
        return None
    return c, float(r), float(inl.mean())


def hull_limb(luma, frac=0.15):
    """Convex-hull limb points of the brightest blob, or None.

    The threshold is a fraction of the frame's 99.999th percentile rather than
    of its maximum, so a single hot pixel cannot set the scale. Holes are
    filled before taking the outline: on a noisy under-exposed disc the mask is
    speckled inside, and without filling those speckles become spurious
    "outline" points scattered across the disc.
    """
    P = outline_points(luma, frac)
    return None if P is None else _hull(P)


def _ellipse_centre(P):
    """Centre of the best-fit ellipse through P (Fitzgibbon), or None.

    Only the centre is wanted, never the shape. Constraining the conic to be an
    ellipse rather than fitting a general conic is what keeps it from
    degenerating: an unconstrained fit on a broken limb returns a hyperbola and
    the centre formula blows up.
    """
    x, y = P[:, 0].copy(), P[:, 1].copy()
    mx, my = x.mean(), y.mean()
    x -= mx
    y -= my
    D1 = np.c_[x * x, x * y, y * y]
    D2 = np.c_[x, y, np.ones(len(x))]
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.inv(S3) @ S2.T
        M = np.linalg.inv(np.array([[0, 0, 2.], [0, -1, 0], [2, 0, 0]])) @ (S1 + S2 @ T)
    except np.linalg.LinAlgError:
        return None
    _, v = np.linalg.eig(M)
    cond = 4 * v[0].real * v[2].real - v[1].real ** 2
    good = np.where(cond > 0)[0]
    if len(good) == 0:
        return None
    a = v[:, good[0]].real
    a = np.concatenate([a, (T @ a).real])
    A, B, C, D, E = a[0], a[1], a[2], a[3], a[4]
    den = B * B - 4 * A * C
    if abs(den) < 1e-12:
        return None
    return np.array([(2 * C * D - B * E) / den + mx, (2 * A * E - B * D) / den + my])


def _ellipse_geometry(P):
    """(centre, semi-major, semi-minor, angle) of the best-fit ellipse, or None."""
    x, y = P[:, 0].copy(), P[:, 1].copy()
    mx, my = x.mean(), y.mean()
    x -= mx
    y -= my
    D1 = np.c_[x * x, x * y, y * y]
    D2 = np.c_[x, y, np.ones(len(x))]
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.inv(S3) @ S2.T
        M = np.linalg.inv(np.array([[0, 0, 2.], [0, -1, 0], [2, 0, 0]])) @ (S1 + S2 @ T)
    except np.linalg.LinAlgError:
        return None
    _, v = np.linalg.eig(M)
    cond = 4 * v[0].real * v[2].real - v[1].real ** 2
    good = np.where(cond > 0)[0]
    if len(good) == 0:
        return None
    a = v[:, good[0]].real
    a = np.concatenate([a, (T @ a).real])
    A, B, C, D, E, F = a
    den = B * B - 4 * A * C
    if abs(den) < 1e-12:
        return None
    cx = (2 * C * D - B * E) / den + mx
    cy = (2 * A * E - B * D) / den + my
    num = 2 * (A * E * E + C * D * D + F * B * B - B * D * E - 4 * A * C * F)
    disc = np.hypot(A - C, B)
    try:
        a1 = np.sqrt(abs(num / (den * ((A + C) + disc))))
        a2 = np.sqrt(abs(num / (den * ((A + C) - disc))))
    except (ZeroDivisionError, FloatingPointError):
        return None
    if not (np.isfinite(a1) and np.isfinite(a2)) or min(a1, a2) < 1e-6:
        return None
    return (np.array([cx, cy]), max(a1, a2), min(a1, a2),
            0.5 * np.arctan2(B, A - C))


def _on_ellipse(P, geom, tol=0.06):
    """Fraction of P lying on the fitted ellipse, within `tol` of its equation.

    An honest agreement score. Counting the points the refit loop happened to
    keep is not one -- it reports ~100% even on an over-exposed frame whose
    "limb" is the image border, because those points do fit *some* ellipse
    beautifully. This asks the different question of how much of the outline
    the fitted curve actually explains.
    """
    c, a, b, ang = geom
    ca, sa = np.cos(-ang), np.sin(-ang)
    xr = (P[:, 0] - c[0]) * ca - (P[:, 1] - c[1]) * sa
    yr = (P[:, 0] - c[0]) * sa + (P[:, 1] - c[1]) * ca
    val = (xr / a) ** 2 + (yr / b) ** 2
    return float((np.abs(val - 1.0) < tol).mean())


def _fit_circle(P):
    """Algebraic (Kasa) circle through P: returns (centre, radius)."""
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    q, *_ = np.linalg.lstsq(A, P[:, 0] ** 2 + P[:, 1] ** 2, rcond=None)
    return np.array([q[0], q[1]]), float(np.sqrt(q[2] + q[0] ** 2 + q[1] ** 2))


def _ransac_circle(P, iters=600, tol=3.0, rmin=None, rmax=None, seed=0):
    """Circle the most outline points agree on: (centre, radius, inlier mask).

    Consensus rather than least squares, because the points that are *not* on
    the limb -- a convex terminator on a gibbous phase, a lower limb washed out
    by extinction -- are a coherent population, not scatter, and least squares
    would be dragged by them rather than reject them.

    `rmin`/`rmax` bound the search. A reference frame is the natural source for
    them: it rejects absurd fits (an over-exposed frame whose corona fills the
    image fits a circle twice the disc's size) without pinning the radius,
    which matters because a real one drifts -- the moon's apparent diameter
    varies ~12% between perigee and apogee, and holding the radius fixed across
    that costs up to 21 px of centre error where letting it float costs 0.3.
    """
    rng = np.random.RandomState(seed)
    n = len(P)
    if n < 3:
        return None, None, None
    lo = rmin if rmin is not None else 0.0
    hi = rmax if rmax is not None else np.inf
    best = None
    for _ in range(iters):
        S = P[rng.choice(n, 3, replace=False)]
        (ax, ay), (bx, by), (cx, cy) = S
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-6:
            continue
        ux = ((ax*ax+ay*ay)*(by-cy) + (bx*bx+by*by)*(cy-ay) + (cx*cx+cy*cy)*(ay-by)) / d
        uy = ((ax*ax+ay*ay)*(cx-bx) + (bx*bx+by*by)*(ax-cx) + (cx*cx+cy*cy)*(bx-ax)) / d
        c = np.array([ux, uy])
        R = float(np.hypot(*(S[0] - c)))
        if not (lo < R < hi):
            continue
        inl = np.abs(np.hypot(*(P - c).T) - R) < tol
        if best is None or inl.sum() > best[2]:
            best = (c, R, int(inl.sum()), inl)
    if best is None:
        return None, None, None
    c, R, _, inl = best
    for _ in range(2):                      # refit on the consensus set
        c, R = _fit_circle(P[inl])
        inl = np.abs(np.hypot(*(P - c).T) - R) < tol
    return c, R, inl


def detect_centre(luma, method="ellipse", frac=0.15, tol=6.0, iters=5,
                  rmin=None, rmax=None):
    """Disc centre from one frame: (centre, inlier_fraction, shape).

    `shape` is the conic the centre came from, in the same pixels as `luma`:
    `{"r": radius}` from a circle, `{"a", "b", "angle"}` from an ellipse, and
    None when nothing was found. Keeping it costs nothing at this point and is
    the only way a person can later see *what was fitted* instead of just where
    its centre landed -- a centre alone cannot show that the curve wrapped
    itself around a terminator.

    Returns None honestly when there is no usable limb -- an over-exposed frame
    whose corona floods it, an under-exposed one with no disc, a limb too
    broken to fit. What to do about a miss is the caller's business.
    """
    Q = hull_limb(luma, frac)
    if Q is None or len(Q) < 20:
        return None, 0.0, None

    if method == "circle":
        c, r, inl = _ransac_circle(Q, rmin=rmin, rmax=rmax)
        if c is None:
            return None, 0.0, None
        return c, float(inl.mean()), {"r": float(r)}

    # ellipse: refit while rejecting points whose radius to the running centre
    # is an outlier, so a limb broken by extinction does not drag the fit.
    keep = np.ones(len(Q), bool)
    geom = None
    for _ in range(iters):
        geom = _ellipse_geometry(Q[keep])
        if geom is None:
            return None, 0.0, None
        c = geom[0]
        r = np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1])
        med = np.median(r[keep])
        s = np.median(np.abs(r[keep] - med)) * 1.4826 + 1e-6
        new = np.abs(r - med) < tol * s
        if new.sum() < 20 or (new == keep).all():
            keep = new if new.sum() >= 20 else keep
            break
        keep = new
    if geom is None:
        return None, 0.0, None
    if rmin is not None and geom[2] < rmin:
        return None, 0.0, None
    if rmax is not None and geom[1] > rmax:
        return None, 0.0, None
    # degrees in the document, because a person reads and edits it
    return geom[0], _on_ellipse(Q, geom), {"a": float(geom[1]),
                                           "b": float(geom[2]),
                                           "angle": float(np.degrees(geom[3]))}


def measure(paths, method="ellipse", step=2, frac=0.15, min_inlier_frac=0.35,
            rmin=None, rmax=None):
    """Per-frame centres in full-resolution pixels: (x, y, 1-inliers, detected).

    Works off `image_io.read_luma`, so the frames can be EXR, DNG, camera raw or JPEG. A
    DNG's luma is one green plane of the mosaic and so already half size; the
    scale factor puts every answer back into full-resolution pixels regardless.

    A frame with no usable limb inherits the last good centre: between adjacent
    frames the subject barely moves, so the previous position is the most
    likely place for it and far safer than a centre invented from noise.
    """
    out = []
    last = None
    for p in paths:
        sc = step * io.luma_scale(p)
        c, f, _ = detect_centre(io.read_luma(p, step), method, frac,
                             rmin=rmin, rmax=rmax)
        if c is not None and f >= min_inlier_frac:
            last = (c[0] * sc, c[1] * sc)
            out.append((last[0], last[1], 1.0 - f, 1))
        elif last is not None:
            out.append((last[0], last[1], np.inf, 0))
        else:
            out.append((np.nan, np.nan, np.inf, 0))
    return np.array(out)
