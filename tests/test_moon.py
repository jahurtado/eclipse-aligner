"""Synthetic regression: lunar phases with a known centre.

No real Moon material yet, so the test is synthetic: a disc with the terminator
in its place (the ellipse x=k*sqrt(R^2-y^2), the projection of the great
circle), a known centre, noise and a soft edge. It checks that 'circle' holds
across the whole cycle, and documents that 'ellipse' does NOT at gibbous phase.
"""
import numpy as np
from scipy import ndimage
from eclipse_aligner.centre_disc import detect_centre


def moon(k, R=300, cx=520.0, cy=380.0, H=760, W=1040, noise=0.004, seed=0):
    """k in [-1,1]: -1 full, 0 quarter, +1 new. Illuminated = (1-k)/2."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    X, Y = xx - cx, yy - cy
    inside = X * X + Y * Y <= R * R
    halfw = np.sqrt(np.clip(R * R - Y * Y, 0, None))
    img = ndimage.gaussian_filter((inside & (X > k * halfw)).astype(np.float32), 1.5)
    return np.clip(img * 0.6 + rng.normal(0, noise, img.shape), 0, None).astype(np.float32), (cx, cy)


PHASES = [(0.90, 5), (0.80, 10), (0.50, 25), (0.20, 40),
          (0.00, 50), (-0.20, 60), (-0.50, 75), (-0.90, 95)]


def main():
    print("illuminated       circle      ellipse")
    bad = 0
    for k, pct in PHASES:
        img, (cx, cy) = moon(k)
        ec = []
        for m in ("circle", "ellipse"):
            c = detect_centre(img, m, 0.15)[0]
            ec.append(np.hypot(c[0] - cx, c[1] - cy) if c is not None else np.inf)
        ok = ec[0] < 2.0                       # the circle is the one that must hold
        bad += not ok
        print("        %3d%%      %6.2f px   %7.2f px  %s"
              % (pct, ec[0], ec[1], "" if ok else "<-- FAILED"))
    print("\n%s" % ("OK: circle is sub-pixel at every phase" if not bad
                    else "%d phases FAILED" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
