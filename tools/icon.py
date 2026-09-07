#!/usr/bin/env python3
"""Draw the application icon, one drawing per size.

    python tools/icon.py

The art is generated rather than committed as an opaque blob, so it can be
argued with: the shapes are here, in numbers, and a change is a diff.

Each size is **drawn at its own size**, never reduced from the big one. A
crosshair that reads at 512 px is noise at 16, so at 16 there is no crosshair --
only the ring, thicker and slightly larger, because two pixels of stroke is the
least that will not shimmer. At 32 the crosshair survives as stubs outside the
ring, where it has room; crossing the ring at that size only smears it.

The ring is the subject at totality and a target at the same time, which is what
the tool does: put the subject where it belongs.
"""

import os

from PIL import Image, ImageDraw

BG, RING, CROSS = (18, 18, 22), (245, 246, 250), (0, 229, 255)
SIZES = (16, 32, 64, 128, 256, 512)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "src", "eclipse_aligner", "icons")


def icon(size):
    """One icon, drawn for `size`."""
    ss = 4 if size <= 32 else 1        # supersample the small ones, then shrink
    w = size * ss
    im = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, w - 1], radius=int(w * 0.22), fill=BG)
    c = w / 2
    r = w * (0.34 if size <= 32 else 0.31)
    stroke = max(2 * ss, int(w * (0.10 if size <= 32 else 0.075)))
    d.ellipse([c - r, c - r, c + r, c + r], outline=RING, width=stroke)
    if size >= 64:
        t = max(1, int(w * 0.014))
        d.line([w * 0.08, c, w * 0.92, c], fill=CROSS, width=t)
        d.line([c, w * 0.08, c, w * 0.92], fill=CROSS, width=t)
    elif size == 32:
        t = max(ss, int(w * 0.05))
        for a, b in ((w * 0.06, w * 0.20), (w * 0.80, w * 0.94)):
            d.line([a, c, b, c], fill=CROSS, width=t)
            d.line([c, a, c, b], fill=CROSS, width=t)
    return im.resize((size, size), Image.LANCZOS) if ss > 1 else im


def main():
    os.makedirs(OUT, exist_ok=True)
    for s in SIZES:
        p = os.path.join(OUT, "icon-%d.png" % s)
        icon(s).save(p)
        print("  %s" % os.path.relpath(p, HERE))
    icns = os.path.join(OUT, "icon.icns")
    icon(512).save(icns, format="ICNS",
                   sizes=[(s, s) for s in (16, 32, 128, 256, 512)])
    ico = os.path.join(OUT, "icon.ico")
    icon(256).save(ico, format="ICO", sizes=[(s, s) for s in SIZES if s <= 256])
    for p in (icns, ico):
        print("  %s" % os.path.relpath(p, HERE))


if __name__ == "__main__":
    main()
