"""Read frames of a timelapse: EXR, DNG, a camera's own raw, or JPEG.

Measuring where the subject is and moving the frame are different problems, and
the formats do not support them equally:

| format | measure | write shifted |
|--------|---------|---------------|
| EXR    | yes     | yes |
| JPEG   | yes     | yes (re-encodes, lossy) |
| DNG    | yes     | yes (per-plane, stays a CFA) |

A DNG holds an undemosaiced CFA mosaic, so it cannot simply be resampled: that
would put green photosites where red ones belong. It *can* be shifted with full
sub-pixel precision by splitting the mosaic into its four Bayer sub-planes,
shifting each on its own grid by half the amount, and re-interleaving. Every
interpolation then happens between photosites of the same colour, which is
strictly cleaner than resampling a demosaiced image -- there is no colour
crosstalk to introduce because no two colours are ever mixed.

Measurement uses a luminance, not colour, and each format gives it differently:
EXR and JPEG average their channels (JPEG's gamma encoding is harmless here --
the estimator is geometric and the threshold relative), and a DNG hands back
**one green channel of the mosaic**. Green because it is the luminance-carrying
half of a Bayer array and the densest, and because taking a single plane costs
nothing: it is already half resolution, which is ample for finding a disc a
thousand pixels across.
"""

import glob
import os
import struct
import sys

import numpy as np

EXR = ".exr"
DNG = (".dng",)
JPG = (".jpg", ".jpeg")
# A camera's own raw. Read through LibRaw and written back out as DNG, because
# a DNG is what this tool makes and a NEF is not something to rebuild. Adding
# another maker is one entry here: nothing below asks which camera it was.
RAW = (".nef", ".cr2", ".cr3")
SUPPORTED = (EXR,) + DNG + JPG + RAW
WRITABLE = {"exr", "jpg", "dng"}

PHOTOMETRIC_CFA = 32803


def kind(path):
    """'exr', 'dng', 'raw' or 'jpg' from the extension, or None if unsupported."""
    e = os.path.splitext(path)[1].lower()
    if e == EXR:
        return "exr"
    if e in DNG:
        return "dng"
    if e in RAW:
        return "raw"
    if e in JPG:
        return "jpg"
    return None


def frames(input_dir):
    """Sorted frames of one clip. **The formats may differ.**

    A clip used to have to be all one format, and that turned out to be the
    wrong rule for the material. A timelapse of an eclipse is single exposures
    for most of its length and bracketed merges through totality, because
    totality is where the range stops fitting in one shot -- so the natural
    sequence is DNG for 241 frames and EXR for the twenty in the middle. They
    are one film, and they are numbered as one.

    Each frame keeps its own format all the way through: what a DNG measures
    is a mosaic and what it writes is a mosaic, what an EXR measures is scene
    linear and what it writes is scene linear. Nothing is converted, and
    nothing needs to be -- the document speaks in full-frame pixels, which
    both of them agree on.

    What must still match is the frame *size*, and that is checked where the
    frames are actually opened rather than here: this lists a directory and
    should not read a hundred files to do it.
    """
    found = []
    for e in SUPPORTED:
        found += glob.glob(os.path.join(input_dir, "*" + e))
        found += glob.glob(os.path.join(input_dir, "*" + e.upper()))
    return sorted(set(found))


def kinds_of(paths):
    """The formats present, in a stable order."""
    return sorted({kind(p) for p in paths if kind(p)})


def format_of(paths):
    """What the document should call the clip's format.

    One name when there is one format, and `mixed` when there is more than
    one. It is a label for a reader and a staleness check for `detect`, not
    something anything dispatches on: every reader and writer asks the file
    in front of it.
    """
    ks = kinds_of(paths)
    return ks[0] if len(ks) == 1 else ("mixed" if ks else "")


def _tag_scalar(tag, default):
    """First value of a TIFF tag as a float, whatever encoding it uses.

    BlackLevel is RATIONAL in Sigma's own DNGs and SHORT in HDRMerge's output,
    and tifffile hands a RATIONAL back as a *flattened* (num, den, ...) tuple,
    so reading value[0] blindly yields 1048576 instead of 1048576/1024 = 1024 --
    which subtracts the whole image to black.
    """
    if tag is None:
        return default
    v = tag.value
    if isinstance(v, (int, float)):
        return float(v)
    if not len(v):
        return default
    if len(v) == 2 * tag.count and float(v[1]):
        return float(v[0]) / float(v[1])
    first = v[0]
    if isinstance(first, tuple) and len(first) == 2 and float(first[1]):
        return float(first[0]) / float(first[1])
    return float(first)


def _cfa_ifd(tf):
    """The IFD holding the CFA, wherever the file keeps it.

    Camera DNGs put the mosaic in a SubIFD and a thumbnail in the main one;
    a DNG written from scratch may put the mosaic in the main IFD. Both are
    legal, so look in both rather than assume.
    """
    for page in tf.pages:
        for sub in (page.pages or []):
            pi = sub.tags.get("PhotometricInterpretation")
            if pi and pi.value == PHOTOMETRIC_CFA:
                return sub
    for page in tf.pages:
        pi = page.tags.get("PhotometricInterpretation")
        if pi and pi.value == PHOTOMETRIC_CFA:
            return page
    return None


def read_cfa(path):
    """(mosaic float32, black, white) from a DNG or a camera's own raw."""
    if kind(path) == "raw":
        import rawpy
        with rawpy.imread(path) as r:
            return (r.raw_image_visible.astype(np.float32),
                    float(min(r.black_level_per_channel)),
                    float(r.white_level))
    import tifffile
    with tifffile.TiffFile(path) as t:
        ifd = _cfa_ifd(t)
        if ifd is None:
            raise ValueError("no CFA image found in %s" % path)
        return (ifd.asarray().astype(np.float32),
                _tag_scalar(ifd.tags.get("BlackLevel"), 0.0),
                _tag_scalar(ifd.tags.get("WhiteLevel"), 16383.0))


def _read_dng_green(path):
    """One green channel of a mosaic, black-subtracted and normalised.

    Which green depends on where the pattern starts, and it matters: taking
    the red plane of an RGGB sensor as if it were green measures a disc that
    is a third as bright and, through a filter, sometimes not there at all.
    """
    cfa, black, white = read_cfa(path)
    dy, dx = _green_offset(path)
    return (cfa[dy::2, dx::2] - black) / max(white - black, 1e-9)


def _green_offset(path):
    """Row and column of a green photosite inside the 2x2 pattern.

    A DNG written by this tool, and every one seen so far, starts RGGB, so the
    green at (0, 1) is the historical answer and stays the default. A camera
    raw says its own pattern and is asked.
    """
    if kind(path) != "raw":
        return 0, 1
    import rawpy
    with rawpy.imread(path) as r:
        desc = r.color_desc.decode()
        pat = r.raw_pattern
    for y in (0, 1):
        for x in (0, 1):
            if desc[pat[y][x]] == "G":
                return y, x
    return 0, 1


def read_luma(path, step=1):
    """A 2-D float array to measure on, downsampled by `step`.

    Note the DNG case is already half resolution (one channel of the mosaic),
    so its `step` compounds with that -- which is why callers work in a
    "measuring space" and scale their answer back out.
    """
    k = kind(path)
    if k == "exr":
        return read_image(path).mean(-1)[::step, ::step]
    if k == "jpg":
        return read_image(path).mean(-1)[::step, ::step]
    if k in ("dng", "raw"):
        return _read_dng_green(path)[::step, ::step]
    raise ValueError("unsupported format: %s" % path)


def luma_scale(path):
    """Pixels of the original frame per pixel of `read_luma(path, 1)`."""
    return 2 if kind(path) in ("dng", "raw") else 1


def bayer_step(path):
    """The smallest move that costs no interpolation, in full-frame pixels.

    On a mosaic that is **2**: the four sub-planes sit on a grid of spacing 2,
    so an even shift moves every photosite by a whole sub-plane pixel and the
    result is the same numbers in different places. Measured on a synthetic
    mosaic, a 2 px move comes back bit-identical to a slice copy, and a 1 px
    move differs by up to 15320 levels -- every photosite a blend of two.

    On anything already demosaiced, EXR or JPEG, there is no phase to keep and
    the answer is 1.
    """
    return 2 if kind(path) in ("dng", "raw") else 1


def shift_cfa(cfa, dy, dx, order=1):
    """Shift a Bayer mosaic by (dy, dx) full-resolution pixels, sub-pixel exact.

    Each of the four sub-planes sits on its own grid of spacing 2, so moving the
    image by (dy, dx) means moving every sub-plane by (dy/2, dx/2) of its own
    pixels. Interpolating inside a sub-plane only ever mixes photosites of the
    same colour, so the result is still a faithful CFA -- no colour crosstalk,
    which is more than can be said for shifting a demosaiced image.
    """
    from scipy.ndimage import shift as ndshift
    out = np.empty_like(cfa)
    for oy in (0, 1):
        for ox in (0, 1):
            plane = cfa[oy::2, ox::2]
            out[oy::2, ox::2] = ndshift(plane, (dy / 2.0, dx / 2.0), order=order,
                                        mode="constant", cval=0.0)
    return out


def _ratio(v):
    """A number out of whatever EXIF hands over: a rational, or a number."""
    try:
        n, d = v
        return float(n) / float(d) if d else 0.0
    except TypeError:
        return float(v)


def _exposure(v):
    t = _ratio(v)
    if t <= 0:
        return None
    return "1/%g s" % round(1.0 / t) if t < 0.5 else "%g s" % t


def shot_info(path):
    """How the frame was taken, as far as the file says: a dict, maybe empty.

    Read from the file's own EXIF, with no exiftool involved -- 0.0008 s for a
    DNG, sixty times cheaper than reading the frame itself, so it can be looked
    up whenever a frame is shown.

    A value that is missing is left out rather than shown as zero. A camera with
    a manual lens on it reports f/0 and 0 mm, and printing that would be
    inventing a measurement nobody made.
    """
    k = kind(path)
    tags = {}
    try:
        if k in ("dng", "raw"):
            # A NEF and a CR2 are TIFF underneath, so their EXIF reads exactly
            # like a DNG's. Only CR3 is a different container; it comes back
            # empty rather than wrong, which is what the caller expects of a
            # frame that will not say.
            import tifffile
            with tifffile.TiffFile(path) as t:
                p0 = t.pages[0]
                e = p0.tags.get("ExifTag")
                tags = dict(e.value) if e else {}
                for name in ("Make", "Model"):
                    if p0.tags.get(name):
                        tags[name] = p0.tags[name].value
        elif k == "jpg":
            from PIL import Image
            from PIL.ExifTags import TAGS
            with Image.open(path) as im:
                ex = im.getexif()
                tags = {TAGS.get(t, t): v for t, v in ex.items()}
                tags.update({TAGS.get(t, t): v
                             for t, v in ex.get_ifd(0x8769).items()})
        else:
            return {}
    except Exception:
        return {}

    out = {}
    if tags.get("ExposureTime") is not None:
        out["exposure"] = _exposure(tags["ExposureTime"])
    iso = tags.get("ISOSpeedRatings") or tags.get("PhotographicSensitivity")
    if iso:
        out["iso"] = int(iso if not isinstance(iso, (tuple, list)) else iso[0])
    f = _ratio(tags["FNumber"]) if tags.get("FNumber") is not None else 0.0
    if f > 0:
        out["aperture"] = f
    mm = _ratio(tags["FocalLength"]) if tags.get("FocalLength") is not None else 0.0
    if mm > 0:
        out["focal"] = mm
    when = tags.get("DateTimeOriginal") or tags.get("DateTime")
    if when:
        out["when"] = str(when)
    make, model = str(tags.get("Make", "")), str(tags.get("Model", ""))
    # "SIGMA" + "SIGMA fp" is how the file says it; a person says "SIGMA fp"
    cam = model if model.upper().startswith(make.upper()) else (make + " " + model)
    if cam.strip():
        out["camera"] = cam.strip()
    return {k2: v for k2, v in out.items() if v}


def shift_frame(in_path, out_path, dy, dx, bits=16, quality=95):
    """Shift one frame by (dy, dx) pixels and write it.

    A mosaic -- a DNG, or a camera's own raw -- goes through the mosaic and
    comes out a DNG. Everything else goes through the pixels and keeps its
    format. A NEF is not rebuilt as a NEF: nobody needs a counterfeit Nikon
    file, and DNG is the format that says "raw, and openly described".
    """
    if abs(dy) < 1e-3 and abs(dx) < 1e-3 and in_path == out_path:
        return
    if kind(in_path) in ("dng", "raw"):
        write_dng_shifted(in_path, out_path, dy, dx)
        return
    a = read_image(in_path)
    if abs(dy) > 1e-3 or abs(dx) > 1e-3:
        from scipy.ndimage import shift as ndshift
        # bilinear, not spline: a cubic spline rings around a highlight sitting
        # decades above its surroundings, which on linear data means negative
        # pixels and a halo.
        a = ndshift(a, (dy, dx, 0.0), order=1, mode="constant", cval=0.0)
    write_image(a, out_path, bits, quality)


# The tags a DNG needs to develop correctly, and where they live. Copying with
# `-all:all` silently copies nothing on a file rebuilt this way, so they are
# named. The second list moves tags that a camera keeps in the raw SubIFD into
# the single IFD this writer produces.
# By number, not by name. These were exiftool's names once, and exiftool calls
# tag 33422 `CFAPattern2` while tifffile calls it `CFAPattern`; asking for the
# wrong one dropped the Bayer order out of the output silently. The file still
# opened -- the developer just guessed the pattern, and 61% of the pixels came
# out a different colour. A number cannot be spelled two ways.
_DNG_IFD0_TAGS = (
    271,    # Make
    272,    # Model
    50706,  # DNGVersion
    50707,  # DNGBackwardVersion
    50708,  # UniqueCameraModel
    50721,  # ColorMatrix1
    50722,  # ColorMatrix2
    50723,  # CameraCalibration1
    50724,  # CameraCalibration2
    50727,  # AnalogBalance
    50728,  # AsShotNeutral
    50778,  # CalibrationIlluminant1
    50779,  # CalibrationIlluminant2
    50730,  # BaselineExposure
    50731,  # BaselineNoise
    50732,  # BaselineSharpness
    50734,  # LinearResponseLimit
)
_DNG_FROM_SUBIFD = (
    50714,  # BlackLevel
    50713,  # BlackLevelRepeatDim
    50717,  # WhiteLevel
    33422,  # CFAPattern
    33421,  # CFARepeatPatternDim
    50710,  # CFAPlaneColor
    50829,  # ActiveArea
    50719,  # DefaultCropOrigin
    50720,  # DefaultCropSize
    50718,  # DefaultScale
    50733,  # BayerGreenSplit
)
# Without these the file is not a usable DNG, whatever else it carries.
_DNG_ESSENTIAL = (50706, 50721, 50728, 33422, 50714, 50717)


# Tags that must not travel. A MakerNote is a maker's own block with absolute
# file offsets inside it: moved to a new file its pointers go stale, and what
# arrives is either ignored or misread. exiftool rewrites those offsets per
# manufacturer, which is most of what exiftool is; reproducing it is a project,
# and nothing downstream of here reads the block. The other three are pointers
# to IFDs of their own, which would have to be written as IFDs, not copied as
# numbers.
_EXIF_SKIP = (
    37500,  # MakerNote
    40965,  # InteroperabilityIFD pointer
    34853,  # GPSInfo IFD pointer
    34665,  # ExifIFD pointer -- this IFD is the thing being written
)

# Shot data that lives in IFD0 rather than in the EXIF IFD. Everything here is
# ASCII or SHORT, which is what keeps the encoder below to two branches; the
# tags that need real types are all in the EXIF IFD, where Pillow does it.
# Anything already written by `_dng_tags` is skipped at the call, not here.
_IFD0_CARRY = {
    271:   2,   # Make   -- `_dng_tags` has these for a DNG source but a camera
    272:   2,   # Model  -- raw has no DNG tags to copy, so they come from here
    270:   2,   # ImageDescription
    274:   3,   # Orientation
    305:   2,   # Software
    306:   2,   # DateTime
    315:   2,   # Artist
    33432: 2,   # Copyright
    50735: 2,   # CameraSerialNumber
}


# Tags whose TIFF type Pillow guesses from the value and gets wrong, with the
# type the EXIF specification actually gives them. Left to itself Pillow writes
# a signed rational as unsigned and an UNDEFINED byte string as BYTE, and
# `exiftool -validate` reports eight non-standard formats on a file that the
# camera writes clean. Readers mostly cope; a file that validates is cheaper
# than finding out which one does not.
#   7 = UNDEFINED, 4 = LONG, 10 = SRATIONAL
_EXIF_TYPE = {
    34866: 4,    # RecommendedExposureIndex
    37121: 7,    # ComponentsConfiguration
    37510: 7,    # UserComment
    41728: 7,    # FileSource
    41729: 7,    # SceneType
    37377: 10,   # ShutterSpeedValue    -- signed: shutters faster than 1 s
    37379: 10,   # BrightnessValue      -- signed: scenes darker than the ref
    37380: 10,   # ExposureBiasValue    -- signed, obviously
    # 37378 ApertureValue is NOT here: an APEX aperture is unsigned, and
    # Pillow's own guess is already the right one.
}


def _encode_ifd0(tag, ttype, value, put):
    """One IFD0 entry, 12 bytes, with anything over four bytes placed by `put`.

    A TIFF entry holds its value inline when it fits in the four bytes of the
    value field and a pointer otherwise, and the two are written the same way
    round -- so the caller cannot tell which happened, which is the point.
    """
    if ttype == 2:
        raw = value.encode("ascii", "replace") if isinstance(value, str) \
            else bytes(value)
        raw = raw.rstrip(b"\0") + b"\0"
        count = len(raw)
    else:                                    # SHORT
        raw = struct.pack("<H", int(value)) + b"\0\0"
        count = 1
    if len(raw) <= 4:
        payload = raw.ljust(4, b"\0")
    else:
        payload = struct.pack("<I", put(raw))
    return struct.pack("<HHI", tag, ttype, count) + payload


def attach_exif(path, src_path, extra_ifd0=()):
    """Copy the source's EXIF into a DNG this program has just written.

    This used to be exiftool's last job, and it is the only reason the tool was
    needed at all: `_dng_tags` already writes everything that makes the file a
    DNG. What was left is the **EXIF** -- exposure, ISO, lens, clock -- which is
    a nested IFD, and tifffile has no EXIF support and refuses to write the tag
    that points at one.

    So it is written here, afterwards, in three moves that TIFF allows because
    the header holds nothing but a pointer to the first IFD:

      1. the EXIF IFD is serialised and appended at the end of the file;
      2. a new IFD0 is appended after it -- the old entries verbatim, whose
         value offsets are still good because nothing moved, plus the pointer
         to the EXIF IFD and whatever `extra_ifd0` adds;
      3. the header is repointed at the new IFD0.

    The old IFD0 is left where it is, unreferenced. Rewriting it in place would
    mean growing it by an entry and shifting every byte after it.

    Verified against exiftool on a Sigma fp DNG: ExposureTime 1/1000, ISO 640,
    DateTimeOriginal and the rest come back reading what the original reads.
    """
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    try:
        src = Image.open(src_path).getexif()
        exif = src.get_ifd(0x8769)
    except Exception:
        return                      # no EXIF to carry: the DNG is still a DNG

    ifd = ImageFileDirectory_v2()
    for code, value in exif.items():
        if code in _EXIF_SKIP:
            continue
        try:
            if code in _EXIF_TYPE:
                ifd.tagtype[code] = _EXIF_TYPE[code]
            ifd[code] = value
        except Exception:
            # One tag a Pillow version cannot type is not worth losing the
            # other fifty over.
            ifd.pop(code, None)

    carry = []
    for code, ttype in sorted(_IFD0_CARRY.items()):
        if code in extra_ifd0:
            continue
        value = src.get(code)
        if value not in (None, ""):
            carry.append((code, ttype, value))
    carry += [(c, t, v) for c, t, v in extra_ifd0]

    with open(path, "r+b") as fh:
        buf = bytearray(fh.read())

    if buf[:2] != b"II":
        return                      # written little-endian here; nothing else
    ifd0 = struct.unpack_from("<I", buf, 4)[0]
    count = struct.unpack_from("<H", buf, ifd0)[0]
    entries = [bytes(buf[ifd0 + 2 + i * 12: ifd0 + 14 + i * 12])
               for i in range(count)]
    nxt = struct.unpack_from("<I", buf, ifd0 + 2 + count * 12)[0]
    # The carried tags replace whatever is there. None of them is a tag
    # `_dng_tags` writes, so the only thing being displaced is a default
    # tifffile put in -- and its Software says "tifffile.py", where the source
    # says which firmware took the frame. The camera's answer is the true one.
    replace = {c for c, _, _ in carry} | {34665}
    entries = [e for e in entries
               if struct.unpack_from("<H", e, 0)[0] not in replace]

    def put(raw):
        """Park a value past the end and return where it went."""
        if len(buf) % 2:
            buf.append(0)           # IFDs and their values start on a word
        at = len(buf)
        buf.extend(raw)
        return at

    if len(buf) % 2:
        buf.append(0)
    exif_at = len(buf)
    buf.extend(ifd.tobytes(exif_at))

    for code, ttype, value in carry:
        entries.append(_encode_ifd0(code, ttype, value, put))
    entries.append(struct.pack("<HHII", 34665, 4, 1, exif_at))

    # TIFF wants the entries in ascending tag order, and readers that binary
    # search the IFD -- LibRaw among them -- quietly miss the ones that are not.
    entries.sort(key=lambda e: struct.unpack_from("<H", e, 0)[0])

    if len(buf) % 2:
        buf.append(0)
    at = len(buf)
    buf.extend(struct.pack("<H", len(entries)))
    buf.extend(b"".join(entries))
    buf.extend(struct.pack("<I", nxt))
    struct.pack_into("<I", buf, 4, at)

    with open(path, "wb") as fh:
        fh.write(bytes(buf))


def _dng_tags(path):
    """Everything that makes the output a DNG, ready for `tifffile`.

    One list, two sources, and the caller does not care which. A DNG is copied
    from the file's own tags -- the real numbers, passed through with their own
    code, type and count, so nothing is re-derived on the way. A camera raw has
    no DNG tags to copy, so they are worked out from what LibRaw knows.

    This used to be exiftool's job, running after the pixels were on disk.
    Writing them here instead means the file is a correct DNG the moment it is
    closed, and copying the tags verbatim turns out to be more faithful than
    asking exiftool to restore them: developing the output against the original
    now differs by exactly nothing, where the old path left a residue.
    """
    if kind(path) == "raw":
        return _dng_tags_from_raw(path)
    import tifffile
    out = []
    with tifffile.TiffFile(path) as t:
        p0 = t.pages[0]
        # A camera keeps the mosaic and its levels in a SubIFD and the colour
        # in IFD0. What this tool writes has one IFD with everything in it, and
        # gets re-read whenever a crop follows an align -- so every tag is
        # looked for in both places rather than assuming the camera's layout.
        # `p0.pages` is None, not empty, when there are no SubIFDs.
        pages = [p0] + list(p0.pages or ())
        for code in _DNG_IFD0_TAGS + _DNG_FROM_SUBIFD:
            for pg in pages:
                tag = pg.tags.get(code)
                if tag is not None:
                    out.append((tag.code, int(tag.dtype), tag.count,
                                tag.value, True))
                    break
    missing = [c for c in _DNG_ESSENTIAL if c not in {t[0] for t in out}]
    if missing:
        raise RuntimeError(
            "%s is missing DNG tags %s, so the output would open and develop "
            "wrong rather than fail" % (os.path.basename(path), missing))
    return out


def _dng_tags_from_raw(path):
    """The DNG tags a camera raw does not carry, worked out from LibRaw.

    A NEF has no `ColorMatrix1`: exiftool asked for one returns nothing. What
    it has is a maker's own encoding, and what LibRaw has is Adobe's published
    matrix for that camera, which is the same number a converter would write.
    So the colour characterisation is not invented here, it is looked up.

    Verified on a Nikon D5600: developing the same frame from the NEF and from
    the DNG this produces differs by 0.009 levels out of 255 on average, and
    98.2% of pixels come out identical. That residue is demosaic rounding.
    """
    import rawpy
    with rawpy.imread(path) as r:
        desc = r.color_desc.decode()
        pattern = bytes({"R": 0, "G": 1, "B": 2}[desc[i]]
                        for i in r.raw_pattern.flatten())
        black = tuple(int(b) for b in r.black_level_per_channel)
        # LibRaw reports two whites: `white_level`, 16383 here, which is what
        # the 14-bit container holds, and `camera_white_level_per_channel`,
        # 15311, where this sensor actually saturates. The second looks more
        # honest and was tried; it is wrong for this purpose. LibRaw develops
        # the NEF itself against 16383, so declaring anything else makes our
        # DNG render brighter than the file it came from -- measured, the mean
        # difference went from 0.006 levels out of 255 to 3.15. The promise
        # here is that the DNG develops like the original, and that is the
        # number that keeps it.
        white = int(r.white_level)
        wb = np.asarray(r.camera_whitebalance[:3], float)
        # AsShotNeutral is the reciprocal of the camera's multipliers,
        # normalised on green: the two say the same thing upside down.
        neutral = (wb[1] / np.where(wb == 0, 1.0, wb))
        matrix = np.asarray(r.rgb_xyz_matrix)[:3]      # XYZ -> camera

    def rat(v, d=10000):
        return [(int(round(x * d)), d) for x in np.asarray(v).flatten()]

    return [
        (50706, "B", 4, b"\x01\x04\x00\x00", True),      # DNGVersion 1.4.0.0
        (50707, "B", 4, b"\x01\x01\x00\x00", True),      # backward to 1.1
        (33421, "H", 2, (2, 2), True),                      # CFARepeatPatternDim
        (33422, "B", 4, pattern, True),                     # CFAPattern
        (50713, "H", 2, (2, 2), True),                      # BlackLevelRepeatDim
        (50714, "I", 4, black, True),                       # BlackLevel
        (50717, "I", 1, white, True),                       # WhiteLevel
        (50721, "2i", 9, rat(matrix), True),                # ColorMatrix1
        (50778, "H", 1, 21, True),                          # illuminant: D65
        (50728, "2I", 3, rat(neutral), True),               # AsShotNeutral
    ]


def write_dng_shifted(in_path, out_path, dy, dx):
    """Write a DNG with its mosaic shifted, keeping the original's metadata.

    The mosaic goes out uncompressed in a single IFD -- legal DNG, and far less
    machinery than reproducing a camera's SubIFD-plus-previews layout. The
    embedded JPEG previews are dropped deliberately: they show the *unshifted*
    frame, so carrying them would be carrying a lie.

    The DNG tags are written with the pixels, by `_dng_tags`; the EXIF follows
    in `attach_exif`. Verified afterwards by LibRaw: pattern RGBG, black 1024,
    white 16383, develops.
    """
    import tifffile
    cfa, _, _ = read_cfa(in_path)
    out = np.clip(np.rint(shift_cfa(cfa, dy, dx)), 0, 65535).astype(np.uint16)
    # Through a temporary and renamed on success, so a failure leaves nothing
    # behind rather than a file that opens and is wrong.
    tmp = out_path + ".part"
    tifffile.imwrite(tmp, out, photometric="cfa", planarconfig="contig",
                     compression=None, extratags=_dng_tags(in_path))

    # All that is left is the EXIF, a nested IFD that tifffile will not write.
    extra = ()
    if kind(in_path) == "raw":
        # rawpy does not expose the camera's name and the DNG spec wants one,
        # so it is built from the two IFD0 tags that do carry it.
        from PIL import Image
        ex = Image.open(in_path).getexif()
        who = ("%s %s" % (ex.get(271, ""), ex.get(272, ""))).strip()
        if who:
            extra = ((50708, 2, who),)
    attach_exif(tmp, in_path, extra)
    os.replace(tmp, out_path)


def read_image(path):
    """Full image as float32 (H, W, C). Not available for DNG."""
    k = kind(path)
    if k == "exr":
        import imagecodecs
        with open(path, "rb") as fh:
            return np.asarray(imagecodecs.exr_decode(fh.read()), np.float32)
    if k == "jpg":
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    raise ValueError("%s cannot be loaded as a continuous image (%s)" % (path, k))


def write_image(a, out_path, bits=16, quality=95):
    """Write a float image back out in the format its extension names."""
    k = kind(out_path)
    if k == "exr":
        import imagecodecs
        dtype = np.float16 if bits == 16 else np.float32
        data = imagecodecs.exr_encode(
            np.ascontiguousarray(a.astype(dtype)), compression="zip")
        with open(out_path, "wb") as fh:
            fh.write(data)
        return
    if k == "jpg":
        from PIL import Image
        b = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(b).save(out_path, quality=quality, subsampling=0)
        return
    raise ValueError("cannot write %s (%s is not a writable format)" % (out_path, k))


def crop_frame(in_path, out_path, top, bottom, left, right, bits=16, quality=95):
    """Trim a frame to a rectangle, in its own format.

    For a DNG the offsets are rounded **down to even numbers** first: the crop
    origin has to land on the same Bayer phase or every colour in the file
    shifts by one photosite, which a developer then renders with red and blue
    swapped. Losing at most one pixel a side is the cheaper mistake.
    """
    if kind(in_path) in ("dng", "raw"):
        import tifffile
        cfa, _, _ = read_cfa(in_path)
        top, left = (top // 2) * 2, (left // 2) * 2
        # ActiveArea and the default crop describe the *original* frame, so
        # they are wrong on a trimmed one. They used to be written and then
        # deleted again; now they are simply not written.
        keep = [t for t in _dng_tags(in_path)
                if t[0] not in (50829, 50719, 50720)]
        tmp = out_path + ".part"
        tifffile.imwrite(tmp, cfa[top:bottom, left:right].astype(np.uint16),
                         photometric="cfa", planarconfig="contig",
                         compression=None, extratags=keep)
        attach_exif(tmp, in_path)
        os.replace(tmp, out_path)
        return
    write_image(read_image(in_path)[top:bottom, left:right], out_path, bits, quality)


def frame_shape(path):
    """(height, width) of a frame in its own full-resolution pixels."""
    luma = read_luma(path, 1)
    sc = luma_scale(path)
    return luma.shape[0] * sc, luma.shape[1] * sc


def write_offsets(path, paths, offsets):
    """Record the measured per-frame shift, in original-frame pixels.

    The output for a format that cannot be written back, and useful on its own:
    the numbers can be applied at develop time, or fed to an editor.
    """
    import json
    rows = [{"file": os.path.basename(p),
             "dx": None if np.isnan(dx) else round(float(dx), 3),
             "dy": None if np.isnan(dy) else round(float(dy), 3)}
            for p, (dy, dx) in zip(paths, offsets)]
    with open(path, "w") as fh:
        json.dump({"unit": "pixels", "convention":
                   "add (dx, dy) to the frame to bring the subject on target",
                   "frames": rows}, fh, indent=1)
    return len(rows)
