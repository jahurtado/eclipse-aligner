"""Align an astronomical timelapse.

The package is three steps kept deliberately apart -- `detect` measures, `gui`
corrects by hand, `apply` moves the pixels -- over four support modules:
`centre_disc` (the estimators), `register_exr` (phase correlation, for when
there is no disc), `image_io` (EXR, DNG, camera raw and JPEG) and `alignment` (the
document format).
"""

__version__ = "1.0.0"
