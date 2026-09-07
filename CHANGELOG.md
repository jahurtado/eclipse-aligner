# Changelog

## 1.0.0

The first release.

Three steps and one document between them: the first pass measures a clip and
writes `alignment.json`, the window corrects the frames the detector got wrong,
and moving the pixels is a separate, deliberate step. The originals are never
written over.

- Reads EXR, DNG, Nikon NEF, Canon CR2/CR3 and JPEG, and writes each back in
  its own format. A camera raw comes out as DNG.
- Shifts a mosaic one Bayer sub-plane at a time, so a DNG stays a legal CFA
  file with its metadata intact.
- Three detection methods: two disc estimators, circle and ellipse, and phase
  correlation for frames with no disc in them.
- The document records one size per clip, and the moon as an offset from the
  alignment rather than in absolute pixels, so the two can be edited in either
  order and land in the same place.
- The window keeps a session journal of everything done in it.
- Desktop builds for Apple Silicon and x64 Windows, each carrying its own
  Python.
