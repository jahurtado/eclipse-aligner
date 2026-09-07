#!/usr/bin/env python3
"""One command, three subcommands, in the order they are meant to be run.

    eclipse-aligner                            # the editor, asking for a clip
    eclipse-aligner CLIP_DIR                   # the editor, on that clip
    eclipse-aligner detect CLIP_DIR -o alignment.json
    eclipse-aligner apply  alignment.json -o OUT_DIR

Naming no command opens the editor, because that is what somebody who typed
the name alone is after, and it is the only thing a double-clicked icon can
ask for.

They are subcommands and not flags of a single run because the separation is
the whole design: measuring is slow and fallible, moving pixels is fast and
irreversible, and a person reads the document in between. `detect --gui` is
the one shortcut offered, and it is safe precisely because the editor writes
no image -- it only edits the document.
"""

import argparse
import sys
from importlib import metadata

from . import apply as apply_cmd
from . import detect, gui, journal

def identity():
    """(author, site) read from the package, so they are written once.

    The version already comes from `__init__` and the author from the project
    metadata, so nothing that shows them -- the terminal, the About box, the
    window -- carries a copy of its own to fall out of date.
    """
    try:
        m = metadata.metadata("eclipse-aligner")
        who = m["Author"] or ", ".join(
            a.split("<")[0].strip() for a in m.get_all("Author-email") or [])
        site = ""
        for u in m.get_all("Project-URL") or []:
            if u.lower().startswith("homepage"):
                site = u.split(",", 1)[1].strip()
    except metadata.PackageNotFoundError:
        who, site = "", ""
    return who, site.replace("https://", "")


def credits():
    """The one-line form, for the terminal."""
    return " \u00b7 ".join(x for x in identity() if x)


COMMANDS = (
    ("detect", detect, "Measure where the subject is in every frame."),
    ("gui", gui, "Inspect and hand-correct an alignment document."),
    ("apply", apply_cmd, "Move the frames according to a document."),
    ("journal", journal, "Read back what was done in an editing session."),
)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="eclipse-aligner",
        description="Align a sequence of frames: measure where the subject "
                    "is, correct by hand what the detector missed, then move "
                    "the pixels.",
        epilog=credits(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version="%(prog)s " + __import__(
                        "eclipse_aligner").__version__ + "\n" + credits())
    sub = ap.add_subparsers(dest="command", required=False, metavar="COMMAND")
    for name, mod, blurb in COMMANDS:
        p = sub.add_parser(
            name, help=blurb, description=mod.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
        mod.add_args(p)
        p.set_defaults(_run=mod.run)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # No command means the editor. Opening a window is what somebody who typed
    # the name alone is after, and it is the only thing a double-clicked icon
    # can ask for -- a Dock has no subcommand to type. A path is a path and not
    # a misspelt command, so that goes to the editor too; anything starting
    # with a dash belongs to argparse, so --help and --version still answer.
    known = {name for name, _, _ in COMMANDS}
    if not argv or (not argv[0].startswith("-") and argv[0] not in known):
        argv = ["gui"] + argv
    args = build_parser().parse_args(argv)
    try:
        return args._run(args)
    except FileNotFoundError as e:
        # A missing document or missing frames is a normal thing to get wrong,
        # not a bug: say what is missing and stop, without a traceback.
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
