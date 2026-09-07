#!/usr/bin/env python3
"""A record of what the hand did, so a session can be read back afterwards.

The editor's whole point is the handful of frames a person fixes by hand, and
until now that work left no trace but its result: the document says a frame is
`manual` and nothing says *how* it got there -- which key, in what order, after
looking at what. That is exactly what is needed to narrate a session later, or
to find out why a stretch came out wrong, or to cut a screencast against the
actions instead of against the clock.

So every action is appended here as one JSON object per line. One line per
event keeps it readable with `tail -f` while the window is open, appendable
without rewriting, and analysable with nothing but `json.loads` in a loop.

Each event carries **where the session was**, not only what was pressed: the
frame on screen, the selection it landed on, which body the arrows were moving,
which view was up, and the sun and moon as they stood *after* the action. That
snapshot is what makes the log stand on its own -- a reader can replay the
session without the document, and without guessing what a `nudge` did.

Writing must never be able to stop the editing. Any failure -- a read-only
folder, a full disc -- disables the journal, says so once, and the window
carries on: a lost log is an annoyance, a lost correction is the work.
"""

import json
import os
import sys
import time


DEFAULT_RETAIN = "1d"

_UNITS = {"m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


def parse_retain(text):
    """`1d`, `12h`, `2w` -> seconds. `off` or `none` -> None, keep everything.

    A bare number is days, because days is the unit anybody reaches for when
    they say how long a log should live.
    """
    if text is None:
        return None
    t = str(text).strip().lower()
    if t in ("off", "none", "never", "forever"):
        return None
    unit = _UNITS.get(t[-1:], None)
    n = t[:-1] if unit else t
    try:
        v = float(n)
    except ValueError:
        raise ValueError("cannot read %r as a retention: try 1d, 12h, 2w, "
                         "or off" % text)
    if v < 0:
        raise ValueError("a retention cannot be negative: %r" % text)
    return v * (unit or _UNITS["d"])


def ended(run):
    """When a session's last event happened, in epoch seconds, or None.

    The `open` event carries the wall clock and everything after it carries an
    offset, so the end is the one plus the other. None when the line cannot be
    read: a session that cannot be dated is never one to throw away.
    """
    if not run:
        return None
    started = run[0].get("started")
    if not started:
        return None
    try:
        t0 = time.mktime(time.strptime(started, "%Y-%m-%dT%H:%M:%S"))
    except (ValueError, OverflowError):
        return None
    return t0 + max(e.get("t", 0) for e in run)


def default_path(document):
    """`alignment.json` -> `alignment.log.jsonl`, beside the document.

    Beside it, and named after it, for the same reason the document lives in
    the clip's folder: the log belongs to *this* clip's alignment, and finding
    it should not need a second thing to remember.
    """
    stem = os.path.splitext(document)[0]
    return stem + ".log.jsonl"


class Journal:
    """Appends events to a JSONL file. Never raises at the caller."""

    def __init__(self, path, header=None, retain=None):
        self.path = path
        self.t0 = time.time()
        self.n = 0
        self._f = None
        # Before the file is opened for appending, not after: pruning renames a
        # new file over the old one, and a handle already held would go on
        # writing into the inode that just stopped being the log.
        if retain is not None and os.path.isfile(path):
            try:
                dropped = prune_older(path, retain)
                if dropped:
                    print("journal: dropped %d session%s older than the "
                          "retention" % (dropped, "" if dropped == 1 else "s"),
                          file=sys.stderr)
            except OSError as e:
                print("journal: could not prune %s: %s" % (path, e),
                      file=sys.stderr)
        try:
            self._f = open(path, "a", encoding="utf-8")
        except OSError as e:
            self._give_up(e)
            return
        h = dict(header or {})
        h["started"] = time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(self.t0))
        self.event("open", **h)

    # ------------------------------------------------------------------ write
    def event(self, do, **detail):
        """One action. `do` is its name; everything else is its detail.

        Times are seconds since the session opened rather than wall clock:
        what a screencast needs is the offset from the start of the recording,
        and the wall clock is in the `open` event for anyone who wants it.
        """
        if self._f is None:
            return
        rec = {"t": round(time.time() - self.t0, 3), "do": do}
        rec.update({k: v for k, v in detail.items() if v is not None})
        try:
            self._f.write(json.dumps(rec, ensure_ascii=False,
                                     default=float) + "\n")
            # Flushed every time: a session ends by killing the window as often
            # as by closing it, and an event written at human speed is far too
            # cheap for the buffering to be worth a truncated log.
            self._f.flush()
        except (OSError, TypeError, ValueError) as e:
            self._give_up(e)
            return
        self.n += 1

    def close(self, why="closed"):
        if self._f is None:
            return
        self.event("close", why=why, events=self.n)
        try:
            self._f.close()
        except OSError:
            pass
        self._f = None

    def _give_up(self, e):
        self._f = None
        print("journal off (%s): %s" % (self.path, e), file=sys.stderr)


def read(path):
    """Every event in a log, bad lines skipped rather than fatal.

    A log is appended to while a window is open and read while it still is, so
    a half-written last line is normal, not corruption.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def sessions(events):
    """Split a log into its sessions: one per `open`.

    The file is appended to across runs on purpose -- a clip's history is one
    story -- so anything counting actions has to know where a run ended.
    """
    out, cur = [], None
    for e in events:
        if e.get("do") == "open" or cur is None:
            cur = []
            out.append(cur)
        cur.append(e)
    return out


# ---------------------------------------------------------------- the command
def _size(b):
    return "%.1f kB" % (b / 1000.0) if b >= 1000 else "%d B" % b


def _clock(t):
    return "%d:%05.2f" % (int(t // 60), t % 60)


def _detail(e):
    """The event's own arguments, without the context every event carries."""
    skip = {"t", "do", "i", "file", "sel", "body", "view", "mode", "px",
            "dirty", "skip", "sun", "moon", "note"}
    out = ["%s=%s" % (k, v) for k, v in e.items() if k not in skip]
    if e.get("note"):
        out.append("\u2014 %s" % e["note"])
    return " ".join(out)


def _timeline(evs, out):
    # Which body, which view, which blend: carried on every event, so printing
    # them all would be a wall of unchanging columns. Printed when they change,
    # which is the only time they are news.
    was = {}
    for e in evs:
        do = e.get("do", "?")
        if do == "open":
            print("--- %s  %s  (%s frames)"
                  % (e.get("started", "?"), e.get("document", "?"),
                     e.get("frames", "?")), file=out)
            was = {}
            continue
        where = e.get("file", "")
        sel = e.get("sel")
        if sel:
            where += "  [%d-%d, %d]" % tuple(sel)
        bits = []
        for k in ("body", "view", "mode", "px"):
            if k in e and e[k] != was.get(k):
                bits.append("%s\u2192%s" % (k, e[k]))
                was[k] = e[k]
        d = " ".join(bits + [_detail(e)]).strip()
        print("%9s  %-16s %-28s %s"
              % (_clock(e.get("t", 0)), do, where, d), file=out)


def _summary(evs, out):
    from collections import Counter
    c = Counter(e["do"] for e in evs if e.get("do") not in ("open", "close"))
    span = max((e.get("t", 0) for e in evs), default=0)
    touched = {e["file"] for e in evs if e.get("do") not in
               ("open", "close", "frame", "focus") and e.get("file")}
    seen = {e["file"] for e in evs if e.get("file")}
    print("%s over %s, %d frames looked at, %d acted on"
          % ("%d actions" % sum(c.values()), _clock(span), len(seen),
             len(touched)), file=out)
    for do, n in c.most_common():
        print("  %-20s %4d" % (do, n), file=out)


def prune(path, keep):
    """Leave only the newest `keep` sessions in a log.

    Rewritten through a temporary file and renamed over, never truncated in
    place: the one thing worse than a log too long is half a log. Even so, do
    it with the window closed -- an editor still appending to the old inode
    would go on writing into a file nobody can find.

    At about 260 bytes an event, a heavy session is a few hundred kilobytes
    next to a clip of raws that is tens of gigabytes, so this exists for when
    somebody wants a clean slate, not because the file is a problem.
    """
    runs = sessions(read(path))
    if len(runs) <= keep:
        return 0, len(runs)
    # `runs[-0:]` is the whole list, not none of it -- the one slice that does
    # not mean what it reads like, and the one that would empty nothing.
    _rewrite(path, runs[-keep:] if keep else [])
    return len(runs) - keep, keep


def prune_older(path, seconds, now=None):
    """Drop the sessions that ended longer ago than `seconds`. Returns how many.

    By age rather than by count because that is what a retention means: a log
    should stop holding what is too old to be about the work in hand, however
    many sessions that turns out to be.

    A session that cannot be dated is kept. Deleting on a timestamp that could
    not be read is the one failure mode with no way back.
    """
    runs = sessions(read(path))
    cut = (now if now is not None else time.time()) - seconds
    keep = [r for r in runs if (ended(r) or float("inf")) >= cut]
    if len(keep) == len(runs):
        return 0
    _rewrite(path, keep)
    return len(runs) - len(keep)


def _rewrite(path, runs):
    """Replace a log with these sessions, atomically.

    Through a temporary and a rename: the one thing worse than a log too long
    is half a log, and a prune interrupted half way leaves the old one whole.
    """
    tmp = path + ".new"
    with open(tmp, "w", encoding="utf-8") as f:
        for run in runs:
            for e in run:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def add_args(ap):
    """The options of `eclipse-aligner journal`."""
    ap.add_argument("log", help="The log file, or the clip, or the document "
                                "it sits beside.")
    ap.add_argument("--session", type=int, metavar="N",
                    help="Which run to read, 1 being the oldest. The last one "
                         "by default.")
    ap.add_argument("--all", action="store_true", help="Every run in the file.")
    ap.add_argument("--summary", action="store_true",
                    help="Counts and elapsed time instead of the timeline.")
    ap.add_argument("--prune", action="store_true",
                    help="Throw away all but the newest sessions and stop. Do "
                         "it with the editor closed.")
    ap.add_argument("--keep", type=int, metavar="N",
                    help="Prune by count instead: how many of the newest "
                         "sessions to leave. 0 empties the log.")
    ap.add_argument("--older-than", default=DEFAULT_RETAIN, metavar="AGE",
                    help="What --prune throws away: sessions that ended "
                         "longer ago than this (default %s). 1d, 12h, 2w; a "
                         "bare number is days." % DEFAULT_RETAIN)


def run(args):
    from . import alignment
    path = args.log
    if not os.path.isfile(path) or not path.endswith(".jsonl"):
        path = default_path(alignment.document_path(path))
    if not os.path.isfile(path):
        raise FileNotFoundError("no journal at %s" % path)
    if args.prune:
        was = os.path.getsize(path)
        if args.keep is not None:
            if args.keep < 0:
                print("--keep takes 0 or more", file=sys.stderr)
                return 1
            dropped, _ = prune(path, args.keep)
        else:
            try:
                secs = parse_retain(args.older_than)
            except ValueError as e:
                print(e, file=sys.stderr)
                return 1
            if secs is None:
                print("--older-than off throws nothing away", file=sys.stderr)
                return 1
            dropped = prune_older(path, secs)
        left = len(sessions(read(path)))
        print("%s: dropped %d session%s, %d left, %s -> %s"
              % (path, dropped, "" if dropped == 1 else "s", left,
                 _size(was), _size(os.path.getsize(path))))
        return 0
    runs = sessions(read(path))
    if not runs:
        print("%s is empty" % path, file=sys.stderr)
        return 1
    if args.all:
        pick = runs
    elif args.session:
        if not 1 <= args.session <= len(runs):
            print("%s holds %d session%s" % (path, len(runs),
                                             "" if len(runs) == 1 else "s"),
                  file=sys.stderr)
            return 1
        pick = [runs[args.session - 1]]
    else:
        pick = [runs[-1]]
    for evs in pick:
        (_summary if args.summary else _timeline)(evs, sys.stdout)
    if len(runs) > len(pick):
        print("(%d of %d sessions; --all for the rest)"
              % (len(pick), len(runs)), file=sys.stderr)
    return 0
