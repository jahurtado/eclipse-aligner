"""Entry point for the macOS bundle.

There is nothing to decide here any more. `cli.main` already treats a missing
command as the editor and a bare path as a clip to open, so a bundle is the
same program with one extra chore: macOS hands some launches a Process Serial
Number, which is neither a command nor a path and must not be taken for one.
"""

import sys

from eclipse_aligner.cli import main

if __name__ == "__main__":
    sys.exit(main([a for a in sys.argv[1:] if not a.startswith("-psn_")]))
