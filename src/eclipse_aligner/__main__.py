"""`python -m eclipse_aligner` -- the same entry point as the installed command."""

import sys

from .cli import main

sys.exit(main())
