#!/usr/bin/env python3
"""
Start the queue worker to process training jobs.

This script starts a background worker that continuously polls the job database
and executes pending training jobs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.jobs.worker import main

if __name__ == "__main__":
    main()
