#!/bin/bash
# Manual fallback for when the watch.py auto-watcher isn't running or isn't
# picking files up. Processes whatever is sitting in this Inbox/ folder right
# now, then archives/moves the results as usual. Trigger it via right-click
# in Nautilus -> Scripts -> "Process Brenna Recorder Inbox" (not by
# double-clicking this file - GNOME Files won't run .sh scripts that way).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
"$ROOT/.venv/bin/python" watch.py --once --diarize
echo
read -p "Done. Press Enter to close..."
