#!/usr/bin/env python3
"""Watch Inbox/ for dropped audio files and run them through the pipeline.

Drop an mp3/wav/m4a/flac into Inbox/, and once its size stops changing (so a
still-copying file isn't picked up mid-write) it gets processed like a normal
pipeline.py run. On success the source file is moved into Converted/, renamed
to match the note/transcript slug. On failure it's moved into Failed/ instead,
so a permanently-broken file doesn't get retried forever.

Usage:
    python watch.py                  # diarize setting comes from config.toml
    python watch.py --diarize
    python watch.py --no-llm
    python watch.py --interval 10    # poll every 10s instead of the 5s default
    python watch.py --once           # process what's there now, then exit
                                      # (still waits for file sizes to stop
                                      # changing first, same as the daemon -
                                      # a manual trigger doesn't guarantee the
                                      # file has finished copying/downloading)
"""
import argparse
import shutil
import sys
import time

import pipeline

ROOT = pipeline.ROOT
INBOX = ROOT / "Inbox"
CONVERTED = ROOT / "Converted"
FAILED = ROOT / "Failed"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def audio_files():
    return sorted(f for f in INBOX.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS)


def stable_files(prev_sizes):
    """Return (files whose size matches the last poll, sizes seen this poll)."""
    ready, current = [], {}
    for f in audio_files():
        size = f.stat().st_size
        current[f] = size
        if prev_sizes.get(f) == size:
            ready.append(f)
    return ready, current


def process_file(f, use_llm, diarize):
    print(f"\n=> picked up {f.name}", flush=True)
    try:
        result = pipeline.process(f, use_llm=use_llm, diarize=diarize)
        dest = pipeline.unique_path(CONVERTED / f"{result['slug']}{f.suffix}")
        shutil.move(str(f), dest)
        print(f"  converted  -> {dest.relative_to(ROOT)}", flush=True)
    except Exception as e:
        print(f"  ! FAILED {f.name}: {e}", file=sys.stderr)
        dest = pipeline.unique_path(FAILED / f.name)
        shutil.move(str(f), dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-llm", action="store_true", help="transcript only")
    ap.add_argument("--diarize", action="store_true", help="speaker labels (needs HF_TOKEN)")
    ap.add_argument("--interval", type=float, default=5.0, help="poll interval in seconds")
    ap.add_argument("--once", action="store_true", help="process what's in Inbox/ now, then exit")
    args = ap.parse_args()

    diarize = args.diarize or pipeline.CFG["whisper"]["diarize"]
    use_llm = not args.no_llm

    for d in (INBOX, CONVERTED, FAILED):
        d.mkdir(parents=True, exist_ok=True)

    if args.once:
        files = audio_files()
        if not files:
            print(f"nothing to process in {INBOX}")
            return 0
        print(f"found {len(files)} file(s), confirming they're done copying ...", flush=True)
        prev_sizes = {}
        while True:
            ready, prev_sizes = stable_files(prev_sizes)
            if set(ready) == set(files):
                break
            time.sleep(args.interval)
        for f in files:
            process_file(f, use_llm, diarize)
        return 0

    print(f"watching {INBOX} (ctrl-c to stop) ...", flush=True)
    prev_sizes = {}
    while True:
        ready, prev_sizes = stable_files(prev_sizes)
        for f in ready:
            process_file(f, use_llm, diarize)
            prev_sizes.pop(f, None)
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped.")
