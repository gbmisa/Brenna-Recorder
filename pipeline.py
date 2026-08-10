#!/usr/bin/env python3
"""Plaud audio -> diarized transcript -> structured Obsidian note.

Usage:
    python pipeline.py recording.wav [more.wav ...]
    python pipeline.py --no-llm recording.wav     # transcript only
    python pipeline.py --force recording.wav      # overwrite existing vault files
"""
import argparse
import datetime as dt
import json
import re
import shutil
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
CFG = tomllib.loads((ROOT / "config.toml").read_text())

VAULT = Path(CFG["paths"]["vault"])
INBOX = VAULT / CFG["paths"]["inbox"]
TRANSCRIPTS = VAULT / CFG["paths"]["transcripts"]
ARCHIVE = VAULT / CFG["paths"]["audio_archive"]


def slugify(text, limit=60):
    s = re.sub(r"[^\w\s-]", "", text).strip()
    return re.sub(r"[\s_]+", "-", s)[:limit] or "untitled"


def hhmmss(seconds):
    return str(dt.timedelta(seconds=int(seconds)))


def yaml_scalar(value):
    """Double-quoted YAML scalar safe for frontmatter injection."""
    s = str(value)
    escaped = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def unique_path(path: Path) -> Path:
    """Return path, or path-2 / path-3 / ... if it already exists."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def require_fresh(path: Path, force: bool) -> Path:
    """Refuse to clobber existing vault files unless force is set."""
    if path.exists() and not force:
        raise FileExistsError(
            f"{path.relative_to(VAULT)} already exists "
            f"(pass --force to overwrite)"
        )
    return path


# --------------------------------------------------------------------------- ASR

def transcribe(audio_path, diarize=False):
    import whisperx

    w = CFG["whisper"]
    device = "cuda"
    print(f"  loading whisper {w['model']} ...", flush=True)
    model = whisperx.load_model(
        w["model"], device, compute_type=w["compute_type"],
        language=w["language"] or None,
    )
    audio = whisperx.load_audio(str(audio_path))
    duration = len(audio) / 16000

    print(f"  transcribing {hhmmss(duration)} of audio ...", flush=True)
    result = model.transcribe(audio, batch_size=w["batch_size"])
    lang = result["language"]

    print("  aligning word timestamps ...", flush=True)
    align_model, meta = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device,
                            return_char_alignments=False)

    if diarize:
        print("  diarizing (this is the slow part) ...", flush=True)
        import os
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("  ! HF_TOKEN unset - skipping diarization", file=sys.stderr)
        else:
            try:
                from whisperx.diarize import DiarizationPipeline
            except ImportError:
                from whisperx import DiarizationPipeline
            dia = DiarizationPipeline(use_auth_token=token, device=device)
            result = whisperx.assign_word_speakers(dia(audio), result)

    return {"segments": result["segments"], "language": lang, "duration": duration}


def transcript_markdown(asr, para_chars=None):
    """Speaker-grouped, timestamped prose.

    Consecutive segments from one speaker merge into a readable paragraph, but a
    new paragraph starts once the block exceeds `para_chars`. Both halves matter:
    merging keeps the transcript readable for a human, while the size cap
    guarantees paragraph breaks for chunk_text() to split on. Without the cap, a
    non-diarized transcript (every segment labelled SPEAKER) collapses into one
    unsplittable blob and map/reduce silently stops working.
    """
    if para_chars is None:
        para_chars = CFG["whisper"].get("paragraph_chars", 1000)

    blocks, speaker, buf, start, size = [], None, [], 0.0, 0
    for seg in asr["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        spk = seg.get("speaker", "SPEAKER")
        if buf and (spk != speaker or size + len(text) > para_chars):
            blocks.append(f"**[{hhmmss(start)}] {speaker}:** {' '.join(buf)}")
            buf, size = [], 0
        if not buf:
            speaker, start = spk, seg["start"]
        buf.append(text)
        size += len(text) + 1
    if buf:
        blocks.append(f"**[{hhmmss(start)}] {speaker}:** {' '.join(buf)}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- LLM

def ollama(prompt, system=None):
    cfg = CFG["llm"]
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": prompt})
    payload = json.dumps({
        "model": cfg["model"], "messages": msgs, "stream": False,
        "think": cfg.get("think", False),
        "options": {"temperature": cfg["temperature"]},
    }).encode()
    req = urllib.request.Request(
        f"{cfg['host']}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["message"]["content"].strip()


SYSTEM = (
    "You turn meeting and interview transcripts into structured notes that a "
    "colleague will use to build workplace training materials. You are precise, "
    "you never invent content that is not in the transcript, and you quote verbatim "
    "when quoting. Output GitHub-flavored Markdown only."
)

STRUCTURE_PROMPT = """Below is a transcript. Produce a structured note with exactly these sections:

## Summary
Three to five sentences. What was discussed and why it matters.

## Key Concepts
Bullet list. Each bullet names one distinct concept, process, or term worth its own
wiki page, formatted as `[[Concept Name]]` followed by " — " and a one-line definition
drawn from the transcript. These become pages in a knowledge wiki, so name them
generally (e.g. `[[Escalation Path]]`, not `[[the escalation thing Bob mentioned]]`).

## Decisions & Action Items
Bullet list. Mark owners if named. Write "None stated" if there are none.

## Training Material Angles
Bullet list. Concrete suggestions for turning this content into training material:
what format suits it, which audience, and which part of the transcript is the source.

## Notable Quotes
Two to five verbatim quotes worth reusing. Each must carry a timestamp that appears
VERBATIM in the transcript above - use the timestamp of the speaker block the quote is
taken from. Never interpolate, estimate, or invent a timestamp.

Transcript:
---
{body}
---"""

REDUCE_PROMPT = """These are structured notes from consecutive chunks of one long recording.
Merge them into a single note using the same section headings. Deduplicate concepts,
combine action items, and keep the strongest quotes. Do not invent anything.

{body}"""


def fill_body(template, body):
    """Insert body without str.format so braces in the transcript are safe."""
    return template.replace("{body}", body)


def _hard_split(text, limit):
    """Split text into pieces of at most `limit` characters, preferring newlines."""
    if len(text) <= limit:
        return [text] if text else []
    pieces = []
    rest = text
    while rest:
        if len(rest) <= limit:
            pieces.append(rest)
            break
        cut = rest.rfind("\n", 0, limit + 1)
        if cut < limit // 2:
            cut = rest.rfind(" ", 0, limit + 1)
        if cut < limit // 2:
            cut = limit
        pieces.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
        if rest.startswith(" "):
            rest = rest.lstrip(" ")
    return pieces


def chunk_text(text, limit):
    """Pack paragraphs into chunks of at most `limit` chars; hard-split oversize ones."""
    if len(text) <= limit:
        return [text]

    paras = text.split("\n\n")
    chunks, cur = [], ""
    for p in paras:
        parts = _hard_split(p, limit) if len(p) > limit else [p]
        for part in parts:
            # room for separator if cur is non-empty
            sep = "\n\n" if cur else ""
            if cur and len(cur) + len(sep) + len(part) > limit:
                chunks.append(cur)
                cur = part
            else:
                cur = cur + sep + part if cur else part
    if cur.strip():
        chunks.append(cur)
    return chunks


def structure(transcript):
    limit = CFG["llm"]["chunk_chars"]
    if len(transcript) <= limit:
        return ollama(fill_body(STRUCTURE_PROMPT, transcript), SYSTEM)

    chunks = chunk_text(transcript, limit)
    print(f"  long transcript - map/reduce over {len(chunks)} chunks", flush=True)
    partials = []
    for i, c in enumerate(chunks, 1):
        print(f"    chunk {i}/{len(chunks)}", flush=True)
        partials.append(ollama(fill_body(STRUCTURE_PROMPT, c), SYSTEM))
    return ollama(fill_body(REDUCE_PROMPT, "\n\n---\n\n".join(partials)), SYSTEM)


# --------------------------------------------------------------------------- main

def process(audio_path, use_llm=True, diarize=False, force=False):
    audio_path = Path(audio_path)
    print(f"\n=> {audio_path.name}")
    asr = transcribe(audio_path, diarize=diarize)
    body = transcript_markdown(asr)

    stamp = dt.datetime.fromtimestamp(audio_path.stat().st_mtime)
    date = stamp.strftime("%Y-%m-%d")
    title = f"{date} {audio_path.stem}"
    slug = slugify(title)

    for d in (INBOX, TRANSCRIPTS, ARCHIVE):
        d.mkdir(parents=True, exist_ok=True)

    tpath = require_fresh(TRANSCRIPTS / f"{slug}-transcript.md", force)
    tpath.write_text(
        f"---\ntitle: {yaml_scalar(title + ' (transcript)')}\ndate: {yaml_scalar(date)}\n"
        f"duration: {yaml_scalar(hhmmss(asr['duration']))}\n"
        f"language: {yaml_scalar(asr['language'])}\n"
        f"source: plaud\ntype: transcript\n---\n\n{body}\n"
    )
    print(f"  transcript -> {tpath.relative_to(VAULT)}")

    if use_llm:
        print("  structuring via ollama ...", flush=True)
        note = structure(body)
        npath = require_fresh(INBOX / f"{slug}.md", force)
        npath.write_text(
            f"---\ntitle: {yaml_scalar(title)}\ndate: {yaml_scalar(date)}\n"
            f"duration: {yaml_scalar(hhmmss(asr['duration']))}\nsource: plaud\n"
            f"type: recording-note\nstatus: unreviewed\n"
            f"transcript: {yaml_scalar('[[' + tpath.stem + ']]')}\n---\n\n"
            f"# {title}\n\n{note}\n\n---\nFull transcript: [[{tpath.stem}]]\n"
        )
        print(f"  note       -> {npath.relative_to(VAULT)}")

    # Date-prefix archive names so same basenames from different folders don't collide.
    dest = unique_path(ARCHIVE / f"{date}-{audio_path.name}")
    if audio_path.resolve() != dest.resolve():
        shutil.copy2(audio_path, dest)
        print(f"  audio      -> {dest.relative_to(VAULT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--no-llm", action="store_true", help="transcript only")
    ap.add_argument("--diarize", action="store_true", help="speaker labels (needs HF_TOKEN)")
    ap.add_argument(
        "--force", action="store_true",
        help="overwrite existing vault notes/transcripts",
    )
    args = ap.parse_args()

    diarize = args.diarize or CFG["whisper"]["diarize"]
    failed = 0
    for a in args.audio:
        try:
            process(a, use_llm=not args.no_llm, diarize=diarize, force=args.force)
        except Exception as e:
            failed += 1
            print(f"  ! FAILED {a}: {e}", file=sys.stderr)
    print(f"\nDone. {len(args.audio) - failed}/{len(args.audio)} succeeded.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
