# Brenna Recorder

Recorder audio → diarized transcript → structured Obsidian note, fully local on the 5090.

Lives at `~/AI projects/Brenna Recorder/`. Both path components contain spaces, so
quote them in the shell.

The Obsidian vault is deliberately NOT here — it stays at `~/Obsidian/Karpathy-Wiki/`.

## Run it

```bash
cd ~/"AI projects/Brenna Recorder"
.venv/bin/python pipeline.py ~/path/to/recording.wav

# several at once
.venv/bin/python pipeline.py ~/Plaud-Dump/*.wav

# transcript only, skip the LLM pass
.venv/bin/python pipeline.py --no-llm recording.wav

# overwrite existing vault notes/transcripts (default is refuse)
.venv/bin/python pipeline.py --force recording.wav

# with speaker labels (needs HF_TOKEN, see below)
HF_TOKEN=hf_xxx .venv/bin/python pipeline.py --diarize recording.wav
```

## Output

Everything lands in `~/Obsidian/Karpathy-Wiki/`:

| Path | Contents |
|---|---|
| `Inbox/<date>-<name>.md` | Structured note — summary, `[[concepts]]`, action items, training angles, quotes |
| `Transcripts/<date>-<name>-transcript.md` | Full timestamped transcript, linked from the note |
| `Audio-Archive/` | Copy of the original audio |

## Configuration

All in `config.toml`. The knobs that matter:

- `whisper.model` — `large-v3`. No reason to go smaller on 32GB.
- `whisper.batch_size` — 16. Can raise if you want more speed.
- `llm.model` — `qwen3.6:35b-a3b` via Ollama.
- `llm.think` — `false`. qwen3.6 is a reasoning model; thinking roughly triples runtime
  for what is an extraction task. Flip to `true` if output quality disappoints.
- `llm.chunk_chars` — transcripts over this are map-reduced (summarize chunks, then merge).

## Verified working

- torch 2.11.0+cu128, sm_120 present → Blackwell fully supported
- WhisperX large-v3 loads on CUDA in ~7s, inference + wav2vec2 alignment OK
- Ollama structuring pass: ~6s for a 2-minute transcript, correct wikilinks and attribution
- End-to-end file writing, frontmatter, archiving

## Not yet verified

- **Real speech audio.** Only tested on a synthetic tone — no TTS on this box.
  First real Plaud recording is the actual test.
- **Diarization.** Wired but never run. Needs `HF_TOKEN` env var and acceptance of the
  pyannote model licences on HuggingFace. Off by default.

## Known benign warnings

`torchcodec` complains that ffmpeg 8.0.1 is newer than it supports (4–7). Harmless —
`whisperx.load_audio` shells out to the `ffmpeg` binary rather than using torchcodec.

## Ingest — still to be decided

Currently you pass files on the command line. The automatic path depends on whether the
NotePin exposes raw audio over USB mass storage:

- **If yes** — watch a folder, drop files in, done. Simplest.
- **If no** — use the [Riffado](https://github.com/riffado/riffado) CLI as a sync layer.
  It authenticates via the same OTP flow as the Plaud app, downloads *only raw audio*,
  and has `recordings list|get|download`. Needs Docker, which is not installed here.
