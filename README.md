# songslike-ai

Find songs that **sound like** a given track — not by metadata or genre tags, but by analysing the audio itself.

The tool converts audio files into text-based acoustic fingerprints, indexes them in a vector database, and lets you retrieve the closest matches using a combined acoustic + lyric similarity score.

---

## How it works

```
Audio (.flac/.mp3/…) + Lyrics (.lrc)
           │
           ▼
  audio_lyrics_process.py
           │
     ┌─────┴──────┐
     ▼            ▼
 .mus.txt      .lrc.txt
 acoustic      lyric
 fingerprint   companion
     │            │
     └─────┬──────┘
           ▼
         rag.py index
           │
     ┌─────┴──────┐
     ▼            ▼
 acoustic      lyric
  index        index
  (ChromaDB)   (ChromaDB)
     │            │
     └─────┬──────┘
           ▼
    rag.py retrieve
           │
           ▼
   ranked similar songs
```

### Stage 1 — Acoustic fingerprinting (`audio_lyrics_process.py`)

Each song is analysed with [librosa](https://librosa.org) and converted to two plain-text files:

**`<stem>.mus.txt`** — acoustic fingerprint indexed for sound similarity
- Global NOTE header: `Tempo`, `Key`, `Mode`, `Instrumentation`
- Per-segment descriptors covering the full song in ~4 s windows, each labelled with:
  - Energy: `explosive` / `loud` / `moderate` / `soft`
  - Brightness: `crisp` / `bright` / `dark`
  - Body: `warm`
  - Tonal character: `gritty` / `clean`
  - Harmonic content: `melodic` / `percussive`
  - Rhythmic feel: `dense driving` / `punchy` / `sparse` / `smooth`
  - Spread: `full`
  - Register: `airy` / `bassy`

**`<stem>.lrc.txt`** — stop-word-filtered lyrics indexed for lyric similarity

Instrumentation is detected heuristically from the audio signal alone (no tags or metadata required):
- `electric guitar` — high ZCR + spectral flatness + mid-range body
- `synthesizer` — low ZCR + tonal purity + presence of drum energy
- `drums` — percussive energy fraction + irregular beat spacing
- `drum machine` — percussive energy fraction + near-perfect beat regularity

### Stage 2 — Indexing (`rag.py index`)

The two text files per song are embedded and stored in two separate [ChromaDB](https://www.trychroma.com) collections via [LlamaIndex](https://www.llamaindex.ai):
- `<name>_acoustic` — built from `.mus.txt` files
- `<name>_lyric` — built from `.lrc.txt` files

### Stage 3 — Retrieval (`rag.py retrieve`)

Both indexes are queried independently and merged by song stem:

```
final_score = alpha × acoustic_score + (1 − alpha) × lyric_score
```

Default `alpha = 0.6` (acoustic slightly dominant). Songs with no lyrics are never silently penalised — their NOTE header still provides acoustic context.

---

## Installation

```bash
pip install librosa numpy tqdm \
            llama-index llama-index-vector-stores-chroma chromadb \
            openai tiktoken
export OPENAI_API_KEY="your-key-here"
```

---

## Usage

### 1. Generate fingerprints

```bash
# Single file (companion .lrc in same directory used automatically)
python audio_lyrics_process.py song.flac ./fingerprints/

# Whole directory
python audio_lyrics_process.py ./music/ ./fingerprints/
```

Supported audio formats: `.mp3`, `.flac`, `.wav`, `.ogg`, `.m4a`, `.aac`, `.opus`, `.aiff`

### 2. Build the index

```bash
# Single directory with both .mus.txt and .lrc.txt
python rag.py index songs --dir ./fingerprints/

# Separate directories
python rag.py index songs --mus-dir ./mus/ --lrc-dir ./lrcs/
```

### 3. Retrieve similar songs

```bash
# Using a fingerprint file as query
python rag.py retrieve songs --file song.mus.txt --top-k 10

# Show individual acoustic/lyric scores
python rag.py retrieve songs --file song.mus.txt --top-k 10 --scores

# Free-text query
python rag.py retrieve songs "fast minor key hard rock"

# Pure acoustic retrieval (ignore lyrics)
python rag.py retrieve songs --file song.mus.txt --alpha 1.0
```

Example output (`--scores`):

```
#      final    acou   lyric  stem
------------------------------------------------------------
1     1.0000  1.0000  1.0000  104.AC_DC-Thunderstruck
2     0.3421  0.4102  0.2301  67.Wolfmother-Joker_And_The_Thief
3     0.2918  0.3550  0.1901  88.ACDC-Back_In_Black
```

---

## File format reference

### `.mus.txt`

```
NOTE
Artist: AC/DC
Title: Thunderstruck
Album: The Razors Edge
Tempo: fast tempo
Key: D#
Mode: minor
Instrumentation: electric guitar, drums

1
00:00:00,000 --> 00:00:04,000
fast tempo, soft, crisp, warm, gritty, smooth, full, airy

2
00:00:04,000 --> 00:00:08,000
fast tempo, loud, bright, warm, gritty, sparse, full, bassy
…
```

### `.lrc.txt`

```
NOTE
Artist: AC/DC
Title: Thunderstruck
Tempo: fast tempo
Key: D#
Mode: minor
Instrumentation: electric guitar, drums

I was caught
In the middle of a railroad track
…
```

---

## Estimate token usage

```bash
python rag.py estimate --dir ./fingerprints/
```

---

## Design notes

- **No chunking** — each song document is kept as a single vector. The default LlamaIndex chunk size is raised to 10 240 tokens so long fingerprints are never split, preserving the temporal arc of the song.
- **Temporal arc** — per-segment descriptors are joined in chronological order (`Arc: … — … — …`). Two songs may share the same vocabulary yet differ in *when* they are loud, bright, or sparse.
- **Instrumentation** — detected from raw audio signals (ZCR, spectral flatness, HPSS percussive fraction, beat regularity) with no external model or metadata dependency. Emitted three times in the acoustic prose to outweigh shared tonal/dynamic terms.
- **Parallel processing** — fingerprint generation uses `ProcessPoolExecutor` across all CPU cores.
