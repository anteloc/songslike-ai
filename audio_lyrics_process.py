#!/usr/bin/env python

from __future__ import annotations

import math
import re
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SubtitleEntry:
    start_ms: int
    end_ms: int
    text: str


# ── LRC parsers ───────────────────────────────────────────────────────────────

LRC_TIME_RE = re.compile(r"\[(\d{1,3}):(\d{2})\.(\d{2})\]")
LRC_META_RE = re.compile(r"\[(\w+):(.+)\]")


def parse_lrc_meta(lrc_text: str) -> dict[str, str]:
    tag_map = {"ar": "Artist", "ti": "Title", "al": "Album"}
    meta: dict[str, str] = {}
    for line in lrc_text.splitlines():
        match = LRC_META_RE.match(line.strip())
        if match:
            key, value = match.group(1).lower(), match.group(2).strip()
            if key in tag_map:
                meta[tag_map[key]] = value
    return meta


def parse_lrc(lrc_text: str) -> list[SubtitleEntry]:
    timed: list[tuple[int, str]] = []
    for line in lrc_text.splitlines():
        line = line.strip()
        m = LRC_TIME_RE.match(line)
        if not m:
            continue
        mins, secs, cs = m.groups()
        start_ms = int(mins) * 60000 + int(secs) * 1000 + int(cs) * 10
        text = line[m.end():].strip()
        if text:
            timed.append((start_ms, text))

    # pair each entry with the next's start as end_ms; last entry uses sentinel
    ends = [t[0] for t in timed[1:]] + [int(1e9)]
    return [
        SubtitleEntry(start, end, text)
        for (start, text), end in zip(timed, ends)
        if end > start
    ]


# ── Output writers ────────────────────────────────────────────────────────────

def _build_note_block(
    meta: dict[str, str] | None,
    global_tokens: dict[str, str] | None,
) -> str:
    """Build the shared NOTE header block used by both output files."""
    note_lines = ["NOTE"]
    if meta:
        note_lines += [f"{k}: {v}" for k, v in meta.items()]
    if global_tokens:
        note_lines += [f"{k}: {v}" for k, v in global_tokens.items()]
    return "\n".join(note_lines)


def format_ms_to_timestamp(ms: int) -> str:
    ms = max(0, ms)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_fp_output(
    entries: list[SubtitleEntry],
    no_ts: bool,
    meta: dict[str, str] | None,
    global_tokens: dict[str, str] | None,
) -> str:
    """
    Build the acoustic fingerprint file content (.fp.txt).

    ┌─────────────────────────────────────────────────────────────────┐
    │  FILE FORMAT: <stem>.fp.txt                                     │
    │─────────────────────────────────────────────────────────────────│
    │  NOTE                                                           │
    │  Artist: <artist>                                               │
    │  Title:  <title>                                                │
    │  Album:  <album>                                                │
    │  Tempo:  slow tempo | mid tempo | fast tempo                    │
    │  Key:    A | A# | … | G#                                        │
    │  Mode:   major | minor                                          │
    │                                                                 │
    │  1                                                              │
    │  00:00:00,000 --> 00:00:04,000   (omitted with --no-ts)         │
    │  fast tempo, loud, crisp, melodic, punchy, full, airy           │
    │  …                                                              │
    └─────────────────────────────────────────────────────────────────┘

    Indexed as the ACOUSTIC index.
    """
    blocks = []
    if meta or global_tokens:
        blocks.append(_build_note_block(meta, global_tokens))

    for i, entry in enumerate(entries, start=1):
        if no_ts:
            block = "\n".join([str(i), entry.text])
        else:
            block = "\n".join([
                str(i),
                f"{format_ms_to_timestamp(entry.start_ms)} --> {format_ms_to_timestamp(entry.end_ms)}",
                entry.text,
            ])
        blocks.append(block)

    return "\n\n".join(blocks) + "\n"


def write_lyrics_output(
    lyric_entries: list[SubtitleEntry],
    meta: dict[str, str] | None,
    global_tokens: dict[str, str] | None,
) -> str:
    """
    Build the plain-lyrics companion file content (.lrc.txt).

    ┌─────────────────────────────────────────────────────────┐
    │  FILE FORMAT: <stem>.lrc.txt                            │
    │─────────────────────────────────────────────────────────│
    │  NOTE                                                   │
    │  Artist: <artist>                                       │
    │  Title:  <title>                                        │
    │  Album:  <album>                                        │
    │  Tempo:  slow tempo | mid tempo | fast tempo            │
    │  Key:    A | A# | … | G#                                │
    │  Mode:   major | minor                                  │
    │                                                         │
    │  <lyric line 1>                                         │
    │  <lyric line 2>                                         │
    │  …                                                      │
    │                                                         │
    │  NOTE: Songs with no lyrics get a single line:          │
    │  instrumental                                           │
    └─────────────────────────────────────────────────────────┘

    Indexed as the LYRIC index. Timestamps are omitted — only the text matters.
    Duplicate lines (repeated chorus etc.) are deduplicated.
    The NOTE header is kept so lyric-based queries still see song-level context.
    """
    blocks = []
    if meta or global_tokens:
        blocks.append(_build_note_block(meta, global_tokens))

    seen: set[str] = set()
    unique_lines: list[str] = []
    for entry in lyric_entries:
        text = entry.text.strip()
        if text and text not in seen:
            seen.add(text)
            unique_lines.append(text)

    blocks.append("\n".join(unique_lines) if unique_lines else "instrumental")

    return "\n\n".join(blocks) + "\n"


# ── Timeline ──────────────────────────────────────────────────────────────────

def ms_to_sample(ms: int, sr: int, total_samples: int) -> int:
    return max(0, min(int(ms * sr / 1000), total_samples))


SUBWINDOW_MS = 4000
HOP_LENGTH   = 512   # STFT hop; all frame-aligned features share this value


def build_timeline(
    lyric_entries: list[SubtitleEntry],
    audio_duration_ms: int,
    max_segment_ms: int = SUBWINDOW_MS,
) -> list[SubtitleEntry]:
    """Build a gapless, subdivided analysis timeline from lyric entries.

    Fills intro/inter-lyric/outro gaps with empty-text entries, clamps everything
    to [0, audio_duration_ms], resolves overlaps, then breaks any segment longer
    than max_segment_ms into equal sub-windows.
    """
    # Build gapless raw timeline
    raw: list[SubtitleEntry] = []
    if not lyric_entries:
        raw.append(SubtitleEntry(0, audio_duration_ms, ""))
    else:
        prev_end = 0
        for entry in lyric_entries:
            if entry.start_ms > prev_end:
                raw.append(SubtitleEntry(prev_end, entry.start_ms, ""))
            raw.append(entry)
            prev_end = max(prev_end, entry.end_ms)
        if prev_end < audio_duration_ms:
            raw.append(SubtitleEntry(prev_end, audio_duration_ms, ""))

    # Clamp to audio bounds and resolve overlaps
    timeline: list[SubtitleEntry] = []
    for entry in raw:
        start = max(0, min(entry.start_ms, audio_duration_ms))
        end   = max(0, min(entry.end_ms,   audio_duration_ms))
        if timeline:
            start = max(start, timeline[-1].end_ms)
        if end > start:
            timeline.append(SubtitleEntry(start, end, entry.text))

    # Subdivide long segments into equal sub-windows
    result: list[SubtitleEntry] = []
    for entry in timeline:
        duration = entry.end_ms - entry.start_ms
        if duration <= max_segment_ms:
            result.append(entry)
            continue
        n = math.ceil(duration / max_segment_ms)
        window = duration // n
        for i in range(n):
            start = entry.start_ms + i * window
            end   = entry.start_ms + (i + 1) * window if i < n - 1 else entry.end_ms
            result.append(SubtitleEntry(start, end, entry.text))

    return result


# ── Song-level global analysis ────────────────────────────────────────────────

_KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key_and_mode(S: np.ndarray, sr: int) -> tuple[str, str]:
    """Detect dominant key and mode (major/minor) via Krumhansl-Schmuckler profiles.

    Accepts the precomputed magnitude STFT S to avoid recomputing it.
    Uses chroma_stft (reuses S) instead of chroma_cqt (separate expensive CQT).
    """
    mean_chroma = librosa.feature.chroma_stft(S=S, sr=sr).mean(axis=1)
    _, best_root, best_mode = max(
        (float(np.dot(np.roll(profile, root), mean_chroma)), root, mode)
        for root in range(12)
        for profile, mode in [(_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")]
    )
    return _KEY_NAMES[best_root], best_mode


# ── Segment-level acoustic analysis ──────────────────────────────────────────

def safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def analyze_segment(
    y: np.ndarray,         # time-domain slice — silence check + RMS only
    S: np.ndarray,         # magnitude STFT slice  [n_bins, n_frames]
    H: np.ndarray,         # harmonic magnitude STFT slice  [n_bins, n_frames]
    zcr_frames: np.ndarray,# ZCR frame slice  [n_frames]
    onset_env: np.ndarray, # onset envelope slice  [n_frames]
    freqs: np.ndarray,     # frequency bin centres  [n_bins]  — constant across segments
    sr: int,
) -> str:
    """Return a musician-style natural-language description for a single audio segment.

    All heavy arrays (S, H, zcr_frames, onset_env) are precomputed globally in
    process_song and sliced to this segment's frame range before calling here.

    Features and their descriptors:
      - RMS energy          → explosive / loud / moderate / soft
      - Spectral centroid   → crisp / bright / dark
      - Warmth (200–800 Hz) → warm  (conditional — emitted only when prominent)
      - Flatness + ZCR      → gritty / clean  (conditional)
      - Harmonic ratio      → melodic / percussive  (conditional)
      - Onset density       → dense driving / punchy / sparse / smooth
      - Spectral spread     → full  (conditional)
      - Peak frequency      → airy / bassy  (conditional)
    """
    if y.size == 0 or float(np.max(np.abs(y))) < 1e-5:
        return "silence"

    eps = 1e-10

    rms = float(np.sqrt(np.mean(y ** 2) + eps))
    zcr = safe_mean(zcr_frames)

    centroid  = safe_mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    bandwidth = safe_mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    rolloff   = safe_mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    flatness  = safe_mean(librosa.feature.spectral_flatness(S=S))

    onset_mean       = safe_mean(onset_env)
    onset_peak_ratio = (float(onset_env.max()) if onset_env.size else 0.0) / (onset_mean + eps)

    mean_mag    = S.mean(axis=1).copy()
    mean_mag[0] = 0.0   # zero DC bin — not musically meaningful
    peak_freq   = float(freqs[np.argmax(mean_mag)])

    # Warmth: 200–800 Hz band energy relative to overall mean
    low_mid_mask = (freqs >= 200) & (freqs <= 800)
    warmth_ratio = float(S[low_mid_mask].mean()) / (float(S.mean()) + eps)

    # Harmonic ratio from precomputed HPSS — equivalent to time-domain ratio by Parseval
    harm_ratio = float(np.mean(H ** 2)) / (float(np.mean(S ** 2)) + eps)

    parts: list[str] = []

    # Energy — 4 tiers
    if rms > 0.20:
        parts.append("explosive")
    elif rms > 0.10:
        parts.append("loud")
    elif rms > 0.04:
        parts.append("moderate")
    else:
        parts.append("soft")

    # Spectral centroid (overall brightness)
    if centroid > 3200:
        parts.append("crisp")
    elif centroid > 1800:
        parts.append("bright")
    else:
        parts.append("dark")

    # Warmth: low-mid body — orthogonal to brightness, e.g. a cello is dark AND warm
    if warmth_ratio > 1.2:
        parts.append("warm")

    # Tonal character (noise / distortion)
    if flatness > 0.25 or zcr > 0.18:
        parts.append("gritty")
    elif bandwidth < 1200 and flatness < 0.12:
        parts.append("clean")

    # Harmonic vs. percussive — orthogonal to gritty/clean, e.g. a distorted lead is melodic AND gritty
    if harm_ratio > 0.70:
        parts.append("melodic")
    elif harm_ratio < 0.30:
        parts.append("percussive")

    # Rhythmic character + onset density (combined: onset_mean captures both)
    if onset_mean > 8.0:
        parts.append("dense driving")
    elif onset_mean > 3.0 and onset_peak_ratio > 5.0:
        parts.append("punchy")
    elif onset_mean < 1.5:
        parts.append("sparse")
    else:
        parts.append("smooth")

    # Spectral spread
    if bandwidth > 2200 or rolloff > 5500:
        parts.append("full")

    # Peak frequency
    if peak_freq > 550:
        parts.append("airy")
    elif 0 < peak_freq < 180:
        parts.append("bassy")

    return ", ".join(parts) if parts else "uncharacterized"


# ── Per-song processing ───────────────────────────────────────────────────────

# Audio formats recognised when scanning an input directory
AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".opus", ".aiff"}


def process_song(
    audio_path: Path,
    lyrics_path: Path | None,
    output_dir: Path,
    no_ts: bool,
) -> None:
    """Process one audio file and write <stem>.mus.txt and <stem>.lrc.txt to output_dir.

    If lyrics_path is None (no matching .lrc found), lyric_entries will be empty
    and the .lrc.txt will contain only the NOTE header plus 'instrumental'.
    """
    mus_path = output_dir / audio_path.with_suffix(".mus.txt").name
    lrc_path = output_dir / audio_path.with_suffix(".lrc.txt").name

    if lyrics_path is not None:
        lyrics_text   = lyrics_path.read_text(encoding="utf-8")
        lyric_entries = parse_lrc(lyrics_text)
        meta          = parse_lrc_meta(lyrics_text) or None
    else:
        lyric_entries = []
        meta          = None

    # ── Load audio ────────────────────────────────────────────────────────
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    sr = int(sr)
    audio_duration_ms = int(len(y) * 1000 / sr)

    # ── Precompute all expensive features once from the full audio ────────
    # Every per-segment call previously recomputed STFT, HPSS, onset, and ZCR.
    # Here we compute each once and slice by frame range inside the loop.
    S_full    = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    H_full, _ = librosa.decompose.hpss(S_full)
    zcr_full  = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    freqs     = librosa.fft_frequencies(sr=sr, n_fft=(S_full.shape[0] - 1) * 2)

    # ── Global song-level properties ──────────────────────────────────────
    tempo_val, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    bpm = float(np.atleast_1d(tempo_val)[0])
    if bpm < 90:
        tempo_token = "slow tempo"
    elif bpm < 130:
        tempo_token = "mid tempo"
    else:
        tempo_token = "fast tempo"

    key_name, mode_name = detect_key_and_mode(S_full, sr)

    global_tokens: dict[str, str] = {
        "Tempo": tempo_token,
        "Key":   key_name,
        "Mode":  mode_name,
    }

    # ── Build segment timeline and analyse each segment ───────────────────
    timeline = build_timeline(lyric_entries, audio_duration_ms)

    mus_entries: list[SubtitleEntry] = []
    for entry in timeline:
        start_sample = ms_to_sample(entry.start_ms, sr, len(y))
        end_sample   = ms_to_sample(entry.end_ms,   sr, len(y))
        f0 = start_sample // HOP_LENGTH
        f1 = end_sample   // HOP_LENGTH
        seg_desc = analyze_segment(
            y[start_sample:end_sample],
            S_full[:, f0:f1],
            H_full[:, f0:f1],
            zcr_full[f0:f1],
            onset_env[f0:f1],
            freqs,
            sr,
        )
        sound_desc = f"{tempo_token}, {seg_desc}"
        mus_entries.append(SubtitleEntry(entry.start_ms, entry.end_ms, sound_desc))

    # ── Write outputs ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    mus_path.write_text(
        write_fp_output(mus_entries, no_ts=no_ts, meta=meta, global_tokens=global_tokens),
        encoding="utf-8",
    )
    print(f"  {mus_path}")

    lrc_path.write_text(
        write_lyrics_output(lyric_entries, meta=meta, global_tokens=global_tokens),
        encoding="utf-8",
    )
    print(f"  {lrc_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate acoustic fingerprint and lyrics files for a directory of songs.\n"
            "\n"
            "Scans input_dir for audio files and matches each with a same-stem .lrc file.\n"
            "Produces TWO output files per song:\n"
            "  <stem>.mus.txt  — acoustic fingerprint (for sound-similarity index)\n"
            "  <stem>.lrc.txt  — plain lyrics companion (for lyric-similarity index)\n"
            "\n"
            f"Recognised audio formats: {', '.join(sorted(AUDIO_EXTENSIONS))}\n"
            "Songs without a matching .lrc get a .lrc.txt with the NOTE header only.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_dir",  help="Directory containing audio + .lrc pairs")
    parser.add_argument("output_dir", help="Directory where output files will be written")
    parser.add_argument(
        "-n", "--no-ts", action="store_true",
        help="Omit timestamps from the .mus.txt output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        return 1

    audio_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"Error: no audio files found in {input_dir}")
        return 1

    print(f"Processing {len(audio_files)} song(s) → {output_dir}")

    for audio_path in audio_files:
        lyrics_path = audio_path.with_suffix(".lrc")
        if not lyrics_path.exists():
            print(f"[{audio_path.name}] (no .lrc found — instrumental)")
            lyrics_path = None
        else:
            print(f"[{audio_path.name}]")
        process_song(audio_path, lyrics_path, output_dir, no_ts=args.no_ts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
