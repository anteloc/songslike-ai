#!/usr/bin/env python

from __future__ import annotations

import math
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import librosa


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SubtitleEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str


# ── Timestamp / subtitle parsers ──────────────────────────────────────────────

TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)
LRC_TIME_RE = re.compile(r"\[(\d{1,3}):(\d{2})\.(\d{2})\]")
LRC_META_RE = re.compile(r"\[(\w+):(.+)\]")


def parse_timestamp_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return (
        int(h) * 3600 * 1000
        + int(m) * 60 * 1000
        + int(s) * 1000
        + int(ms)
    )


def format_ms_to_timestamp(ms: int) -> str:
    ms = max(0, ms)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(srt_text: str) -> List[SubtitleEntry]:
    blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.MULTILINE)
    entries: List[SubtitleEntry] = []
    for block in blocks:
        lines = [line.rstrip("\n\r") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        try:
            index = int(lines[0].strip())
            time_line = lines[1].strip()
            text = "\n".join(lines[2:]).strip()
        except ValueError:
            index = len(entries) + 1
            time_line = lines[0].strip()
            text = "\n".join(lines[1:]).strip()
        match = TIME_RE.match(time_line)
        if not match:
            raise ValueError(f"Invalid SRT time line: {time_line}")
        start_ms = parse_timestamp_to_ms(*match.groups()[0:4])
        end_ms   = parse_timestamp_to_ms(*match.groups()[4:8])
        if end_ms <= start_ms:
            continue
        entries.append(SubtitleEntry(index=index, start_ms=start_ms, end_ms=end_ms, text=text))
    entries.sort(key=lambda e: (e.start_ms, e.end_ms))
    return entries


def parse_lrc_meta(lrc_text: str) -> dict[str, str]:
    tag_map = {"ar": "Artist", "ti": "Title", "al": "Album"}
    meta: dict[str, str] = {}
    for line in lrc_text.splitlines():
        match = LRC_META_RE.match(line.strip())
        if match and not LRC_TIME_RE.match(line.strip()):
            key, value = match.group(1).lower(), match.group(2).strip()
            if key in tag_map:
                meta[tag_map[key]] = value
    return meta


def parse_lrc(lrc_text: str) -> List[SubtitleEntry]:
    timed: List[tuple[int, str]] = []
    for line in lrc_text.splitlines():
        line = line.strip()
        match = LRC_TIME_RE.match(line)
        if not match:
            continue
        m, s, cs = match.groups()
        start_ms = int(m) * 60000 + int(s) * 1000 + int(cs) * 10
        text = line[match.end():].strip()
        if text:
            timed.append((start_ms, text))
    entries: List[SubtitleEntry] = []
    for i, (start_ms, text) in enumerate(timed):
        end_ms = timed[i + 1][0] if i + 1 < len(timed) else int(1e9)
        if end_ms <= start_ms:
            continue
        entries.append(SubtitleEntry(index=i + 1, start_ms=start_ms, end_ms=end_ms, text=text))
    return entries


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


def write_fp_output(
    entries: List[SubtitleEntry],
    no_ts: bool,
    meta: dict[str, str] | None,
    global_tokens: dict[str, str] | None,
) -> str:
    """
    Write the acoustic fingerprint file (.fp.txt).

    ┌─────────────────────────────────────────────────────────┐
    │  FILE FORMAT: <stem>.fp.txt                             │
    │─────────────────────────────────────────────────────────│
    │  NOTE                                                   │
    │  Artist: <artist>                                       │
    │  Title:  <title>                                        │
    │  Album:  <album>                                        │
    │  Tempo:  bpm:fast | bpm:mid | bpm:slow                  │
    │  Key:    key:A | key:A# | … | key:G#                    │
    │  Mode:   mode:major | mode:minor                        │
    │                                                         │
    │  1                                                      │
    │  00:00:00,000 --> 00:00:04,000   (omitted with --no-ts) │
    │  {bpm:fast DOOM tsing ahhh shaa dum}                    │
    │                                                         │
    │  2                                                      │
    │  00:00:04,000 --> 00:00:08,000                          │
    │  Lyric line text {bpm:fast DOOM tsing ahhh shaa dum}    │
    │  …                                                      │
    └─────────────────────────────────────────────────────────┘

    Indexed by rag_solution.py as the ACOUSTIC index.
    The {…} token blocks are the only content read by fp_to_acoustic_prose().
    Lyric text present on the same line as a token block is ignored by the
    acoustic indexer — it exists only for human readability.
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
    lyric_entries: List[SubtitleEntry],
    meta: dict[str, str] | None,
    global_tokens: dict[str, str] | None,
) -> str:
    """
    Write the plain-lyrics companion file (.lrc.txt).

    ┌─────────────────────────────────────────────────────────┐
    │  FILE FORMAT: <stem>.lrc.txt                            │
    │─────────────────────────────────────────────────────────│
    │  NOTE                                                   │
    │  Artist: <artist>                                       │
    │  Title:  <title>                                        │
    │  Album:  <album>                                        │
    │  Tempo:  bpm:fast | bpm:mid | bpm:slow                  │
    │  Key:    key:A | key:A# | … | key:G#                    │
    │  Mode:   mode:major | mode:minor                        │
    │                                                         │
    │  <lyric line 1>                                         │
    │  <lyric line 2>                                         │
    │  …                                                      │
    │                                                         │
    │  NOTE: Songs with no lyrics get a single line:          │
    │  instrumental                                           │
    └─────────────────────────────────────────────────────────┘

    Indexed by rag_solution.py as the LYRIC index.
    No acoustic token blocks appear here.  The NOTE header (Tempo/Key/Mode)
    is kept so that even purely lyric-based queries still benefit from the
    song-level acoustic context when the two indexes are merged.
    Timestamps are intentionally omitted — only the text matters.
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


# ── Timeline helpers ──────────────────────────────────────────────────────────

def safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size else 0.0


def ms_to_sample(ms: int, sr: int, total_samples: int) -> int:
    sample = int(ms * sr / 1000)
    return max(0, min(sample, total_samples))


def make_gap_entry(start_ms: int, end_ms: int) -> SubtitleEntry | None:
    if end_ms <= start_ms:
        return None
    return SubtitleEntry(index=0, start_ms=start_ms, end_ms=end_ms, text="")


def build_full_timeline_entries(
    lyric_entries: List[SubtitleEntry],
    audio_duration_ms: int,
) -> List[SubtitleEntry]:
    if not lyric_entries:
        gap = make_gap_entry(0, audio_duration_ms)
        return [gap] if gap else []

    result: List[SubtitleEntry] = []
    intro = make_gap_entry(0, lyric_entries[0].start_ms)
    if intro:
        result.append(intro)

    prev_end = lyric_entries[0].end_ms
    result.append(lyric_entries[0])

    for entry in lyric_entries[1:]:
        gap = make_gap_entry(prev_end, entry.start_ms)
        if gap:
            result.append(gap)
        result.append(entry)
        prev_end = max(prev_end, entry.end_ms)

    outro = make_gap_entry(lyric_entries[-1].end_ms, audio_duration_ms)
    if outro:
        result.append(outro)

    cleaned: List[SubtitleEntry] = []
    for entry in result:
        start_ms = max(0, min(entry.start_ms, audio_duration_ms))
        end_ms   = max(0, min(entry.end_ms,   audio_duration_ms))
        if end_ms <= start_ms:
            continue
        if cleaned and start_ms < cleaned[-1].end_ms:
            start_ms = cleaned[-1].end_ms
        if end_ms <= start_ms:
            continue
        cleaned.append(SubtitleEntry(index=0, start_ms=start_ms, end_ms=end_ms, text=entry.text))
    return cleaned


SUBWINDOW_MS = 4000


def subdivide_segments(entries: List[SubtitleEntry], max_ms: int) -> List[SubtitleEntry]:
    result: List[SubtitleEntry] = []
    for entry in entries:
        duration = entry.end_ms - entry.start_ms
        if duration <= max_ms:
            result.append(entry)
            continue
        n = math.ceil(duration / max_ms)
        window = duration // n
        for i in range(n):
            start = entry.start_ms + i * window
            end   = entry.start_ms + (i + 1) * window if i < n - 1 else entry.end_ms
            result.append(SubtitleEntry(index=0, start_ms=start, end_ms=end, text=entry.text))
    return result


# ── Song-level global analysis ────────────────────────────────────────────────

_KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key_and_mode(y: np.ndarray, sr: int) -> tuple[str, str]:
    """Detect dominant key and mode (major/minor) from the full audio."""
    chroma     = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = chroma.mean(axis=1)

    best_key   = 0
    best_mode  = "major"
    best_score = -np.inf

    for root in range(12):
        major_score = float(np.dot(np.roll(_MAJOR_PROFILE, root), mean_chroma))
        minor_score = float(np.dot(np.roll(_MINOR_PROFILE, root), mean_chroma))
        if major_score > best_score:
            best_score = major_score
            best_key   = root
            best_mode  = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_key   = root
            best_mode  = "minor"

    return f"key:{_KEY_NAMES[best_key]}", f"mode:{best_mode}"


# ── Segment-level acoustic analysis ──────────────────────────────────────────

def analyze_segment(y: np.ndarray, sr: int) -> str:
    """Return acoustic syllable tokens for a single audio segment."""
    if y.size == 0 or float(np.max(np.abs(y))) < 1e-5:
        return "shhh"

    eps = 1e-10

    rms      = float(np.sqrt(np.mean(y ** 2) + eps))
    zcr      = safe_mean(librosa.feature.zero_crossing_rate(y))

    S         = np.abs(librosa.stft(y))
    centroid  = safe_mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    bandwidth = safe_mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    rolloff   = safe_mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    flatness  = safe_mean(librosa.feature.spectral_flatness(S=S))

    onset_env        = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean       = safe_mean(onset_env)
    onset_peak_ratio = (float(onset_env.max()) if onset_env.size else 0.0) / (onset_mean + eps)

    mean_mag    = S.mean(axis=1)
    mean_mag[0] = 0.0
    freqs       = librosa.fft_frequencies(sr=sr, n_fft=(S.shape[0] - 1) * 2)
    peak_freq   = float(freqs[np.argmax(mean_mag)])

    y_harm, _   = librosa.effects.hpss(y)
    total_power = float(np.mean(y ** 2)) + eps
    harm_ratio  = float(np.mean(y_harm ** 2)) / total_power

    syllables: List[str] = []

    # Energy — 4 tiers
    if rms > 0.20:
        syllables.append("DOOM")
    elif rms > 0.10:
        syllables.append("voom")
    elif rms > 0.04:
        syllables.append("meh")
    else:
        syllables.append("hmm")

    # Spectral centroid
    if centroid > 3200:
        syllables.append("tsee")
    elif centroid > 1800:
        syllables.append("tsing")
    else:
        syllables.append("bwoom")

    # Tonal character
    if flatness > 0.25 or zcr > 0.18:
        syllables.append("bzzra")
    elif bandwidth < 1200 and flatness < 0.12:
        syllables.append("ooh")

    # Harmonic vs. percussive
    if harm_ratio > 0.70:
        syllables.append("tonal")
    elif harm_ratio < 0.30:
        syllables.append("noisy")

    # Rhythmic attacks
    if onset_mean > 8.0 or (onset_mean > 3.0 and onset_peak_ratio > 5.0):
        syllables.append("tak-tak")
    else:
        syllables.append("ahhh")

    # Spectral spread / brightness
    if bandwidth > 2200 or rolloff > 5500:
        syllables.append("shaa")

    # Peak frequency
    if peak_freq > 550:
        syllables.append("weee")
    elif 0 < peak_freq < 180:
        syllables.append("dum")

    seen:  set[str]  = set()
    final: List[str] = []
    for s in syllables:
        if s not in seen:
            seen.add(s)
            final.append(s)

    return " ".join(final) if final else "mmm"


def combine_text(lyric_text: str, sound_text: str) -> str:
    lyric_text = lyric_text.strip()
    sound_text = sound_text.strip()
    if lyric_text:
        return f"{lyric_text} {{{sound_text}}}"
    return f"{{{sound_text}}}"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge lyric subtitles with acoustic syllable tokens.\n"
            "\n"
            "Produces TWO output files per song:\n"
            "  <stem>.fp.txt   — acoustic fingerprint (for sound-similarity index)\n"
            "  <stem>.lrc.txt  — plain lyrics companion (for lyric-similarity index)\n"
            "\n"
            "Both files share the same NOTE header (Artist/Title/Album/Tempo/Key/Mode)\n"
            "so that song-level context is available in both indexes.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_audio",  help="Input audio file (.mp3, .flac, etc.)")
    parser.add_argument("input_lyrics", help="Input lyric file (.srt or .lrc)")
    parser.add_argument(
        "output_stem",
        help=(
            "Output path stem WITHOUT extension. "
            "E.g. 'out/104.AC_DC-Thunderstruck' produces "
            "'out/104.AC_DC-Thunderstruck.fp.txt' and "
            "'out/104.AC_DC-Thunderstruck.lrc.txt'."
        ),
    )
    parser.add_argument(
        "-n", "--no-ts", action="store_true",
        help="Omit timestamps from the .fp.txt output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audio_path = Path(args.input_audio)
    lyrics_path = Path(args.input_lyrics)
    stem        = Path(args.output_stem)
    fp_path     = stem.with_suffix(".fp.txt")
    lrc_path    = stem.with_suffix(".lrc.txt")

    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}")
        return 1
    if not lyrics_path.exists():
        print(f"Error: lyrics file not found: {lyrics_path}")
        return 1

    suffix = lyrics_path.suffix.lower()
    if suffix not in (".srt", ".lrc"):
        print(f"Error: unsupported lyrics format '{suffix}' (expected .srt or .lrc)")
        return 1

    lyrics_text = lyrics_path.read_text(encoding="utf-8")

    if suffix == ".srt":
        lyric_entries = parse_srt(lyrics_text)
        meta = None
    else:
        lyric_entries = parse_lrc(lyrics_text)
        meta = parse_lrc_meta(lyrics_text) or None

    # ── Load audio and compute global song-level properties ───────────────
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    sr = int(sr)
    audio_duration_ms = int(len(y) * 1000 / sr)

    tempo_val, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo_val)[0])
    if bpm < 90:
        tempo_token = "bpm:slow"
    elif bpm < 130:
        tempo_token = "bpm:mid"
    else:
        tempo_token = "bpm:fast"

    key_token, mode_token = detect_key_and_mode(y, sr)

    global_tokens: dict[str, str] = {
        "Tempo": tempo_token,
        "Key":   key_token,
        "Mode":  mode_token,
    }

    # ── Build segment timeline and analyse each segment ───────────────────
    timeline_entries = build_full_timeline_entries(lyric_entries, audio_duration_ms)
    timeline_entries = subdivide_segments(timeline_entries, SUBWINDOW_MS)

    fp_entries: List[SubtitleEntry] = []
    for entry in timeline_entries:
        start_sample = ms_to_sample(entry.start_ms, sr, len(y))
        end_sample   = ms_to_sample(entry.end_ms,   sr, len(y))
        segment      = y[start_sample:end_sample]

        sound_desc  = f"{tempo_token} {analyze_segment(segment, sr)}"
        merged_text = combine_text(entry.text, sound_desc)

        fp_entries.append(SubtitleEntry(
            index=0,
            start_ms=entry.start_ms,
            end_ms=entry.end_ms,
            text=merged_text,
        ))

    # ── Write .fp.txt ─────────────────────────────────────────────────────
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(
        write_fp_output(fp_entries, no_ts=args.no_ts, meta=meta, global_tokens=global_tokens),
        encoding="utf-8",
    )
    print(f"Wrote acoustic fingerprint : {fp_path}")

    # ── Write .lrc.txt ────────────────────────────────────────────────────
    lrc_path.write_text(
        write_lyrics_output(lyric_entries, meta=meta, global_tokens=global_tokens),
        encoding="utf-8",
    )
    print(f"Wrote lyrics companion     : {lrc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())