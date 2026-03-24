#!/usr/bin/env python
from __future__ import annotations

import math
import sys
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import librosa


@dataclass
class SubtitleEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str


TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)
LRC_TIME_RE = re.compile(r"\[(\d{1,3}):(\d{2})\.(\d{2})\]")


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
        lines = [line.rstrip("\n\r") for line in block.splitlines() if line.strip() != ""]
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
        end_ms = parse_timestamp_to_ms(*match.groups()[4:8])

        if end_ms <= start_ms:
            continue

        entries.append(
            SubtitleEntry(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                text=text,
            )
        )

    entries.sort(key=lambda e: (e.start_ms, e.end_ms))
    return entries


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


def write_output(entries: List[SubtitleEntry], no_ts: bool = False) -> str:
    blocks = []

    for i, entry in enumerate(entries, start=1):
        if no_ts:
            block = "\n".join(
                [
                    str(i),
                    entry.text,
                ]
            )
        else:
            block = "\n".join(
                [
                    str(i),
                    f"{format_ms_to_timestamp(entry.start_ms)} --> {format_ms_to_timestamp(entry.end_ms)}",
                    entry.text,
                ]
            )
        blocks.append(block)

    return "\n\n".join(blocks) + "\n"


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
        end_ms = max(0, min(entry.end_ms, audio_duration_ms))
        if end_ms <= start_ms:
            continue

        if cleaned and start_ms < cleaned[-1].end_ms:
            start_ms = cleaned[-1].end_ms
            if end_ms <= start_ms:
                continue

        cleaned.append(
            SubtitleEntry(
                index=0,
                start_ms=start_ms,
                end_ms=end_ms,
                text=entry.text,
            )
        )

    return cleaned


SUBWINDOW_MS = 4000  # max segment duration before subdivision


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
            end = entry.start_ms + (i + 1) * window if i < n - 1 else entry.end_ms
            result.append(SubtitleEntry(index=0, start_ms=start, end_ms=end, text=entry.text))
    return result


def analyze_segment(y: np.ndarray, sr: int) -> str:
    """
    Return fake sung syllables that imitate the sound of the segment.
    """
    if y.size == 0 or float(np.max(np.abs(y))) < 1e-5:
        return "shhh"

    eps = 1e-10

    rms = float(np.sqrt(np.mean(y ** 2) + eps))
    zcr = safe_mean(librosa.feature.zero_crossing_rate(y))

    # Single STFT pass — reused by all spectral features
    S = np.abs(librosa.stft(y))
    centroid = safe_mean(librosa.feature.spectral_centroid(S=S, sr=sr))
    bandwidth = safe_mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    rolloff = safe_mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    flatness = safe_mean(librosa.feature.spectral_flatness(S=S))

    # Onset strength + peak-to-mean ratio as percussiveness indicator (replaces HPSS)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = safe_mean(onset_env)
    onset_peak_ratio = (float(onset_env.max()) if onset_env.size else 0.0) / (onset_mean + eps)

    # Dominant frequency via FFT peak (replaces expensive pyin)
    mean_mag = S.mean(axis=1)
    mean_mag[0] = 0.0  # exclude DC
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(S.shape[0] - 1) * 2)
    peak_freq = float(freqs[np.argmax(mean_mag)])

    syllables: List[str] = []

    if rms > 0.18:
        syllables.append("DOOM")
    elif rms > 0.08:
        syllables.append("voom")
    else:
        syllables.append("hmm")

    if centroid > 3200:
        syllables.append("tsee")
    elif centroid > 1800:
        syllables.append("tsing")
    else:
        syllables.append("bwoom")

    if flatness > 0.25 or zcr > 0.18:
        syllables.append("bzzra")
    elif bandwidth < 1200 and flatness < 0.12:
        syllables.append("ooh")

    if onset_mean > 8.0 or (onset_mean > 3.0 and onset_peak_ratio > 5.0):
        syllables.append("tak-tak")
    else:
        syllables.append("ahhh")

    if bandwidth > 2200 or rolloff > 5500:
        syllables.append("shaa")

    if peak_freq > 550:
        syllables.append("weee")
    elif 0 < peak_freq < 180:
        syllables.append("dum")

    seen = set()
    final_syllables: List[str] = []
    for s in syllables:
        if s not in seen:
            seen.add(s)
            final_syllables.append(s)

    return " ".join(final_syllables) if final_syllables else "mmm"


def combine_text(lyric_text: str, sound_text: str) -> str:
    lyric_text = lyric_text.strip()
    sound_text = sound_text.strip()

    if lyric_text:
        return f"{lyric_text} {{{sound_text}}}"
    return f"{{{sound_text}}}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge lyric subtitles with fake sung instrument-imitating syllables."
    )
    parser.add_argument("input_audio", help="Input audio file, e.g. song.mp3 or song.flac, etc.")
    parser.add_argument("input_lyrics", help="Input lyric file (.srt or .lrc)")
    parser.add_argument("output_file", help="Output file path")
    parser.add_argument(
        "-n",
        "--no-ts",
        action="store_true",
        help="Omit timestamps from the output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audio_path = Path(args.input_audio)
    lyrics_path = Path(args.input_lyrics)
    output_path = Path(args.output_file)

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
    lyric_entries = parse_srt(lyrics_text) if suffix == ".srt" else parse_lrc(lyrics_text)

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

    timeline_entries = build_full_timeline_entries(lyric_entries, audio_duration_ms)
    timeline_entries = subdivide_segments(timeline_entries, SUBWINDOW_MS)

    output_entries: List[SubtitleEntry] = []

    for entry in timeline_entries:
        start_sample = ms_to_sample(entry.start_ms, sr, len(y))
        end_sample = ms_to_sample(entry.end_ms, sr, len(y))
        segment = y[start_sample:end_sample]

        sound_description = f"{tempo_token} {analyze_segment(segment, sr)}"
        merged_text = combine_text(entry.text, sound_description)

        output_entries.append(
            SubtitleEntry(
                index=0,
                start_ms=entry.start_ms,
                end_ms=entry.end_ms,
                text=merged_text,
            )
        )

    output_text = write_output(output_entries, no_ts=args.no_ts)
    output_path.write_text(output_text, encoding="utf-8")

    if args.no_ts:
        print(f"Wrote output without timestamps to: {output_path}")
    else:
        print(f"Wrote output with timestamps to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())