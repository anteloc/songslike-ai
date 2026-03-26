#!/bin/bash

script_dir="$(dirname "$(realpath "$0")")"

src="$1"
output_dir="$2"

if [ -z "$src" ] || [ -z "$output_dir" ]; then
    echo "Usage: $0 <audio_file_with_lyrics | audio_files_with_lyrics_dir> <output_directory>"
    echo "Processes a single or all audio files in the given source directory (non-recursively) and writes the .mus.txt and .lrc.txt outputs to the specified output directory."
    echo "Example: $0 /path/to/audio_file.flac /path/to/output_dir"
    echo "Example: $0 /path/to/audio_files_with_lyrics /path/to/output_dir"
    echo "Note: The audio file or source directory should contain pairs of audio file and lyrics with the same stem, e.g. 'song.mp3' and 'song.lrc'."
    exit 1
fi

# verify that src exists
if [ ! -e "$src" ]; then
    echo "Error: Source '$src' not found."
    exit 1
fi

src="$(realpath "$src")"

# If src is a directory, process all audio files in it (non-recursively)
if [ -d "$src" ]; then
    audio_files=$(find "$src" -maxdepth 1 -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.wav" \))
    total=$(echo "$audio_files" | wc -l)

    echo "Found $total audio files in directory '$src'."
fi

python $script_dir/audio_lyrics_process.py "$src" "$output_dir" --no-ts

