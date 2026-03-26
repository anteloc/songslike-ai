#!/bin/bash

script_dir="$(dirname "$(realpath "$0")")"

source_dir="$1"
output_dir="$2"

if [ -z "$source_dir" ] || [ -z "$output_dir" ]; then
    echo "Usage: $0 <audio_files_with_lyrics_dir> <output_directory>"
    echo "Processes all audio files in the given source directory (non-recursively) and writes the .mus.txt and .lrc.txt outputs to the specified output directory."
    echo "Example: $0 /path/to/audio_files_with_lyrics /path/to/output_dir"
    echo "Note: The source directory should contain pairs of audio file and lyrics with the same stem, e.g. 'song.mp3' and 'song.lrc'."
    exit 1
fi

# verify that source_dir exists
if [ ! -d "$source_dir" ]; then
    echo "Error: Source directory '$source_dir' not found."
    exit 1
fi

source_dir="$(realpath "$source_dir")"

# If source_dir is a directory, process all audio files in it (non-recursively)
audio_files=$(find "$source_dir" -maxdepth 1 -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.wav" \))
total=$(echo "$audio_files" | wc -l)

echo "Found $total audio files in directory '$source_dir'."

python $script_dir/audio_lyrics_process.py "$source_dir" "$output_dir"

