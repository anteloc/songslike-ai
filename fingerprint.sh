#!/bin/bash

script_dir="$(dirname "$(realpath "$0")")"

# Create fingerprints for every given audio file in the given directory (non-recursively), or single file if a file is given

function fingerprint_file() {
    local audio_file="$1"
    local base_audio_name="$(basename "$audio_file")"
    local lyrics_file="${audio_file%.*}.lrc"
    local fingerprint_stem="${audio_file%.*}"
    echo "Processing: '$base_audio_name'..."
    
    echo "Downloading lyrics for '$base_audio_name'..."
    $script_dir/download_lyrics.sh "$audio_file" "$lyrics_file"

    echo "Generating fingerprint for '$base_audio_name', stem: '$fingerprint_stem'..."
    python $script_dir/singing_srt_audio_merge.py "$audio_file" "$lyrics_file" "$fingerprint_stem" --no-ts
}

target="$1"

if [ -z "$target" ]; then
    echo "Usage: $0 <audio_file_or_directory>"
    exit 1
fi

target="$(realpath "$target")"

# verify that target exists
if [ ! -e "$target" ]; then
    echo "Error: Target '$target' not found."
    exit 1
fi

# If target is a file, process it and early exit
if [ -f "$target" ]; then
    fingerprint_file "$target"
    exit 0
fi

# If target is a directory, process all audio files in it (non-recursively)
audio_files=$(find "$target" -maxdepth 1 -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.wav" \))
counter=1
total=$(echo "$audio_files" | wc -l)

echo "Found $total audio files in directory '$target'."

echo "$audio_files" | while read -r audio_file; do
    echo "Processing file $counter of $total: '$audio_file'"
    fingerprint_file "$audio_file"
    counter=$((counter + 1))
done

