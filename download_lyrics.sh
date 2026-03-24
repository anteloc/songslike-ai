#!/bin/bash

songslike_header="songslike-ai (https://github.com/anteloc/songslike-ai)"
lrclib_api="https://lrclib.net/api"

function lrc_length() {
    echo "$1" | awk '{printf "%d:%05.2f\n", int($1/60), $1%60}'
}

# Download lyrics for the given input song (audio file) from lrclib.net

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <audio_song_file> [<output_lrc_file>]"
    echo "Downloads lyrics from lrclib.net for the given audio song file and saves it as an .lrc file."
    echo "If no output LRC file is specified, it will be saved alongside the audio file."
    exit 1
fi

audio_file="$1"

if [ ! -f "$audio_file" ]; then
    echo "Error: Audio file '$audio_file' not found."
    exit 1
fi

audio_filename="$(basename "$audio_file")"
lrc_default_filename="${audio_filename%.*}.lrc"

audio_file="$(realpath "$audio_file")"
audio_file_dir="$(dirname "$audio_file")"

lrc_default_file="$audio_file_dir/$lrc_default_filename"

lrc_file="${2:-$lrc_default_file}"

# get the track info: duration, title, artist, album
track_info="$(ffprobe -v error -show_entries format=duration:format_tags=title,artist,album -of csv=p=0:s='|' "$audio_file")"
duration=$(echo "$track_info" | cut -d '|' -f 1)
title=$(echo "$track_info" | cut -d '|' -f 2)
artist=$(echo "$track_info" | cut -d '|' -f 3)
album=$(echo "$track_info" | cut -d '|' -f 4)

lrc_ar="[ar:$artist]"
lrc_ti="[ti:$title]"
lrc_al="[al:$album]"
lrc_length="[length:$(lrc_length $duration)]"

# [ar:Some Artist]
# [ti:Some song title]
# [al:Some album name]
# [length:3:54.72]
printf "%s\n%s\n%s\n%s\n" "$lrc_ar" "$lrc_ti" "$lrc_al" "$lrc_length" > "$lrc_file"

curl -s -G "$lrclib_api/get" \
    --data-urlencode "artist_name=$artist" \
    --data-urlencode "track_name=$title" \
    --data-urlencode "album_name=$album" \
    --data-urlencode "duration=$duration" \
    --header "User-Agent: $songslike_header" \
    | jq -r '.syncedLyrics' >> "$lrc_file"



