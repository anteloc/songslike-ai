#!/bin/bash

songslike_header="songslike-ai (https://github.com/anteloc/songslike-ai)"
lrclib_api="https://lrclib.net/api"

function lrc_length() {
    echo "$1" | awk '{printf "%d:%05.2f\n", int($1/60), $1%60}'
}

# Download lyrics for the given input song (audio file) from lrclib.net

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <audio_file>"
    exit 1
fi

audio_file="$1"
lrc_file="${audio_file%.*}.lrc"

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



