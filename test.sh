#!/bin/bash

script_dir="$(dirname "$(realpath "$0")")"

function retrieve() {
    local fingerprint_file="$1"
    local flag="$2"

    python $script_dir/rag_solution.py retrieve $flag --top-k "$TOP_K" songslike-openai "$(cat "$fingerprint_file")"
}

# Accept as input a fingerprint file, perform a RAG search, and copy the audio files of the top results to an output directory.

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <fingerprint_file> <output_directory> [<top_k>]"
    echo "Performs a RAG search using the given fingerprint file and copies the audio files of the top results to the output directory."
    echo "AUDIO_DIR environment variable must be set to the directory (recursive search) containing the audio files, always in .flac format, corresponding to the fingerprints"
    echo "If <top_k> is not specified, it defaults to 3."
    exit 1
fi

FINGERPRINT_FILE=$1
OUTPUT_DIR=$2
TOP_K=${3:-3}

if [ ! -f "$FINGERPRINT_FILE" ]; then
    echo "Error: Fingerprint file '$FINGERPRINT_FILE' not found."
    exit 1
fi

if [ -z "$AUDIO_DIR" ]; then
    echo "Error: AUDIO_DIR environment variable is not set."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Query by the contents of the fingerprint file, excluding the metadata (Author, title. etc) to avoid biasing the search results towards "same artist" matches
retrieve "$FINGERPRINT_FILE" "--metadata" \
    | tee $OUTPUT_DIR/__rag-results.json | jq -r '.[].file_name' | sed 's/.fp.txt//' \
    | while IFS= read -r f; do 
        find "$AUDIO_DIR" -name "$f.*" -exec cp {} "$OUTPUT_DIR" \;
      done
    
retrieve "$FINGERPRINT_FILE" "--full" > "$OUTPUT_DIR/__rag-full-matches.txt"

# python $script_dir/rag_solution.py retrieve --metadata --top-k "$TOP_K" songslike-openai "$(cat "$FINGERPRINT_FILE" | tail +5)" \
#     | tee $OUTPUT_DIR/rag-results.json | jq -r '.[].file_name' | sed 's/.fp.txt//' \
#     | while IFS= read -r f; do 
#         find "$AUDIO_DIR" -name "$f.*" -exec cp {} "$OUTPUT_DIR" \;
#       done

open "$OUTPUT_DIR"