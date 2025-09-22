#!/bin/bash
#
# mixmp3.sh - Mix two MP3 files into one MP3
#
# Usage:
#   ./mixmp3.sh input1.mp3 input2.mp3 output.mp3
#
# Parameters:
#   input1.mp3   First input audio file
#   input2.mp3   Second input audio file
#   output.mp3   Desired output file
#
# Example:
#   ./mixmp3.sh voice.mp3 music.mp3 mixed.mp3
#

# --- Check parameters ---
if [ "$#" -ne 3 ]; then
    echo "Error: You must provide exactly 3 arguments."
    echo "Usage: $0 input1.mp3 input2.mp3 output.mp3"
    exit 1
fi

INPUT1="$1"
INPUT2="$2"
OUTPUT="$3"

# --- Verify input files exist ---
if [ ! -f "$INPUT1" ]; then
    echo "Error: File '$INPUT1' not found."
    exit 1
fi

if [ ! -f "$INPUT2" ]; then
    echo "Error: File '$INPUT2' not found."
    exit 1
fi

# --- Perform mixing ---
ffmpeg -i "$INPUT1" -i "$INPUT2" \
  -filter_complex "amix=inputs=2:duration=longest:dropout_transition=2" \
  -c:a libmp3lame -q:a 2 "$OUTPUT"

# --- Check result ---
if [ $? -eq 0 ]; then
    echo "Mixing completed successfully! Output file: $OUTPUT"
else
    echo "Error: Something went wrong during mixing."
    exit 1
fi