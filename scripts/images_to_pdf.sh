#!/bin/bash
# Usage: ./images_to_pdf.sh [input_directory] [output.pdf]
# Example: ./images_to_pdf.sh ./images combined.pdf

set -e

# Check for ImageMagick
if ! command -v convert &> /dev/null; then
    echo "Error: ImageMagick 'convert' is not installed."
    echo "Install it using: sudo apt install imagemagick"
    exit 1
fi

INPUT_DIR="${1:-.}"
OUTPUT_FILE="${2:-output.pdf}"
TMP_LIST=$(mktemp)

# Find supported image types
find "$INPUT_DIR" -maxdepth 1 -type f \( \
    -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \
\) | sort > "$TMP_LIST"

if [ ! -s "$TMP_LIST" ]; then
    echo "No image files found in $INPUT_DIR."
    rm "$TMP_LIST"
    exit 1
fi

echo "Combining images from $INPUT_DIR into $OUTPUT_FILE ..."

# A4 size in points (at 72 dpi) = 595x842
# The key is -resize 595x\> which fits width to A4 while keeping aspect ratio
# Then -extent 595x842 ensures it’s centered vertically if smaller
convert @${TMP_LIST} -units PixelsPerInch -density 300 \
    -resize 595x\> -background white -gravity center -extent 595x842 \
    "$OUTPUT_FILE"

echo "✅ Done: $OUTPUT_FILE created."

rm "$TMP_LIST"

