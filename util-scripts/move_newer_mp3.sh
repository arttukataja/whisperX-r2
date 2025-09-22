#!/bin/bash
shopt -s nullglob

FORCE_MOVE=false
if [[ "$1" == "-y" ]]; then
    FORCE_MOVE=true
fi

# Collect files to be moved into an array
to_move=()

for mp3 in */*.mp3; do
    base="${mp3%.mp3}"
    md="${base}.md"
    if [[ -f "$md" && "$mp3" -nt "$md" ]]; then
        to_move+=("$mp3")
    fi
done

if [[ ${#to_move[@]} -eq 0 ]]; then
    echo "No mp3 files to move were found."
    exit 0
fi

echo "The following files would be moved to the root ./ :"
for f in "${to_move[@]}"; do
    echo "  $f"
done

if [[ "$FORCE_MOVE" == true ]]; then
    answer="y"
else
    # Confirmation
    read -p "Do you want to move these files? (Y/N): " answer
fi

case "$answer" in
    [Yy]* )
        for f in "${to_move[@]}"; do
            echo "Moving: $f -> ./"
            mv "$f" ./
        done
        echo "Move complete."
        ;;
    * )
        echo "Move not performed."
        ;;
esac