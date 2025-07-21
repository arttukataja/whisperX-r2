#!/usr/bin/env python3
"""
Program to organize ./input/ subdirectories based on identifier words.

Rules:
- Organization directories: contain only subdirectories, no .mp3 files
- Transcript directories: contain .mp3 files, located directly under ./input/
- Identifier word: text after first '-' and before second '-' or '.'
- Move transcript directories under matching organization directories
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


def extract_identifier(directory_name: str) -> str:
    """Extract identifier word from directory name."""
    if '-' not in directory_name:
        return ""

    # Split by '-' and take the second part
    parts = directory_name.split('-')
    if len(parts) < 2:
        return ""

    identifier = parts[1]

    # Remove everything after '.' if present
    if '.' in identifier:
        identifier = identifier.split('.')[0]

    return identifier


def is_transcript_directory(path: Path) -> bool:
    """Check if directory contains .mp3 files (transcript directory)."""
    if not path.is_dir():
        return False

    for file in path.iterdir():
        if file.suffix.lower() == '.mp3':
            return True
    return False


def is_organization_directory(path: Path) -> bool:
    """Check if directory contains only subdirectories and no .mp3 files. Empty directories are also considered organization directories."""
    if not path.is_dir():
        return False

    # Check that there are no .mp3 files
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() == '.mp3':
            return False

    # If we get here, there are no .mp3 files, so it's an organization directory
    # (whether it's empty or contains only subdirectories)
    return True


def find_organization_directories(input_path: Path) -> Dict[str, Path]:
    """Find all organization directories and their identifiers."""
    org_dirs = {}

    for item in input_path.iterdir():
        if item.is_dir() and is_organization_directory(item):
            # Check if the organization directory name is directly an identifier
            # (e.g., "ilmassa")
            org_dirs[item.name] = item

            # Also check if it contains an identifier with dashes
            # (e.g., "category-ilmassa-something")
            identifier = extract_identifier(item.name)
            if identifier:
                org_dirs[identifier] = item

            # Check subdirectories of organization directory
            for subdir in item.iterdir():
                if subdir.is_dir():
                    # Subdirectory name as identifier
                    org_dirs[subdir.name] = item

                    # Subdirectory identifier with dashes
                    sub_identifier = extract_identifier(subdir.name)
                    if sub_identifier:
                        org_dirs[sub_identifier] = item

    return org_dirs


def find_transcript_directories(input_path: Path) -> List[Tuple[Path, str]]:
    """Find transcript directories directly under input path."""
    transcript_dirs = []

    for item in input_path.iterdir():
        if item.is_dir() and is_transcript_directory(item):
            identifier = extract_identifier(item.name)
            if identifier:
                transcript_dirs.append((item, identifier))

    return transcript_dirs


def plan_moves(input_path: Path) -> List[Tuple[Path, Path]]:
    """Plan all directory moves."""
    org_dirs = find_organization_directories(input_path)
    transcript_dirs = find_transcript_directories(input_path)

    moves = []

    for transcript_path, identifier in transcript_dirs:
        if identifier in org_dirs:
            target_path = org_dirs[identifier] / transcript_path.name
            moves.append((transcript_path, target_path))

    return moves


def display_move_plan(moves: List[Tuple[Path, Path]]) -> None:
    """Display the planned moves to the user."""
    if not moves:
        print("Ei siirrettäviä hakemistoja löytynyt.")
        return

    print("\nSuunnitellut hakemistosiirrot:")
    print("=" * 50)

    for i, (source, target) in enumerate(moves, 1):
        print(f"{i}. {source.name}")
        print(f"   -> {target}")
        print()


def execute_moves(moves: List[Tuple[Path, Path]]) -> None:
    """Execute the planned moves."""
    for source, target in moves:
        try:
            # Create target directory's parent if it doesn't exist
            target.parent.mkdir(parents=True, exist_ok=True)

            # Move the directory
            shutil.move(str(source), str(target))
            print(f"✓ Siirretty: {source.name} -> {target}")

        except Exception as e:
            print(f"✗ Virhe siirrettäessä {source.name}: {e}")


def main():
    """Main function."""
    input_path = Path("./input")

    if not input_path.exists():
        print("Virhe: ./input hakemistoa ei löydy!")
        return

    print("Analysoidaan ./input hakemiston rakennetta...")

    # Plan moves
    moves = plan_moves(input_path)

    # Display plan
    display_move_plan(moves)

    if not moves:
        return

    # Ask for confirmation
    while True:
        response = input("Haluatko suorittaa nämä siirrot? (k/e): ").lower().strip()
        if response in ['k', 'kyllä', 'y', 'yes']:
            print("\nSuoritetaan siirrot...")
            execute_moves(moves)
            print("\nSiirrot suoritettu!")
            break
        elif response in ['e', 'ei', 'n', 'no']:
            print("Siirrot peruutettu.")
            break
        else:
            print("Vastaa 'k' (kyllä) tai 'e' (ei).")


if __name__ == "__main__":
    main()
