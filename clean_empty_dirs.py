#!/usr/bin/env python3
"""
Ohjelma joka listaa input/ alla olevat tyhjät hakemistot ja kysyy käyttäjältä
poistetaanko nämä hakemistot.
"""

import os
import shutil
from pathlib import Path


def find_empty_directories(root_path):
    """Löytää kaikki tyhjät hakemistot annetun polun alta."""
    empty_dirs = []
    root = Path(root_path)

    if not root.exists() or not root.is_dir():
        print(f"Virhe: Hakemisto {root_path} ei ole olemassa tai ei ole hakemisto.")
        return empty_dirs

    # Käy läpi kaikki hakemistot rekursiivisesti
    for dirpath, dirnames, filenames in os.walk(root_path):
        dir_path = Path(dirpath)

        # Tarkista onko hakemisto tyhjä (ei tiedostoja eikä hakemistoja)
        if not dirnames and not filenames:
            empty_dirs.append(dir_path)
        # Tarkista onko hakemistossa vain .DS_Store (macOS)
        elif len(filenames) == 1 and filenames[0] == '.DS_Store' and not dirnames:
            empty_dirs.append(dir_path)

    return sorted(empty_dirs)


def ask_user_confirmation(empty_dirs):
    """Kysyy käyttäjältä haluaako poistaa tyhjät hakemistot."""
    if not empty_dirs:
        print("Ei löytynyt tyhjiä hakemistoja input/ alla.")
        return False

    print(f"\nLöytyi {len(empty_dirs)} tyhjää hakemistoa:")
    print("-" * 50)

    for i, dir_path in enumerate(empty_dirs, 1):
        # Näytä suhteellinen polku input/ hakemistosta
        rel_path = dir_path.relative_to(Path("input"))
        print(f"{i:3}. {rel_path}")

    print("-" * 50)

    while True:
        response = input(f"\nHaluatko poistaa nämä {len(empty_dirs)} tyhjää hakemistoa? (k/e): ").strip().lower()
        if response in ['k', 'kyllä', 'y', 'yes']:
            return True
        elif response in ['e', 'ei', 'n', 'no']:
            return False
        else:
            print("Vastaa 'k' (kyllä) tai 'e' (ei)")


def delete_directories(empty_dirs):
    """Poistaa annetut hakemistot."""
    deleted_count = 0
    failed_count = 0

    for dir_path in empty_dirs:
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                rel_path = dir_path.relative_to(Path("input"))
                print(f"Poistettu: {rel_path}")
                deleted_count += 1
            else:
                print(f"Hakemisto ei ole enää olemassa: {dir_path}")
        except Exception as e:
            rel_path = dir_path.relative_to(Path("input"))
            print(f"Virhe poistettaessa {rel_path}: {e}")
            failed_count += 1

    print(f"\nValmis! Poistettu {deleted_count} hakemistoa.")
    if failed_count > 0:
        print(f"Epäonnistui {failed_count} hakemiston poisto.")


def main():
    """Pääohjelma."""
    input_dir = "input"

    print("Etsitään tyhjiä hakemistoja input/ alla...")

    # Etsi tyhjät hakemistot
    empty_dirs = find_empty_directories(input_dir)

    # Kysy käyttäjältä haluaako poistaa
    if ask_user_confirmation(empty_dirs):
        print("\nPoistetaan hakemistot...")
        delete_directories(empty_dirs)
    else:
        print("Peruutettu. Hakemistoja ei poistettu.")


if __name__ == "__main__":
    main()
