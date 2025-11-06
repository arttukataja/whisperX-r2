# Projektin kuvaus

Tämä projekti on bugikorjattu versio alkuperäisestä [WhisperX projektista](https://github.com/m-bain/whisperX), jossa pitkät suomenkieliset äänitiedostot toimivat.

# Serveri audio_file_monitor.py

`audio_file_monitor.py`-skripti toimii serverinä, joka kuuntelee input-hakemistoja ja tuottaa mp3-, m4a- ja mp4-tiedostoista transkriptit alihakemistoihin `[input-hakemisto]/[äänitiedosto-nimi]/`. 

Oletuksena skripti löytää ja kuuntelee automaattisesti **kaikkia** `input*`-alkuisia hakemistoja (esim. `./input`, `./input-test`, `./input-foo`). Jos yhtään hakemistoa ei löydy, se luo ja kuuntelee `./input/` hakemistoa. Voit myös määrittää tarkasti, mitä hakemistoja haluat kuunnella käynnistysparametrilla `--input-dirs`.

Transkripti tehdään oletuksella suomen kielellä. Jos äänitiedoston nimessä on merkkijono `-en-` tai `-en.mp3`/`-en.m4a`/`-en.mp4`, tehdään transkripti englanniksi.

# GPU-tuki

Skripti tukee NVidian GPU:ta, jos ympäristöön on asennettu CUDA. NVidia RTX 4090 transkriptin nopeus on 30x realiaikaiseen äänen verrattuna. 

MacbookPro M4 Maxilla transkriptin nopeus on 0,8x realiaikaiseen verrattuna.

# Asennusohje (MacOS ja Linux)

1. Kloonaa projekti GitHubista itsellesi
2. Luo uusi virtuaaliympäristö `python -m venv .venv`
3. Aktivoi virtuaaliympäristö `source .venv/bin/activate`
4. Asenna riippuvuudet pyproject.toml-tiedostosta `pip install .`
5. Asenna HF_TOKEN ympäristömuuttuja `export HF_TOKEN=your_huggingface_token`
6. (Valinnainen) Tee symbolisia linkkejä äänitiedostohakemistoihin:
   ```bash
   ln -s /path/to/your/audio/files ./input
   ln -s /path/to/more/audio/files ./input-test
   ```

# Ajo-ohje

1. Siirry projektin juurikansioon
2. Aktivoi virtuaaliympäristö `source .venv/bin/activate`
3. Aja serveri jommalla kummalla tavalla:
   - **Automaattinen löytäminen (suositeltu)**: `python audio_file_monitor.py`
   - **Tietyt hakemistot**: `python audio_file_monitor.py --input-dirs ./input ./input-test ./input-foo`

## Käynnistysparametrit

- `--input-dirs [HAKEMISTO ...]` : Määrittää luettelon kuunneltavista hakemistoista. Jos parametria ei anneta, skripti löytää automaattisesti kaikki `input*`-alkuiset hakemistot.

## Esimerkkejä

```bash
# Automaattinen löytäminen - löytää kaikki input*-hakemistot
python audio_file_monitor.py

# Käytä tiettyjä hakemistoja
python audio_file_monitor.py --input-dirs ./input ./input-test

# Käytä absoluuttisia polkuja
python audio_file_monitor.py --input-dirs /Users/username/audio-files /Users/username/more-audio

# Käytä suhteellisia polkuja
python audio_file_monitor.py --input-dirs ../my-audio-files ./test-audio
```
