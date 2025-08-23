# Projektin kuvaus

Tämä projekti on bugikorjattu versio alkuperäisestä [WhisperX projektista](https://github.com/m-bain/whisperX), jossa pitkät suomenkieliset äänitiedostot toimivat.

# Serveri audio_file_monitor.py

`audio_file_monitor.py`-skripti toimii serverinä, joka kuuntelee input-hakemistoa ja tuottaa mp3- ja m4a-tiedostoista transkriptit alihakemistoihin `[input-hakemisto]/[äänitiedosto-nimi]/`. 

Oletuksena skripti kuuntelee `./input/` hakemistoa, mutta voit määrittää minkä tahansa hakemiston käynnistysparametrilla `--input-dir`.

Transkripti tehdään oletuksella suomen kielellä. Jos äänitiedoston nimessä on merkkijono -en- tai -en.mp3/-en.m4a, tehdään transkripti englanniksi.

# GPU-tuki

Skripti tukee NVidian GPU:ta, jos ympäristöön on asennettu CUDA. NVidia RTX 4090 transkriptin nopeus on 30x realiaikaiseen äänen verrattuna. 

MacbookPro M4 Maxilla transkriptin nopeus on 0,8x realiaikaiseen verrattuna.

# Asennusohje (MacOS ja Linux)

1. Kloonaa projekti GitHubista itsellesi
2. Luo uusi virtuaaliympäristö `python -m venv .venv`
3. Aktivoi virtuaaliympäristö `source .venv/bin/activate`
4. Asenna riippuvuudet pyproject.toml-tiedostosta `pip install .`
5. Asenna HF_TOKEN ympäristömuuttuja `export HF_TOKEN=your_huggingface_token`
6. Tee symbolinen linkki `ln -s /path/to/your/audio/files ./input`

# Ajo-ohje

1. Siirry projektin juurikansioon
2. Aktivoi virtuaaliympäristö `source .venv/bin/activate`
3. Aja serveri jommalla kummalla tavalla:
   - Oletushakemisto: `python audio_file_monitor.py`
   - Oma hakemisto: `python audio_file_monitor.py --input-dir /polku/äänitiedostoihin`

## Käynnistysparametrit

- `--input-dir` : Määrittää input-hakemiston polun. Oletusarvo on `./input`

## Esimerkkejä

```bash
# Käytä oletushakemistoa ./input
python audio_file_monitor.py

# Käytä omaa hakemistoa
python audio_file_monitor.py --input-dir /Users/username/audio-files

# Käytä suhteellista polkua
python audio_file_monitor.py --input-dir ../my-audio-files
```
