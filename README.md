# Projektin kuvaus

Tämä projekti on bugikorjattu versio alkuperäisestä [WhisperX projektista](https://github.com/m-bain/whisperX), jossa pitkät suomenkieliset äänitiedostot toimivat.

# Serveri mp3_file_monitor.py

`mp3_file_monitor.py`-skripti toimii serverinä, joka kuuntelee input-hakemistoa ja tuottaa mp3-tiedostoista transkriptit alihakemistoihin `[input-hakemisto]/[mp3-nimi]/`. 

Oletuksena skripti kuuntelee `./input/` hakemistoa, mutta voit määrittää minkä tahansa hakemiston käynnistysparametrilla `--input-dir`.

Skripti tukee NVidian GPU:ta, jos ympäristöön on asennettu CUDA. 

Transkripti tehdään oletuksella suomen kielellä. Jos mp3-tiedoston nimessä on merkkijono -en- tai -en.mp3, tehdään transkripti englanniksi.

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
   - Oletushakemisto: `python mp3_file_monitor.py`
   - Oma hakemisto: `python mp3_file_monitor.py --input-dir /polku/äänitiedostoihin`

## Käynnistysparametrit

- `--input-dir` : Määrittää input-hakemiston polun. Oletusarvo on `./input`

## Esimerkkejä

```bash
# Käytä oletushakemistoa ./input
python mp3_file_monitor.py

# Käytä omaa hakemistoa
python mp3_file_monitor.py --input-dir /Users/username/audio-files

# Käytä suhteellista polkua
python mp3_file_monitor.py --input-dir ../my-audio-files
```

