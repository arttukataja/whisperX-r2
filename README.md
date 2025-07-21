# Projektin kuvaus

Tämä projekti on bugikorjattu versio alkuperäisestä [WhisperX projektista](https://github.com/m-bain/whisperX), jossa pitkät suomenkieliset äänitiedostot toimivat.

# mp3_file_monitor.py

mp3_file_monitor.py-skripti toimii serverinä, joka kuuntelee ./input/ hakemistoa ja tuottaa mp3-tiedostoista transkriptit alihakemistoihin ./input/<mp3-nimi>/. Skripti tukee CUDAa, jos se on asennettu.

Transkripti tehdään oletuksella suomen kielellä. Jos mp3-tiedoston nimessä on merkkijono -en- tai -en.mp3, tehdään transkripti englanniksi.