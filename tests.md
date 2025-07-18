# CPU test (3min audio)

python whisperx audio/tukevasti-ilmassa-3min.mp3 --output_dir output --output_format all --diarize --language fi --model large-v3 --device cpu --compute_type int8 --hf_token $HF_TOKEN --interpolate_method linear

# GPU test (3min audio)

python whisperx audio/tukevasti-ilmassa-3min.mp3 --output_dir output --output_format all --diarize --language fi --model large-v3 --device cuda --hf_token $HF_TOKEN --interpolate_method linear

