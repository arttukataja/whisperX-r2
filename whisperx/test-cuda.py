import whisperx
import gc
import time
import librosa
import os



device = "cuda"
#device = "cpu"
audio_file = "whisperx-audio/musiikkiluvat-3min.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# Start timing
start_time = time.time()

# 1. Transcribe with original whisper (batched)
##model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
model_dir = "/Users/arttu/dev/models"
model = whisperx.load_model("large-v3", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)

# Get audio duration
audio_duration = len(audio) / 16000  # whisperx uses 16kHz sample rate

result = model.transcribe(audio, batch_size=batch_size, language="fi")
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code="fi", device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs

# Calculate and display timing results
end_time = time.time()
processing_time = end_time - start_time
speed_ratio = audio_duration / processing_time

print(f"\n=== TIMING RESULTS ===")
print(f"Audio duration: {audio_duration:.2f} seconds")
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Speed ratio: {speed_ratio:.2f}x (processing time vs audio length)")
if speed_ratio > 1:
    print(f"Processing was {speed_ratio:.2f}x faster than real-time")
else:
    print(f"Processing was {1/speed_ratio:.2f}x slower than real-time")