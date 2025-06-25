import whisperx
import gc
import time
import librosa
import os

print("hello cpu")
print("=" * 50)
print("WHISPERX CPU PROCESSING PIPELINE")
print("=" * 50)


#device = "cuda"
device = "cpu"
audio_file = "whisperx-audio/musiikkiluvat-3min.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

print(f"\n[1/5] CONFIGURATION:")
print(f"  Device: {device}")
print(f"  Audio file: {audio_file}")
print(f"  Batch size: {batch_size}")
print(f"  Compute type: {compute_type}")

# Start timing
start_time = time.time()

print(f"\n[2/5] LOADING WHISPER MODEL...")
# 1. Transcribe with original whisper (batched)
##model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
model_dir = os.path.expanduser("~/dev/models")
model = whisperx.load_model("large-v3", device, compute_type=compute_type, download_root=model_dir)
print(f"  ✓ Whisper model loaded successfully")

print(f"\n[3/5] LOADING AUDIO...")
audio = whisperx.load_audio(audio_file)
# Get audio duration
audio_duration = len(audio) / 16000  # whisperx uses 16kHz sample rate
print(f"  ✓ Audio loaded: {audio_duration:.2f} seconds duration")

print(f"\n[4/5] TRANSCRIBING AUDIO...")
transcribe_start = time.time()
result = model.transcribe(audio, batch_size=batch_size, language="fi")
transcribe_time = time.time() - transcribe_start
print(f"  ✓ Transcription completed in {transcribe_time:.2f}s")
print(f"  ✓ Found {len(result['segments'])} segments")
print(f"\n  SEGMENTS BEFORE ALIGNMENT:")
for i, segment in enumerate(result["segments"][:3]):  # Show first 3 segments
    print(f"    [{i+1}] {segment['start']:.2f}s-{segment['end']:.2f}s: {segment['text'][:50]}...")
if len(result["segments"]) > 3:
    print(f"    ... and {len(result['segments']) - 3} more segments")

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

print(f"\n[5/5] LOADING ALIGNMENT MODEL...")
# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code="fi", device=device)
print(f"  ✓ Alignment model loaded for Finnish")

print(f"\n[6/5] ALIGNING TRANSCRIPT...")
align_start = time.time()
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
align_time = time.time() - align_start
print(f"  ✓ Alignment completed in {align_time:.2f}s")
print(f"\n  SEGMENTS AFTER ALIGNMENT:")
for i, segment in enumerate(result["segments"][:3]):  # Show first 3 segments
    print(f"    [{i+1}] {segment['start']:.2f}s-{segment['end']:.2f}s: {segment['text'][:50]}...")

# delete model if low on GPU resources
# import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

print(f"\n[7/5] LOADING DIARIZATION MODEL...")
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
print(f"  ✓ Diarization model loaded")

print(f"\n[8/5] PERFORMING SPEAKER DIARIZATION...")
diarize_start = time.time()
# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
diarize_time = time.time() - diarize_start
print(f"  ✓ Diarization completed in {diarize_time:.2f}s")

print(f"\n[9/5] ASSIGNING SPEAKERS TO WORDS...")
assign_start = time.time()
result = whisperx.assign_word_speakers(diarize_segments, result)
assign_time = time.time() - assign_start
print(f"  ✓ Speaker assignment completed in {assign_time:.2f}s")

# Calculate and display timing results
end_time = time.time()
processing_time = end_time - start_time
speed_ratio = audio_duration / processing_time

print(f"\n" + "=" * 50)
print(f"DETAILED TIMING BREAKDOWN:")
print(f"=" * 50)
print(f"Transcription: {transcribe_time:.2f}s ({transcribe_time/processing_time*100:.1f}%)")
print(f"Alignment: {align_time:.2f}s ({align_time/processing_time*100:.1f}%)")
print(f"Diarization: {diarize_time:.2f}s ({diarize_time/processing_time*100:.1f}%)")
print(f"Speaker assignment: {assign_time:.2f}s ({assign_time/processing_time*100:.1f}%)")
print(f"Other operations: {processing_time - transcribe_time - align_time - diarize_time - assign_time:.2f}s")

print(f"\n=== FINAL TIMING RESULTS ===")
print(f"Audio duration: {audio_duration:.2f} seconds")
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Speed ratio: {speed_ratio:.2f}x (processing time vs audio length)")
if speed_ratio > 1:
    print(f"Processing was {speed_ratio:.2f}x faster than real-time")
else:
    print(f"Processing was {1/speed_ratio:.2f}x slower than real-time")