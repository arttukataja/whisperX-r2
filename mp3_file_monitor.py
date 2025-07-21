#!/usr/bin/env python3
"""
MP3 File Monitor and Transcription System
Monitors ./input directory for MP3 files and transcribes them using WhisperX
"""

import os
import time
import shutil
import threading
import json
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import whisperx
import gc
import torch
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mp3_file_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MP3Handler(FileSystemEventHandler):
    def __init__(self):
        self.processing_lock = threading.Lock()
        self.current_language = None
        self.model_a = None
        self.metadata = None
        self.setup_whisperx()

    def detect_language_from_filename(self, filename):
        """Detect language from filename patterns"""
        filename_lower = filename.lower()
        if "-en.mp3" in filename_lower or "-en-" in filename_lower:
            return "en"
        return "fi"  # Default to Finnish

    def setup_whisperx(self):
        """Initialize WhisperX models"""
        logger.info("Initializing WhisperX models...")

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"

        self.batch_size = 16

        logger.info(f"Device: {self.device} ({'CUDA detected' if self.device == 'cuda' else 'CUDA not available'})")
        logger.info(f"Compute type: {self.compute_type}")

        try:
            # Load Whisper model
            self.model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)
            logger.info("✓ Whisper model loaded successfully")

            # Load diarization model
            hftoken = os.getenv("HF_TOKEN")
            self.diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hftoken, device=self.device)
            logger.info("✓ Diarization model loaded")

            # Note: Alignment model will be loaded dynamically based on detected language

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def load_alignment_model(self, language_code):
        """Load alignment model for specific language"""
        if self.current_language != language_code:
            logger.info(f"Loading alignment model for language: {language_code}")
            try:
                self.model_a, self.metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
                self.current_language = language_code
                logger.info(f"✓ Alignment model loaded for {language_code}")
            except Exception as e:
                logger.error(f"Error loading alignment model for {language_code}: {e}")
                raise
        else:
            logger.info(f"Alignment model for {language_code} already loaded")

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.mp3'):
            # Small delay to ensure file is fully written
            time.sleep(2)
            self.process_mp3_file(event.src_path)

    def process_mp3_file(self, file_path):
        """Process a single MP3 file"""
        with self.processing_lock:
            try:
                file_path = Path(file_path)
                logger.info(f"Processing MP3 file: {file_path.name}")

                # Create directory structure: ./input/[filename]/
                filename_base = file_path.stem  # filename without extension
                target_dir = Path("./input") / filename_base
                target_dir.mkdir(parents=True, exist_ok=True)

                # Move MP3 file to target directory
                new_mp3_path = target_dir / file_path.name
                if file_path.exists():
                    shutil.move(str(file_path), str(new_mp3_path))
                    logger.info(f"Moved {file_path.name} to {target_dir}")
                else:
                    logger.warning(f"File {file_path} no longer exists")
                    return

                # Detect language from filename
                language_code = self.detect_language_from_filename(file_path.name)
                logger.info(f"Detected language: {language_code}")

                # Load alignment model for the detected language
                self.load_alignment_model(language_code)

                # Transcribe the file
                self.transcribe_file(new_mp3_path, target_dir)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Save error log to target directory if it exists
                if 'target_dir' in locals():
                    error_file = target_dir / "error.log"
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(f"Error processing {file_path.name}\n")
                        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Error: {str(e)}\n")

    def transcribe_file(self, mp3_path, output_dir):
        """Transcribe MP3 file using WhisperX pipeline"""
        start_time = time.time()

        logger.info(f"Starting transcription of {mp3_path.name}")

        try:
            # Load audio
            logger.info("Loading audio...")
            audio = whisperx.load_audio(str(mp3_path))
            audio_duration = len(audio) / 16000
            logger.info(f"Audio loaded: {audio_duration:.2f} seconds duration")

            # Transcribe with detected language
            logger.info(f"Transcribing audio (language: {self.current_language})...")
            transcribe_start = time.time()
            result = self.model.transcribe(audio, batch_size=self.batch_size, language=self.current_language)
            transcribe_time = time.time() - transcribe_start
            logger.info(f"Transcription completed in {transcribe_time:.2f}s ({len(result['segments'])} segments)")

            # Align
            logger.info("Aligning transcript...")
            align_start = time.time()
            result = whisperx.align(result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)
            align_time = time.time() - align_start
            logger.info(f"Alignment completed in {align_time:.2f}s")

            # Diarize
            logger.info("Performing speaker diarization...")
            diarize_start = time.time()
            diarize_segments = self.diarize_model(audio)
            diarize_time = time.time() - diarize_start
            logger.info(f"Diarization completed in {diarize_time:.2f}s")

            # Assign speakers
            logger.info("Assigning speakers to words...")
            assign_start = time.time()
            result = whisperx.assign_word_speakers(diarize_segments, result)
            assign_time = time.time() - assign_start
            logger.info(f"Speaker assignment completed in {assign_time:.2f}s")

            # Save results
            self.save_transcript(result, mp3_path, output_dir, audio_duration, start_time)

            # Cleanup
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            processing_time = time.time() - start_time
            speed_ratio = audio_duration / processing_time
            logger.info(f"✓ Transcription completed for {mp3_path.name}")
            logger.info(f"Processing time: {processing_time:.2f}s (speed ratio: {speed_ratio:.2f}x)")

        except Exception as e:
            logger.error(f"Error during transcription of {mp3_path.name}: {e}")
            raise

    def save_transcript(self, result, mp3_path, output_dir, audio_duration, start_time):
        """Save transcript to file"""
        # Use base filename without extension for output files
        base_filename = mp3_path.stem
        transcript_file = output_dir / f"{base_filename}.txt"
        markdown_file = output_dir / f"{base_filename}.md"
        json_file = output_dir / f"{base_filename}.json"

        processing_time = time.time() - start_time
        speed_ratio = audio_duration / processing_time

        # Save .txt format (existing)
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"Diarized Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Audio file: {mp3_path.name}\n")
            f.write(f"Language: {self.current_language}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Duration: {audio_duration:.2f} seconds\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n")
            f.write(f"Speed ratio: {speed_ratio:.2f}x\n")
            f.write("=" * 50 + "\n\n")

            for i, segment in enumerate(result["segments"]):
                speaker = segment.get('speaker', 'UNKNOWN')
                start_time_seg = segment['start']
                end_time_seg = segment['end']
                text = segment['text']
                f.write(f"[{i + 1:03d}] {speaker} ({start_time_seg:.2f}s-{end_time_seg:.2f}s): {text}\n")

        # Save .md format (new)
        self.save_markdown_transcript(result, mp3_path, markdown_file, audio_duration, start_time)

        # Save .json format (new)
        self.save_json_transcript(result, mp3_path, json_file, audio_duration, start_time)

        logger.info(f"Transcript saved to {transcript_file}")
        logger.info(f"Markdown transcript saved to {markdown_file}")
        logger.info(f"JSON transcript saved to {json_file}")

    def save_markdown_transcript(self, result, mp3_path, markdown_file, audio_duration, start_time):
        """Save transcript in markdown format grouped by speakers"""
        processing_time = time.time() - start_time
        speed_ratio = audio_duration / processing_time

        # Group segments by speaker
        speaker_segments = {}
        for segment in result["segments"]:
            speaker = segment.get('speaker', 'UNKNOWN')
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)

        with open(markdown_file, 'w', encoding='utf-8') as f:
            # Write header information
            f.write(f"# Diarized Transcript\n\n")
            f.write(f"**Audio file:** {mp3_path.name}  \n")
            f.write(f"**Language:** {self.current_language}  \n")
            f.write(f"**Device:** {self.device}  \n")
            f.write(f"**Duration:** {audio_duration:.2f} seconds  \n")
            f.write(f"**Processing time:** {processing_time:.2f} seconds  \n")
            f.write(f"**Speed ratio:** {speed_ratio:.2f}x  \n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            f.write("---\n\n")

            # Write segments grouped by speaker
            for speaker in sorted(speaker_segments.keys()):
                segments = speaker_segments[speaker]

                # Get first and last timestamps for this speaker
                first_timestamp = segments[0]['start']
                last_timestamp = segments[-1]['end']

                f.write(f"## {speaker} ({first_timestamp:.2f}s-{last_timestamp:.2f}s):\n\n")

                for segment in segments:
                    text = segment['text'].strip()
                    if text:  # Only write non-empty text
                        f.write(f"{text}\n\n")

                f.write("---\n\n")

    def save_json_transcript(self, result, mp3_path, json_file, audio_duration, start_time):
        """Save transcript in JSON format"""
        processing_time = time.time() - start_time
        speed_ratio = audio_duration / processing_time

        # Prepare JSON data
        json_data = {
            "audio_file": mp3_path.name,
            "language": self.current_language,
            "device": self.device,
            "duration": audio_duration,
            "processing_time": processing_time,
            "speed_ratio": speed_ratio,
            "segments": result["segments"],
            "generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    def process_existing_files(self, input_dir):
        """Process any existing MP3 files in the input directory"""
        input_path = Path(input_dir)
        mp3_files = list(input_path.glob("*.mp3"))

        if mp3_files:
            logger.info(f"Found {len(mp3_files)} existing MP3 file(s) to process:")
            for mp3_file in mp3_files:
                logger.info(f"  - {mp3_file.name}")
                self.process_mp3_file(str(mp3_file))
        else:
            logger.info("No existing MP3 files found in input directory")

def main():
    """Main function to start file monitoring"""
    # Ensure input directory exists
    input_dir = Path("./input")
    input_dir.mkdir(exist_ok=True)

    logger.info("=" * 50)
    logger.info("MP3 FILE MONITOR AND TRANSCRIPTION SYSTEM")
    logger.info("=" * 50)
    logger.info(f"Monitoring directory: {input_dir.absolute()}")
    logger.info("Language detection:")
    logger.info("  - Files with '-en.mp3' or '-en-' in filename: English transcription")
    logger.info("  - All other files: Finnish transcription (default)")
    logger.info("Drop MP3 files into ./input directory to start transcription")
    logger.info("Press Ctrl+C to stop monitoring")

    # Create event handler and observer
    event_handler = MP3Handler()
    observer = Observer()
    observer.schedule(event_handler, str(input_dir), recursive=False)

    # Start monitoring
    observer.start()

    # Process any existing files in the input directory
    event_handler.process_existing_files(input_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file monitor...")
        observer.stop()

    observer.join()
    logger.info("File monitor stopped")

if __name__ == "__main__":
    main()
