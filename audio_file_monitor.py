#!/usr/bin/env python3
"""
Audio File Monitor and Transcription System
Monitors input directory for MP3 and M4A files and transcribes them using WhisperX
"""

import os
import time
import shutil
import threading
import json
import argparse
from pathlib import Path
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
        logging.FileHandler('audio_file_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioFileMonitor:
    def __init__(self, input_dir="./input"):
        self.input_dir = Path(input_dir)
        self.processing_lock = threading.Lock()
        self.current_language = None
        self.model_a = None
        self.metadata = None
        self.file_sizes = {}  # Track file sizes to detect when writing is complete
        self.supported_extensions = {'.mp3', '.m4a'}
        self.setup_whisperx()

    def detect_language_from_filename(self, filename):
        """Detect language from filename patterns"""
        filename_lower = filename.lower()
        if "-en.mp3" in filename_lower or "-en.m4a" in filename_lower or "-en-" in filename_lower:
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

    def process_audio_file(self, file_path):
        """Process a single audio file (MP3 or M4A)"""
        with self.processing_lock:
            try:
                file_path = Path(file_path)
                logger.info(f"Processing audio file: {file_path.name}")

                # Create directory structure: [input_dir]/[filename]/
                filename_base = file_path.stem  # filename without extension
                target_dir = self.input_dir / filename_base
                target_dir.mkdir(parents=True, exist_ok=True)

                # Move audio file to target directory
                new_audio_path = target_dir / file_path.name
                if file_path.exists():
                    shutil.move(str(file_path), str(new_audio_path))
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
                self.transcribe_file(new_audio_path, target_dir)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                # Save error log to target directory if it exists
                if 'target_dir' in locals():
                    error_file = target_dir / "error.log"
                    with open(error_file, 'w', encoding='utf-8') as f:
                        f.write(f"Error processing {file_path.name}\n")
                        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Error: {str(e)}\n")

    def transcribe_file(self, audio_path, output_dir):
        """Transcribe audio file using WhisperX pipeline"""
        start_time = time.time()

        logger.info(f"Starting transcription of {audio_path.name}")

        try:
            # Load audio
            logger.info("Loading audio...")
            audio = whisperx.load_audio(str(audio_path))
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
            self.save_transcript(result, audio_path, output_dir, audio_duration, start_time)

            # Cleanup
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            processing_time = time.time() - start_time
            speed_ratio = audio_duration / processing_time
            logger.info(f"✓ Transcription completed for {audio_path.name}")
            logger.info(f"Processing time: {processing_time:.2f}s (speed ratio: {speed_ratio:.2f}x)")

        except Exception as e:
            logger.error(f"Error during transcription of {audio_path.name}: {e}")
            raise

    def save_transcript(self, result, audio_path, output_dir, audio_duration, start_time):
        """Save transcript to file"""
        # Use base filename without extension for output files
        base_filename = audio_path.stem
        transcript_file = output_dir / f"{base_filename}.txt"
        markdown_file = output_dir / f"{base_filename}.md"
        json_file = output_dir / f"{base_filename}.json"

        processing_time = time.time() - start_time
        speed_ratio = audio_duration / processing_time

        # Save .txt format (existing)
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"Diarized Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Audio file: {audio_path.name}\n")
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
        self.save_markdown_transcript(result, audio_path, markdown_file, audio_duration, start_time)

        # Save .json format (new)
        self.save_json_transcript(result, audio_path, json_file, audio_duration, start_time)

        logger.info(f"Transcript saved to {transcript_file}")
        logger.info(f"Markdown transcript saved to {markdown_file}")
        logger.info(f"JSON transcript saved to {json_file}")

    def save_markdown_transcript(self, result, audio_path, markdown_file, audio_duration, start_time):
        """Save transcript in markdown format in chronological order"""
        processing_time = time.time() - start_time
        speed_ratio = audio_duration / processing_time

        with open(markdown_file, 'w', encoding='utf-8') as f:
            # Write header information
            f.write(f"# Diarized Transcript\n\n")
            f.write(f"**Audio file:** {audio_path.name}  \n")
            f.write(f"**Language:** {self.current_language}  \n")
            f.write(f"**Device:** {self.device}  \n")
            f.write(f"**Duration:** {audio_duration:.2f} seconds  \n")
            f.write(f"**Processing time:** {processing_time:.2f} seconds  \n")
            f.write(f"**Speed ratio:** {speed_ratio:.2f}x  \n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            f.write("---")

            # Write segments in chronological order
            current_speaker = None
            for i, segment in enumerate(result["segments"]):
                speaker = segment.get('speaker', 'UNKNOWN')
                start_time_seg = segment['start']
                end_time_seg = segment['end']
                text = segment['text'].strip()

                # Add speaker header when speaker changes
                if speaker != current_speaker:
                    start_time_str = time.strftime('%H:%M:%S', time.gmtime(start_time_seg))
                    f.write(f"\n\n## {speaker} ({start_time_str})\n\n")
                    current_speaker = speaker

                # Write the segment text
                if text:  # Only write non-empty text
                    if not text.endswith(" "):
                        text += " "
                    f.write(f"{text}")

    def save_json_transcript(self, result, audio_path, json_file, audio_duration, start_time):
        """Save transcript in JSON format"""
        processing_time = time.time() - start_time
        speed_ratio = audio_duration / processing_time

        # Prepare JSON data
        json_data = {
            "audio_file": audio_path.name,
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
        """Process any existing audio files in the input directory"""
        input_path = Path(input_dir)
        audio_files = list(input_path.glob("*.[mM][pP]3")) + list(input_path.glob("*.[mM]4[aA]"))

        if audio_files:
            logger.info(f"Found {len(audio_files)} existing audio file(s) to process:")
            for audio_file in audio_files:
                logger.info(f"  - {audio_file.name}")
                self.process_audio_file(str(audio_file))
        else:
            logger.info("No existing audio files found in input directory")

    def monitor_directory(self, input_dir):
        """Monitor the input directory for new audio files"""
        logger.info(f"Monitoring directory: {input_dir.absolute()}")
        logger.info("Language detection:")
        logger.info("  - Files with '-en.mp3', '-en.m4a' or '-en-' in filename: English transcription")
        logger.info("  - All other files: Finnish transcription (default)")
        logger.info(f"Drop MP3 or M4A files into {input_dir} directory to start transcription")
        logger.info("Files will be processed only after they are stable (not growing) for 10 seconds")
        logger.info("Press Ctrl+C to stop monitoring")

        try:
            while True:
                # Check for new audio files
                input_path = Path(input_dir)
                audio_files = list(input_path.glob("*.[mM][pP]3")) + list(input_path.glob("*.[mM]4[aA]"))

                for audio_file in audio_files:
                    # Check if file is stable before processing
                    if self.is_file_stable(audio_file):
                        logger.info(f"File is stable, processing: {audio_file.name}")
                        self.process_audio_file(str(audio_file))
                    else:
                        # File detected but not stable yet
                        if str(audio_file) not in self.file_sizes:
                            logger.info(f"New audio file detected, waiting for stability: {audio_file.name}")
                        else:
                            current_size = audio_file.stat().st_size
                            logger.debug(f"File {audio_file.name} still growing (size: {current_size} bytes)")

                time.sleep(5)  # Polling interval

        except KeyboardInterrupt:
            logger.info("Stopping directory monitor...")
        except Exception as e:
            logger.error(f"Error monitoring directory: {e}")

    def is_file_stable(self, file_path, stability_duration=10):
        """
        Check if a file is stable (not growing) by monitoring its size.
        Returns True if file size hasn't changed for stability_duration seconds.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return False

        current_size = file_path.stat().st_size

        # If we haven't seen this file before, start tracking it
        if str(file_path) not in self.file_sizes:
            self.file_sizes[str(file_path)] = {
                'size': current_size,
                'last_change': time.time()
            }
            return False

        # Check if size has changed
        last_info = self.file_sizes[str(file_path)]
        if current_size != last_info['size']:
            # Size changed, update tracking info
            self.file_sizes[str(file_path)] = {
                'size': current_size,
                'last_change': time.time()
            }
            return False

        # Size hasn't changed, check if enough time has passed
        time_since_change = time.time() - last_info['last_change']
        if time_since_change >= stability_duration:
            # File is stable, we can remove it from tracking
            del self.file_sizes[str(file_path)]
            return True

        return False

def main():
    """Main function to start file monitoring"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio File Monitor and Transcription System")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./input",
        help="Input directory to monitor for MP3 and M4A files (default: ./input)"
    )
    args = parser.parse_args()

    # Ensure input directory exists
    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("AUDIO FILE MONITOR AND TRANSCRIPTION SYSTEM")
    logger.info("=" * 50)

    # Create event handler and observer
    event_handler = AudioFileMonitor(input_dir=args.input_dir)

    # Process any existing files in the input directory
    event_handler.process_existing_files(input_dir)

    # Start monitoring
    event_handler.monitor_directory(input_dir)

    logger.info("File monitor stopped")

if __name__ == "__main__":
    main()
