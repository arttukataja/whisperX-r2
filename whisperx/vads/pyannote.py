import os
from typing import Callable, Text, Union
from typing import Optional

import numpy as np
import torch
from pyannote.audio import Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.core import Segment

from whisperx.diarize import Segment as SegmentX
from whisperx.vads.vad import Vad
from whisperx.utils import suppress_reproducibility_warnings


def load_vad_model(device, vad_onset=0.500, vad_offset=0.363, use_auth_token=None, model_fp=None):
    model_dir = torch.hub._get_torch_home()

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(model_dir, exist_ok = True)
    if model_fp is None:
        # Dynamically resolve the path to the model file
        model_fp = os.path.join(main_dir, "assets", "pytorch_model.bin")
        model_fp = os.path.abspath(model_fp)  # Ensure the path is absolute
    else:
        model_fp = os.path.abspath(model_fp)  # Ensure any provided path is absolute

    # Check if the resolved model file exists
    if not os.path.exists(model_fp):
        raise FileNotFoundError(f"Model file not found at {model_fp}")

    if os.path.exists(model_fp) and not os.path.isfile(model_fp):
        raise RuntimeError(f"{model_fp} exists and is not a regular file")

    model_bytes = open(model_fp, "rb").read()

    suppress_reproducibility_warnings()
    vad_model = Model.from_pretrained(model_fp, use_auth_token=use_auth_token)
    hyperparameters = {"onset": vad_onset,
                    "offset": vad_offset,
                    "min_duration_on": 0.1,
                    "min_duration_off": 0.1}
    vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
    vad_pipeline.instantiate(hyperparameters)

    # Re-enable TF32 after pyannote modified it
    try:
        from whisperx.utils import enable_tf32

        enable_tf32()
    except Exception:
        pass

    return vad_pipeline

class Binarize:
    """Binarize detection scores using hysteresis thresholding, with min-cut operation
    to ensure not segments are longer than max_duration.

    Parameters
    ----------
    onset : float, optional
        Onset threshold. Defaults to 0.5.
    offset : float, optional
        Offset threshold. Defaults to `onset`.
    min_duration_on : float, optional
        Remove active regions shorter than that many seconds. Defaults to 0s.
    min_duration_off : float, optional
        Fill inactive regions shorter than that many seconds. Defaults to 0s.
    pad_onset : float, optional
        Extend active regions by moving their start time by that many seconds.
        Defaults to 0s.
    pad_offset : float, optional
        Extend active regions by moving their end time by that many seconds.
        Defaults to 0s.
    max_duration: float
        The maximum length of an active segment, divides segment at timestamp with lowest score.
    Reference
    ---------
    Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
    RNN-based Voice Activity Detection", InterSpeech 2015.

    Modified by Max Bain to include WhisperX's min-cut operation
    https://arxiv.org/abs/2303.00747

    Pyannote-audio
    """

    def __init__(
            self,
            onset: float = 0.5,
            offset: Optional[float] = None,
            min_duration_on: float = 0.0,
            min_duration_off: float = 0.0,
            pad_onset: float = 0.0,
            pad_offset: float = 0.0,
            max_duration: float = float('inf')
    ):

        super().__init__()

        self.onset = onset
        self.offset = offset or onset

        self.pad_onset = pad_onset
        self.pad_offset = pad_offset

        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off

        self.max_duration = max_duration

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """Binarize detection scores
        Parameters
        ----------
        scores : SlidingWindowFeature
            Detection scores.
        Returns
        -------
        active : Annotation
            Binarized scores.
        """

        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        # annotation meant to store 'active' regions
        active = Annotation()
        for k, k_scores in enumerate(scores.data.T):

            label = k if scores.labels is None else scores.labels[k]

            # initial state
            start = timestamps[0]
            is_active = k_scores[0] > self.onset
            curr_scores = [k_scores[0]]
            curr_timestamps = [start]
            t = start
            for t, y in zip(timestamps[1:], k_scores[1:]):
                # currently active
                if is_active:
                    curr_duration = t - start
                    if curr_duration > self.max_duration:
                        search_after = len(curr_scores) // 2
                        # divide segment
                        min_score_div_idx = search_after + np.argmin(curr_scores[search_after:])
                        min_score_t = curr_timestamps[min_score_div_idx]
                        region = Segment(start - self.pad_onset, min_score_t + self.pad_offset)
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx + 1:]
                        curr_timestamps = curr_timestamps[min_score_div_idx + 1:]
                    # switching from active to inactive
                    elif y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                    curr_scores.append(y)
                    curr_timestamps.append(t)
                # currently inactive
                else:
                    # switching from inactive to active
                    if y > self.onset:
                        start = t
                        is_active = True

            # if active at the end, add final region
            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        # because of padding, some active regions might be overlapping: merge them.
        # also: fill same speaker gaps shorter than min_duration_off
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            if self.max_duration < float("inf"):
                raise NotImplementedError(f"This would break current max_duration param")
            active = active.support(collar=self.min_duration_off)

        # remove tracks shorter than min_duration_on
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active


class VoiceActivitySegmentation(VoiceActivityDetection):
    def __init__(
            self,
            segmentation: PipelineModel = "pyannote/segmentation",
            fscore: bool = False,
            use_auth_token: Union[Text, None] = None,
            **inference_kwargs,
    ):

        super().__init__(segmentation=segmentation, fscore=fscore, use_auth_token=use_auth_token, **inference_kwargs)

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file)

        return segmentations


class Pyannote(Vad):

    def __init__(self, device, use_auth_token=None, model_fp=None, **kwargs):
        print(">>Performing voice activity detection using Pyannote...")
        super().__init__(kwargs['vad_onset'])
        self.vad_pipeline = load_vad_model(device, use_auth_token=use_auth_token, model_fp=model_fp)

    def __call__(self, audio: AudioFile, **kwargs):
        return self.vad_pipeline(audio)

    @staticmethod
    def preprocess_audio(audio):
        return torch.from_numpy(audio).unsqueeze(0)

    @staticmethod
    def merge_chunks(segments,
                     chunk_size,
                     onset: float = 0.5,
                     offset: Optional[float] = None,
                     ):
        assert chunk_size > 0
        binarize = Binarize(max_duration=chunk_size, onset=onset, offset=offset)
        segments = binarize(segments)
        segments_list = []
        for speech_turn in segments.get_timeline():
            segments_list.append(SegmentX(speech_turn.start, speech_turn.end, "UNKNOWN"))

        if len(segments_list) == 0:
            print("No active speech found in audio")
            return []
        assert segments_list, "segments_list is empty."
        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)
