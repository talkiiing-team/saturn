import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike
from pydub import AudioSegment


def extract_audio_frames(audio_path: str) -> Tuple[ArrayLike, int]:
    segment = AudioSegment.from_file(audio_path)
    frames = np.array(segment.get_array_of_samples())

    return frames, segment.frame_rate
