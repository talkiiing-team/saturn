from dataclasses import dataclass
from typing import List

from saturn.models.MemesFeature import MemesFeature


@dataclass
class Features:
    video_embedding: List[float]
    audio_embedding: List[float]
    ocr: str
    speech_to_text: str
    memes: MemesFeature
