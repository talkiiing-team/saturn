from saturn.libs.broker import broker
from saturn.processors.helpers import PerformanceMeasurer
from saturn.processors.extract_audio_frames import extract_audio_frames
from saturn.processors.extract_audio_embedding import extract_audio_embedding


@broker.task
def audio_embedding_extractor(audio_path: str):
    with PerformanceMeasurer("audio_embedding_extractor"):
        with PerformanceMeasurer("extract_audio_frames"):
            audio_frames, sample_rate = extract_audio_frames(audio_path)

        with PerformanceMeasurer("extract_audio_embedding"):
            embedding = extract_audio_embedding(audio_frames, sample_rate)

        return embedding
