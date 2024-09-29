from saturn.libs.broker import broker
from saturn.processors.helpers import PerformanceMeasurer
from saturn.processors.extract_text_by_stt import extract_text_by_stt


@broker.task
def sst_extractor(audio_path: str):
    with PerformanceMeasurer("sst_extractor"):
        stt = extract_text_by_stt(audio_path)

        return stt
