from saturn.libs.broker import broker
from saturn.processors.helpers import PerformanceMeasurer
from saturn.processors.extract_video_frames import extract_video_frames
from saturn.processors.extract_video_embedding import extract_video_embedding
from saturn.processors.extract_text_by_ocr import extract_text_by_ocr
from saturn.processors.detect_memes import detect_memes


@broker.task
def video_processing(video_path: str):
    with PerformanceMeasurer("video_processing"):
        with PerformanceMeasurer("extract_video_frames"):
            video_frames = extract_video_frames(video_path=video_path)

        first_frame, last_frame = video_frames[0], video_frames[-1]

        with PerformanceMeasurer("extract_video_embedding"):
            embedding = extract_video_embedding(video_frames)

        with PerformanceMeasurer("extract_text_by_ocr"):
            ocr_by_first_frame = extract_text_by_ocr(first_frame)
            ocr_by_last_frame = extract_text_by_ocr(last_frame)

        with PerformanceMeasurer("detect_memes"):
            memes = detect_memes(first_frame)

        ocr = ocr_by_first_frame or ""
        if ocr_by_last_frame is not None:
            ocr += " " + ocr_by_last_frame

        return embedding, ocr, memes
