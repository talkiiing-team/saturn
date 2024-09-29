from saturn.libs.broker import broker
from saturn.processors.helpers import (
    PerformanceMeasurer,
    download_video,
    extract_audio_track,
)


@broker.task
def source_media_preparator(temp_dir: str, video_url: str):
    with PerformanceMeasurer("prepare sources"):
        video_path = download_video(temp_dir=temp_dir, video_url=video_url)
        audio_path = extract_audio_track(video_path=video_path)

        return video_path, audio_path
