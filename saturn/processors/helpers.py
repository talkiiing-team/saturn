import moviepy.editor as mp
from urllib.request import urlretrieve
from os import path
from time import perf_counter


def download_video(temp_dir: str, video_url: str) -> str:
    video_path, _ = urlretrieve(video_url, path.join(temp_dir, "video.mp4"))

    return video_path


def extract_audio_track(video_path: str) -> str:
    audio_path = path.join(path.dirname(video_path), "audio.wav")

    video_clip = mp.VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)

    return audio_path


class PerformanceMeasurer:
    start_at: float
    end_at: float
    name: str

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start_at = perf_counter()
        print(f"started {self.name}")

    def __exit__(self, type, value, traceback):
        self.end_at = perf_counter()
        elapsed = (self.end_at - self.start_at) * 1000
        print(f"end {self.name} took {elapsed} ms")

    def __int__(self):
        return int((self.end_at - self.start_at) * 1000)
