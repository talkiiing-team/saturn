from os import path
from numpy.typing import ArrayLike

from saturn.libs.broker import broker
from saturn.models.MemesFeature import MemesFeature, MemeCoordinates

model = None
if broker.is_worker_process:
    from ultralytics import YOLO

    model = YOLO(path.join(path.dirname(__file__), "models/memes_detector.pt"))


def detect_memes(img: ArrayLike) -> MemesFeature:
    """
    Функция для поиска мемов на изображении.
    """

    results = model(img, verbose=False)  # predict on an image

    if len(results[0].boxes) > 0:
        coordinates = []

        for res in results[0].boxes:
            x, y, w, h = res.xywh[0].cpu().numpy().astype(int)
            coordinates.append(
                MemeCoordinates(x=int(x), y=int(y), width=int(w), height=int(h))
            )

        return MemesFeature(has_meme=True, coordinates=coordinates)

    return MemesFeature(has_meme=False, coordinates=[])


def crop_image(image, xywh):
    if xywh is None:
        return image
    x, y, w, h = map(int, xywh)
    return x, y, w, h
