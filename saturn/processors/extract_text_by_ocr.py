import cv2
import pytesseract
import re
from numpy.typing import ArrayLike
from PIL import Image


def extract_text_by_ocr(image: ArrayLike) -> str | None:
    """
    Функция для извлечения текста из изображения с помощью модели Tesseract
    """

    preprocessed_image = preprocess_image(image)
    pil_image = Image.fromarray(preprocessed_image)

    text = pytesseract.image_to_string(pil_image, lang="rus")
    text = " ".join(
        [
            x
            for x in clean_text(
                text.replace("\n", " ").replace("^", "").strip()
            ).split()
            if len(x) > 1
        ]
    )

    if len(text) == 0:
        return None

    return text


def clean_text(text):
    regex = re.compile("[^а-яА-Яa-zA-Z ]")
    text = regex.sub("", text).strip().lower()
    text = re.sub(" +", " ", text)

    return text


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    return binary
