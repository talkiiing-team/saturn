from os import path
import librosa
import numpy as np
import numpy.typing
import pickle

from saturn.libs.broker import broker

scaler = None
if broker.is_worker_process:
    scaler = pickle.load(
        open(
            path.join(path.dirname(__file__), "models/audio_embeds_scaler.pickle"),
            "rb",
        )
    )


def extract_audio_embedding(audio: numpy.typing.ArrayLike, sampling_rate: int):
    """
    Функция для извлечения нормализованной Мел-спектрограммы из аудио.
    """

    if len(audio) > 0:
        mfccs = librosa.feature.mfcc(
            y=audio.astype(float), sr=sampling_rate, n_mfcc=128
        )
        mfccs_orig = np.mean(mfccs.T, axis=0)
    else:
        mfccs_orig = np.zeros(128)

    return scaler.transform(mfccs_orig.reshape(1, -1))[0].tolist()
