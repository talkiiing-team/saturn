import os
import shutil
import uuid

from saturn.models.Features import Features
from saturn.processors.tasks.sst_extractor import sst_extractor
from saturn.processors.tasks.audio_embedding_extractor import audio_embedding_extractor
from saturn.processors.tasks.video_processing import video_processing
from saturn.processors.tasks.source_media_preparator import source_media_preparator


async def feature_extraction(video_url: str):
    temp_dir = f"/opt/saturn/tmp/{uuid.uuid4()}"
    os.mkdir(temp_dir)

    prepare_media_source_task = await source_media_preparator.kiq(temp_dir, video_url)
    prepare_media_source_result = await prepare_media_source_task.wait_result()
    video_path, audio_path = prepare_media_source_result.return_value

    sst_extraction_task = await sst_extractor.kiq(audio_path)
    audio_embedding_extraction_task = await audio_embedding_extractor.kiq(audio_path)
    video_processing_task = await video_processing.kiq(video_path)

    video_processing_result = await video_processing_task.wait_result()
    sst_extraction_result = await sst_extraction_task.wait_result()
    audio_embedding_extraction_result = (
        await audio_embedding_extraction_task.wait_result()
    )

    video_embedding, ocr, memes = video_processing_result.return_value
    stt = sst_extraction_result.return_value
    audio_embedding = audio_embedding_extraction_result.return_value

    shutil.rmtree(temp_dir)

    features = Features(
        video_embedding=video_embedding,
        audio_embedding=audio_embedding,
        ocr=ocr,
        speech_to_text=stt,
        memes=memes,
    )

    return features
