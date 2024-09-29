import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import Annotated, Tuple
from fastapi import Depends
from chromadb import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection

from saturn.libs.chroma import get_chroma_connection
from saturn.models.Features import Features
from saturn.processors.extract_text_by_ocr import clean_text


async def get_videos_collection(
    db: Annotated[AsyncClientAPI, Depends(get_chroma_connection)]
):
    collection = await db.get_or_create_collection(
        name="videos", metadata={"hnsw:space": "cosine", "hnsw:M": 1024}
    )

    yield collection


async def get_audios_collection(
    db: Annotated[AsyncClientAPI, Depends(get_chroma_connection)]
):
    collection = await db.get_or_create_collection(
        name="audios", metadata={"hnsw:space": "cosine", "hnsw:M": 1024}
    )

    yield collection


async def get_screen_texts_collection(
    db: Annotated[AsyncClientAPI, Depends(get_chroma_connection)]
):
    collection = await db.get_or_create_collection(
        name="screen_texts", metadata={"hnsw:M": 1024}
    )

    yield collection


async def get_speechs_collection(
    db: Annotated[AsyncClientAPI, Depends(get_chroma_connection)]
):
    collection = await db.get_or_create_collection(
        name="speechs", metadata={"hnsw:M": 1024}
    )

    yield collection


class EmbeddingsRepository:
    db: AsyncClientAPI
    videos: AsyncCollection

    def __init__(
        self,
        db: Annotated[AsyncClientAPI, Depends(get_chroma_connection)],
        videos: Annotated[AsyncCollection, Depends(get_videos_collection)],
        audios: Annotated[AsyncCollection, Depends(get_audios_collection)],
        screen_texts: Annotated[AsyncCollection, Depends(get_screen_texts_collection)],
        speechs: Annotated[AsyncCollection, Depends(get_speechs_collection)],
    ) -> None:
        self.db = db
        self.videos = videos
        self.audios = audios
        self.screen_texts = screen_texts
        self.speechs = speechs

    async def add(self, id: str, features: Features, created_at: datetime | None):
        created_at = (created_at or datetime.now()).timestamp()

        await self.videos.add(
            ids=[id],
            embeddings=[features.video_embedding],
            metadatas=[
                {
                    "id": id,
                    "created_at": created_at,
                    "has_meme": features.memes["has_meme"],
                }
            ],
        )

        await self.audios.add(
            ids=[id],
            embeddings=[features.audio_embedding],
            metadatas=[
                {
                    "id": id,
                    "created_at": created_at,
                    "has_meme": features.memes["has_meme"],
                }
            ],
        )

        await self.screen_texts.add(
            ids=[id],
            documents=[features.ocr],
            metadatas=[
                {
                    "id": id,
                    "created_at": created_at,
                    "has_meme": features.memes["has_meme"],
                }
            ],
        )

        await self.speechs.add(
            ids=[id],
            documents=[features.speech_to_text],
            metadatas=[
                {
                    "id": id,
                    "created_at": created_at,
                    "has_meme": features.memes["has_meme"],
                }
            ],
        )

    async def get_features(self, id: str) -> Features:
        videos = await self.videos.get(ids=[id], include=["metadatas", "embeddings"])
        if len(videos["embeddings"]) < 1:
            raise LookupError(
                f"not found video embeddings of video {id} for getting features"
            )

        audios = await self.audios.get(ids=[id], include=["metadatas", "embeddings"])
        if len(audios["embeddings"]) < 1:
            raise LookupError(
                f"not found audio embeddings of video {id} for getting features"
            )

        screen_texts = await self.screen_texts.get(ids=[id])
        ocr = screen_texts["documents"][0] if len(screen_texts["documents"]) > 0 else ""

        speechs = await self.speechs.get(ids=[id])
        speech_to_text = (
            speechs["documents"][0] if len(speechs["documents"]) > 0 else ""
        )

        return Features(
            video_embedding=videos["embeddings"][0],
            audio_embedding=audios["embeddings"][0],
            ocr=ocr,
            speech_to_text=speech_to_text,
            memes=videos["metadatas"][0]["has_meme"],
        )


    async def has(self, id: str):
        found = await self.videos.get(ids=[id])

        return len(found["ids"]) > 0

    async def get_creation_datetime(self, id: str) -> float:
        found = await self.videos.get(ids=[id])

        if len(found["ids"]) < 1:
            raise LookupError("not found video for getting creation datetime")

        if not isinstance(found["metadatas"][0]["created_at"], float):
            raise TypeError("video metadatas.created_at is not float")

        return found["metadatas"][0]["created_at"]

    async def find_original_source(
        self, id: str, features: Features
    ) -> Tuple[bool, str | None]:
        created_at = await self.get_creation_datetime(id=id)

        pick_items = 200

        videos = await self.videos.query(
            where={
                "$and": [
                    {"id": {"$ne": id}},
                    {"created_at": {"$lte": created_at}},
                ]
            },
            query_embeddings=[features.video_embedding],
            n_results=pick_items,
            include=["metadatas", "distances", "embeddings"],
        )

        if (len(videos["ids"])) < 2:
            return False, None

        audios = await self.audios.query(
            where={
                "$and": [
                    {"id": {"$ne": id}},
                    {"created_at": {"$lte": created_at}},
                ]
            },
            query_embeddings=[features.audio_embedding],
            n_results=pick_items,
            include=["distances", "embeddings"],
        )

        screen_texts = await self.screen_texts.query(
            where={
                "$and": [
                    {"id": {"$ne": id}},
                    {"created_at": {"$lte": created_at}},
                ]
            },
            query_texts=[features.ocr],
            n_results=pick_items,
            include=["distances", "documents"],
        )

        speechs = await self.speechs.query(
            where={
                "$and": [
                    {"id": {"$ne": id}},
                    {"created_at": {"$lte": created_at}},
                ]
            },
            query_texts=[features.speech_to_text],
            n_results=pick_items,
            include=["distances", "documents"],
        )

        candidates = pd.DataFrame(
            {
                "uuid": [ids[0] for ids in videos["ids"]],
                "metadata": [metadatas[0] for metadatas in videos["metadatas"]],
                "video_similarity": [
                    (1 - distances[0]) for distances in videos["distances"]
                ],
                "video_embedding": [
                    embeddings[0] for embeddings in videos["embeddings"]
                ],
                "audio_similarity": [
                    (1 - distances[0]) for distances in audios["distances"]
                ],
                "audio_embedding": [
                    embeddings[0] for embeddings in audios["embeddings"]
                ],
                "screen_text_similarity": [
                    (1 - distances[0]) for distances in screen_texts["distances"]
                ],
                "screen_text": [
                    documents[0] for documents in screen_texts["documents"]
                ],
                "speech_similarity": [
                    (1 - distances[0]) for distances in speechs["distances"]
                ],
                "speech": [documents[0] for documents in speechs["documents"]],
            }
        )

        video_threshold = 0.90 if max(candidates["video_similarity"]) >= 0.90 else 0.75
        audio_threshold = 0.95 if max(candidates["audio_similarity"]) >= 0.95 else 0.80

        candidates = candidates[
            (
                (
                    (candidates["audio_similarity"] > audio_threshold)
                    & (candidates["video_similarity"] > 0.4)
                )
                | (candidates["video_similarity"] > video_threshold)
            )
        ]

        source_uuid = self._find_source_uuid(candidates, features)

        if source_uuid is not None:
            return True, source_uuid
        else:
            return False, None

    def _find_source_uuid(self, all_candidates: pd.DataFrame, features: Features):
        candidate_uuid = None

        if len(all_candidates) >= 1:
            ratios_stt = [
                fuzz.token_set_ratio(features.speech_to_text, x)
                for x in all_candidates["speech"]
            ]

            if max(ratios_stt) > 85:
                all_candidates = all_candidates[np.array(ratios_stt) > 85]

        if len(all_candidates) >= 1 and features.ocr != " ":
            candidates_texts = all_candidates[
                all_candidates["screen_text"] != " "
            ].copy()
            candidates_texts["screen_text"] = candidates_texts["screen_text"].map(
                clean_text
            )

            ratios_ocr = [
                fuzz.token_set_ratio(clean_text(features.ocr), x)
                for x in candidates_texts
            ]

            if max(ratios_ocr) >= 80:
                all_candidates = candidates_texts[np.array(ratios_ocr) >= 80]

        if len(all_candidates) > 1:
            video_similarity = cosine_similarity(
                np.stack(all_candidates["video_embedding"].values),
                features.video_embedding.reshape(1, -1),
            ).flatten()

            audio_similarity = cosine_similarity(
                np.stack(all_candidates["audio_embedding"].values),
                features.audio_embedding.reshape(1, -1),
            ).flatten()

            mixed_similarity = (audio_similarity + 2 * video_similarity) / 3

            if max(mixed_similarity) >= 0.7:
                candidate_text = all_candidates[
                    np.array(mixed_similarity) == max(mixed_similarity)
                ]["speech"].iloc[0]

                if (
                    candidate_text is not None
                    and len(set(candidate_text.lower().split())) > 20
                    and fuzz.token_set_ratio(candidate_text, features.speech_to_text)
                    < 70
                ):
                    pass
                else:
                    candidate_uuid = all_candidates["uuid"].tolist()[
                        np.argmax(mixed_similarity)
                    ]
        elif len(all_candidates) == 1:
            candidate_text = all_candidates["speech"].iloc[0]

            if (
                candidate_text is not None
                and len(set(candidate_text.lower().split())) > 20
                and fuzz.token_set_ratio(candidate_text, features.speech_to_text) < 70
            ):
                pass
            else:
                candidate_uuid = all_candidates.iloc[0]["uuid"]

        return candidate_uuid
