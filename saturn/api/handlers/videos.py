from typing import Annotated, Optional
from datetime import datetime
from fastapi import APIRouter, Depends
from pydantic import BaseModel, HttpUrl

from saturn.api.repositories.EmbeddingsRepository import EmbeddingsRepository
from saturn.models.Features import Features
from saturn.processors.pipeline import feature_extraction
from saturn.processors.helpers import PerformanceMeasurer

router = APIRouter()


class UploadVideoBody(BaseModel):
    link: HttpUrl
    created_at: Optional[datetime] = None


class UploadVideoResponse(BaseModel):
    is_duplicate: bool
    duplicate_for: Optional[str | None] = None


def get_uuid_from_url(url: HttpUrl) -> str:
    return url.path.split("/")[-1].split(".")[0]


@router.post("")
async def process_video(
    body: UploadVideoBody,
    embeddings_repository: Annotated[EmbeddingsRepository, Depends()],
) -> UploadVideoResponse:
    with PerformanceMeasurer("POST /video"):
        id: str = get_uuid_from_url(body.link)

        features: Features

        if not await embeddings_repository.has(id=id):
            with PerformanceMeasurer("feature_extraction"):
                features = await feature_extraction(video_url=str(body.link))

            await embeddings_repository.add(
                id=id, features=features, created_at=body.created_at
            )
        else:
            features = await embeddings_repository.get_features(id=id)

        with PerformanceMeasurer("find_original_source"):
            is_duplicate, duplicate_for = (
                await embeddings_repository.find_original_source(
                    id=id, features=features
                )
            )

        return UploadVideoResponse(
            is_duplicate=is_duplicate, duplicate_for=duplicate_for
        )
