from dataclasses import dataclass
from typing import List


@dataclass
class MemeCoordinates:
    x: int
    y: int
    width: int
    height: int


@dataclass
class MemesFeature:
    """
    has_meme: bool - есть ли мем на изображении;
    coordinates: List[MemeCoordinates] - список координат найденых мемов в формате ;
    """

    has_meme: bool
    coordinates: List[MemeCoordinates]
