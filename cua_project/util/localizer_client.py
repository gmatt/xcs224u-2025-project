from typing import TypedDict

import requests

LOCALIZE_URL = "http://localhost:8001/localize"


class Coordinates(TypedDict):
    x: float
    y: float


class LocalizerClient:
    def localize(
        self,
        image: bytes,
        label: str,
    ) -> dict:
        response = requests.post(
            LOCALIZE_URL,
            files={"image": image},
            data={"label": label},
        )
        response.raise_for_status()
        coordinates = response.json()
        return coordinates
