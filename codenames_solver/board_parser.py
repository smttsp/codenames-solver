from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from codenames_solver.config import VISION_MODEL


class BoardColors(BaseModel):
    blue: list[str]
    red: list[str]
    other: list[str]  # neutral / tan cards
    black: list[str]  # assassin card


def parse_screenshot(image_path: str | Path, team: str = "blue") -> BoardColors:  # noqa: ARG001
    image_path = Path(image_path)
    raw = image_path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode()

    suffix = image_path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/png")

    model = ChatOpenAI(model=VISION_MODEL, temperature=0)
    structured = model.with_structured_output(BoardColors)

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
            {
                "type": "text",
                "text": (
                    "This is a Codenames board. Extract every visible word and classify it by card color:\n"
                    "- blue: blue team cards\n"
                    "- red: red team cards\n"
                    "- other: neutral/tan/beige cards\n"
                    "- black: the assassin card\n\n"
                    "Return all words in lowercase. Only include words whose card color is already revealed."
                ),
            },
        ]
    )

    result = structured.invoke([message])
    # Normalise to lowercase
    return BoardColors(
        blue=[w.lower() for w in result.blue],
        red=[w.lower() for w in result.red],
        other=[w.lower() for w in result.other],
        black=[w.lower() for w in result.black],
    )
