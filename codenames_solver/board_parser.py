from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from codenames_solver.config import VISION_MODEL

_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

_BOARD_PROMPT = (
    "This is a Codenames board. Extract every visible word and classify it by card color:\n"
    "- blue: blue team cards\n"
    "- red: red team cards\n"
    "- other: neutral/tan/beige cards\n"
    "- black: the assassin card\n\n"
    "Return all words in lowercase. Only include words whose card color is already revealed."
)


class BoardColors(BaseModel):
    blue: list[str]
    red: list[str]
    other: list[str]  # neutral / tan cards
    black: list[str]  # assassin card


class BoardParser:
    def __init__(self, model: str = VISION_MODEL) -> None:
        llm = ChatOpenAI(model=model, temperature=0)
        self._model = llm.with_structured_output(BoardColors)

    def parse(self, image_path: str | Path) -> BoardColors:
        image_path = Path(image_path)
        b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
        mime = _MIME_MAP.get(image_path.suffix.lower(), "image/png")

        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": _BOARD_PROMPT},
            ]
        )
        result = self._model.invoke([message])
        return BoardColors(
            blue=[w.lower() for w in result.blue],
            red=[w.lower() for w in result.red],
            other=[w.lower() for w in result.other],
            black=[w.lower() for w in result.black],
        )
