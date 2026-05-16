from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path.home() / ".codenames_solver"
CHROMA_DIR = DATA_DIR / "chroma"

EMBEDDING_MODEL = "voyage-4-large"
VISION_MODEL = "gpt-5.4"
COLLECTION_NAME = "english_words"

EMBEDDING_BATCH_SIZE = 128  # Voyage AI accepts up to 128 inputs per request
MIN_WORD_LEN = 3
MAX_WORD_LEN = 15

ASSASSIN_PENALTY = 0.3  # extra additive penalty if assassin is closer than avoid
RERANKER_MODEL = "claude-haiku-4-5-20251001"  # fast + cheap; swap for claude-sonnet-4-6 for higher quality
# How many of the top avoid-word similarities a clue must beat to count as covering a target.
# 1 = must beat every avoid word (strict); 2 = one avoid word may be closer (recommended).
AVOID_DANGER_TOP_K = 2
