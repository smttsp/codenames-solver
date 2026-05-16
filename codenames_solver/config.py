from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path.home() / ".codenames_solver"
CHROMA_DIR = DATA_DIR / "chroma"

EMBEDDING_MODEL = "voyage-3-large"
VISION_MODEL = "gpt-4o"
COLLECTION_NAME = "english_words"

EMBEDDING_BATCH_SIZE = 500  # OpenAI accepts up to 2048 inputs per request
MIN_WORD_LEN = 3
MAX_WORD_LEN = 15

# Scoring penalties (multiplied by cosine similarity)
AVOID_PENALTY = 1.0   # kept for API compat; no longer used as multiplier
ASSASSIN_PENALTY = 0.3  # extra additive penalty if assassin is closer than avoid
