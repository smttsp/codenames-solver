from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path.home() / ".codenames_solver"
CHROMA_DIR = DATA_DIR / "chroma"

EMBEDDING_MODEL = "text-embedding-3-small"
VISION_MODEL = "gpt-4o"
COLLECTION_NAME = "english_words"

EMBEDDING_BATCH_SIZE = 500  # OpenAI accepts up to 2048 inputs per request
MIN_WORD_LEN = 3
MAX_WORD_LEN = 15

# Scoring penalties (multiplied by cosine similarity)
AVOID_PENALTY = 1.5  # opponent + neutral words
ASSASSIN_PENALTY = 5.0  # instant-loss assassin word
