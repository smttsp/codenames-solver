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

# Coverage margin: a target only counts as "covered" if its similarity beats the
# max avoid/assassin similarity by this margin. Prevents phantom coverage.
COVERAGE_MARGIN = 0.02

# Drop candidate clues whose corpus frequency (Brown+Reuters+Webtext) is below
# this floor. Filters out obscure/archaic words and proper-noun residue.
MIN_CANDIDATE_FREQ = 5
# Training-time floor (more lenient; the runtime filter still applies later).
MIN_TRAINING_FREQ = 2
