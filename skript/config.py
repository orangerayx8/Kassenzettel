from pathlib import Path


BASE_DIR = Path(r"C:\Users\micha\Dropbox\Apps\Server Kassenzettel")

INCOMING_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
CSV_DIR = BASE_DIR / "csv"

SCRIPT_DIR = BASE_DIR / "skript"
PROMPT_FILE = SCRIPT_DIR / "prompt.txt"
CATEGORIES_FILE = SCRIPT_DIR / "allowed_categories.txt"
MERCHANTS_FILE = SCRIPT_DIR / "allowed_merchants.txt"

REGISTRY_FILE = SCRIPT_DIR / "receipt_id_registry.csv"

VISION_KEY_FILE = SCRIPT_DIR / "vision_key.json"
OPENAI_KEY_FILE = SCRIPT_DIR / "openai_key.txt"

DEBUG_TABLE_DIR = SCRIPT_DIR / "debug_tables"
DEBUG_OCR_DIR = SCRIPT_DIR / "debug_vision_ocr"
DEBUG_LLM_DIR = SCRIPT_DIR / "debug_llm_raw"

OPENAI_MODEL = "gpt-4o-mini"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

DESKEW_MIN_ABS_DEG = 0.4
DESKEW_MAX_ABS_DEG = 12.0

LINE_Y_THRESHOLD_MULT = 0.60
X_GAP_MULT = 1.8
MIN_BUCKETS = 4
MAX_BUCKETS = 16

CSV_HEADER = [
    "receipt_id",
    "source_image",
    "imported_at",
    "merchant",
    "receipt_date",
    "item_index",
    "name",
    "category",
    "quantity",
    "unit_price",
    "total_price",
]
